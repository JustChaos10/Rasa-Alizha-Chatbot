#!/usr/bin/env python3
"""
Knowledge Base MCP Server - Document Q&A with Schema-Driven Visualization

Architecture: "Brutal Efficiency Refactor"
- Single unified LLM call for answer + visualization config
- Multi-provider LLM with automatic failover (GROQ primary, Gemini fallback)
- Schema-driven chart rendering (no LLM code generation)
- FAISS vector store with sentence-transformers embeddings

Tools:
- knowledgebase_query: Answer questions about documents (budget speech, policies, etc.)

Transport: stdio (auto-managed by Flask app)
"""

import sys
import os
import asyncio
import json
import logging
import time
import pickle
import re
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
load_dotenv()

# Use sentence-transformers directly (avoids NumPy conflicts)
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization_tool import SmartVisualizationTool, create_chart_from_config, parse_sql_result_to_data

# Import llm_utils directly to avoid triggering architecture/__init__.py
import importlib.util
_llm_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'architecture', 'llm_utils.py')
_spec = importlib.util.spec_from_file_location('llm_utils', _llm_utils_path)
_llm_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm_utils)
MultiProviderLLM = _llm_utils.MultiProviderLLM
get_llm = _llm_utils.get_llm


def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "knowledgebase_server.log"
    
    logger = logging.getLogger("knowledgebase-server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("[KB] %(message)s"))

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


logger = setup_logging()


class KBConfig:
    docs_path = os.getenv("KB_DOCS_PATH", "documents/")
    vectordb_path = os.getenv("KB_VECTORDB_PATH", "data/vectordb")
    embedding_model = os.getenv("KB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("KB_CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("KB_CHUNK_OVERLAP", "100"))
    top_k = int(os.getenv("KB_TOP_K", "5"))
    llm_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.getenv("KB_TEMPERATURE", "0"))


class VectorStoreRetriever:
    """Simple vector store using sentence-transformers and numpy (no langchain dependencies)."""
    
    def __init__(self, config):
        self.config = config
        self.vectordb = None
        self.embeddings_model = None
        self.documents = []
        self.embeddings = None
        
        try:
            self.embeddings_model = SentenceTransformer(config.embedding_model)
            logger.info(f"Loaded embedding model: {config.embedding_model}")
            self.vectordb = self._initialize_vector_store()
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            import traceback
            traceback.print_exc()
            self.vectordb = None

    def _initialize_vector_store(self):
        vectordb_path = Path(self.config.vectordb_path)
        
        # Try to load existing embeddings
        embeddings_file = vectordb_path / "embeddings.pkl"
        if embeddings_file.exists():
            try:
                logger.info(f"Loading existing embeddings from {embeddings_file}")
                with open(embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = data['embeddings']
                logger.info(f"Loaded {len(self.documents)} document chunks")
                return True
            except Exception as e:
                logger.warning(f"Could not load existing embeddings: {e}")

        # Load from FAISS if available (convert to simple format)
        faiss_index = vectordb_path / "index.faiss"
        faiss_pkl = vectordb_path / "index.pkl"
        if faiss_index.exists() and faiss_pkl.exists():
            try:
                logger.info("Loading from existing FAISS store...")
                import faiss
                with open(faiss_pkl, 'rb') as f:
                    faiss_data = pickle.load(f)
                
                if hasattr(faiss_data, 'docstore'):
                    docs = []
                    for idx in range(len(faiss_data.index_to_docstore_id)):
                        doc_id = faiss_data.index_to_docstore_id[idx]
                        doc = faiss_data.docstore.search(doc_id)
                        if doc:
                            docs.append({
                                'content': doc.page_content,
                                'metadata': doc.metadata
                            })
                    self.documents = docs
                    
                    index = faiss.read_index(str(faiss_index))
                    self.embeddings = index.reconstruct_n(0, index.ntotal)
                    
                    self._save_embeddings(vectordb_path)
                    logger.info(f"Converted FAISS store: {len(self.documents)} chunks")
                    return True
            except Exception as e:
                logger.warning(f"Could not convert FAISS store: {e}")

        # Build new embeddings from documents
        return self._build_from_documents()

    def _build_from_documents(self):
        docs_path = Path(self.config.docs_path)
        if not docs_path.exists():
            logger.warning(f"Documents directory not found: {docs_path}")
            return None

        try:
            logger.info(f"Loading documents from {docs_path}")
            raw_texts = []
            
            # Load PDFs
            for pdf_file in docs_path.glob("**/*.pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(pdf_file))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    if text.strip():
                        raw_texts.append({
                            'content': text,
                            'metadata': {'source': str(pdf_file)}
                        })
                except Exception as e:
                    logger.warning(f"Error loading PDF {pdf_file}: {e}")

            # Load text files
            for txt_file in docs_path.glob("**/*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if text.strip():
                        raw_texts.append({
                            'content': text,
                            'metadata': {'source': str(txt_file)}
                        })
                except Exception as e:
                    logger.warning(f"Error loading text file {txt_file}: {e}")

            if not raw_texts:
                logger.warning("No documents found to index")
                return None

            logger.info(f"Loaded {len(raw_texts)} raw documents")

            # Simple text splitting
            self.documents = []
            chunk_size = self.config.chunk_size
            chunk_overlap = self.config.chunk_overlap
            
            for doc in raw_texts:
                text = doc['content']
                chunks = self._split_text(text, chunk_size, chunk_overlap)
                for chunk in chunks:
                    self.documents.append({
                        'content': chunk,
                        'metadata': doc['metadata']
                    })
            
            logger.info(f"Split into {len(self.documents)} chunks")

            # Generate embeddings
            texts = [d['content'] for d in self.documents]
            self.embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
            
            # Save embeddings
            vectordb_path = Path(self.config.vectordb_path)
            self._save_embeddings(vectordb_path)
            
            return True

        except Exception as e:
            logger.error(f"Error building embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text splitting by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_text = ' '.join(current_chunk)[-overlap:] if overlap > 0 else ''
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]

    def _save_embeddings(self, vectordb_path: Path):
        """Save embeddings to disk."""
        vectordb_path.mkdir(parents=True, exist_ok=True)
        embeddings_file = vectordb_path / "embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        logger.info(f"Saved embeddings to {embeddings_file}")

    def retrieve(self, query: str) -> str:
        """Retrieve relevant documents for a query."""
        if not self.vectordb or self.embeddings is None:
            return "Knowledge base not available. Please check configuration."

        try:
            # Encode query
            query_embedding = self.embeddings_model.encode([query])[0]
            
            # Compute cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k
            top_indices = np.argsort(similarities)[-self.config.top_k:][::-1]
            
            if len(top_indices) == 0:
                return "No relevant documents found for your query."

            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                content = doc['content'].strip()
                source = doc['metadata'].get('source', 'Unknown')
                
                source_info = f"[Source: {Path(source).name}]"
                results.append(f"{content}\n{source_info}")

            return "\n\n---\n\n".join(results)

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error retrieving documents: {str(e)}"


class RAGSystem:
    """
    RAG System with SmartVisualizationTool for intelligent chart decisions.
    
    Flow:
    1. User query → Retrieve relevant documents (embedding search, no LLM)
    2. Context + query → LLM generates answer
    3. SmartVisualizationTool decides if/how to visualize (LLM-powered)
    
    Visualization is automatic when data is suitable - no explicit keywords needed.
    """
    
    def __init__(self, config):
        self.config = config
        self.retriever = VectorStoreRetriever(config)
        self.charts_dir = "charts/kb"
        
        # Ensure charts directory exists
        Path(self.charts_dir).mkdir(parents=True, exist_ok=True)

        # Initialize Multi-Provider LLM
        self.llm = MultiProviderLLM(
            groq_model=config.llm_model,
            temperature=config.temperature,
            max_tokens=1024
        )
        logger.info(f"✅ MultiProviderLLM initialized")
        logger.info(f"   LLM Status: {self.llm.get_status()}")
        
        # Initialize SmartVisualizationTool for automatic chart decisions
        self.viz_tool = SmartVisualizationTool(self.llm, charts_dir=self.charts_dir)
        logger.info(f"✅ SmartVisualizationTool initialized (auto-detect charts)")

        # Answer-only prompt - visualization is handled separately by SmartVisualizationTool
        self.answer_prompt = """You are an expert AI assistant. Answer the user's question based ONLY on the provided context.

CONTEXT:
{context}

USER'S QUESTION: {query}

CRITICAL RULES:
1. Answer must be based ONLY on the provided context - never speculate
2. If context lacks relevant information, say "The information is not available in the provided context."
3. Answer must be plain text - NO tables, NO markdown formatting, NO ASCII art
4. Keep answer to MAX 100 words, 3-5 sentences
5. Include specific numbers/dates/names from context when available
6. If the data has structure (like tax brackets, rates, categories), LIST THEM CLEARLY with their values
7. Be comprehensive - include ALL relevant data points from the context

Answer:"""

    async def _check_visualization_requested_llm(self, query: str) -> bool:
        """Use LLM to understand if visualization is requested (language-agnostic)."""
        prompt = f"""Does the user want a chart, graph, visualization, table, or diagram?

User Query: {query}

Respond with ONLY: YES or NO"""
        try:
            response = await self.llm.invoke(prompt)
            if response.success:
                return response.content.strip().upper().startswith('YES')
        except Exception as e:
            logger.warning(f"LLM visualization detection failed: {e}")
        return self._check_visualization_requested_keywords(query)
    
    def _check_visualization_requested_keywords(self, query: str) -> bool:
        """Fallback: Check for visualization keywords (supports English and Arabic)."""
        query_lower = query.lower()
        # English keywords
        viz_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'visualization', 'diagram',
            'table', 'tabular', 'show me', 'display', 'make a', 'create a',
            'bar', 'pie', 'line', 'histogram', 'scatter'
        ]
        # Arabic keywords for chart/visualization
        arabic_keywords = [
            'رسم', 'مخطط', 'جدول', 'بياني', 'رسم بياني', 'اعرض', 'أظهر', 'عرض',
            'توضيح', 'شكل', 'إحصائية', 'احصائيات', 'تصور', 'رسوم', 'انشئ'
        ]
        all_keywords = viz_keywords + arabic_keywords
        return any(kw in query_lower or kw in query for kw in all_keywords)

    def _create_fallback_answer(self, context: str) -> str:
        """
        Create a useful fallback answer from retrieved context when LLM fails.
        Extracts key information without requiring an LLM call.
        """
        docs = context.split("---")
        
        if not docs or not context.strip():
            return "I couldn't find relevant information for your query. Please try rephrasing your question."
        
        # Extract first document's content (most relevant)
        first_doc = docs[0].strip()
        
        # Clean up the document text
        lines = first_doc.split("\n")
        content_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("[Source") or not line:
                continue
            content_lines.append(line)
        
        # Get first meaningful content (up to 300 chars)
        content = " ".join(content_lines)[:300]
        
        if not content:
            return "I found some documents but couldn't extract the relevant information. Please try again."
        
        return f"Based on the documents: {content}..."

    async def generate_answer(self, query: str, state_id: str) -> Tuple[str, Optional[str]]:
        """
        Generate answer with SmartVisualizationTool for automatic chart decisions.
        
        Flow:
        1. Retrieve relevant documents (embedding search, no LLM)
        2. LLM call for answer generation
        3. SmartVisualizationTool decides if/how to visualize (LLM-powered)
        
        Returns:
            Tuple of (answer_text, chart_path or None)
        """
        if not self.retriever.vectordb:
            return "Knowledge base not properly initialized.", None

        logger.info(f"Processing KB query: {query[:100]}... [state_id={state_id}]")

        try:
            # Step 1: Retrieve relevant context (no LLM call)
            context = self.retriever.retrieve(query)
            
            if "not available" in context.lower() or "error" in context.lower():
                return context, None

            # Step 2: Build prompt and call LLM for answer
            prompt = self.answer_prompt.format(
                context=context[:3000],  # Limit context size
                query=query
            )
            
            # Call LLM (with automatic failover)
            response = await self.llm.invoke(prompt)
            
            if not response.success:
                logger.warning(f"LLM call failed: {response.error}")
                return self._create_fallback_answer(context), None
            
            answer = response.content.strip()
            logger.info(f"✅ KB answer generated via {response.provider.value}")
            
            # Step 3: Use SmartVisualizationTool to decide and create visualization
            # Pass both the query and the answer for intelligent chart decision
            chart_path = None
            viz_requested = await self._check_visualization_requested_llm(query)
            
            try:
                viz_result = await self.viz_tool.create_visualization(
                    query=query,
                    result=answer,  # Pass the answer as data source
                    query_id=state_id,
                    force_visualize=viz_requested  # Force visualization if explicitly requested
                )
                
                if viz_result.get("success") and viz_result.get("chart_path"):
                    chart_path = viz_result["chart_path"]
                    logger.info(f"✅ SmartVisualizationTool created chart: {chart_path}")
                elif viz_result.get("message"):
                    logger.info(f"Visualization: {viz_result['message']}")
                    
            except Exception as viz_error:
                logger.warning(f"Visualization attempt failed: {viz_error}")
                # Continue without visualization - answer is still valid

            return answer, chart_path

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {str(e)}", None


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("knowledgebase-server")
config = KBConfig()
rag_system = RAGSystem(config)


@mcp.tool()
async def knowledgebase_query(query: str, state_id: str = "default") -> str:
    """
    Answer questions about documents in the knowledge base with automatic visualization.
    
    IMPORTANT: Use this tool for ANY question about:
    - Budget speech, tax brackets, tax slabs, revised tax structure, fiscal policies
    - Financial reports, policy documents, government announcements
    - Company policies, procedures, guidelines, HR documents
    - Any uploaded documents (PDF, TXT, DOCX)
    
    This tool should be used when:
    - User asks about content from uploaded documents
    - User mentions "budget", "tax", "policy", "document", "speech", "announcement"
    - User asks to "make a table of" or "show" or "visualize" data from documents
    - User asks about tax rates, tax structure, tax brackets
    
    DO NOT use SQL tool for these queries - they are about document content, not database data.
    
    Args:
        query: The user's question about the documents
        state_id: Session identifier for tracking
        
    Returns:
        JSON string with answer and optional chart path (auto-generated when data is suitable)
    """
    try:
        logger.info(f"KB query received: '{query}' [state_id={state_id}]")
        answer, viz_path = await rag_system.generate_answer(query, state_id)
        response = {"answer": answer, "chart": viz_path}
        logger.info(f"KB response [state_id={state_id}]: chart={viz_path}")
        return json.dumps(response)
    except Exception as e:
        logger.error(f"KB query error [state_id={state_id}]: {e}", exc_info=True)
        return json.dumps({"answer": f"Error processing knowledge base query: {str(e)}", "chart": None})


@mcp.tool()
async def knowledgebase_health() -> str:
    """Check the health status of the knowledge base server."""
    try:
        status = {
            "server": "KnowledgeBase MCP Server",
            "status": "healthy" if rag_system.retriever.vectordb else "degraded",
            "vectordb_loaded": rag_system.retriever.vectordb is not None,
            "documents_count": len(rag_system.retriever.documents),
            "llm_status": rag_system.llm.get_status(),
            "documents_path": str(config.docs_path),
            "vectordb_path": str(config.vectordb_path),
            "top_k": config.top_k,
            "charts_dir": rag_system.charts_dir
        }
        return json.dumps(status)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


if __name__ == "__main__":
    logger.info("Starting Knowledge Base MCP Server (stdio)...")
    logger.info("Architecture: Schema-Driven Visualization + Multi-Provider LLM")
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
