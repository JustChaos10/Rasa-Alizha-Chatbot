#!/usr/bin/env python3
"""
SQL MCP Server - Natural Language to SQL with Schema-Driven Visualization

Architecture: "Brutal Efficiency Refactor"
- Single unified LLM call for summary + visualization config
- Multi-provider LLM with automatic failover (GROQ primary, Gemini fallback)
- Schema-driven chart rendering (no LLM code generation)
- Heuristic-based schema pruning (no LLM table selection)

Tools:
- sql_query: Convert natural language questions to SQL and execute them

Transport: stdio (auto-managed by Flask app)
"""

import sys
import os
import asyncio
import json
import logging
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import SQLDatabase
from mcp.server.fastmcp import FastMCP
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization_tool import create_chart_from_config, parse_sql_result_to_data

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
    log_file = logs_dir / "sql_server.log"
    
    logger = logging.getLogger("sql-server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("[SQL] %(message)s"))

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


class SQLConfig:
    # Support both PG_* and DB_* environment variable naming conventions
    pg_host = os.getenv("PG_HOST", os.getenv("DB_HOST", "localhost"))
    pg_port = os.getenv("PG_PORT", os.getenv("DB_PORT", "5432"))
    pg_database = os.getenv("PG_DATABASE", os.getenv("DB_NAME", "adventureworks"))
    pg_user = os.getenv("PG_USER", os.getenv("DB_USER", "dhrutipurushotham"))
    pg_password = os.getenv("PG_PASSWORD", os.getenv("DB_PASSWORD", ""))
    # AdventureWorks uses multiple schemas - we'll include the main ones
    pg_schema = os.getenv("PG_SCHEMA", "production,sales,person,humanresources,purchasing")
    # LLM settings
    llm_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature = float(os.getenv("SQL_TEMPERATURE", "0"))
    pool_size = int(os.getenv("SQL_POOL_SIZE", "5"))
    max_retries = int(os.getenv("SQL_MAX_RETRIES", "3"))

    @classmethod
    def get_db_uri(cls):
        return f"postgresql://{cls.pg_user}:{cls.pg_password}@{cls.pg_host}:{cls.pg_port}/{cls.pg_database}"


class SQLServer:
    """
    SQL Server with schema-driven visualization and multi-provider LLM.
    
    Flow:
    1. User query → Heuristic schema pruning (no LLM)
    2. Pruned schema + query → LLM generates SQL (1 LLM call)
    3. Execute SQL → Get raw results
    4. Results + query → LLM returns JSON {summary, visualization} (1 LLM call)
    5. If visualization config present → Schema-driven chart rendering (no LLM)
    
    Total LLM calls per query: 2 (SQL generation + unified analysis)
    """
    
    def __init__(self, config):
        self.config = config
        self.db_uri = config.get_db_uri()
        self.target_schemas = [s.strip() for s in config.pg_schema.split(",") if s.strip()]
        self.db_executor = ThreadPoolExecutor(max_workers=config.pool_size)
        self.pool = asyncio.Queue(maxsize=config.pool_size)
        self._schema_cache = {}
        self._cache_time = {}
        self._table_list_cache = None
        self.initialized = False
        self.charts_dir = "charts/sql"
        
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

        # ====================================================================
        # PROMPTS
        # ====================================================================
        
        # Prompt for generating SQL (LLM Call #1)
        self.sql_prompt = """You are a PostgreSQL expert. Generate an EXACT, CORRECT SQL query.

Database Schema (ONLY these tables exist):
{schema}

CRITICAL RULES:
1. Return ONLY the SQL query - no explanations, no markdown code blocks
2. Use ONLY tables and columns from the schema above - DO NOT invent tables that don't exist
3. ALWAYS include schema prefix (e.g., sales.salesorderdetail NOT salesorderdetail)
4. For sales/orders data, use sales.salesorderdetail (has productid, orderqty, unitprice) and sales.salesorderheader
5. For product info, use production.product (has productid, name, productnumber)
6. DO NOT use tables that aren't in the schema (no productsales, no productsubcategoryhistory)
7. Use proper PostgreSQL syntax with GROUP BY for aggregations
8. Add ORDER BY and LIMIT 15 for meaningful results

User Question: {question}

SQL Query:"""

        # Unified analysis prompt (LLM Call #2) - Returns JSON with summary + visualization
        self.unified_analysis_prompt = """Analyze these SQL query results and respond with JSON.

User Question: {question}

SQL Results:
{result}

USER EXPLICITLY REQUESTED CHART: {chart_requested}

Respond with ONLY this JSON structure (no markdown, no explanation):
{{
    "summary": "A 2-3 sentence summary answering the user's question. Include key numbers. MAX 50 words.",
    "visualization": {viz_instruction}
}}

RULES:
1. Summary must directly answer the question in plain text (no tables, no markdown)
2. Summary must be concise - maximum 50 words, 2-3 sentences
3. Include specific numbers from the results in the summary
4. Visualization is ONLY included if user explicitly asked for a chart OR data has 3+ rows of comparable numbers
5. For visualization.data, use format {{"label1": number1, "label2": number2, ...}}
6. Visualization type should be "bar" for comparisons, "pie" for distributions, "line" for trends

JSON Response:"""

    async def initialize(self):
        """Initialize database connection pool."""
        if self.initialized:
            return
        try:
            logger.info(f"Connecting to PostgreSQL: {self.config.pg_host}:{self.config.pg_port}/{self.config.pg_database}")
            logger.info(f"Target schemas: {self.target_schemas}")
            for _ in range(self.config.pool_size):
                db = SQLDatabase.from_uri(
                    self.db_uri,
                    sample_rows_in_table_info=3
                )
                await self.pool.put(db)
            self.initialized = True
            logger.info(f"Database pool initialized ({self.config.pool_size} connections)")
            
            # Pre-fetch and cache schema on startup
            schema = await self.get_schema()
            logger.info(f"Schema cached: {len(schema)} characters")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def get_connection(self):
        if not self.initialized:
            await self.initialize()
        return await asyncio.wait_for(self.pool.get(), timeout=30)

    async def release_connection(self, db):
        await self.pool.put(db)

    async def get_schema(self):
        """Get database schema for specified schemas using direct SQL queries."""
        cache_key = "schema"
        cache_ttl = 3600
        if cache_key in self._schema_cache:
            if time.time() - self._cache_time.get(cache_key, 0) < cache_ttl:
                return self._schema_cache[cache_key]
        
        db = await self.get_connection()
        try:
            loop = asyncio.get_event_loop()
            
            def fetch_schema_info():
                conn = psycopg2.connect(
                    host=self.config.pg_host,
                    port=self.config.pg_port,
                    database=self.config.pg_database,
                    user=self.config.pg_user,
                    password=self.config.pg_password
                )
                cur = conn.cursor()
                
                schema_parts = []
                for schema_name in self.target_schemas:
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = %s AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """, (schema_name,))
                    tables = [row[0] for row in cur.fetchall()]
                    
                    for table in tables[:20]:
                        cur.execute("""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s
                            ORDER BY ordinal_position
                        """, (schema_name, table))
                        columns = cur.fetchall()
                        
                        col_defs = [f"  {col[0]} {col[1]}" for col in columns[:15]]
                        schema_parts.append(f"TABLE {schema_name}.{table}:\n" + "\n".join(col_defs))
                
                conn.close()
                return "\n\n".join(schema_parts)
            
            schema = await loop.run_in_executor(self.db_executor, fetch_schema_info)
            self._schema_cache[cache_key] = schema
            self._cache_time[cache_key] = time.time()
            logger.info(f"Schema fetched: {len(schema)} chars, schemas: {self.target_schemas}")
            return schema
        finally:
            await self.release_connection(db)

    async def get_table_list(self) -> List[str]:
        """Get a simple list of all tables in the database."""
        if self._table_list_cache:
            return self._table_list_cache
            
        loop = asyncio.get_event_loop()
        
        def fetch_tables():
            conn = psycopg2.connect(
                host=self.config.pg_host,
                port=self.config.pg_port,
                database=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password
            )
            cur = conn.cursor()
            
            if self.target_schemas:
                schema_list = ",".join([f"'{s}'" for s in self.target_schemas])
                schema_filter = f"WHERE table_schema IN ({schema_list})"
            else:
                schema_filter = ""
            
            cur.execute(f"""
                SELECT DISTINCT table_schema || '.' || table_name as full_name
                FROM information_schema.tables
                {schema_filter}
                ORDER BY full_name
            """)
            tables = [row[0] for row in cur.fetchall()]
            conn.close()
            return tables
        
        self._table_list_cache = await loop.run_in_executor(self.db_executor, fetch_tables)
        logger.info(f"Table list cached: {len(self._table_list_cache)} tables")
        return self._table_list_cache

    async def get_schema_for_tables(self, table_names: List[str]) -> str:
        """Get schema information for specific tables only."""
        loop = asyncio.get_event_loop()
        
        def fetch_specific_schema():
            conn = psycopg2.connect(
                host=self.config.pg_host,
                port=self.config.pg_port,
                database=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password
            )
            cur = conn.cursor()
            schema_parts = []
            
            for full_table in table_names:
                if '.' in full_table:
                    schema_name, table_name = full_table.split('.', 1)
                else:
                    schema_name, table_name = 'public', full_table
                
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema_name, table_name))
                columns = cur.fetchall()
                
                if columns:
                    col_defs = [f"  {col[0]} {col[1]}" for col in columns]
                    schema_parts.append(f"TABLE {schema_name}.{table_name}:\n" + "\n".join(col_defs))
            
            conn.close()
            return "\n\n".join(schema_parts)
        
        return await loop.run_in_executor(self.db_executor, fetch_specific_schema)

    async def prune_schema(self, question: str) -> str:
        """
        Use heuristics to select only relevant tables for the query.
        NO LLM calls - pure keyword matching.
        """
        table_list = await self.get_table_list()
        question_lower = question.lower()
        
        # Common table mappings based on keywords
        keyword_tables = {
            'product': ['production.product', 'production.productcategory', 'production.productsubcategory'],
            'category': ['production.productcategory', 'production.productsubcategory'],
            'subcategory': ['production.productsubcategory', 'production.productcategory'],
            'order': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'sales': ['sales.salesorderheader', 'sales.salesorderdetail', 'sales.salesperson', 'production.product'],
            'sold': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'selling': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'best': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'top': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'revenue': ['sales.salesorderheader', 'sales.salesorderdetail', 'production.product'],
            'quantity': ['sales.salesorderdetail', 'production.product', 'production.productinventory'],
            'customer': ['sales.customer', 'person.person', 'sales.salesorderheader'],
            'employee': ['humanresources.employee', 'person.person'],
            'person': ['person.person', 'person.address'],
            'address': ['person.address', 'person.stateprovince'],
            'inventory': ['production.productinventory', 'production.product'],
            'vendor': ['purchasing.vendor', 'purchasing.productvendor'],
            'purchase': ['purchasing.purchaseorderheader', 'purchasing.purchaseorderdetail'],
            'territory': ['sales.salesterritory', 'sales.salesperson'],
        }
        
        matched_tables = set()
        for keyword, tables in keyword_tables.items():
            if keyword in question_lower:
                for t in tables:
                    if t in table_list:
                        matched_tables.add(t)
        
        if not matched_tables:
            default_tables = ['production.product', 'production.productcategory', 'production.productsubcategory']
            matched_tables = set(t for t in default_tables if t in table_list)
            logger.info(f"No keyword matches, using default tables: {matched_tables}")
        
        logger.info(f"Schema pruned (heuristic): {len(matched_tables)} tables")
        return await self.get_schema_for_tables(list(matched_tables))

    async def generate_sql(self, question: str) -> str:
        """
        Generate SQL query using LLM with pruned schema.
        LLM CALL #1
        """
        # Get pruned schema (heuristic, no LLM)
        schema = await self.prune_schema(question)
        logger.info(f"Using schema: {len(schema)} chars")
        
        prompt = self.sql_prompt.format(schema=schema, question=question)
        
        # Call LLM (with automatic failover)
        response = await self.llm.invoke(prompt)
        
        if not response.success:
            raise Exception(f"SQL generation failed: {response.error}")
        
        sql = response.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"✅ SQL generated via {response.provider.value}: {sql[:100]}...")
        return sql

    async def execute_sql(self, sql: str) -> str:
        """Execute SQL query and return results."""
        db = await self.get_connection()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.db_executor, db.run, sql)
            return result
        finally:
            await self.release_connection(db)

    async def _check_chart_requested_llm(self, query: str) -> bool:
        """Use LLM to understand if visualization is requested (language-agnostic)."""
        prompt = f"""Does the user want a chart, graph, visualization, or table?

User Query: {query}

Respond with ONLY: YES or NO"""
        try:
            response = await self.llm.invoke(prompt)
            if response.success:
                return response.content.strip().upper().startswith('YES')
        except Exception as e:
            logger.warning(f"LLM chart detection failed: {e}")
        return self._check_chart_requested_keywords(query)
    
    def _check_chart_requested_keywords(self, query: str) -> bool:
        """Fallback: Check for chart keywords (supports English and Arabic)."""
        query_lower = query.lower()
        # English keywords
        chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'diagram', 'show', 'display']
        # Arabic keywords for chart/visualization
        arabic_keywords = [
            'رسم', 'مخطط', 'جدول', 'بياني', 'رسم بياني', 'اعرض', 'أظهر', 'عرض',
            'توضيح', 'شكل', 'إحصائية', 'احصائيات', 'تصور', 'رسوم'
        ]
        all_keywords = chart_keywords + arabic_keywords
        return any(kw in query_lower or kw in query for kw in all_keywords)

    async def _analyze_results(self, question: str, result: str) -> Tuple[str, Optional[Dict]]:
        """
        Analyze SQL results and return summary + optional visualization config.
        LLM CALL #2 - Single unified call for both summary and chart config.
        
        Returns:
            Tuple of (summary_text, visualization_config or None)
        """
        chart_requested = await self._check_chart_requested_llm(question)
        
        # Build visualization instruction based on whether chart was requested
        if chart_requested:
            viz_instruction = '{"type": "bar|pie|line", "data": {"label": value, ...}, "title": "Chart Title"} OR null if data not suitable'
        else:
            viz_instruction = 'null (only include if data has 5+ comparable rows)'
        
        prompt = self.unified_analysis_prompt.format(
            question=question,
            result=result[:2000],  # Limit result size
            chart_requested="YES" if chart_requested else "NO",
            viz_instruction=viz_instruction
        )
        
        # Call LLM (with automatic failover)
        response = await self.llm.invoke(prompt)
        
        if not response.success:
            logger.warning(f"Analysis LLM call failed: {response.error}")
            # Fallback: return truncated result as summary
            return self._create_fallback_summary(result), None
        
        # Parse JSON response
        return self._parse_analysis_response(response.content, result)

    def _parse_analysis_response(self, content: str, raw_result: str) -> Tuple[str, Optional[Dict]]:
        """
        Parse LLM response to extract summary and visualization config.
        Handles various JSON formats and malformed responses gracefully.
        """
        try:
            # Clean up response - remove markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            # Try to parse JSON
            data = json.loads(content)
            
            summary = data.get("summary", "").strip()
            viz_config = data.get("visualization")
            
            # Validate summary
            if not summary:
                summary = self._create_fallback_summary(raw_result)
            
            # Truncate if too long (50 word limit)
            words = summary.split()
            if len(words) > 60:
                summary = " ".join(words[:55]) + "..."
            
            # Validate visualization config
            if viz_config:
                if not isinstance(viz_config, dict):
                    viz_config = None
                elif not viz_config.get("type") or not viz_config.get("data"):
                    viz_config = None
                elif not isinstance(viz_config.get("data"), dict):
                    viz_config = None
            
            return summary, viz_config
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            # Try to extract summary from malformed response
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', content)
            if summary_match:
                return summary_match.group(1), None
            return self._create_fallback_summary(raw_result), None
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return self._create_fallback_summary(raw_result), None

    def _create_fallback_summary(self, result: str) -> str:
        """Create a simple fallback summary from raw results."""
        # Count rows
        lines = [l for l in result.strip().split('\n') if l.strip()]
        row_count = len(lines)
        
        # Extract first few values
        data = parse_sql_result_to_data(result)
        if data:
            items = list(data.items())[:3]
            sample = ", ".join([f"{k}: {v:,.0f}" for k, v in items])
            return f"Query returned {row_count} rows. Sample: {sample}."
        
        return f"Query returned {row_count} results."

    async def process_query(self, question: str, state_id: str) -> Tuple[str, Optional[str]]:
        """
        Process a natural language query end-to-end.
        
        Sequential flow:
        1. Generate SQL (LLM call #1)
        2. Execute SQL
        3. Analyze results (LLM call #2) → returns {summary, visualization}
        4. If visualization config present, render chart (no LLM)
        
        Returns:
            Tuple of (response_text, chart_path or None)
        """
        logger.info(f"Processing SQL query: {question[:100]}... [state_id={state_id}]")
        
        try:
            # Step 1: Generate SQL (LLM call #1)
            sql = await self.generate_sql(question)
            logger.info(f"Generated SQL: {sql[:200]}...")

            # Safety check
            sql_upper = sql.upper()
            if any(kw in sql_upper for kw in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']):
                return "Only SELECT queries are allowed for safety reasons.", None

            # Step 2: Execute SQL
            result = await self.execute_sql(sql)
            if not result or result.strip() == "":
                return "Query executed but returned no results.", None

            # Step 3: Analyze results (LLM call #2) - SEQUENTIAL, not parallel
            summary, viz_config = await self._analyze_results(question, result)

            # Step 4: Render chart if config present (no LLM call)
            chart_path = None
            if viz_config:
                logger.info(f"Rendering chart: {viz_config.get('type')}")
                chart_path = create_chart_from_config(
                    config=viz_config,
                    charts_dir=self.charts_dir,
                    query_id=state_id
                )
                if chart_path:
                    logger.info(f"✅ Chart rendered: {chart_path}")
                else:
                    logger.warning("Chart rendering returned None")

            return summary, chart_path

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing SQL query: {str(e)}", None


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("sql-server")
config = SQLConfig()
sql_server = SQLServer(config)


@mcp.tool()
async def sql_query(query: str, state_id: str = "default") -> str:
    """
    Convert natural language questions to SQL and execute them against the AdventureWorks database.
    
    Use this tool ONLY for questions about DATABASE data:
    - Sales data (orders, revenue, transactions)
    - Customer information (contacts, addresses, history)
    - Product catalog (products, categories, inventory)
    - Employee data (staff, departments, territories)
    - Business analytics (top sellers, trends, comparisons)
    
    DO NOT use this tool for:
    - Budget speech, tax brackets, tax structure (use knowledgebase_query instead)
    - Document content, policies, announcements (use knowledgebase_query instead)
    - Any question about uploaded PDFs or documents (use knowledgebase_query instead)
    
    Args:
        query: Natural language question about DATABASE data
        state_id: Session identifier for tracking
        
    Returns:
        JSON string with answer and optional chart path
    """
    try:
        logger.info(f"SQL query received: '{query}' [state_id={state_id}]")
        answer, viz_path = await sql_server.process_query(query, state_id)
        response = {"answer": answer, "chart": viz_path}
        logger.info(f"SQL response [state_id={state_id}]: chart={viz_path}")
        return json.dumps(response)
    except Exception as e:
        logger.error(f"SQL query error [state_id={state_id}]: {e}", exc_info=True)
        return json.dumps({"answer": f"Error processing SQL query: {str(e)}", "chart": None})


@mcp.tool()
async def sql_schema() -> str:
    """Get the database schema information."""
    try:
        schema = await sql_server.get_schema()
        return json.dumps({"schema": schema, "database": config.pg_database, "host": config.pg_host})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def sql_health() -> str:
    """Check the health status of the SQL server."""
    try:
        status = {
            "server": "SQL MCP Server",
            "status": "healthy",
            "database": config.pg_database,
            "host": config.pg_host,
            "port": config.pg_port,
            "initialized": sql_server.initialized,
            "llm_status": sql_server.llm.get_status(),
            "charts_dir": sql_server.charts_dir
        }
        return json.dumps(status)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


if __name__ == "__main__":
    logger.info("Starting SQL MCP Server (stdio)...")
    logger.info("Architecture: Schema-Driven Visualization + Multi-Provider LLM")
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
