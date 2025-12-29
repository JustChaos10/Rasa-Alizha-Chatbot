#!/usr/bin/env python3
"""
Consolidated Secure RAG Module

This single file contains all the essential components for the Secure RAG pipeline:
- InputGuard: Blocks harmful/injection queries
- OutputGuard: Filters sensitive data from responses
- RBAC: Role-Based Access Control
- VectorStore: Simple in-memory document storage
- Pipeline: Orchestrates all components

No heavy dependencies (llm-guard, spacy) - uses lightweight regex-based guards.
"""

import os
import re
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class UserRole(Enum):
    ADMIN = "admin"
    HR_ADMIN = "hr_admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    GUEST = "guest"


class AccessLevel(Enum):
    FULL = "full"
    READ = "read"
    LIMITED = "limited"
    DENIED = "denied"


class ScanResult(Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


class PipelineResult(Enum):
    SUCCESS = "success"
    BLOCKED_INPUT = "blocked_input"
    BLOCKED_ACCESS = "blocked_access"
    BLOCKED_OUTPUT = "blocked_output"
    ERROR = "error"


@dataclass
class User:
    user_id: str
    role: UserRole
    access_level: AccessLevel
    allowed_categories: Set[str]
    department: str = ""
    name: str = ""


@dataclass
class InputScanResult:
    is_valid: bool
    result_type: ScanResult
    sanitized_prompt: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class OutputScanResult:
    is_valid: bool
    sanitized_response: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class AccessResult:
    allowed: bool
    user: Optional[User]
    reason: str
    filtered_documents: List[Dict]
    access_level: AccessLevel


@dataclass
class SecureRAGResult:
    success: bool
    result_type: PipelineResult
    response: str
    user_id: str
    query: str
    processing_time: float
    security_summary: Dict


# =============================================================================
# INPUT GUARD - Lightweight regex-based
# =============================================================================

class InputGuard:
    """Blocks harmful queries and injection attempts using regex patterns."""
    
    def __init__(self):
        self.blocked_patterns = [
            # Jailbreak/injection
            re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
            re.compile(r"ignore\s+.*system\s+prompts?", re.IGNORECASE),
            # Fixed: Only block "show/reveal/print" when followed by "system prompt" or "hidden prompt"
            re.compile(r"(reveal|show|print)\s+.*(system\s+prompt|hidden\s+prompt|internal\s+prompt)", re.IGNORECASE),
            re.compile(r"pretend\s+to\s+be|act\s+as\s+.*(developer|root|system)", re.IGNORECASE),
            re.compile(r"jailbreak|bypass\s+.*(safety|guard|filter)", re.IGNORECASE),
            # Harmful content
            re.compile(r"how\s+to\s+(make|build|create)\s+a?\s*bomb", re.IGNORECASE),
            re.compile(r"how\s+to\s+(kill|murder|harm)", re.IGNORECASE),
            re.compile(r"how\s+to\s+hack", re.IGNORECASE),
            re.compile(r"illegal\s+drugs?\s+recipe", re.IGNORECASE),
        ]
    
    def scan(self, prompt: str) -> InputScanResult:
        """Scan input for harmful patterns."""
        if not prompt:
            return InputScanResult(True, ScanResult.SAFE, prompt)
        
        for pattern in self.blocked_patterns:
            if pattern.search(prompt):
                return InputScanResult(
                    is_valid=False,
                    result_type=ScanResult.BLOCKED,
                    sanitized_prompt=prompt,
                    warnings=[f"Blocked: harmful content detected"]
                )
        
        return InputScanResult(True, ScanResult.SAFE, prompt)


# =============================================================================
# OUTPUT GUARD - Filters sensitive data
# =============================================================================

class OutputGuard:
    """Filters sensitive data from LLM responses."""
    
    def __init__(self):
        self.sensitive_patterns = [
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),  # SSN
            (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD REDACTED]"),  # Credit Card
            (re.compile(r"password\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE), "[PASSWORD REDACTED]"),
            (re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE), "[API_KEY REDACTED]"),
        ]
    
    def scan(self, response: str) -> OutputScanResult:
        """Scan and sanitize output."""
        if not response:
            return OutputScanResult(True, response)
        
        sanitized = response
        warnings = []
        
        for pattern, replacement in self.sensitive_patterns:
            if pattern.search(sanitized):
                sanitized = pattern.sub(replacement, sanitized)
                warnings.append(f"Redacted sensitive data")
        
        return OutputScanResult(True, sanitized, warnings)


# =============================================================================
# SIMPLE VECTOR STORE - In-memory with cosine similarity
# =============================================================================

EMBEDDING_DIM = 128

def _embed_text(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Create deterministic embedding using SHA-256 (no external model needed)."""
    payload = text.strip().encode("utf-8") or b"empty"
    buffer = bytearray()
    seed = hashlib.sha256(payload).digest()
    buffer.extend(seed)
    while len(buffer) < dim * 4:
        seed = hashlib.sha256(seed + payload).digest()
        buffer.extend(seed)
    
    vector = np.frombuffer(bytes(buffer[:dim * 4]), dtype=np.uint32).astype(np.float32)
    vector /= np.float32(2**32)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


class SimpleVectorStore:
    """In-memory vector store for document retrieval."""
    
    def __init__(self):
        self.documents: List[Dict] = []
        self.vectors: List[np.ndarray] = []
    
    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents with text and metadata."""
        count = 0
        for doc in documents:
            text = doc.get("text", "")
            if text:
                vector = _embed_text(text)
                self.documents.append(doc)
                self.vectors.append(vector)
                count += 1
        return count
    
    def search(self, query: str, top_k: int = 5, filter_fn=None) -> List[Dict]:
        """Search for similar documents, optionally filtering."""
        if not self.documents:
            return []
        
        query_vec = _embed_text(query)
        scores = []
        
        for i, (doc, vec) in enumerate(zip(self.documents, self.vectors)):
            # Apply filter if provided
            if filter_fn and not filter_fn(doc):
                continue
            score = float(np.dot(query_vec, vec))
            scores.append((score, i, doc))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in scores[:top_k]]


# =============================================================================
# RBAC SERVICE - Role-Based Access Control
# =============================================================================

class RBACService:
    """Manages users and document access permissions."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.vector_store = SimpleVectorStore()
        self._setup_default_users()
        self._setup_test_documents()
    
    def _setup_default_users(self):
        """Setup default users matching the app.db roles."""
        # Map database roles to RBAC roles
        # admin -> full access
        # user -> employee access
        
        default_users = [
            User("admin", UserRole.ADMIN, AccessLevel.FULL, 
                 {"hr", "employee_data", "confidential", "public", "general", "salary"}, "Admin", "Admin User"),
            User("hr_admin", UserRole.HR_ADMIN, AccessLevel.FULL,
                 {"hr", "employee_data", "confidential", "public", "general", "salary"}, "HR", "HR Admin"),
            User("manager", UserRole.MANAGER, AccessLevel.READ,
                 {"employee_data", "public", "general"}, "Management", "Manager"),
            User("employee", UserRole.EMPLOYEE, AccessLevel.LIMITED,
                 {"public", "general"}, "General", "Employee"),
            User("guest", UserRole.GUEST, AccessLevel.LIMITED,
                 {"public"}, "General", "Guest"),
        ]
        
        for user in default_users:
            self.users[user.user_id] = user
    
    def _setup_test_documents(self):
        """Setup test documents for RAG."""
        documents = [
            {
                "text": "Employee Details: Akash works at ITC Infotech as IS2 level engineer in the Engineering department. He specializes in cloud architecture and has 5 years of experience.",
                "category": "employee_data",
                "sensitivity": "medium",
                "department": "Engineering"
            },
            {
                "text": "Employee Details: Arpan works at ITC Infotech as IS1 level Data Scientist in the MOC Innovation Team. He works on machine learning projects.",
                "category": "employee_data",
                "sensitivity": "medium",
                "department": "Data Science"
            },
            {
                "text": "Company Policies: All employees must follow security guidelines, maintain confidentiality, and report security incidents. Work hours are flexible with core hours 10 AM to 4 PM.",
                "category": "public",
                "sensitivity": "low",
                "department": "General"
            },
            {
                "text": "ITC Infotech provides digital transformation and IT services to clients worldwide. The company focuses on innovation and technology solutions.",
                "category": "public",
                "sensitivity": "low",
                "department": "General"
            },
            {
                "text": "Salary Information: IS1 level employees earn between 8-12 LPA. IS2 level employees earn between 12-18 LPA. This information is confidential.",
                "category": "salary",
                "sensitivity": "high",
                "department": "HR"
            },
        ]
        self.vector_store.add_documents(documents)
    
    def get_user(self, user_id: str, db_role: str = None) -> Optional[User]:
        """
        Get user by ID, creating dynamic user if needed based on database role.
        
        Args:
            user_id: The user identifier
            db_role: The role from the database (e.g., 'admin', 'user')
        """
        # First check if we have a predefined user
        if user_id in self.users:
            return self.users[user_id]
        
        # Map database role to RBAC user
        if db_role == "admin":
            return User(user_id, UserRole.ADMIN, AccessLevel.FULL,
                       {"hr", "employee_data", "confidential", "public", "general", "salary"},
                       "Admin", f"Admin-{user_id}")
        elif db_role == "user":
            return User(user_id, UserRole.EMPLOYEE, AccessLevel.LIMITED,
                       {"public", "general"},
                       "General", f"Employee-{user_id}")
        else:
            # Default to guest
            return User(user_id, UserRole.GUEST, AccessLevel.LIMITED,
                       {"public"},
                       "General", f"Guest-{user_id}")
    
    def check_access(self, user: User, query: str) -> AccessResult:
        """Check what documents a user can access for a query."""
        if not user:
            return AccessResult(False, None, "User not found", [], AccessLevel.DENIED)
        
        # Check if this is a personal/contact card query (not for knowledge base)
        personal_keywords = ["contact card", "contact details", "contact information", "my contact", "personal contact"]
        query_lower = query.lower()
        is_personal_query = any(kw in query_lower for kw in personal_keywords)
        
        if is_personal_query:
            # Allow through - not a knowledge base query
            return AccessResult(
                allowed=True,
                user=user,
                reason="Personal contact query - not applicable to knowledge base",
                filtered_documents=[],  # Empty - no KB docs needed
                access_level=user.access_level
            )
        
        def filter_fn(doc):
            category = doc.get("category", "public")
            # Admin and HR_ADMIN see everything
            if user.role in [UserRole.ADMIN, UserRole.HR_ADMIN]:
                return True
            # Others only see their allowed categories
            return category in user.allowed_categories
        
        # First, check if ANY relevant documents exist for this query (without filter)
        all_relevant_docs = self.vector_store.search(query, top_k=5, filter_fn=None)
        
        # Then get filtered docs based on user access
        filtered_docs = self.vector_store.search(query, top_k=5, filter_fn=filter_fn)
        
        # Check if user is trying to access restricted content
        # If there ARE relevant docs but user can't see them, deny access
        if all_relevant_docs and not filtered_docs:
            return AccessResult(
                allowed=False,
                user=user,
                reason=f"Access denied: You don't have permission to view this information. Your role ({user.role.value}) does not have access to these documents.",
                filtered_documents=[],
                access_level=user.access_level
            )
        
        # Check if user is asking about employee-specific data without proper access
        employee_keywords = ["akash", "arpan", "employee", "role", "salary", "level", "is1", "is2"]
        is_employee_query = any(kw in query_lower for kw in employee_keywords)
        
        if is_employee_query and user.role not in [UserRole.ADMIN, UserRole.HR_ADMIN]:
            # Check if filtered docs contain employee_data
            has_employee_data = any(d.get("category") == "employee_data" for d in all_relevant_docs)
            if has_employee_data:
                return AccessResult(
                    allowed=False,
                    user=user,
                    reason=f"Access denied: Employee information is restricted. Only administrators and HR can view this data.",
                    filtered_documents=[],
                    access_level=user.access_level
                )
        
        return AccessResult(
            allowed=True,
            user=user,
            reason=f"Access granted for role {user.role.value}",
            filtered_documents=filtered_docs,
            access_level=user.access_level
        )
    
    def get_all_users(self) -> List[Dict]:
        """Get list of all predefined users."""
        return [
            {
                "user_id": u.user_id,
                "role": u.role.value,
                "access_level": u.access_level.value,
                "categories": list(u.allowed_categories)
            }
            for u in self.users.values()
        ]


# =============================================================================
# LLM SERVICE - Simple response generation
# =============================================================================

class LLMService:
    """Simple LLM service with fallback responses."""
    
    def __init__(self):
        self.groq_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client if available."""
        api_key = os.environ.get('GROQ_API_KEY', '')
        if api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
            except ImportError:
                pass
    
    def generate(self, query: str, context: str) -> str:
        """Generate response using LLM or fallback."""
        if self.groq_client:
            try:
                prompt = f"""Based on the following context, answer the question.
If the information is not in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
                
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
        
        # Fallback: return context summary
        if context:
            return f"Based on available information: {context[:500]}..."
        return "I don't have enough information to answer that question."


# =============================================================================
# SECURE RAG PIPELINE - Main orchestrator
# =============================================================================

class SecureRAGPipeline:
    """
    Main pipeline that orchestrates:
    1. Input Guard (block harmful queries)
    2. RBAC (filter documents by user permissions)
    3. LLM Generation (answer from filtered context)
    4. Output Guard (redact sensitive data)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.rbac = RBACService()
        self.llm = LLMService()
        
        if verbose:
            logger.info("SecureRAGPipeline initialized")
    
    def process_query(self, query: str, user_id: str, db_role: str = None) -> SecureRAGResult:
        """
        Process a query through the full secure pipeline.
        
        Args:
            query: The user's question
            user_id: User identifier
            db_role: Role from database ('admin' or 'user')
        """
        start_time = time.time()
        security_summary = {
            "checks_passed": 0,
            "total_checks": 4,
            "risk_level": "low",
            "warnings": [],
            "blocked_reasons": []
        }
        
        # EARLY CHECK: Skip personal contact queries (not for knowledge base)
        personal_keywords = ["contact card", "contact details", "contact information", "my contact", "personal contact"]
        query_lower = query.lower()
        is_personal_query = any(kw in query_lower for kw in personal_keywords)
        
        if is_personal_query:
            # Not applicable to Secure RAG - let other tools handle it
            return SecureRAGResult(
                success=False,
                result_type=PipelineResult.ERROR,
                response="This query is not applicable to the knowledge base. Try asking the contact form tool.",
                user_id=user_id,
                query=query,
                processing_time=time.time() - start_time,
                security_summary={"note": "Personal query - not for KB"}
            )
        
        # Step 1: Input Guard
        input_result = self.input_guard.scan(query)
        if not input_result.is_valid:
            security_summary["blocked_reasons"].append("Input blocked by security policy")
            return SecureRAGResult(
                success=False,
                result_type=PipelineResult.BLOCKED_INPUT,
                response="I cannot answer that query as it violates security policies.",
                user_id=user_id,
                query=query,
                processing_time=time.time() - start_time,
                security_summary=security_summary
            )
        security_summary["checks_passed"] += 1
        
        # Step 2: Get user and check RBAC
        user = self.rbac.get_user(user_id, db_role)
        if not user:
            security_summary["blocked_reasons"].append("User not found")
            return SecureRAGResult(
                success=False,
                result_type=PipelineResult.BLOCKED_ACCESS,
                response="Access denied: User not recognized.",
                user_id=user_id,
                query=query,
                processing_time=time.time() - start_time,
                security_summary=security_summary
            )
        security_summary["checks_passed"] += 1
        
        # Step 3: Get accessible documents
        access_result = self.rbac.check_access(user, query)
        if not access_result.allowed:
            security_summary["blocked_reasons"].append(access_result.reason)
            return SecureRAGResult(
                success=False,
                result_type=PipelineResult.BLOCKED_ACCESS,
                response="Access denied: You don't have permission to access this information.",
                user_id=user_id,
                query=query,
                processing_time=time.time() - start_time,
                security_summary=security_summary
            )
        security_summary["checks_passed"] += 1
        
        # Build context from filtered documents
        context_parts = []
        for doc in access_result.filtered_documents:
            context_parts.append(doc.get("text", ""))
        context = "\n\n".join(context_parts)
        
        # Step 4: Generate response
        if not context:
            response = "I don't have information available to answer that question based on your access level."
        else:
            response = self.llm.generate(query, context)
        
        # Step 5: Output Guard
        output_result = self.output_guard.scan(response)
        if output_result.warnings:
            security_summary["warnings"].extend(output_result.warnings)
        security_summary["checks_passed"] += 1
        
        # Determine risk level
        if security_summary["warnings"]:
            security_summary["risk_level"] = "medium"
        
        return SecureRAGResult(
            success=True,
            result_type=PipelineResult.SUCCESS,
            response=output_result.sanitized_response,
            user_id=user_id,
            query=query,
            processing_time=time.time() - start_time,
            security_summary=security_summary
        )
    
    def get_user_permissions(self, user_id: str, db_role: str = None) -> Dict:
        """Get permissions for a user."""
        user = self.rbac.get_user(user_id, db_role)
        if not user:
            return {"error": f"User '{user_id}' not found"}
        
        return {
            "user_id": user.user_id,
            "role": user.role.value,
            "access_level": user.access_level.value,
            "allowed_categories": list(user.allowed_categories),
            "department": user.department
        }
    
    def list_users(self) -> Dict:
        """List all predefined users."""
        return {
            "users": self.rbac.get_all_users(),
            "total": len(self.rbac.users)
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "total_documents": len(self.rbac.vector_store.documents),
            "total_users": len(self.rbac.users),
            "input_guard": "active",
            "output_guard": "active"
        }
