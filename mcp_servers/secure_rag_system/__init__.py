#!/usr/bin/env python3
"""
Secure RAG Package - Consolidated Module

All components are in secure_rag.py:
- InputGuard: Blocks harmful/injection queries  
- OutputGuard: Filters sensitive data from responses
- RBACService: Role-Based Access Control
- SimpleVectorStore: In-memory document storage
- LLMService: Response generation
- SecureRAGPipeline: Main orchestrator
"""

from .secure_rag import (
    # Enums
    UserRole,
    AccessLevel,
    ScanResult,
    PipelineResult,
    # Data classes
    User,
    InputScanResult,
    OutputScanResult,
    AccessResult,
    SecureRAGResult,
    # Components
    InputGuard,
    OutputGuard,
    SimpleVectorStore,
    RBACService,
    LLMService,
    SecureRAGPipeline,
)

__all__ = [
    "UserRole",
    "AccessLevel", 
    "ScanResult",
    "PipelineResult",
    "User",
    "InputScanResult",
    "OutputScanResult",
    "AccessResult",
    "SecureRAGResult",
    "InputGuard",
    "OutputGuard",
    "SimpleVectorStore",
    "RBACService",
    "LLMService",
    "SecureRAGPipeline",
]


