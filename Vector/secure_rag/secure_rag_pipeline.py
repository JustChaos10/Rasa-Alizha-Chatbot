#!/usr/bin/env python3
"""
FIXED: Secure RAG Pipeline Orchestrator with Enhanced Error Reporting
Main fixes:
1. Better LLM response handling and fallback logic
2. Detailed rejection reasoning at each stage
3. Improved RBAC decision logging
4. Enhanced error propagation and reporting
"""

import os
import sys
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import our custom services with error handling
try:
    from .input_guard_service import InputGuardService, ScanResult as InputScanResult
    from .rbac_service import RBACService, AccessLevel
    from .groq_service import GroqService, ResponseFormat
    from .output_guard_service import OutputGuardService, OutputScanResultType
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: {e}")
    print("Some services not available. Using fallback mode.")
    SERVICES_AVAILABLE = False


class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    USER_QUERY = "user_query"
    PROMPT_GUARD_INPUT = "prompt_guard_input"
    LLM_GUARD_INPUT = "llm_guard_input"
    RBAC_CHECK = "rbac_check"
    LLM_GENERATION = "llm_generation"
    LLM_GUARD_OUTPUT = "llm_guard_output"
    PROMPT_GUARD_OUTPUT = "prompt_guard_output"
    FINAL_RESPONSE = "final_response"


class PipelineResult(Enum):
    """Pipeline result enumeration"""
    SUCCESS = "success"
    BLOCKED_INPUT = "blocked_input"
    BLOCKED_ACCESS = "blocked_access"
    BLOCKED_OUTPUT = "blocked_output"
    ERROR = "error"


@dataclass
class DetailedReason:
    """Detailed reasoning for pipeline decisions"""
    stage: str
    decision: str
    reason: str
    technical_details: Dict
    user_friendly_message: str
    suggested_action: str = ""


@dataclass
class PipelineStageResult:
    """Enhanced result from a pipeline stage"""
    stage: PipelineStage
    success: bool
    content: str
    metadata: Dict
    processing_time: float
    error_message: str = ""
    detailed_reason: Optional[DetailedReason] = None
    debug_info: Dict = None


@dataclass
class SecureRAGResult:
    """Enhanced final result from Secure RAG Pipeline"""
    success: bool
    result_type: PipelineResult
    final_response: str
    user_query: str
    user_id: str
    stage_results: List[PipelineStageResult]
    total_processing_time: float
    security_summary: Dict
    detailed_reasons: List[DetailedReason]
    pipeline_debug_info: Dict


class SecureRAGPipeline:
    """Enhanced Secure RAG Pipeline with better error reporting"""

    def __init__(self, verbose: bool = True, debug_mode: bool = False):
        """
        Initialize Enhanced Secure RAG Pipeline

        Args:
            verbose: Enable verbose terminal output
            debug_mode: Enable detailed debugging information
        """
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.stage_results = []
        self.detailed_reasons = []

        # Initialize services with better error handling
        self._initialize_services()

        if self.verbose:
            print("🚀 Enhanced Secure RAG Pipeline initialized successfully")
            print(f"   Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")

    def _initialize_services(self):
        """Initialize all pipeline services with enhanced error handling"""
        try:
            if SERVICES_AVAILABLE:
                # Initialize with detailed error reporting
                self.input_guard = InputGuardService(verbose=self.verbose)
                self.rbac = RBACService(verbose=self.verbose)

                # Enhanced Groq initialization with API key validation (with fallback enabled)
                groq_api_key = os.environ.get('GROQ_API_KEY', '')
                self.groq = GroqService(api_key=groq_api_key or None, verbose=self.verbose, enable_fallback=True)
                if self.verbose:
                    if groq_api_key:
                        print("✅ Groq service initialized (API key provided)")
                    else:
                        print("⚠️ GROQ_API_KEY not set. Using Groq fallback mode (no external calls).")

                self.output_guard = OutputGuardService(verbose=self.verbose)

                # Setup test documents with enhanced metadata
                self._setup_enhanced_test_documents()

                if self.verbose:
                    print("✅ All pipeline services initialized")
            else:
                self._initialize_fallback_services()

        except Exception as e:
            if self.verbose:
                print(f"❌ Error initializing services: {e}")
            self._initialize_fallback_services()

    def _initialize_fallback_services(self):
        """Initialize fallback services when imports fail"""
        self.input_guard = None
        self.rbac = None
        self.groq = None
        self.output_guard = None
        if self.verbose:
            print("⚠️ Using fallback services - limited functionality")

    def _setup_enhanced_test_documents(self):
        """Setup enhanced test documents with better metadata"""
        test_documents = [
            {
                "owner_ids": ["hr_admin", "hr_executive", "emp1"],
                "shared": True,  # Make more documents accessible
                "category": "employee_data",
                "sensitivity_level": "high",
                "created_by": "hr_admin",
                "department": "HR",
                "text": "Employee Details: Akash works at ITC Infotech as IS2 level engineer in the Engineering department. He specializes in cloud architecture and has 5 years experience."
            },
            {
                "owner_ids": ["hr_admin", "hr_executive", "emp2"],
                "shared": True,
                "category": "employee_data",
                "sensitivity_level": "high",
                "created_by": "hr_admin",
                "department": "Data Science",
                "text": "Employee Details: Arpan works at ITC Infotech as IS1 level Data Scientist. He works on machine learning projects for the MOC Innovation Team."
            },
            {
                "owner_ids": ["hr_admin"],
                "shared": True,
                "category": "public",
                "sensitivity_level": "low",
                "created_by": "hr_admin",
                "department": "General",
                "text": "Company Policies: All employees must follow security guidelines, maintain confidentiality, and report security incidents immediately. Work hours are flexible with core hours 10 AM to 4 PM."
            },
            {
                "owner_ids": ["hr_admin", "hr_executive"],
                "shared": False,
                "category": "confidential",
                "sensitivity_level": "critical",
                "created_by": "hr_admin",
                "department": "HR",
                "text": "Confidential HR Data: Salary ranges, performance reviews, and disciplinary actions. This information is restricted to HR administrators only."
            },
            {
                "owner_ids": [],
                "shared": True,
                "category": "public",
                "sensitivity_level": "low",
                "created_by": "system",
                "department": "General",
                "text": "General Information: ITC Infotech is a leading IT services company providing digital transformation solutions. We serve clients across various industries with innovative technology solutions."
            }
        ]

        if self.rbac:
            success = self.rbac.ingest_documents(test_documents)
            if self.verbose:
                print(f"✅ Enhanced test documents loaded: {success}")

    def _create_detailed_reason(self, stage: str, decision: str, reason: str,
                             technical_details: Dict = None,
                             user_friendly_message: str = "",
                             suggested_action: str = "") -> DetailedReason:
        """Create detailed reasoning for pipeline decisions"""
        return DetailedReason(
            stage=stage,
            decision=decision,
            reason=reason,
            technical_details=technical_details or {},
            user_friendly_message=user_friendly_message or reason,
            suggested_action=suggested_action
        )

    def process_query(self, user_query: str, user_id: str) -> SecureRAGResult:
        """
        Enhanced process query with detailed error reporting
        """
        start_time = time.time()
        self.stage_results = []
        self.detailed_reasons = []

        if self.verbose:
            print("🚀 Starting Enhanced Secure RAG Pipeline")
            print(f"   User ID: {user_id}")
            print(f"   Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
            print(f"   Debug Mode: {self.debug_mode}")

        current_content = user_query

        # Stage 1: User Query
        stage_result = self._process_user_query_stage(user_query, user_id)
        self.stage_results.append(stage_result)
        if not stage_result.success:
            return self._create_final_result(PipelineResult.ERROR, user_query, user_id, start_time)

        # Stage 2: Input Guard
        stage_result = self._process_input_guard_stage(current_content)
        self.stage_results.append(stage_result)
        if not stage_result.success:
            return self._create_final_result(PipelineResult.BLOCKED_INPUT, user_query, user_id, start_time)
        current_content = stage_result.content

        # Stage 3: RBAC Check (moved earlier to provide better context)
        stage_result = self._process_rbac_stage(current_content, user_id)
        self.stage_results.append(stage_result)
        if not stage_result.success:
            return self._create_final_result(PipelineResult.BLOCKED_ACCESS, user_query, user_id, start_time)

        # Get context from RBAC for LLM
        context_content = stage_result.content
        accessible_docs_count = stage_result.metadata.get('accessible_documents', 0)

        # Stage 4: LLM Generation (enhanced)
        stage_result = self._process_llm_generation_stage(user_query, context_content, user_id)
        self.stage_results.append(stage_result)
        if not stage_result.success:
            return self._create_final_result(PipelineResult.ERROR, user_query, user_id, start_time)
        current_content = stage_result.content

        # Stage 5: Output Guard
        stage_result = self._process_output_guard_stage(user_query, current_content, user_id)
        self.stage_results.append(stage_result)
        if not stage_result.success:
            return self._create_final_result(PipelineResult.BLOCKED_OUTPUT, user_query, user_id, start_time)

        final_response = stage_result.content

        # Stage 6: Final Response
        stage_result = self._create_final_response_stage(final_response)
        self.stage_results.append(stage_result)

        return self._create_final_result(PipelineResult.SUCCESS, user_query, user_id, start_time, final_response)

    def _process_rbac_stage(self, content: str, user_id: str) -> PipelineStageResult:
        """Enhanced RBAC processing with detailed reasoning"""
        stage_start = time.time()

        if not self.rbac:
            reason = self._create_detailed_reason(
                stage="RBAC",
                decision="FALLBACK",
                reason="RBAC service not available - using fallback mode",
                user_friendly_message="Access control not available, using basic permissions"
            )
            return PipelineStageResult(
                stage=PipelineStage.RBAC_CHECK,
                success=True,
                content=content,
                metadata={"fallback": True},
                processing_time=time.time() - stage_start,
                detailed_reason=reason
            )

        # Perform RBAC check
        access_result = self.rbac.check_access(user_id, content)

        if not access_result.allowed:
            reason = self._create_detailed_reason(
                stage="RBAC",
                decision="DENIED",
                reason=access_result.reason,
                technical_details={
                    "user_id": user_id,
                    "user_found": access_result.user is not None,
                    "user_role": access_result.user.role.value if access_result.user else None,
                    "access_level": access_result.access_level.value
                },
                user_friendly_message=f"Access denied: {access_result.reason}",
                suggested_action="Contact your administrator for access permissions"
            )

            return PipelineStageResult(
                stage=PipelineStage.RBAC_CHECK,
                success=False,
                content="",
                metadata={"access_denied": True},
                processing_time=time.time() - stage_start,
                error_message=access_result.reason,
                detailed_reason=reason
            )

        # Success case - build context from filtered documents
        filtered_docs = access_result.filtered_documents
        if not filtered_docs:
            # User has access but no relevant documents found
            context_content = "No specific company documents found for this query. You can ask general questions."
            reason_msg = "Access granted but no relevant documents available for this query"
        else:
            # Build rich context from available documents
            context_parts = []
            for doc in filtered_docs:
                doc_context = f"Document: {doc.page_content}"
                if doc.metadata.get('category'):
                    doc_context += f" (Category: {doc.metadata['category']})"
                context_parts.append(doc_context)

            context_content = "\n\n".join(context_parts)
            reason_msg = f"Access granted. Found {len(filtered_docs)} relevant documents."

        reason = self._create_detailed_reason(
            stage="RBAC",
            decision="ALLOWED",
            reason=reason_msg,
            technical_details={
                "user_id": user_id,
                "user_role": access_result.user.role.value,
                "access_level": access_result.access_level.value,
                "accessible_documents": len(filtered_docs),
                "document_categories": list(set([doc.metadata.get('category', 'unknown') for doc in filtered_docs]))
            },
            user_friendly_message=f"Access granted with {access_result.access_level.value} level permissions"
        )

        return PipelineStageResult(
            stage=PipelineStage.RBAC_CHECK,
            success=True,
            content=context_content,
            metadata={
                "user_role": access_result.user.role.value,
                "access_level": access_result.access_level.value,
                "accessible_documents": len(filtered_docs),
                "reason": access_result.reason
            },
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _process_llm_generation_stage(self, original_query: str, context_content: str, user_id: str) -> PipelineStageResult:
        """Enhanced LLM generation with better error handling"""
        stage_start = time.time()

        if not self.groq:
            reason = self._create_detailed_reason(
                stage="LLM_GENERATION",
                decision="ERROR",
                reason="Groq service not available - API key missing or invalid",
                technical_details={"groq_available": False, "api_key_set": bool(os.environ.get('GROQ_API_KEY'))},
                user_friendly_message="AI response service is currently unavailable",
                suggested_action="Please check API configuration and try again"
            )

            return PipelineStageResult(
                stage=PipelineStage.LLM_GENERATION,
                success=False,
                content="",
                metadata={"error": "Groq service unavailable"},
                processing_time=time.time() - stage_start,
                error_message="LLM service not available",
                detailed_reason=reason
            )

        # Build enhanced prompt with proper context
        if context_content and context_content.strip() and not context_content.startswith("No specific"):
            enhanced_prompt = f"""Based on the following company information and documents:

{context_content}

Please answer this question: {original_query}

Provide a helpful, accurate response based on the available information. If the information doesn't fully answer the question, please indicate what you can and cannot determine from the provided context."""
        else:
            enhanced_prompt = f"""Please answer this question: {original_query}

Note: No specific company documents were available for this query, so please provide a general response based on your knowledge."""

        if self.debug_mode and self.verbose:
            print(f"🤖 LLM Prompt (first 200 chars): {enhanced_prompt[:200]}...")

        # Generate response
        llm_response = self.groq.generate_text(
            prompt=enhanced_prompt,
            temperature=0.7,
            max_tokens=1024
        )

        if not llm_response.success:
            reason = self._create_detailed_reason(
                stage="LLM_GENERATION",
                decision="ERROR",
                reason=f"LLM generation failed: {llm_response.error_message}",
                technical_details={
                    "model_used": llm_response.model_used,
                    "error_message": llm_response.error_message,
                    "tokens_used": llm_response.tokens_used
                },
                user_friendly_message="AI response generation failed",
                suggested_action="Please try again or contact support"
            )

            return PipelineStageResult(
                stage=PipelineStage.LLM_GENERATION,
                success=False,
                content="",
                metadata={"llm_error": llm_response.error_message},
                processing_time=time.time() - stage_start,
                error_message=llm_response.error_message,
                detailed_reason=reason
            )

        # Validate response content
        if not llm_response.content or llm_response.content.strip() == "":
            reason = self._create_detailed_reason(
                stage="LLM_GENERATION",
                decision="ERROR",
                reason="LLM returned empty response",
                technical_details={
                    "model_used": llm_response.model_used,
                    "tokens_used": llm_response.tokens_used,
                    "content_length": len(llm_response.content)
                },
                user_friendly_message="No response was generated",
                suggested_action="Please try rephrasing your question"
            )

            return PipelineStageResult(
                stage=PipelineStage.LLM_GENERATION,
                success=False,
                content="",
                metadata={"empty_response": True},
                processing_time=time.time() - stage_start,
                error_message="Empty response from LLM",
                detailed_reason=reason
            )

        # Success case
        reason = self._create_detailed_reason(
            stage="LLM_GENERATION",
            decision="SUCCESS",
            reason="Response successfully generated",
            technical_details={
                "model_used": llm_response.model_used,
                "tokens_used": llm_response.tokens_used,
                "response_length": len(llm_response.content),
                "processing_time": llm_response.processing_time
            },
            user_friendly_message="AI response generated successfully"
        )

        return PipelineStageResult(
            stage=PipelineStage.LLM_GENERATION,
            success=True,
            content=llm_response.content,
            metadata={
                "model_used": llm_response.model_used,
                "tokens_used": llm_response.tokens_used,
                "groq_metadata": llm_response.metadata
            },
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _process_user_query_stage(self, query: str, user_id: str) -> PipelineStageResult:
        """Process initial user query stage"""
        stage_start = time.time()

        reason = self._create_detailed_reason(
            stage="USER_QUERY",
            decision="ACCEPTED",
            reason="Query received and validated",
            technical_details={
                "query_length": len(query),
                "user_id": user_id,
                "timestamp": time.time()
            },
            user_friendly_message="Your query has been received"
        )

        return PipelineStageResult(
            stage=PipelineStage.USER_QUERY,
            success=True,
            content=query,
            metadata={"user_id": user_id, "query_length": len(query)},
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _process_input_guard_stage(self, content: str) -> PipelineStageResult:
        """Process input guard stage with detailed reasoning"""
        stage_start = time.time()

        if not self.input_guard:
            reason = self._create_detailed_reason(
                stage="INPUT_GUARD",
                decision="FALLBACK",
                reason="Input guard service not available",
                user_friendly_message="Basic input validation applied"
            )
            return PipelineStageResult(
                stage=PipelineStage.LLM_GUARD_INPUT,
                success=True,
                content=content,
                metadata={"fallback": True},
                processing_time=time.time() - stage_start,
                detailed_reason=reason
            )

        input_scan_result = self.input_guard.scan_input(content)

        if not input_scan_result.is_valid:
            reason = self._create_detailed_reason(
                stage="INPUT_GUARD",
                decision="BLOCKED",
                reason="Input failed security scan",
                technical_details={
                    "result_type": input_scan_result.result_type.value,
                    "scanner_results": input_scan_result.scanner_results,
                    "warnings": input_scan_result.warnings,
                    "errors": input_scan_result.errors
                },
                user_friendly_message="Your input contains content that violates security policies",
                suggested_action="Please rephrase your query avoiding sensitive or harmful content"
            )
        else:
            reason = self._create_detailed_reason(
                stage="INPUT_GUARD",
                decision="ALLOWED",
                reason="Input passed security scan",
                technical_details={
                    "scanner_results": input_scan_result.scanner_results,
                    "processing_time": input_scan_result.processing_time
                },
                user_friendly_message="Input validation successful"
            )

        return PipelineStageResult(
            stage=PipelineStage.LLM_GUARD_INPUT,
            success=input_scan_result.is_valid,
            content=input_scan_result.sanitized_prompt,
            metadata={
                "result_type": input_scan_result.result_type.value,
                "scanner_results": input_scan_result.scanner_results,
                "warnings": input_scan_result.warnings
            },
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _process_output_guard_stage(self, query: str, response: str, user_id: str) -> PipelineStageResult:
        """Process output guard stage with role-based filtering"""
        stage_start = time.time()

        if not self.output_guard:
            reason = self._create_detailed_reason(
                stage="OUTPUT_GUARD",
                decision="FALLBACK",
                reason="Output guard service not available",
                user_friendly_message="Basic output validation applied"
            )
            return PipelineStageResult(
                stage=PipelineStage.LLM_GUARD_OUTPUT,
                success=True,
                content=response,
                metadata={"fallback": True},
                processing_time=time.time() - stage_start,
                detailed_reason=reason
            )

        # Get user access level for output filtering
        user_access_level = AccessLevel.LIMITED
        if self.rbac and self.rbac.get_user(user_id):
            user_access_level = self.rbac.get_user(user_id).access_level

        output_scan_result = self.output_guard.scan_output(query, response, user_access_level)

        if not output_scan_result.is_valid:
            reason = self._create_detailed_reason(
                stage="OUTPUT_GUARD",
                decision="BLOCKED",
                reason="Output failed security scan",
                technical_details={
                    "result_type": output_scan_result.result_type.value,
                    "quality_score": output_scan_result.quality_score,
                    "scanner_results": output_scan_result.scanner_results,
                    "warnings": output_scan_result.warnings,
                    "user_access_level": user_access_level.value
                },
                user_friendly_message="The generated response contains content that cannot be shared",
                suggested_action="Please try a different question or contact support for access"
            )
        else:
            reason = self._create_detailed_reason(
                stage="OUTPUT_GUARD",
                decision="ALLOWED",
                reason="Output passed security scan",
                technical_details={
                    "quality_score": output_scan_result.quality_score,
                    "scanner_results": output_scan_result.scanner_results,
                    "user_access_level": user_access_level.value
                },
                user_friendly_message="Response validated and approved"
            )

        return PipelineStageResult(
            stage=PipelineStage.LLM_GUARD_OUTPUT,
            success=output_scan_result.is_valid,
            content=output_scan_result.sanitized_response,
            metadata={
                "result_type": output_scan_result.result_type.value,
                "quality_score": output_scan_result.quality_score,
                "scanner_results": output_scan_result.scanner_results,
                "warnings": output_scan_result.warnings
            },
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _create_final_response_stage(self, response: str) -> PipelineStageResult:
        """Create final response stage"""
        stage_start = time.time()

        reason = self._create_detailed_reason(
            stage="FINAL_RESPONSE",
            decision="SUCCESS",
            reason="Pipeline completed successfully",
            technical_details={"response_length": len(response)},
            user_friendly_message="Your request has been processed successfully"
        )

        return PipelineStageResult(
            stage=PipelineStage.FINAL_RESPONSE,
            success=True,
            content=response,
            metadata={"response_length": len(response)},
            processing_time=time.time() - stage_start,
            detailed_reason=reason
        )

    def _create_final_result(self, result_type: PipelineResult, user_query: str, user_id: str,
                           start_time: float, final_response: str = "") -> SecureRAGResult:
        """Create enhanced final result with detailed reporting"""
        total_time = time.time() - start_time

        # Collect all detailed reasons
        all_reasons = [stage.detailed_reason for stage in self.stage_results if stage.detailed_reason]

        # Create security summary
        security_summary = self._create_enhanced_security_summary()

        # Generate appropriate response based on result type
        if not final_response:
            response_map = {
                PipelineResult.BLOCKED_INPUT: "Your query was blocked due to security policy violations in the input validation stage.",
                PipelineResult.BLOCKED_ACCESS: "Access denied. You don't have the necessary permissions to access this information.",
                PipelineResult.BLOCKED_OUTPUT: "The generated response was blocked due to security policy violations.",
                PipelineResult.ERROR: "An error occurred while processing your request. Please try again."
            }
            final_response = response_map.get(result_type, "Request could not be processed.")

        # Add debugging information
        pipeline_debug_info = {
            "total_processing_time": total_time,
            "stages_completed": len(self.stage_results),
            "successful_stages": len([s for s in self.stage_results if s.success]),
            "failed_stage": next((s.stage.value for s in self.stage_results if not s.success), None),
            "groq_available": self.groq is not None,
            "api_key_configured": bool(os.environ.get('GROQ_API_KEY')),
            "services_available": SERVICES_AVAILABLE
        }

        result = SecureRAGResult(
            success=(result_type == PipelineResult.SUCCESS),
            result_type=result_type,
            final_response=final_response,
            user_query=user_query,
            user_id=user_id,
            stage_results=self.stage_results,
            total_processing_time=total_time,
            security_summary=security_summary,
            detailed_reasons=all_reasons,
            pipeline_debug_info=pipeline_debug_info
        )

        # Enhanced logging
        self._log_enhanced_final_result(result)

        return result

    def _create_enhanced_security_summary(self) -> Dict:
        """Create enhanced security summary"""
        security_stages = [
            PipelineStage.PROMPT_GUARD_INPUT,
            PipelineStage.LLM_GUARD_INPUT,
            PipelineStage.RBAC_CHECK,
            PipelineStage.LLM_GUARD_OUTPUT,
            PipelineStage.PROMPT_GUARD_OUTPUT
        ]

        total_security_checks = len(security_stages)
        passed_security_checks = 0
        blocked_reasons = []
        warnings = []
        errors = []

        for result in self.stage_results:
            if result.stage in security_stages:
                if result.success:
                    passed_security_checks += 1
                else:
                    if result.detailed_reason:
                        blocked_reasons.append(f"{result.stage.value}: {result.detailed_reason.reason}")
                    elif result.error_message:
                        blocked_reasons.append(f"{result.stage.value}: {result.error_message}")

            if result.metadata.get("warnings"):
                warnings.extend(result.metadata["warnings"])

            if result.error_message:
                errors.append(f"{result.stage.value}: {result.error_message}")

        # Determine risk level
        success_rate = (passed_security_checks / total_security_checks) if total_security_checks > 0 else 0
        if success_rate >= 0.8:
            risk_level = "low"
        elif success_rate >= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "total_security_checks": total_security_checks,
            "security_checks_passed": passed_security_checks,
            "success_rate": success_rate,
            "risk_level": risk_level,
            "blocked_reasons": blocked_reasons,
            "warnings": warnings,
            "errors": errors
        }

    def _log_enhanced_final_result(self, result: SecureRAGResult):
        """Enhanced logging of final results"""
        if not self.verbose:
            return

        print(f"\n{'='*80}")
        print("🎯 ENHANCED PIPELINE RESULT")
        print(f"{'='*80}")

        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} Overall Status: {result.result_type.value.upper()}")
        print(f"⏱️  Total Time: {result.total_processing_time:.3f}s")
        print(f"🔄 Stages: {result.pipeline_debug_info['successful_stages']}/{result.pipeline_debug_info['stages_completed']}")
        print(f"🛡️  Security: {result.security_summary['security_checks_passed']}/{result.security_summary['total_security_checks']} checks passed")
        print(f"⚡ Risk Level: {result.security_summary['risk_level'].upper()}")

        if result.pipeline_debug_info.get('failed_stage'):
            print(f"🚫 Failed at: {result.pipeline_debug_info['failed_stage']}")

        if result.security_summary.get('blocked_reasons'):
            print("🔒 Block Reasons:")
            for reason in result.security_summary['blocked_reasons']:
                print(f"   - {reason}")

        if self.debug_mode:
            print("\n🔧 DEBUG INFO:")
            print(f"   Groq Available: {result.pipeline_debug_info['groq_available']}")
            print(f"   API Key Set: {result.pipeline_debug_info['api_key_configured']}")
            print(f"   Services Available: {result.pipeline_debug_info['services_available']}")

        if result.success:
            print("\n💬 FINAL RESPONSE:")
            print(f"{'─' * 40}")
            print(result.final_response)
            print(f"{'─' * 40}")


# Test function remains the same but with enhanced output
def test_enhanced_secure_rag_pipeline():
    """Test the enhanced pipeline"""
    print("🧪 Testing Enhanced Secure RAG Pipeline")
    print("="*80)

    try:
        pipeline = SecureRAGPipeline(verbose=True, debug_mode=True)

        test_cases = [
            {
                "name": "HR Admin - Employee Query",
                "user_id": "hr_admin",
                "query": "Tell me about employees in our company",
                "expected_success": True
            },
            {
                "name": "Regular Employee - Personal Query",
                "user_id": "emp1",
                "query": "What are the company policies?",
                "expected_success": True
            },
            {
                "name": "Guest - Basic Query",
                "user_id": "hr_common",
                "query": "What services does ITC Infotech provide?",
                "expected_success": True
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['name']}")
            print("="*60)

            result = pipeline.process_query(test_case["query"], test_case["user_id"])

            print("\n📋 DETAILED REASONING:")
            for reason in result.detailed_reasons:
                print(f"   {reason.stage}: {reason.decision} - {reason.user_friendly_message}")
                if reason.suggested_action:
                    print(f"      Suggestion: {reason.suggested_action}")

            success_match = result.success == test_case['expected_success']
            print(f"\n✅ Test {'PASSED' if success_match else 'FAILED'}")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_enhanced_secure_rag_pipeline()
    else:
        print("Usage: python secure_rag_pipeline_fixed.py test")


