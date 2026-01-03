import base64
import io
import json
import logging
import os
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import PyPDF2
from docx import Document

# Import telemetry
from architecture.telemetry import trace_llm_call, log_llm_event
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Avoid logging full request URLs (can include secrets like `?key=...`).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    # Redact common query-parameter secrets (Gemini style: ?key=...)
    text = re.sub(r"(?i)(key=)[^&\s]+", r"\1***", text)
    # Redact common API key patterns
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,]+", r"\1***", text)
    # Redact bearer tokens if ever present in exception strings
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._-]+", r"\1***", text)
    return text
 
# Import langdetect for language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language detection will use LLM only.")
 
class Config:
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 30
    VISION_TIMEOUT = 60
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_DIMENSION = 2048
    JPEG_QUALITY = 85
    MAX_STORAGE_SIZE = 1000
    UPLOAD_FOLDER = (os.getenv("UPLOAD_FOLDER") or "uploads").strip() or "uploads"
    MIN_PHONE_LENGTH = 10
    MAX_PHONE_LENGTH = 15
    MIN_NAME_LENGTH = 2
    MIN_ADDRESS_LENGTH = 3
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    DEFAULT_TEXT_MODEL = "llama-3.3-70b-versatile"
    AVAILABLE_MODELS = [
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'meta-llama/llama-guard-4-12b',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        'moonshotai/kimi-k2-instruct',
        'qwen/qwen3-32b',
    ]
    
    # Gemini Configuration (fallback provider)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_MODEL)

    # Bedrock configuration (AWS)
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-1-8b-instruct-v1:0")
    BEDROCK_REGION = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
   
    # Translation Configuration
    TRANSLATION_MODEL = "llama-3.3-70b-versatile"
    TRANSLATION_TIMEOUT = 30
    TRANSLATION_MAX_TOKENS = 1000
    TRANSLATION_TEMPERATURE = 0.3
    SUPPORTED_LANGUAGES = ["en", "ar"]
    LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.7

    # Image search / visual enrichment
    IMAGE_SEARCH_ENABLED = os.getenv("ENABLE_IMAGE_SEARCH", "1").strip() not in {"0", "false", "False"}
    IMAGE_SEARCH_PROVIDER = os.getenv("IMAGE_SEARCH_PROVIDER", "google_cse").strip().lower()
    IMAGE_SEARCH_MAX_RESULTS = int(os.getenv("IMAGE_SEARCH_MAX_RESULTS", "3") or 3)
    IMAGE_SEARCH_TIMEOUT = float(os.getenv("IMAGE_SEARCH_TIMEOUT", "4.0") or 4.0)
    IMAGE_SEARCH_CACHE_TTL = int(os.getenv("IMAGE_SEARCH_CACHE_TTL", "900") or 900)
 
# Secure RAG JSONL logger helper
def _sr_log(entry: Dict[str, Any]) -> None:
    try:
        path = Path((os.getenv("SECURE_RAG_LOG_PATH") or "logs/secure_rag.log").strip() or "logs/secure_rag.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
 
class ServiceManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._storage = []
        return cls._instance
    def get_llm_service(self):
        if 'llm' not in self._services:
            self._services['llm'] = LLMService()
        return self._services['llm']
    def get_storage_service(self):
        if 'storage' not in self._services:
            self._services['storage'] = StorageService(self._storage)
        return self._services['storage']
    def get_file_service(self):
        if 'file' not in self._services:
            self._services['file'] = FileService()
        return self._services['file']
    def get_validation_service(self):
        if 'validation' not in self._services:
            self._services['validation'] = ValidationService()
        return self._services['validation']
    def get_translation_service(self):
        if 'translation' not in self._services:
            self._services['translation'] = LLMTranslationService()
        return self._services['translation']
    def get_image_service(self):
        if 'image' not in self._services:
            self._services['image'] = ImageSearchService()
        return self._services['image']
 
def get_service_manager():
    return ServiceManager()


def _gemini_vertex_enabled() -> bool:
    provider = (os.getenv("GEMINI_PROVIDER") or "").strip().lower()
    if provider in {"vertex", "vertexai", "gcp"}:
        return True
    if (os.getenv("GEMINI_USE_VERTEX") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    creds_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    # Auto-enable Vertex only when no API key is present (avoid accidental switch)
    return bool(creds_path and Path(creds_path).exists() and not Config.GEMINI_API_KEY)


def _infer_gcp_project_from_credentials() -> str:
    creds_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not creds_path:
        return ""
    try:
        with open(creds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("project_id") or "").strip()
    except Exception:
        return ""


def _get_gcp_project_and_location() -> Tuple[str, str]:
    project = (
        (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
        or (os.getenv("GOOGLE_PROJECT_ID") or "").strip()
        or (os.getenv("GCP_PROJECT") or "").strip()
        or (os.getenv("VERTEX_PROJECT") or "").strip()
        or _infer_gcp_project_from_credentials()
    )
    location = (
        (os.getenv("GOOGLE_CLOUD_LOCATION") or "").strip()
        or (os.getenv("GCP_LOCATION") or "").strip()
        or (os.getenv("VERTEX_LOCATION") or "").strip()
        or "us-central1"
    )
    return project, location


def _gemini_generate_text(prompt: str, temperature: float, max_output_tokens: int) -> str:
    """
    Gemini generation supporting:
    - Vertex/Service Account via `google-genai` (preferred when configured)
    - API key via `google-generativeai` (legacy)
    """
    if _gemini_vertex_enabled():
        from google import genai
        from google.genai import types

        project, location = _get_gcp_project_and_location()
        if not project:
            raise ValueError("Vertex Gemini enabled but no GCP project configured")

        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return getattr(response, "text", "") or ""

    import google.generativeai as genai

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("No Gemini API key configured")

    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text if getattr(response, "text", None) else ""


def _gemini_analyze_vision(image_base64: str, query: str, temperature: float, max_output_tokens: int) -> str:
    if _gemini_vertex_enabled():
        import base64
        from google import genai
        from google.genai import types

        project, location = _get_gcp_project_and_location()
        if not project:
            raise ValueError("Vertex Gemini enabled but no GCP project configured")

        client = genai.Client(vertexai=True, project=project, location=location)
        image_bytes = base64.b64decode(image_base64)
        parts = [
            types.Part.from_text(text=query),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ]
        response = client.models.generate_content(
            model=Config.GEMINI_VISION_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return getattr(response, "text", "") or ""

    # API-key vision via google-generativeai
    import base64
    import google.generativeai as genai

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("No Gemini API key configured for vision")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(Config.GEMINI_VISION_MODEL)
    image_bytes = base64.b64decode(image_base64)
    response = model.generate_content(
        [
            query,
            {"mime_type": "image/jpeg", "data": image_bytes},
        ],
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text if getattr(response, "text", None) else ""
 
class LLMService:
    def __init__(self):
        # Try to load API key, but don't hard-fail: enable graceful fallback
        try:
            self._api_key = self._get_api_key()
        except Exception as e:
            logger.warning(f"GROQ_API_KEY not configured; enabling fallback mode ({e})")
            self._api_key = ""
 
        # Build headers (omit Authorization if no key)
        self._headers = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"
 
        # Lazy init for Secure RAG components
        self._input_guard = None
        self._output_guard = None
        self._rbac = None
    def _get_api_key(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or len(api_key.strip()) < 20:
            raise ValueError("GROQ_API_KEY not configured properly")
        return api_key.strip()
    def generate_text(self, messages, model=None, trace_name: str = "shared-llm-call", **kwargs):
        """Generate text via Groq HTTP API; fallback to secure_rag GroqService if unavailable.
 
        messages: list of {role, content}. We will extract the latest user prompt and any system
        context for fallback mode.
        """
        prefer_env = (os.getenv("PREFER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
        if prefer_env in {"aws", "bedrock"}:
            from aws_bedrock import BedrockNotConfigured, invoke_llama31_text

            try:
                return invoke_llama31_text(
                    messages=list(messages or []),
                    max_tokens=int(kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS)),
                    temperature=float(kwargs.get("temperature", Config.DEFAULT_TEMPERATURE)),
                )
            except BedrockNotConfigured as e:
                raise RuntimeError(str(e)) from e

        used_model = model or Config.DEFAULT_TEXT_MODEL
        
        # Extract query preview for telemetry
        query_preview = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query_preview = msg.get("content", "")[:100]
                break
        
        # If API key is missing, go straight to fallback
        if not self._api_key:
            return self._generate_text_fallback(messages, model=model, trace_name=trace_name, **kwargs)
 
        payload = {
            "model": used_model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
        }
        if 'response_format' in kwargs and kwargs['response_format']:
            payload["response_format"] = kwargs['response_format']
 
        # Telemetry: trace this LLM call
        with trace_llm_call(
            name=trace_name,
            model=f"groq/{used_model}",
            input_data={"messages": messages},
            model_parameters={"temperature": payload["temperature"], "max_tokens": payload["max_tokens"]},
            metadata={"source": "shared_utils.LLMService"}
        ) as trace:
            try:
                response = requests.post(
                    Config.GROQ_API_URL,
                    json=payload,
                    headers=self._headers,
                    timeout=kwargs.get('timeout', Config.DEFAULT_TIMEOUT)
                )
                response.raise_for_status()
                response_data = response.json()
                result = response_data["choices"][0]["message"]["content"]
                
                # Extract token usage
                usage = None
                if "usage" in response_data:
                    usage = {
                        "prompt_tokens": response_data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": response_data["usage"].get("completion_tokens", 0),
                        "total_tokens": response_data["usage"].get("total_tokens", 0)
                    }
                
                trace.update(
                    output=result,
                    usage=usage,
                    metadata={"success": True, "provider": "groq"}
                )
                return result
            except Exception as e:
                logger.warning(f"Groq HTTP call failed; using fallback ({_redact_secrets(str(e))})")
                log_llm_event("groq-http-fallback", {"error": _redact_secrets(str(e))}, level="WARNING")
                trace.update(
                    output=f"Fallback triggered: {e}",
                    metadata={"success": False, "fallback": True}
                )
                return self._generate_text_fallback(messages, model=model, trace_name=f"{trace_name}-fallback", **kwargs)
 
    def _generate_text_fallback(self, messages, model=None, trace_name: str = "shared-llm-fallback", **kwargs) -> str:
        """Use Gemini as fallback when GROQ fails."""
        prefer_env = (os.getenv("PREFER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
        if prefer_env in {"aws", "bedrock"}:
            from aws_bedrock import BedrockNotConfigured, invoke_llama31_text

            try:
                return invoke_llama31_text(
                    messages=list(messages or []),
                    max_tokens=int(kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS)),
                    temperature=float(kwargs.get("temperature", Config.DEFAULT_TEMPERATURE)),
                )
            except BedrockNotConfigured as e:
                raise RuntimeError(str(e)) from e

        try:
            # Convert messages to Gemini format
            prompt_parts = []
            for m in (messages or []):
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role == "system":
                        prompt_parts.insert(0, f"Instructions: {content}\n")
                    else:
                        prompt_parts.append(content)
            
            full_prompt = "\n".join(prompt_parts)
            
            # Telemetry: trace fallback call
            with trace_llm_call(
                name=trace_name,
                model=f"gemini/{Config.GEMINI_MODEL}",
                input_data={"prompt": full_prompt[:500]},
                model_parameters={"temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)},
                metadata={"source": "shared_utils.LLMService.fallback"}
            ) as trace:
                result = _gemini_generate_text(
                    full_prompt,
                    temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
                    max_output_tokens=kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                )
                
                trace.update(
                    output=result[:500],
                    metadata={"success": True, "fallback_provider": "gemini"}
                )
                logger.info(f"✅ Fallback to Gemini successful")
                return result
        except Exception as e:
            logger.error(f"Fallback generation failed: {_redact_secrets(str(e))}")
            log_llm_event("fallback-generation-failed", {"error": _redact_secrets(str(e))}, level="ERROR")
            return "I'm sorry, I'm unable to generate a response right now."
 
    # --- Secure RAG integration wrappers ---
    def _init_secure_services(self):
        if self._input_guard is None or self._output_guard is None or self._rbac is None:
            try:
                from secure_rag.input_guard_service import InputGuardService
                from secure_rag.output_guard_service import OutputGuardService
                from secure_rag.rbac_service import RBACService, AccessLevel
                self._InputGuardServiceCls = InputGuardService
                self._OutputGuardServiceCls = OutputGuardService
                self._RBACServiceCls = RBACService
                self._AccessLevelCls = AccessLevel
                # Create instances (verbose False to reduce console noise in prod)
                self._input_guard = self._InputGuardServiceCls(verbose=False)
                self._output_guard = self._OutputGuardServiceCls(verbose=False)
                self._rbac = self._RBACServiceCls(verbose=False)
 
                # Seed RBAC with a minimal document set if empty, so context building is useful
                try:
                    if hasattr(self._rbac, 'documents') and not self._rbac.documents:
                        sample_docs = [
                            {
                                "owner_ids": ["hr_admin", "hr_executive"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Engineering",
                                "text": "Employee Details: Akash works at ITC Infotech as IS2 level engineer in Engineering department"
                            },
                            {
                                "owner_ids": ["hr_admin", "emp2"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Data Science",
                                "text": "Employee Details: Arpan works at ITC Infotech as IS1 level Data Scientist in MOC Innovation Team"
                            },
                            {
                                "owner_ids": [],
                                "shared": True,
                                "category": "public",
                                "sensitivity_level": "low",
                                "created_by": "system",
                                "department": "General",
                                "text": "Company Policies: Security guidelines, work hours, and professional conduct standards"
                            },
                            {
                                "owner_ids": ["hr_admin"],
                                "shared": True,
                                "category": "general",
                                "sensitivity_level": "low",
                                "created_by": "hr_admin",
                                "department": "General",
                                "text": "ITC Infotech provides digital transformation and IT services to clients worldwide"
                            }
                        ]
                        try:
                            self._rbac.ingest_documents(sample_docs)
                        except Exception as ie:
                            logger.warning(f"RBAC document seeding failed: {ie}")
                except Exception as seed_err:
                    logger.warning(f"RBAC initialization warning: {seed_err}")
            except Exception as e:
                logger.error(f"Secure RAG init failed: {e}")
                self._input_guard = None
                self._output_guard = None
                self._rbac = None
 
    def _map_role_to_user_id(self, user_role: Optional[str]) -> str:
        role = (user_role or "user").strip().lower()
        return "hr_admin" if role == "admin" else "hr_common"
 
    def _should_include_rbac_context(self, prompt: str) -> bool:
        """Heuristic to include RBAC document context only for document-relevant queries.
        Reduces leakage of internal names into unrelated answers (e.g., jokes).
        """
        text = (prompt or "").lower()
        triggers = [
            "employee", "employees", "hr", "confidential", "policy", "policies",
            "document", "documents", "docs", "company policy", "company policies",
            "details for", "itc infotech", "akash", "arpan"
        ]
        return any(t in text for t in triggers)
 
    def generate_text_secure(
        self,
        user_text: str,
        user_role: Optional[str] = None,
        model: Optional[str] = None,
        context_tail: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Secure RAG wrapper around generate_text with input/output guards and RBAC.
 
        - user_role: "admin" or "user" (defaults to user)
        - model: keep your adapter's chosen model (auto/llama/qwen/gptoss)
        - context_tail: optional recent conversation to include for tone/continuity
        """
        # Initialize Secure RAG services (lazy)
        self._init_secure_services()
 
        # If Secure RAG unavailable, fall back to vanilla generation
        if not self._input_guard or not self._output_guard or not self._rbac:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text},
            ]
            return self.generate_text(messages, model=model, **kwargs)
 
        # 1) Input guard
        scan = self._input_guard.scan_input(user_text or "")
        scan_results = getattr(scan, "scanner_results", {}) if scan else {}
        sanitized_prompt = getattr(scan, "sanitized_prompt", user_text)
        sanitized_changed = sanitized_prompt != user_text
 
        invalid_scanners = [name for name, ok in scan_results.items() if not ok]
        admin_override = False
        if not getattr(scan, "is_valid", True):
            if (user_role or "").strip().lower() == "admin" and invalid_scanners and all(name.lower().startswith("anonymize") for name in invalid_scanners):
                admin_override = True
            else:
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "input_guard",
                        "result": "blocked",
                        "scanner_results": scan_results,
                        "scanner_scores": getattr(scan, 'scanner_scores', {}),
                        "warnings": getattr(scan, 'warnings', []),
                        "errors": getattr(scan, 'errors', [])
                    })
                except Exception:
                    pass
                return "Your input was blocked by security policy. Please rephrase."
 
        if admin_override:
            sanitized_prompt = user_text
            sanitized_changed = False
 
        if (user_role or "").strip().lower() == "admin" and sanitized_changed:
            # Preserve full context for administrators to enable RBAC evaluation.
            sanitized_prompt = user_text
            sanitized_changed = False
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "input_guard",
                "result": "ok",
                "scanner_results": scan_results,
                "scanner_scores": getattr(scan, 'scanner_scores', {}),
                "warnings": getattr(scan, 'warnings', []),
                "errors": getattr(scan, 'errors', []),
                "sanitized": sanitized_changed
            })
        except Exception:
            pass
 
        # 2) RBAC context (admin gets FULL, user LIMITED)
        user_id_for_rbac = self._map_role_to_user_id(user_role)
        access = self._rbac.check_access(user_id_for_rbac, sanitized_prompt)
        # Build context from accessible docs (only when relevant)
        include_docs = self._should_include_rbac_context(sanitized_prompt)
        rbac_context = ""
        if include_docs and access and getattr(access, "filtered_documents", None):
            ctx_parts = []
            for doc in access.filtered_documents:
                part = f"Document: {doc.page_content}"
                cat = doc.metadata.get('category') if isinstance(doc.metadata, dict) else None
                if cat:
                    part += f" (Category: {cat})"
                ctx_parts.append(part)
            rbac_context = "\n\n".join(ctx_parts)
 
        # Optionally include conversation tail
        composed_context = rbac_context
        if context_tail and isinstance(context_tail, str) and len(context_tail.strip()) > 0:
            # Only append conversation tail; do not inject default doc text when none
            composed_context = (composed_context + ("\n\nRecent conversation:\n" + context_tail.strip() if composed_context else f"Recent conversation:\n{context_tail.strip()}"))
 
        # 3) Call existing adapter generate_text (preserve your model selection)
        # Add explicit role-based policy instructions to avoid contradictory answers
        AccessLevel = self._AccessLevelCls  # type: ignore[attr-defined]
        level = AccessLevel.FULL if (user_role or "").strip().lower() == "admin" else AccessLevel.LIMITED
        base_system = "You are a helpful AI assistant."
        if level.name == "LIMITED":
            policy = (
                " Do not disclose any sensitive or employee-specific personal details. "
                "If asked for confidential information, politely refuse and offer only general, public information."
            )
        else:
            policy = (
                " You may answer using the provided document context when the question is about policies, employees, or company documents. "
                "Do not claim you are unauthorized when the context allows answering. Do not introduce internal names in unrelated topics."
            )
        if composed_context:
            sys_content = f"{base_system}{policy} Use this context if relevant: {composed_context}"
        else:
            sys_content = f"{base_system}{policy}"
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": sanitized_prompt},
        ]
        raw = self.generate_text(messages, model=model, **kwargs)
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "llm_generation",
                "model": model or Config.DEFAULT_TEXT_MODEL,
                "prompt_len": len(sanitized_prompt or ""),
                "context_len": len(composed_context or "")
            })
        except Exception:
            pass
 
        # 4) Output guard with role-based enforcement
        try:
            out = self._output_guard.scan_output(sanitized_prompt, raw, access_level=level)
            if not getattr(out, "is_valid", True):
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "output_guard",
                        "result": "blocked",
                        "warnings": getattr(out, 'warnings', []),
                        "errors": getattr(out, 'errors', []),
                        "quality": getattr(out, 'quality_score', None)
                    })
                except Exception:
                    pass
                return "The response was blocked by security policy. Please try a different question."
            final = getattr(out, "sanitized_response", raw) or raw
            try:
                _sr_log({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "output_guard",
                    "result": "ok",
                    "warnings": getattr(out, 'warnings', []),
                    "errors": getattr(out, 'errors', []),
                    "quality": getattr(out, 'quality_score', None),
                    "final_len": len(final or "")
                })
            except Exception:
                pass
            return MessageFormatter.clean_markdown_text(final)
        except Exception as e:
            logger.error(f"Output guard failed, returning raw: {e}")
            return MessageFormatter.clean_markdown_text(raw)
    def analyze_vision(self, image_base64, query, **kwargs):
        for model in Config.VISION_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
                }
                response = requests.post(Config.GROQ_API_URL, json=payload, headers=self._headers, timeout=Config.VISION_TIMEOUT)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception:
                continue
        # Fallback to Gemini vision if configured (Vertex or API key)
        try:
            return _gemini_analyze_vision(
                image_base64=image_base64,
                query=query,
                temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
                max_output_tokens=kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
            )
        except Exception:
            raise Exception("All vision models failed")
 
class ModelSelector:
    @staticmethod
    def choose_model(user_text: Optional[str], preferred_model: Optional[str], available_models: Optional[List[str]] = None, default_model: Optional[str] = None) -> str:
        # Honor explicit user preference when provided
        if preferred_model and isinstance(preferred_model, str) and preferred_model.strip():
            return preferred_model.strip()
 
        text = (user_text or "").lower()
        available_models = available_models or Config.AVAILABLE_MODELS
        default_model = default_model or Config.DEFAULT_TEXT_MODEL
 
        # Helper to pick best available option among candidates, else default
        def pick(*candidates: str) -> str:
            for cand in candidates:
                if cand in available_models:
                    return cand
            return default_model
 
        reasoning_triggers = [
            "why", "explain", "step by step", "think", "reason", "justify", "proof", "derive",
            "analyze", "compare", "evaluate", "plan", "strategy", "trade-off", "pros and cons",
            "eli5", "explain like i'm", "explain like im"
        ]
        coding_triggers = [
            "code", "python", "javascript", "typescript", "bug", "error", "stack trace", "function", "class", "regex",
            "refactor", "optimize", "snippet"
        ]
        math_triggers = [
            "calculate", "compute", "solve", "equation", "integral", "derivative", "proof"
        ]
        json_triggers = ["json", "return json", "json_object", "valid json", "response_format"]
        creative_triggers = ["essay", "story", "poem", "creative", "blog", "rewrite", "paraphrase"]
        quick_triggers = ["hi", "hello", "hey", "thanks", "thank you", "help", "who are you", "what can you do"]
 
        long_query = len(text.split()) >= 60 or len(text) >= 300
 
        # Heuristics
        if any(k in text for k in reasoning_triggers) or "chain of thought" in text or "cot" in text:
            # Heavy reasoning → prefer Qwen first, then 120B, then 70B
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in coding_triggers) or "explain code" in text or "refactor" in text:
            # Coding → prefer Qwen first for code tasks
            return pick("qwen/qwen3-32b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in math_triggers):
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in json_triggers):
            return pick("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in creative_triggers):
            return pick("moonshotai/kimi-k2-instruct", "llama-3.3-70b-versatile", "openai/gpt-oss-20b")
 
        if any(text.strip().startswith(t) or t == text.strip() for t in quick_triggers):
            return pick("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if long_query or "summarize" in text or "analyze" in text or "write" in text:
            # Long/summary → prefer 70B or Qwen before 120B to reduce cost/noise
            return pick("llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b")
 
        # Default balanced choice
        return pick("llama-3.3-70b-versatile", "openai/gpt-oss-20b", default_model)
 
class StorageService:
    def __init__(self, storage_list):
        self._storage = storage_list
    def store_user_info(self, user_id, data):
        data.update({"user_id": user_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        existing = next((item for item in self._storage if item["user_id"] == user_id), None)
        if existing:
            existing.update(data)
        else:
            self._storage.append(data)
        self._cleanup()
    def get_user_info(self, user_id):
        return next((item for item in self._storage if item["user_id"] == user_id), None)
    def _cleanup(self):
        if len(self._storage) >= Config.MAX_STORAGE_SIZE:
            keep_count = int(Config.MAX_STORAGE_SIZE * 0.8)
            self._storage[:] = self._storage[-keep_count:]
 
class ValidationService:
    def validate_name(self, name):
        return bool(name and len(name.strip()) >= Config.MIN_NAME_LENGTH)
    def normalize_phone(self, phone):
        """Return a normalized representation for phone numbers."""
        if phone is None:
            return ""
        text = str(phone).strip()
        if not text:
            return ""
        digits = ''.join(ch for ch in text if ch.isdigit())
        if not digits:
            return ""
        return digits
 
    def validate_phone(self, phone):
        normalized = self.normalize_phone(phone)
        if not normalized:
            return False
        digits = ''.join(ch for ch in normalized if ch.isdigit())
        return Config.MIN_PHONE_LENGTH <= len(digits) <= Config.MAX_PHONE_LENGTH
    def validate_address(self, address):
        return bool(address and len(address.strip()) >= Config.MIN_ADDRESS_LENGTH)

class ImageSearchService:
    """Fetch illustrative images using the configured provider (Google CSE by default)."""

    GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        self.cx = os.getenv("GOOGLE_CSE_CX", "").strip()
        self.enabled = (
            Config.IMAGE_SEARCH_ENABLED
            and Config.IMAGE_SEARCH_PROVIDER == "google_cse"
            and bool(self.api_key and self.cx)
        )
        self.cache_ttl = Config.IMAGE_SEARCH_CACHE_TTL
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(key)
        if not cached:
            return None
        if time.time() - cached["ts"] > self.cache_ttl:
            self._cache.pop(key, None)
            return None
        return cached["value"]

    def _cache_set(self, key: str, value: Optional[Dict[str, Any]]) -> None:
        self._cache[key] = {"ts": time.time(), "value": value}

    def fetch_image(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a representative image dict for the given query."""
        if not self.is_enabled():
            return None
        normalized = (query or "").strip()
        if len(normalized) < 2:
            return None
        cache_key = normalized.lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached or None

        params = {
            "q": normalized,
            "searchType": "image",
            "num": max(1, min(Config.IMAGE_SEARCH_MAX_RESULTS, 5)),
            "safe": "active",
            "key": self.api_key,
            "cx": self.cx,
        }
        try:
            response = requests.get(
                self.GOOGLE_ENDPOINT,
                params=params,
                timeout=Config.IMAGE_SEARCH_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items") or []
            for item in items:
                link = item.get("link")
                if not link:
                    continue
                image_meta = item.get("image") or {}
                result = {
                    "image_url": link,
                    "thumbnail_url": image_meta.get("thumbnailLink") or link,
                    "width": image_meta.get("width"),
                    "height": image_meta.get("height"),
                    "context_url": image_meta.get("contextLink") or item.get("link"),
                    "attribution": item.get("displayLink") or item.get("title") or "",
                    "alt": (item.get("snippet") or item.get("title") or normalized).strip(),
                }
                self._cache_set(cache_key, result)
                return result
        except Exception as exc:
            logger.warning("Image search failed for '%s': %s", normalized, exc)
        self._cache_set(cache_key, None)
        return None

    def clear_cache(self) -> None:
        self._cache.clear()

    def analyze_vision(self, image_base64, query, **kwargs):
        for model in Config.VISION_MODELS:
            try:
                headers = {"Content-Type": "application/json"}
                api_key = os.getenv("GROQ_API_KEY", "").strip()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
                }
                response = requests.post(Config.GROQ_API_URL, json=payload, headers=headers, timeout=Config.VISION_TIMEOUT)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception:
                continue
        # Fallback to Gemini vision if configured (Vertex or API key)
        try:
            return _gemini_analyze_vision(
                image_base64=image_base64,
                query=query,
                temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
                max_output_tokens=kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
            )
        except Exception:
            raise Exception("All vision models failed")
 
class FileService:
    def __init__(self):
        self.upload_dir = Path(Config.UPLOAD_FOLDER)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Track last processed file for sticky context
        self._last_processed_file = None
        self._last_file_text = None
        self._last_file_language = None
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        if not text:
            return "en"
        # Count Arabic characters (Unicode range for Arabic)
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "en"
        arabic_ratio = arabic_chars / total_alpha
        return "ar" if arabic_ratio > 0.3 else "en"
    
    def process_image(self, image_path, target_language: str = None):
        try:
            if not self._is_valid_image(image_path):
                return ""
            
            # Store for sticky context
            self._last_processed_file = str(image_path)
            
            llm_service = ServiceManager().get_llm_service()
            prefer_env = (os.getenv("PREFER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()

            # Bedrock Llama 3.1 8B is text-only; use AWS Textract for OCR then summarize with Bedrock.
            if prefer_env in {"aws", "bedrock"}:
                try:
                    import boto3
                except Exception as e:
                    logger.error(f"Bedrock mode requires boto3 for Textract OCR: {e}")
                    return ""

                region = (
                    (os.getenv("AWS_REGION") or "").strip()
                    or (os.getenv("AWS_DEFAULT_REGION") or "").strip()
                    or (os.getenv("BEDROCK_REGION") or "").strip()
                )
                if not region:
                    logger.error("Bedrock mode requires AWS_REGION (Textract OCR)")
                    return ""

                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    textract = boto3.client("textract", region_name=region)
                    resp = textract.detect_document_text(Document={"Bytes": image_bytes})
                    lines = [b.get("Text", "") for b in (resp.get("Blocks") or []) if (b or {}).get("BlockType") == "LINE"]
                    extracted = "\n".join([ln for ln in lines if ln]).strip()
                except Exception as e:
                    logger.error(f"Textract OCR failed: {e}")
                    return ""

                detected_lang = target_language or self._detect_language(extracted)
                self._last_file_language = detected_lang
                self._last_file_text = extracted[:8000]

                if detected_lang == "ar":
                    system_prompt = "أنت مساعد مفيد. لخص النص المستخرج من الصورة في 30-50 كلمة بالعربية. كن مباشرًا ولا تضف معلومات غير موجودة."
                    user_prompt = f"النص المستخرج من الصورة:\n\n{extracted[:4000]}"
                else:
                    system_prompt = "You are a helpful assistant. Summarize the OCR text from the image in 30-50 words. Be concise and do not add extra information."
                    user_prompt = f"OCR text from image:\n\n{extracted[:4000]}"

                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                result = llm_service.generate_text(messages, max_tokens=180, temperature=0.2)
                cleaned = MessageFormatter.clean_markdown_text(result)
                if detected_lang == "ar":
                    try:
                        ts = get_service_manager().get_translation_service()
                        if cleaned and not ts.is_arabic(cleaned):
                            cleaned = ts.translate_english_to_arabic(cleaned)
                    except Exception:
                        pass
                self._last_file_text = extracted[:8000]
                return cleaned

            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                return ""
             
            # If no target language specified, first detect if image contains Arabic text
            if not target_language:
                # Ask the model to detect language in the image
                detect_prompt = "Look at this image. Does it contain Arabic text/writing? Reply with ONLY 'ar' if it contains Arabic text, or 'en' if it contains English or no text."
                lang_response = llm_service.analyze_vision(base64_image, detect_prompt)
                detected_lang = "ar" if lang_response and "ar" in lang_response.lower()[:10] else "en"
                target_language = detected_lang
                logger.info(f"🖼️ Image language detection: {detected_lang}")
            
            self._last_file_language = target_language
            
            # Generate description in the detected/target language
            if target_language == "ar":
                prompt = "صف هذه الصورة في 30-50 كلمة بالعربية. كن موجزاً ومباشراً. ابدأ بـ 'الملف الذي رفعته هو' ثم أضف المزيد من المعلومات."
            else:
                prompt = "Describe this image in 30-50 words in English. Be concise. Give the summary directly, don't add extra information. Start with 'The file you uploaded is' and add more info about it."
            
            result = llm_service.analyze_vision(base64_image, prompt)
            self._last_file_text = result or ""  # Store for follow-up questions
            return result
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return ""
    
    def process_document(self, doc_path, target_language: str = None):
        try:
            logger.info(f"📄 Processing document: {doc_path}")
            text = self._extract_document_text(doc_path)
            logger.info(f"📄 Extracted text length: {len(text) if text else 0} chars")
            
            # Log Arabic character count for debugging
            if text:
                arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
                logger.info(f"📄 Arabic characters in extracted text: {arabic_chars}")
            
            if not text or len(text.strip()) < 10:
                logger.warning(f"⚠️ Document text too short or empty: {doc_path}")
                return ""
            
            # Detect language if not specified
            detected_lang = self._detect_language(text)
            use_lang = target_language or detected_lang
            logger.info(f"📄 Detected language: {detected_lang}, using: {use_lang}")
            
            # Store for sticky context
            self._last_processed_file = str(doc_path)
            self._last_file_text = text[:8000]  # Store more for follow-up questions
            self._last_file_language = use_lang
            
            # Language-specific prompts
            if use_lang == "ar":
                system_prompt = "أنت مساعد ذكي. قم بتلخيص المستند في 30-50 كلمة بالعربية. كن موجزاً ومباشراً."
                user_prompt = f"المستند:\n\n{text[:4000]}"
            else:
                system_prompt = "You are a helpful assistant. Summarize the document in 30-50 words. Be concise and direct. dont add extra information."
                user_prompt = f"Document:\n\n{text[:4000]}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            llm_service = ServiceManager().get_llm_service()
            result = llm_service.generate_text(messages, max_tokens=150, temperature=0.3)
            logger.info(f"✅ Document analysis complete: {len(result) if result else 0} chars")
            cleaned = MessageFormatter.clean_markdown_text(result)
            if use_lang == "ar":
                try:
                    ts = get_service_manager().get_translation_service()
                    if cleaned and not ts.is_arabic(cleaned):
                        cleaned = ts.translate_english_to_arabic(cleaned)
                except Exception:
                    pass
            return cleaned
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return ""
    
    def get_last_file_context(self) -> dict:
        """Get context of the last processed file for sticky follow-up questions."""
        return {
            "file_path": self._last_processed_file,
            "text": self._last_file_text,
            "language": self._last_file_language
        }
    
    def answer_followup(self, question: str, target_language: str = None) -> str:
        """Answer a follow-up question about the last processed file."""
        if not self._last_processed_file or not self._last_file_text:
            return "No file context available. Please upload a file first."
        
        use_lang = target_language or self._last_file_language or "en"
        
        if use_lang == "ar":
            system_prompt = "أنت مساعد ذكي. أجب على السؤال بناءً على محتوى المستند. أجب بالعربية في 30-50 كلمة.  dont add extra information."
        else:
            system_prompt = "You are a helpful assistant. Answer the question based on the document content. Respond in English in 30-50 words. dont add extra information."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document content:\n{self._last_file_text[:4000]}\n\nQuestion: {question}"}
        ]
        
        llm_service = ServiceManager().get_llm_service()
        return llm_service.generate_text(messages, max_tokens=150, temperature=0.3)
    
    def find_recent_files(self, file_type):
        extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp'},
            'document': {'.pdf', '.docx', '.txt'}
        }
        if file_type not in extensions:
            return []
        files = []
        if self.upload_dir.exists():
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions[file_type]:
                    files.append(file_path)
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    def _encode_image_to_base64(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            if len(image_bytes) > Config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                if image.width > Config.MAX_IMAGE_DIMENSION or image.height > Config.MAX_IMAGE_DIMENSION:
                    image.thumbnail((Config.MAX_IMAGE_DIMENSION, Config.MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=Config.JPEG_QUALITY)
                image_bytes = img_buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    def _extract_document_text(self, doc_path):
        if doc_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(doc_path)
        elif doc_path.suffix.lower() == '.docx':
            return self._extract_from_docx(doc_path)
        elif doc_path.suffix.lower() == '.txt':
            return self._extract_from_txt(doc_path)
        else:
            raise Exception(f"Unsupported file type: {doc_path.suffix}")
    def _extract_from_pdf(self, pdf_path):
        """Extract text from PDF with Arabic/Unicode support using PyMuPDF."""
        text = ""
        
        # Try PyMuPDF first (better Arabic/Unicode support)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            logger.info(f"📄 PDF has {page_count} pages (using PyMuPDF)")
            
            for i, page in enumerate(doc):
                try:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_err:
                    logger.warning(f"⚠️ PyMuPDF failed on page {i+1}: {page_err}")
                    continue
            
            doc.close()
            
            # Check if we got meaningful Arabic text
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
            logger.info(f"📄 PyMuPDF extracted: {len(text)} chars, {arabic_chars} Arabic chars")
            
            if text.strip():
                return text.strip()
                
        except ImportError:
            logger.warning("⚠️ PyMuPDF not installed, falling back to PyPDF2")
        except Exception as e:
            logger.warning(f"⚠️ PyMuPDF extraction failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                page_count = len(pdf_reader.pages)
                logger.info(f"📄 PDF has {page_count} pages (using PyPDF2)")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_err:
                        logger.warning(f"⚠️ Failed to extract page {i+1}: {page_err}")
                        continue
                
                logger.info(f"📄 PyPDF2 extracted: {len(text)} chars from {page_count} pages")
                return text.strip()
        except Exception as e:
            logger.error(f"❌ PDF extraction failed: {e}")
            return ""
    def _extract_from_docx(self, docx_path):
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    def _extract_from_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
 
class MessageFormatter:
    @staticmethod
    def clean_markdown_text(text):
        if not text:
            return text
        # Strip chain-of-thought style tags and obvious reasoning lead-ins
        try:
            import re
            # Remove <think>...</think> blocks if model leaked internal reasoning
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Remove stray HTML-like tags that may break frontend rendering
            text = re.sub(r"<\/?[a-zA-Z][^>]*>", "", text)
            # Trim common meta lead-ins like "Okay," "Let's" when leaking planning
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pruned = []
            dropping = True
            for ln in lines:
                if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I will|I cannot|The user|We need to|Reasoning|Analysis|Plan)\b", ln, flags=re.IGNORECASE):
                    dropping = False
                if not dropping:
                    pruned.append(ln)
            text = (" ".join(pruned) if pruned else text).strip()
        except Exception:
            # Best-effort only; continue with other cleanups
            pass
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        text = re.sub(r'\*\*([^*]*?)\*\*', r'\1', text)
        text = re.sub(r'__([^_]*?)__', r'\1', text)
        text = re.sub(r'\*([^*]*?)\*', r'\1', text)
        text = re.sub(r'_([^_]*?)_', r'\1', text)
        text = re.sub(r'^#{1,6}\s*(.*)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', '', text)
        text = text.replace('**', '').replace('__', '')
        return ' '.join(text.split()).strip()
 
    @staticmethod
    def truncate_sentences(text: str, max_sentences: int = 2) -> str:
        """Return only the first `max_sentences` sentences from text."""
        if not text or max_sentences <= 0:
            return text
        try:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if not sentences:
                return text.strip()
            truncated = ' '.join(sentences[:max_sentences]).strip()
            return truncated or text.strip()
        except Exception:
            return text.strip()
    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
 
class LLMTranslationService:
    """Translation service using GROQ Qwen3.3 32B for Arabic ↔ English translation"""
   
    def __init__(self):
        self.llm_service = None  # Will be initialized lazily to avoid circular imports
        self.cache = {}  # In-memory cache as fallback
        self.redis_cache = None  # Redis cache (will be initialized if available)
        self._init_redis_cache()
        self.translation_prompts = {
            "ar_to_en": """You are a professional Arabic to English translator. Translate the following Arabic text to clear, natural English while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
Arabic: {text}
 
English:""",
           
            "en_to_ar": """You are a professional English to Arabic translator. Translate the following English text to clear, natural Arabic while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
English: {text}
 
Arabic:"""
        }
        self._markdown_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self._ar_to_en_fixed = {
            # Contact card / user info flows
            "اعرض معلوماتي": "Show my info",
            "أعرض معلوماتي": "Show my info",
            "اظهر معلوماتي": "Show my info",
            "أظهر معلوماتي": "Show my info",
            "اعرض بطاقة الاتصال": "Show my info",
            "اعرض بطاقة اتصالي": "Show my info",
            "أعرض بطاقة الاتصال": "Show my info",
            # Contact info collection
            "اجمع معلوماتي": "Collect my info",
            "أجمع معلوماتي": "Collect my info",
            "اجمع بياناتي": "Collect my info",
            # Common tool triggers
            "أنشئ استبياناً": "Generate a survey",
            "أنشئ استبيانا": "Generate a survey",
            "انشئ استبيانا": "Generate a survey",
            "أعطني أهم الأخبار حول الذكاء الاصطناعي": "Give me top headlines about AI",
        }
        self.COMMON_PHRASES = {
        "learn more": "اعرف المزيد",
        "read more": "اقرأ المزيد",
        "view details": "عرض التفاصيل",
        "submit": "إرسال",
        "cancel": "إلغاء",
        "close": "إغلاق",
        "open": "فتح",
        "search": "بحث",
        "menu": "القائمة",
        "home": "الرئيسية",
        "back": "رجوع",
        "next": "التالي",
        "previous": "السابق",
        "yes": "نعم",
        "no": "لا",
        "ok": "موافق",
        "confirm": "تأكيد",
        "apply": "تطبيق",
        "reset": "إعادة تعيين",
        "save": "حفظ",
        "delete": "حذف",
        "edit": "تعديل",
        "update": "تحديث",
        "create": "إنشاء",
        "new": "جديد",
        "add": "إضافة",
        "remove": "إزالة",
        "select": "تحديد",
        "choose": "اختر",
        "upload": "رفع",
        "download": "تحميل",
        "share": "مشاركة",
        "send": "إرسال",
        "reply": "رد",
        "forward": "إعادة توجيه",
        "copy": "نسخ",
        "paste": "لصق",
        "cut": "قص",
        "undo": "تراجع",
        "redo": "إعادة",
        "print": "طباعة",
        "export": "تصدير",
        "import": "استيراد",
        "filter": "تصفية",
        "sort": "فرز",
        "group": "تجميع",
        "list": "قائمة",
        "grid": "شبكة",
        "table": "جدول",
        "chart": "رسم بياني",
        "graph": "رسم بياني",
        "map": "خريطة",
        "calendar": "تقويم",
        "timeline": "خط زمن",
        "form": "نموذج",
        "report": "تقرير",
        "dashboard": "لوحة القيادة",
        "brochure": "كتيب",
        "click here": "اضغط هنا",
        "error": "خطأ",
        "success": "نجاح",
        "warning": "تحذير",
        "info": "معلومات",
        "help": "مساعدة",
        "settings": "الإعدادات",
        "profile": "الملف الشخصي",
        "logout": "تسجيل الخروج",
        "login": "تسجيل الدخول",
        "welcome": "مرحباً"
    }

    def _translate_plain_en_to_ar(self, english_text: str) -> str:
        """Translate text without preserving Markdown (used internally)."""
        if not english_text or not english_text.strip():
            return english_text

        # Check common phrases first (case-insensitive)
        lower_text = english_text.lower().strip().strip(".,!?:;")
        if lower_text in self.COMMON_PHRASES:
            return self.COMMON_PHRASES[lower_text]

        cached = self._get_cached_translation(english_text, "en_to_ar_plain")
        if cached:
            return cached

        def _extract_emojis(text: str) -> List[str]:
            if not text:
                return []
            emojis: List[str] = []
            i = 0
            while i < len(text):
                code = ord(text[i])
                is_emoji = (0x1F000 <= code <= 0x1FAFF) or (0x2600 <= code <= 0x27BF)
                if is_emoji:
                    emoji = text[i]
                    if i + 1 < len(text) and ord(text[i + 1]) == 0xFE0F:
                        emoji += text[i + 1]
                        i += 1
                    emojis.append(emoji)
                i += 1
            return emojis

        def _replace_emojis_with_placeholders(text: str) -> Tuple[str, Dict[str, str], List[str]]:
            emojis = _extract_emojis(text)
            if not emojis:
                return text, {}, []
            mapping: Dict[str, str] = {}
            replaced = text
            # Replace in order of appearance; stable placeholders.
            for idx, emoji in enumerate(emojis, start=1):
                # Keep token ASCII-only and compatible with `_strip_reasoning()`'s Arabic sanitizer.
                token = f"[[{idx}]]"
                mapping[token] = emoji
                replaced = replaced.replace(emoji, token)
            return replaced, mapping, emojis

        prepared_text, emoji_map, source_emojis = _replace_emojis_with_placeholders(english_text)

        # Use a more robust prompt to prevent hallucination/garbage output
        prompt = (
            "You are a professional translator. Translate the following English text to Arabic.\n"
            "Rules:\n"
            "1. Provide ONLY the Arabic translation.\n"
            "2. Do NOT add any explanations, notes, or extra text.\n"
            "3. Do NOT repeat the input text.\n"
            "4. If the text is a proper noun or technical term that should not be translated, keep it as is.\n"
            "5. Ensure the translation is natural and grammatically correct.\n"
            "6. If the text is a short UI label (like a button), translate it concisely (1-3 words).\n"
            "7. Preserve punctuation and symbols exactly as they appear.\n"
            "8. Do NOT remove or alter placeholder tokens like [[1]]. Keep them exactly.\n\n"
            f"English: {prepared_text}\n\n"
            "Arabic:"
        )
        messages = [{"role": "user", "content": prompt}]

        # Use a lower temperature for more deterministic output, and retry on rate limits
        # or clearly invalid (non-Arabic) "translations".
        translation: Optional[str] = None
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                response_text = self._get_llm_service().generate_text(
                    messages=messages,
                    model=Config.TRANSLATION_MODEL,
                    max_tokens=Config.TRANSLATION_MAX_TOKENS,
                    temperature=0.1,  # Lower temperature to reduce hallucinations
                    timeout=Config.TRANSLATION_TIMEOUT,
                )
                if response_text is None:
                    raise RuntimeError("Empty translation response")

                candidate = str(response_text).strip()
                candidate = self._strip_reasoning(candidate, target_lang="ar")
                if candidate.startswith('"') and candidate.endswith('"'):
                    candidate = candidate[1:-1]

                # Restore emoji placeholders (when present)
                if emoji_map:
                    for token, emoji in emoji_map.items():
                        candidate = candidate.replace(token, emoji)

                    # If the model dropped placeholders, re-add any missing emojis so UI keeps them.
                    for emoji in source_emojis:
                        if emoji and emoji not in candidate:
                            candidate = f"{candidate} {emoji}".strip()

                # If the output is clearly not Arabic for an input that contains letters,
                # treat as failure (common with LLM fallbacks during rate limits).
                if any(ch.isalpha() for ch in english_text) and not self.is_arabic(candidate):
                    raise ValueError("Non-Arabic translation output")

                translation = candidate
                break
            except Exception as e:
                last_error = e
                translation = None
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg:
                    delay = 0.8 * (2**attempt)
                    logger.warning(
                        f"EN->AR translation rate-limited, retrying in {delay:.1f}s (attempt {attempt + 1}/3)"
                    )
                    time.sleep(delay)
                    continue
                if "Non-Arabic translation output" in msg:
                    delay = 0.4 * (2**attempt)
                    logger.warning(
                        f"EN->AR translation invalid, retrying in {delay:.1f}s (attempt {attempt + 1}/3)"
                    )
                    time.sleep(delay)
                    continue
                raise

        if translation is None:
            logger.error(f"EN->AR translation failed after retries: {_redact_secrets(str(last_error))}")
            return english_text

        self._cache_translation(english_text, translation, "en_to_ar_plain")
        return translation

    def _segment_text_with_links(self, text: str) -> List[Dict[str, str]]:
        segments: List[Dict[str, str]] = []
        if not text:
            return segments
        last_index = 0
        for match in self._markdown_link_pattern.finditer(text):
            if match.start() > last_index:
                segments.append({"type": "text", "value": text[last_index:match.start()]})
            segments.append({"type": "link", "label": match.group(1), "url": match.group(2)})
            last_index = match.end()
        if last_index < len(text):
            segments.append({"type": "text", "value": text[last_index:]})
        return segments

    def _sanitize_markdown_label(self, translated_label: str, fallback: str) -> str:
        """Ensure translated link labels cannot break markdown syntax."""
        clean = (translated_label or "").strip()
        if not clean:
            return fallback
        clean = re.sub(r"\s+", " ", clean)
        clean = clean.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        clean = clean.strip()
        return clean if clean else fallback

    def _translate_text_segment_en_to_ar(self, segment: str) -> str:
        if not segment:
            return segment
        if segment.strip() == "":
            return segment
        prefix_match = re.match(r"^\s*", segment)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix_match = re.search(r"\s*$", segment)
        suffix = suffix_match.group(0) if suffix_match else ""
        core_start = len(prefix)
        core_end = len(segment) - len(suffix) if suffix else len(segment)
        core = segment[core_start:core_end]
        translated_core = self._translate_plain_en_to_ar(core) if core else core
        return f"{prefix}{translated_core}{suffix}"

    def _normalize_arabic_digits(self, text: str) -> str:
        """Convert Eastern Arabic digits to ASCII digits so Markdown numbering stays intact."""
        if not text:
            return text
        digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
        return text.translate(digit_map)
   
    def _init_redis_cache(self):
        """Initialize Redis cache if available"""
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_db = int(os.getenv('REDIS_DB', '0'))
           
            self.redis_cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_cache.ping()
            logger.info("✅ Redis cache initialized successfully")
        except ImportError:
            logger.warning("⚠️ Redis not installed. Using in-memory cache only.")
            self.redis_cache = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using in-memory cache only.")
            self.redis_cache = None
   
    def _get_llm_service(self):
        """Lazy initialization of LLM service to avoid circular imports"""
        if self.llm_service is None:
            self.llm_service = get_service_manager().get_llm_service()
        return self.llm_service
   
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        Returns: (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 2:
            return "en", 0.5  # Default to English for very short text
       
        text = text.strip()
       
        # Check for Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
       
        if total_chars == 0:
            return "en", 0.5  # Default for non-alphabetic text
       
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
       
        # If more than 30% Arabic characters, likely Arabic
        if arabic_ratio > 0.3:
            confidence = min(0.9, 0.5 + arabic_ratio)
            return "ar", confidence
       
        # Use langdetect if available for better detection
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(text)
                for lang_obj in detected_langs:
                    if lang_obj.lang == "ar" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "ar", lang_obj.prob
                    elif lang_obj.lang == "en" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "en", lang_obj.prob
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
       
        # Default to English if uncertain
        return "en", 0.6
   
    def _get_cached_translation(self, text: str, direction: str) -> Optional[str]:
        """Get cached translation from Redis or in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
       
        # Try Redis first
        if self.redis_cache:
            try:
                cached = self.redis_cache.get(cache_key)
                if cached:
                    logger.info(f"Redis cache hit ({direction.upper()})")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
       
        # Fallback to in-memory cache
        if cache_key in self.cache:
            logger.info(f"Memory cache hit ({direction.upper()})")
            return self.cache[cache_key]
       
        return None
   
    def _cache_translation(self, text: str, translation: str, direction: str):
        """Cache translation in Redis and in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
        ttl = int(os.getenv('TRANSLATION_CACHE_TTL', '3600'))  # 1 hour default
       
        # Cache in Redis
        if self.redis_cache:
            try:
                self.redis_cache.setex(cache_key, ttl, translation)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
       
        # Cache in memory (with size limit)
        self.cache[cache_key] = translation
        if len(self.cache) > 1000:  # Limit memory cache size
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
 
    def translate_arabic_to_english(self, arabic_text: str) -> str:
        """Translate Arabic text to English using GROQ LLM"""
        if not arabic_text or not arabic_text.strip():
            return arabic_text

        # Deterministic short-phrase translations (more reliable than LLM and helps routing).
        normalized = self._normalize_arabic_for_lookup(arabic_text)
        fixed = self._ar_to_en_fixed.get(normalized)
        if fixed:
            self._cache_translation(arabic_text, fixed, "ar_to_en")
            return fixed
       
        # Check cache first
        cached_translation = self._get_cached_translation(arabic_text, "ar_to_en")
        if cached_translation:
            return cached_translation
       
        try:
            prompt = self.translation_prompts["ar_to_en"].format(text=arabic_text)
            messages = [{"role": "user", "content": prompt}]
           
            translation = self._get_llm_service().generate_text(
                messages=messages,
                model=Config.TRANSLATION_MODEL,
                max_tokens=Config.TRANSLATION_MAX_TOKENS,
                temperature=Config.TRANSLATION_TEMPERATURE,
                timeout=Config.TRANSLATION_TIMEOUT
            )
           
            # Clean up the translation
            translation = translation.strip()
            translation = self._strip_reasoning(translation, target_lang="en")
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
           
            # Cache the result
            self._cache_translation(arabic_text, translation, "ar_to_en")
            logger.info(f"Translated AR→EN: {arabic_text[:50]}... → {translation[:50]}...")
           
            return translation
           
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {e}")
            return f"[Translation Error] {arabic_text}"

    @staticmethod
    def _normalize_arabic_for_lookup(text: str) -> str:
        """Normalize Arabic input for deterministic phrase lookup."""
        if not text:
            return ""
        # Remove tatweel and common punctuation, normalize whitespace
        cleaned = re.sub(r"[\u0640\u200f\u200e]", "", text)
        cleaned = re.sub(r"[؟?!.,،؛:()\\[\\]\"'“”‘’]", "", cleaned)
        cleaned = " ".join(cleaned.strip().split())
        # Remove diacritics
        cleaned = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", cleaned)
        return cleaned
   
    def translate_english_to_arabic(self, english_text: str) -> str:
        """Translate English text to Arabic while preserving links (Markdown and HTML)."""
        if not english_text or not english_text.strip():
            return english_text

        cached_translation = self._get_cached_translation(english_text, "en_to_ar")
        if cached_translation:
            return cached_translation

        try:
            # Pattern for Markdown links: [text](url)
            md_pattern = re.compile(r'^(\d+)\.\s\[(.+?)\]\((https?://[^\s)]+)\)(.*)$')
            # Pattern for HTML links: <a href='url'>text</a>
            html_link_pattern = re.compile(r"<a\s+href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</a>", re.IGNORECASE)
            
            lines_en = english_text.splitlines()
            
            # Check if text contains HTML links
            has_html_links = bool(html_link_pattern.search(english_text))
            use_md_structured = any(md_pattern.match(line.strip()) for line in lines_en)

            if has_html_links:
                # Handle HTML links - translate text parts only, preserve links
                translated_lines: List[str] = []
                for line in lines_en:
                    stripped = line.strip()
                    if not stripped:
                        translated_lines.append('')
                        continue
                    
                    # Check if line has HTML link
                    html_match = html_link_pattern.search(stripped)
                    if html_match:
                        # Extract parts before, the link, and after
                        # Translate the link text only
                        def translate_link_text(match):
                            url = match.group(1)
                            link_text = match.group(2)
                            # Translate the link text
                            translated_text = self._translate_plain_en_to_ar(link_text)
                            return f"<a href='{url}' target='_blank' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{translated_text}</a>"
                        
                        # Replace HTML links with translated versions
                        translated_line = html_link_pattern.sub(translate_link_text, stripped)
                        
                        # Also translate the prefix (like "1. ")
                        prefix_match = re.match(r'^(\d+)\.\s*', translated_line)
                        if prefix_match:
                            # Keep the number prefix as-is
                            pass
                        
                        translated_lines.append(translated_line)
                    elif stripped.startswith("Source:") or "Source:" in stripped:
                        # Translate source line
                        translated_lines.append(self._translate_plain_en_to_ar(stripped))
                    elif stripped == "---MESSAGE_SPLIT---":
                        translated_lines.append(stripped)
                    else:
                        # Translate other lines (descriptions, etc.)
                        translated_lines.append(self._translate_plain_en_to_ar(stripped))
                
                translation = "\n".join(line for line in translated_lines if line is not None)
                
            elif use_md_structured:
                translated_lines: List[str] = []
                for line in lines_en:
                    stripped = line.strip()
                    if not stripped:
                        translated_lines.append('')
                        continue
                    match = md_pattern.match(stripped)
                    if match:
                        index, label, url, remainder = match.groups()
                        arabic_label = self._translate_plain_en_to_ar(label)
                        arabic_label = self._sanitize_markdown_label(arabic_label, label)
                        translated_lines.append(f"{index}. {arabic_label}")
                        translated_lines.append(f"الرابط: <{url}>")
                        translated_lines.append('')
                        remainder = remainder.strip()
                        if remainder:
                            arabic_extra = self._translate_plain_en_to_ar(remainder)
                            translated_lines.append(arabic_extra)
                    else:
                        translated_lines.append(self._translate_plain_en_to_ar(stripped))
                translation = "\n".join(line for line in translated_lines if line is not None)
            else:
                translation = self._translate_plain_en_to_ar(english_text)

            translation = self._normalize_arabic_digits(translation)
            self._cache_translation(english_text, translation, "en_to_ar")
            logger.info(f"Translated EN->AR: {english_text[:50]}... -> {translation[:50]}...")

            return translation
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {e}")
            return english_text  # Return original English if translation fails
   
    def translate_with_fallback(self, text: str, direction: str) -> str:
        """Translate with fallback handling"""
        try:
            if direction == "ar_to_en":
                return self.translate_arabic_to_english(text)
            elif direction == "en_to_ar":
                return self.translate_english_to_arabic(text)
            else:
                logger.error(f"Unknown translation direction: {direction}")
                return text
        except Exception as e:
            logger.error(f"Translation failed ({direction}): {e}")
            if direction == "ar_to_en":
                return f"[Translation Error] {text}"
            else:
                return text  # Return English if Arabic translation fails
   
    def is_arabic(self, text: str) -> bool:
        """Quick check if text contains Arabic characters"""
        if not text:
            return False
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_chars > 0

    def translate_adaptive_card(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate all text content in an Adaptive Card from English to Arabic.
        
        This recursively walks the card JSON structure and translates:
        - TextBlock "text" fields
        - Action "title" fields
        - Input "label" and "placeholder" fields
        - FactSet "title" and "value" fields
        - Column/Container items recursively
        
        Returns a new card dict with translated content.
        """
        if not card or not isinstance(card, dict):
            return card
        
        from copy import deepcopy
        translated_card = deepcopy(card)
        
        # Text fields that should be translated
        # Note: "value" is excluded to protect Input.Choice, Input.Date, etc.
        # FactSet values are handled explicitly.
        TEXT_FIELDS = {"text", "title", "label", "placeholder", "altText", "fallbackText", "summary"}
        # Fields that should NOT be translated (URLs, IDs, types, etc.)
        SKIP_FIELDS = {"type", "url", "id", "$schema", "version", "style", "size", "weight", 
                       "color", "horizontalAlignment", "verticalContentAlignment", "spacing",
                       "separator", "wrap", "isSubtle", "width", "height", "data", "verb",
                       "iconUrl", "backgroundImage", "bleed", "minHeight", "fallback",
                       "requires", "lang", "speak", "selectAction", "refresh", "authentication"}
        
        def should_translate(text: str) -> bool:
            """Check if text should be translated (not a URL, number-only, or too short)"""
            if not text or not isinstance(text, str):
                return False
            text = text.strip()
            if len(text) < 2:
                return False
            # Skip URLs
            if text.startswith(("http://", "https://", "data:", "mailto:")):
                return False
            # Skip if already Arabic
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            if arabic_chars > len(text) * 0.3:
                return False
            # Skip pure numbers/dates
            if re.match(r'^[\d\s\-\/\:\.\,\$\%]+$', text):
                return False
            return True
        
        def translate_value(value: str) -> str:
            """Translate a single text value"""
            if not should_translate(value):
                return value
            try:
                translated = self._translate_plain_en_to_ar(value)
                return translated if translated else value
            except Exception as e:
                logger.warning(f"Failed to translate card text '{value[:30]}...': {e}")
                return value
        
        def translate_node(node: Any) -> Any:
            """Recursively translate a node in the card structure"""
            if isinstance(node, dict):
                result = {}
                for key, val in node.items():
                    if key == "data" and isinstance(val, dict):
                        # Keep action payload structure intact, but translate known textual fields
                        # like `summary` used by the frontend to trigger intents.
                        out = {}
                        for dk, dv in val.items():
                            if dk in TEXT_FIELDS and isinstance(dv, str):
                                out[dk] = translate_value(dv)
                            else:
                                out[dk] = dv
                        result[key] = out
                        continue
                    if key in SKIP_FIELDS:
                        result[key] = val
                    elif key in TEXT_FIELDS and isinstance(val, str):
                        result[key] = translate_value(val)
                    elif key == "facts" and isinstance(val, list):
                        # FactSet facts need special handling
                        result[key] = [
                            {
                                "title": translate_value(f.get("title", "")) if isinstance(f.get("title"), str) else f.get("title"),
                                "value": translate_value(f.get("value", "")) if isinstance(f.get("value"), str) else f.get("value")
                            } if isinstance(f, dict) else f
                            for f in val
                        ]
                    elif key in ("body", "items", "columns", "actions", "card", "inlines"):
                        result[key] = translate_node(val)
                    else:
                        result[key] = translate_node(val)
                return result
            elif isinstance(node, list):
                return [translate_node(item) for item in node]
            else:
                return node
        
        translated_card = translate_node(translated_card)
        
        # Add RTL lang attribute to the card for proper rendering
        if "lang" not in translated_card:
            translated_card["lang"] = "ar"
        
        logger.info("✅ Adaptive Card translated to Arabic")
        return translated_card
   
    def clear_cache(self):
        """Clear translation cache"""
        self.cache.clear()
        logger.info("Translation cache cleared")
 
    def _strip_reasoning(self, text: str, target_lang: str) -> str:
        """Remove common chain-of-thought artifacts and keep the likely translation body.
        For EN->AR, keep content from the first Arabic character onward.
        For AR->EN, remove <think> blocks and leading meta lines like 'Okay,' or 'Let's'.
        """
        if not text:
            return text
        try:
            # Remove <think>...</think> blocks
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Trim whitespace
            text = text.strip()
            if target_lang == "ar":
                # Keep from first Arabic char onward and drop stray non-Arabic symbols
                for idx, ch in enumerate(text):
                    if '\u0600' <= ch <= '\u06FF':
                        ar = text[idx:]
                        # Remove any characters outside Arabic block, digits, whitespace and common punctuation
                        ar = re.sub(r"[^\u0600-\u06FF0-9\s\.,!\?;:\-\[\]\(\)]+", "", ar)
                        # Normalize spaces while preserving line breaks for markdown structure
                        ar = re.sub(r"[ \t]+", " ", ar)
                        ar = re.sub(r"\r\n", "\n", ar)
                        ar = re.sub(r"\n{3,}", "\n\n", ar)
                        return ar.strip()
                return text.strip()
            else:
                # Remove obvious reasoning lead-ins in English
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pruned = []
                dropping = True
                for ln in lines:
                    # stop dropping when the line looks like a sentence that could be a translation
                    if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I need to|I will|The user|We need to|Reasoning|Analysis)\b", ln, flags=re.IGNORECASE):
                        dropping = False
                    if not dropping:
                        pruned.append(ln)
                return (" ".join(pruned) if pruned else text).strip()
        except Exception:
            return text
