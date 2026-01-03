"""
Brochure Generation MCP Server

Generates professional brochure content with real images using LLM and Google Image Search.

Tools:
- generate_brochure: Create complete brochure with content and images
- get_brochure_template: Get the brochure structure schema
"""

import json
import os
import sys
import re
import time
import logging
import httpx
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Import telemetry
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.telemetry import trace_llm_call, log_llm_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("brochure-server")

# Initialize MCP server
mcp = FastMCP("brochure-server")

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BROCHURE_MODEL = "llama-3.3-70b-versatile"

# Google CSE Configuration for Image Search
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


# ============================================================================
# IMAGE SEARCH SERVICE
# ============================================================================

class ImageSearchService:
    """Fetch images using Google Custom Search Engine."""
    
    def __init__(self):
        self.api_key = GOOGLE_CSE_API_KEY
        self.cx = GOOGLE_CSE_CX
        self.enabled = bool(self.api_key and self.cx)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 900  # 15 minutes
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(key)
        if not cached:
            return None
        if time.time() - cached["ts"] > self._cache_ttl:
            del self._cache[key]
            return None
        return cached["value"]
    
    def _cache_set(self, key: str, value: Optional[Dict[str, Any]]) -> None:
        self._cache[key] = {"ts": time.time(), "value": value}
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid direct image URL."""
        if not url:
            return False
        
        # Common image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')
        url_lower = url.lower()
        
        # Check for common image hosting patterns
        valid_patterns = [
            'upload.wikimedia.org',
            'images.unsplash.com',
            'images.pexels.com',
            'i.imgur.com',
            '.staticflickr.com',
            'media.gettyimages.com',
            '/images/',
            '/photos/',
            '/img/',
        ]
        
        # Skip known problematic patterns
        invalid_patterns = [
            'tiktok.com',
            'instagram.com/p/',
            'twitter.com',
            'facebook.com',
            'youtube.com',
            '/api/',
            '/v1/',
            '/v2/',
        ]
        
        for pattern in invalid_patterns:
            if pattern in url_lower:
                return False
        
        # Has valid image extension
        if any(url_lower.endswith(ext) or f"{ext}?" in url_lower for ext in image_extensions):
            return True
        
        # From known good image sources
        for pattern in valid_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    async def fetch_image(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch a representative image for the query."""
        if not self.is_enabled():
            logger.warning("Image search not enabled (missing API keys)")
            return None
        
        normalized = (query or "").strip()
        if len(normalized) < 2:
            return None
        
        cache_key = normalized.lower()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        params = {
            "q": normalized,
            "searchType": "image",
            "num": 5,  # Request more to filter
            "safe": "active",
            "imgType": "photo",  # Prefer photos over clipart
            "key": self.api_key,
            "cx": self.cx,
        }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(GOOGLE_CSE_ENDPOINT, params=params)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("items", [])
                
                # Find the first valid image URL
                for item in items:
                    link = item.get("link", "")
                    image_meta = item.get("image", {})
                    thumbnail = image_meta.get("thumbnailLink")
                    
                    # Prefer direct image URL that passes validation
                    if self._is_valid_image_url(link):
                        result = {
                            "url": link,
                            "title": item.get("title", ""),
                            "thumbnail": thumbnail,
                            "context": image_meta.get("contextLink"),
                        }
                        self._cache_set(cache_key, result)
                        logger.info(f"   🖼️ Found valid image for '{normalized}': {link[:80]}...")
                        return result
                
                # Fallback: use thumbnail if no valid direct link found
                # Thumbnails are Google-hosted and always work
                for item in items:
                    image_meta = item.get("image", {})
                    thumbnail = image_meta.get("thumbnailLink")
                    if thumbnail:
                        result = {
                            "url": thumbnail,  # Use thumbnail as main image
                            "title": item.get("title", ""),
                            "thumbnail": thumbnail,
                            "context": image_meta.get("contextLink"),
                        }
                        self._cache_set(cache_key, result)
                        logger.info(f"   🖼️ Using thumbnail for '{normalized}': {thumbnail[:80]}...")
                        return result
                
                logger.warning(f"   No valid images found for '{normalized}'")
                        
        except Exception as e:
            logger.warning(f"Image search failed for '{normalized}': {e}")
        
        self._cache_set(cache_key, None)
        return None


# Global image service instance
_image_service = ImageSearchService()


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

async def call_llm(
    messages: List[Dict[str, str]], 
    max_tokens: int = 1000, 
    temperature: float = 0.6,
    json_mode: bool = False,
    trace_name: str = "brochure-llm"
) -> str:
    """Call an LLM API with telemetry (Groq primary, Gemini/Vertex fallback)."""
    prefer_env = (os.getenv("PREFER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
    force_gemini = prefer_env in {"gemini", "google", "vertex", "vertexai"}
    force_bedrock = prefer_env in {"aws", "bedrock"}

    # Extract query preview for telemetry
    query_preview = ""
    for msg in messages:
        if msg.get("role") == "user":
            query_preview = msg.get("content", "")[:100]
            break
    
    async def _call_gemini() -> str:
        # Use google-genai directly so we can request strict JSON when json_mode=True.
        from google import genai
        from google.genai import types

        # Infer Vertex mode from env (preferred) vs API key.
        provider = (os.getenv("GEMINI_PROVIDER") or "").strip().lower()
        use_vertex = provider in {"vertex", "vertexai", "gcp"} or (os.getenv("GEMINI_USE_VERTEX") or "").strip().lower() in {"1", "true", "yes", "on"}
        creds_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if creds_path and Path(creds_path).exists() and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            use_vertex = True

        # Extract system + user content
        system_parts = [m.get("content", "") for m in messages if (m.get("role") == "system")]
        user_parts = [m.get("content", "") for m in messages if (m.get("role") != "system")]
        system_instruction = "\n".join([p for p in system_parts if p]).strip() or None
        prompt = "\n".join([p for p in user_parts if p]).strip()

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

        if use_vertex:
            project = (
                (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
                or (os.getenv("GOOGLE_PROJECT_ID") or "").strip()
                or (os.getenv("GCP_PROJECT") or "").strip()
                or (os.getenv("VERTEX_PROJECT") or "").strip()
            )
            if not project and creds_path and Path(creds_path).exists():
                try:
                    with open(creds_path, "r", encoding="utf-8") as f:
                        project = (json.load(f).get("project_id") or "").strip()
                except Exception:
                    project = ""
            location = (
                (os.getenv("GOOGLE_CLOUD_LOCATION") or "").strip()
                or (os.getenv("GCP_LOCATION") or "").strip()
                or (os.getenv("VERTEX_LOCATION") or "").strip()
                or "us-central1"
            )
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("Gemini is not configured (no service account and no API key)")
            client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.systemInstruction = system_instruction
        if json_mode:
            config.responseMimeType = "application/json"

        response = await asyncio.to_thread(
            client.models.generate_content,
            model,
            prompt,
            config,
        )
        return (getattr(response, "text", "") or "").strip()

    async def _call_bedrock() -> str:
        from aws_bedrock import invoke_llama31_text

        # Best-effort JSON enforcement via instruction.
        msgs = list(messages or [])
        if json_mode:
            msgs = [
                {"role": "system", "content": "Return ONLY valid JSON. No markdown, no explanation."},
                *msgs,
            ]
        return await asyncio.to_thread(
            invoke_llama31_text,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    payload = {
        "model": BROCHURE_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    if force_bedrock:
        with trace_llm_call(
            name=trace_name,
            model=f"bedrock/{os.getenv('BEDROCK_MODEL_ID','meta.llama3-1-8b-instruct-v1:0')}",
            input_data={"messages": messages},
            model_parameters={"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode},
            metadata={"source": "brochure_server", "provider": "bedrock"}
        ) as trace:
            result = await _call_bedrock()
            trace.update(output=(result or "")[:500], metadata={"success": bool(result), "response_length": len(result or "")})
            return (result or "").strip()

    if force_gemini or not GROQ_API_KEY:
        with trace_llm_call(
            name=trace_name,
            model="gemini",
            input_data={"messages": messages},
            model_parameters={"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode},
            metadata={"source": "brochure_server", "provider": "gemini"}
        ) as trace:
            result = await _call_gemini()
            trace.update(output=(result or "")[:500], metadata={"success": bool(result), "response_length": len(result or "")})
            return (result or "").strip()

    with trace_llm_call(
        name=trace_name,
        model=f"groq/{BROCHURE_MODEL}",
        input_data={"messages": messages},
        model_parameters={"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode},
        metadata={"source": "brochure_server"}
    ) as trace:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    GROQ_API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip()

                usage = None
                if "usage" in data:
                    usage = {
                        "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": data["usage"].get("completion_tokens", 0),
                        "total_tokens": data["usage"].get("total_tokens", 0)
                    }

                trace.update(output=result, usage=usage, metadata={"success": True, "response_length": len(result)})
                return result
        except Exception as e:
            logger.warning(f"Groq brochure LLM failed, falling back to Gemini: {e}")
            result = await _call_gemini()
            trace.update(output=(result or "")[:500], metadata={"success": bool(result), "fallback": True, "provider": "gemini"})
            return (result or "").strip()


async def extract_subject_with_llm(user_prompt: str) -> str:
    """Use LLM to extract the brochure subject from user's request."""
    messages = [
        {
            "role": "system",
            "content": (
                "Extract the main subject/topic for a brochure from the user's request. "
                "Return ONLY the subject name, nothing else. No quotes, no explanation. "
                "Examples:\n"
                "- 'create a brochure about Riyadh Season' → Riyadh Season\n"
                "- 'make me a brochure on the Eiffel Tower' → Eiffel Tower\n"
                "- 'brochure for Saudi heritage and culture' → Saudi Heritage and Culture\n"
                "- 'I need a brochure featuring AlUla ancient sites' → AlUla Ancient Sites"
            )
        },
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        subject = await call_llm(messages, max_tokens=50, temperature=0.1)
        # Clean up any quotes or extra whitespace
        subject = subject.strip().strip('"\'')
        if subject:
            return subject.title()
    except Exception as e:
        logger.warning(f"LLM subject extraction failed: {e}")
    
    # Fallback: return cleaned prompt
    return user_prompt.strip()[:100].title()


def build_brochure_prompt(subject: str, user_prompt: str) -> List[Dict[str, str]]:
    """Build the LLM prompt for brochure generation."""
    schema = f'''Return only valid JSON with this schema:
{{
  "subject": "{subject}",
  "title": "Concise brochure headline (<= 80 characters)",
  "short_description": "One complete sentence summarizing the subject (<= 160 characters)",
  "detailed_description": "Two to three sentences expanding on the story (<= 350 characters)",
  "image_query": "Best search query to find a representative image for this subject",
  "facts": [
    {{"title": "Label", "detail": "One-sentence supporting fact"}},
    {{"title": "Label", "detail": "One-sentence supporting fact"}},
    {{"title": "Label", "detail": "One-sentence supporting fact"}}
  ],
  "more_info": {{
    "title": "Link label (<= 40 characters)",
    "url": "https://official-or-reputable-website.com"
  }}
}}

Rules:
- Title and descriptions should NOT mention "brochure" or reference the user's request
- Use {subject} as the central theme
- Provide exactly 3 facts with accurate, interesting information
- image_query should be descriptive (e.g., "Hampi stone chariot temple ruins India")
- URL must be a real, reputable source related to {subject}'''

    return [
        {
            "role": "system",
            "content": (
                "You are an expert brochure copywriter. Create original, informative, "
                "and engaging content about the specified subject. Use accurate details "
                "and adapt the tone to fit the locale or context."
            )
        },
        {
            "role": "user",
            "content": f"Create brochure content about '{subject}'.\n\nUser's request: {user_prompt}\n\n{schema}"
        }
    ]


def validate_brochure(payload: Dict[str, Any], subject: str) -> Dict[str, Any]:
    """Validate and clean the LLM response."""
    
    def clean_text(text: str, max_len: int = 500) -> str:
        if not text:
            return ""
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = text.strip()
        # Ensure ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        return text[:max_len]
    
    def clean_facts(facts: Any) -> List[Dict[str, str]]:
        if not isinstance(facts, list):
            return []
        cleaned = []
        for fact in facts[:5]:
            if isinstance(fact, dict):
                title = clean_text(fact.get("title", ""), 60)
                detail = clean_text(fact.get("detail") or fact.get("value", ""), 240)
                if title and detail:
                    cleaned.append({"title": title, "detail": detail})
        return cleaned
    
    title = clean_text(payload.get("title") or subject, 120)
    short_desc = clean_text(payload.get("short_description") or f"Discover {subject}.", 200)
    detailed_desc = clean_text(payload.get("detailed_description") or short_desc, 500)
    image_query = (payload.get("image_query") or subject).strip()[:100]
    
    facts = clean_facts(payload.get("facts"))
    if not facts:
        facts = [{"title": "Highlight", "detail": f"{subject} offers memorable experiences."}]
    
    more_info = payload.get("more_info") or {}
    if not isinstance(more_info, dict):
        more_info = {}
    more_info_title = (more_info.get("title") or "Learn More")[:50]
    more_info_url = more_info.get("url") or "https://www.google.com/search?q=" + subject.replace(" ", "+")
    if not more_info_url.startswith(("http://", "https://")):
        more_info_url = "https://" + more_info_url
    
    return {
        "subject": subject,
        "title": title,
        "short_description": short_desc,
        "detailed_description": detailed_desc,
        "image_query": image_query,
        "facts": facts,
        "more_info_title": more_info_title,
        "more_info_url": more_info_url,
    }


# ============================================================================
# ADAPTIVE CARD BUILDER
# ============================================================================

def build_brochure_card(brochure: Dict[str, Any], image: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an Adaptive Card for the brochure with optional image."""
    
    # Build facts as FactSet
    facts_data = []
    for fact in brochure.get("facts", []):
        if isinstance(fact, dict):
            facts_data.append({
                "title": fact.get("title", ""),
                "value": fact.get("detail", "")
            })
    
    more_info = brochure.get("more_info", {})
    if isinstance(more_info, dict):
        more_info_url = more_info.get("url", brochure.get("more_info_url", ""))
        more_info_title = more_info.get("title", brochure.get("more_info_title", "Explore"))
    else:
        more_info_url = brochure.get("more_info_url", "")
        more_info_title = brochure.get("more_info_title", "Explore")
    
    # Card body elements
    body = []
    
    # Hero image (if available)
    if isinstance(image, dict) and image.get("url"):
        body.append({
            "type": "Image",
            "url": image["url"],
            "size": "Stretch",
            "altText": brochure.get("title", "Brochure image")
        })
    
    # Title
    body.append({
        "type": "TextBlock",
        "text": brochure.get("title", ""),
        "weight": "Bolder",
        "size": "Large",
        "wrap": True,
        "spacing": "Medium" if image else "None"
    })
    
    # Short description
    body.append({
        "type": "TextBlock",
        "text": brochure.get("short_description", ""),
        "wrap": True,
        "spacing": "Small"
    })
    
    # Detailed description
    body.append({
        "type": "TextBlock",
        "text": brochure.get("detailed_description", ""),
        "wrap": True,
        "spacing": "Medium",
        "isSubtle": True
    })
    
    # Highlights header
    body.append({
        "type": "TextBlock",
        "text": "Highlights",
        "weight": "Bolder",
        "size": "Medium",
        "spacing": "Large"
    })
    
    # Facts
    if facts_data:
        body.append({
            "type": "FactSet",
            "facts": facts_data,
            "spacing": "Small"
        })
    
    card = {
        "type": "AdaptiveCard",
        "version": "1.5",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "body": body,
        "actions": [
            {
                "type": "Action.OpenUrl",
                "title": more_info_title,
                "url": more_info_url,
                "style": "positive"
            }
        ]
    }
    
    return card


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
async def generate_brochure(topic: str) -> dict:
    """
    Generate a professional brochure with content and images.
    
    Creates complete brochure content including title, descriptions,
    key facts, and fetches a relevant image from Google.
    
    Examples:
    - "create a brochure about Riyadh Season"
    - "brochure on the Taj Mahal"
    - "make a brochure featuring Hampi ruins"
    
    Args:
        topic: The brochure topic as a simple string (e.g., "Hampi", "Riyadh Season")
    
    Returns:
        Adaptive card with brochure content and image
    """
    logger.info("🔌 MCP TOOL: generate_brochure")
    
    # Defensive: Handle if topic comes as dict (shouldn't happen but be safe)
    if isinstance(topic, dict):
        logger.warning("   Topic received as dict, extracting string...")
        topic = topic.get("content") or topic.get("topic") or topic.get("subject") or str(topic)
    
    topic = str(topic).strip()
    logger.info(f"   Topic: {topic}")
    
    try:
        # Extract subject using LLM
        subject = await extract_subject_with_llm(topic)
        logger.info(f"   Subject: {subject}")
        
        # Generate brochure content
        messages = build_brochure_prompt(subject, topic)
        logger.info("   Calling LLM for content...")
        raw_response = await call_llm(messages, max_tokens=800, temperature=0.6, json_mode=True)
        
        # Parse and validate
        try:
            cleaned = (raw_response or "").strip()
            # Strip common Markdown code fences
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                cleaned = cleaned.strip()
            # Extract first JSON object if extra text is present
            if not cleaned.startswith("{"):
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if match:
                    cleaned = match.group(0).strip()
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                # Some models return Python-dict-like output (single quotes). Try a safe literal eval.
                import ast
                payload = ast.literal_eval(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"   JSON parse error: {e}")
            return {
                "type": "error",
                "message": f"Failed to generate brochure content: {e}",
            }
        except Exception as e:
            logger.error(f"   JSON parse error: {e}")
            return {
                "type": "error",
                "message": f"Failed to generate brochure content: {e}",
            }
        
        brochure = validate_brochure(payload, subject)
        logger.info(f"   ✅ Content generated: {brochure['title']}")
        
        # Fetch image
        image = None
        if _image_service.is_enabled():
            image = await _image_service.fetch_image(brochure["image_query"])
            if not image:
                # Try with just the subject
                image = await _image_service.fetch_image(subject)
        
        # Build card
        card = build_brochure_card(brochure, image)
        
        return {
            "type": "adaptive_card",
            "card": card,
            "brochure": brochure,
            "image": image,
            "message": f"Here's your brochure for '{subject}'",
            "metadata": {"template": "brochure"}
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"   API error: {e}")
        return {"type": "error", "message": f"API error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"   Error: {e}", exc_info=True)
        return {"type": "error", "message": f"Failed to generate brochure: {e}"}


@mcp.tool()
def get_brochure_template() -> dict:
    """
    Get the structure of a generated brochure.
    
    Returns the schema showing what fields a brochure contains.
    """
    logger.info("🔌 MCP TOOL: get_brochure_template")
    
    return {
        "type": "template",
        "template": {
            "subject": "Main topic of the brochure",
            "title": "Headline (max 80 chars)",
            "short_description": "One sentence summary (max 160 chars)",
            "detailed_description": "2-3 sentences detail (max 350 chars)",
            "image": "Fetched from Google Image Search",
            "facts": [{"title": "Label", "detail": "Supporting fact"}],
            "more_info": {"title": "Link text", "url": "https://..."}
        },
        "message": "This is the structure of a generated brochure."
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mcp.run()
