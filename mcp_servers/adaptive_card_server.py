"""
Adaptive Card Generation MCP Server

Generates professional Microsoft Adaptive Cards using LLM with real images.

Tools:
- generate_adaptive_card: Create custom Adaptive Cards from natural language descriptions
"""

import json
import os
import sys
import time
import logging
import httpx
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GLOBAL LLM Service - THE ONLY WAY TO CALL LLMs
from shared_utils import get_global_llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("adaptive-card-server")

# Initialize MCP server
mcp = FastMCP("adaptive-card-server")

# Get the global LLM service (handles rate limiting, failover, etc.)
_llm_service = get_global_llm_service()
logger.info("🤖 Adaptive Card server using GlobalLLMService")

# Google CSE Configuration for Image Search
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "").strip()
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


# ============================================================================
# IMAGE SEARCH SERVICE (Same pattern as brochure_server - with caching)
# ============================================================================

class ImageSearchService:
    """Fetch images using Google Custom Search Engine with caching."""
    
    def __init__(self):
        self.api_key = GOOGLE_CSE_API_KEY
        self.cx = GOOGLE_CSE_CX
        self.enabled = bool(self.api_key and self.cx)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 900  # 15 minutes cache
    
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
    
    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = {"ts": time.time(), "value": value}
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid direct image URL."""
        if not url:
            return False
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')
        url_lower = url.lower()
        
        valid_patterns = [
            'upload.wikimedia.org', 'images.unsplash.com', 'images.pexels.com',
            'i.imgur.com', '.staticflickr.com', '/images/', '/photos/', '/img/',
        ]
        
        invalid_patterns = [
            'tiktok.com', 'instagram.com/p/', 'twitter.com', 'facebook.com',
            'youtube.com', '/api/', '/v1/', '/v2/',
        ]
        
        for pattern in invalid_patterns:
            if pattern in url_lower:
                return False
        
        if any(url_lower.endswith(ext) or f"{ext}?" in url_lower for ext in image_extensions):
            return True
        
        for pattern in valid_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    async def fetch_images(self, query: str, num_images: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch multiple images for the query (max 3-5 to save API quota).
        Uses caching to minimize API calls.
        """
        if not self.is_enabled():
            logger.warning("Image search not enabled (missing API keys)")
            return []
        
        normalized = (query or "").strip()
        if len(normalized) < 2:
            return []
        
        cache_key = f"{normalized.lower()}:{num_images}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info(f"   📦 Using cached images for '{normalized[:40]}...'")
            return cached
        
        params = {
            "q": normalized,
            "searchType": "image",
            "num": min(num_images + 2, 10),  # Request a few extra to filter
            "safe": "active",
            "imgType": "photo",
            "key": self.api_key,
            "cx": self.cx,
        }
        
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.get(GOOGLE_CSE_ENDPOINT, params=params)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("items", [])
                results = []
                
                for item in items:
                    if len(results) >= num_images:
                        break
                    
                    link = item.get("link", "")
                    image_meta = item.get("image", {})
                    thumbnail = image_meta.get("thumbnailLink")
                    
                    # Prefer direct image URL
                    if self._is_valid_image_url(link):
                        results.append({
                            "url": link,
                            "title": item.get("title", ""),
                            "thumbnail": thumbnail,
                        })
                    elif thumbnail:
                        # Fallback to thumbnail
                        results.append({
                            "url": thumbnail,
                            "title": item.get("title", ""),
                            "thumbnail": thumbnail,
                        })
                
                if results:
                    logger.info(f"   🖼️ Found {len(results)} images for '{normalized[:40]}...'")
                else:
                    logger.warning(f"   No valid images found for '{normalized}'")
                
                self._cache_set(cache_key, results)
                return results
                
        except Exception as e:
            logger.warning(f"Image search failed for '{normalized}': {e}")
        
        self._cache_set(cache_key, [])
        return []


# Global image service instance
_image_service = ImageSearchService()


# ============================================================================
# SYSTEM PROMPT FOR CARD GENERATION (SIMPLIFIED for reliable JSON output)
# ============================================================================

CARD_SYSTEM_PROMPT = """Generate a Microsoft Adaptive Card as valid JSON.

RULES:
1. Use placeholder.com URLs for images (e.g., "https://placeholder.com/hero.jpg")
2. Include 2-3 images in body
3. Actions go at card ROOT level only (never inside containers)
4. Always include image_queries array with 3 search terms

REQUIRED JSON FORMAT:
{
  "card": {
    "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
    "type": "AdaptiveCard",
    "version": "1.5",
    "body": [
      {"type": "Image", "url": "https://placeholder.com/hero.jpg", "size": "Large"},
      {"type": "Container", "style": "emphasis", "items": [
        {"type": "TextBlock", "text": "Title Here", "size": "ExtraLarge", "weight": "Bolder", "wrap": true}
      ]},
      {"type": "FactSet", "facts": [{"title": "Label", "value": "Value"}]}
    ],
    "actions": [{"type": "Action.OpenUrl", "title": "Learn More", "url": "https://example.com"}]
  },
  "image_queries": ["search term 1", "search term 2", "search term 3"]
}

Return ONLY valid JSON. No markdown, no explanation."""


# ============================================================================
# LLM FUNCTION - Uses GlobalLLMService
# ============================================================================

async def call_llm(
    messages: List[Dict[str, str]], 
    max_tokens: int = 2500, 
    temperature: float = 0.7,
    json_mode: bool = True
) -> str:
    """Call LLM using GlobalLLMService (handles rate limiting + failover)."""
    return await _llm_service.call_with_messages_async(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        trace_name="adaptive-card-llm"
    )


# ============================================================================
# CARD GENERATION WITH IMAGES
# ============================================================================

def get_fallback_card(description: str, error: str = "") -> Dict[str, Any]:
    """Return a basic fallback card when generation fails."""
    return {
        "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {"type": "TextBlock", "text": "⚠️ Card Generation Failed", "size": "Large", "weight": "Bolder", "wrap": True},
                    {"type": "TextBlock", "text": f"Request: {description[:150]}", "wrap": True, "isSubtle": True}
                ]
            },
            {"type": "TextBlock", "text": f"Error: {error[:200]}" if error else "Please try again.", "wrap": True, "color": "Attention", "spacing": "Medium"}
        ],
        "actions": [{"type": "Action.Submit", "title": "Retry", "data": {"action": "retry"}}]
    }


def ensure_card_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure card has all required Adaptive Card fields."""
    card["$schema"] = "https://adaptivecards.io/schemas/adaptive-card.json"
    card["type"] = "AdaptiveCard"
    if "version" not in card:
        card["version"] = "1.5"
    if "body" not in card or not card["body"]:
        card["body"] = [{"type": "TextBlock", "text": "Generated Card", "weight": "Bolder", "wrap": True}]
    return card


def build_basic_card(description: str) -> Dict[str, Any]:
    """Build a basic card from description when LLM fails."""
    # Extract a title from description
    title = description[:60].split('.')[0].strip()
    if len(title) < 5:
        title = "Information Card"
    
    return {
        "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {"type": "TextBlock", "text": title, "size": "Large", "weight": "Bolder", "wrap": True},
                    {"type": "TextBlock", "text": description[:300], "wrap": True, "isSubtle": True}
                ]
            }
        ],
        "actions": [{"type": "Action.Submit", "title": "OK", "data": {"action": "acknowledge"}}]
    }


async def build_card_with_images(description: str) -> Dict[str, Any]:
    """Build a card with images when LLM JSON parsing fails."""
    card = build_basic_card(description)
    
    # Try to add images
    if _image_service.is_enabled():
        images = await _image_service.fetch_images(description[:100], num_images=2)
        if images:
            # Add hero image at the start
            card["body"].insert(0, {
                "type": "Image",
                "url": images[0]["url"],
                "size": "Large",
                "horizontalAlignment": "Center"
            })
            logger.info(f"🖼️ Added fallback image to card")
    
    return card


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    
    return json.loads(text.strip())


async def replace_placeholder_images(card: Dict[str, Any], images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace placeholder image URLs with real ones."""
    if not images:
        return card
    
    image_idx = 0
    
    def replace_in_node(node: Any) -> Any:
        nonlocal image_idx
        
        if isinstance(node, dict):
            # Check if this is an Image element with placeholder
            if node.get("type") == "Image":
                url = node.get("url", "")
                if "placeholder" in url.lower() and image_idx < len(images):
                    node["url"] = images[image_idx]["url"]
                    image_idx += 1
            
            # Recurse into all dict values
            for key, value in node.items():
                node[key] = replace_in_node(value)
        
        elif isinstance(node, list):
            return [replace_in_node(item) for item in node]
        
        return node
    
    return replace_in_node(card)


async def generate_card(description: str, tone: str = "professional") -> Dict[str, Any]:
    """
    Generate an Adaptive Card using LLM with real images.
    
    Flow:
    1. LLM generates card JSON with placeholder images + image_queries
    2. Fetch 3-5 real images using Google CSE (cached)
    3. Replace placeholders with real image URLs
    """
    logger.info(f"📝 Generating card for: {description[:80]}...")
    
    # Simplified user prompt for better JSON generation
    user_prompt = f"""Create an Adaptive Card for: {description}
Tone: {tone}

Include placeholder.com image URLs and image_queries array. Return ONLY valid JSON."""

    messages = [
        {"role": "system", "content": CARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Use higher max_tokens to avoid truncation
        response = await call_llm(messages, max_tokens=3500, temperature=0.7, json_mode=True)
        logger.info(f"✅ LLM response received ({len(response)} chars)")
        
        # Check for obviously truncated response
        if len(response) < 500:
            logger.warning(f"⚠️ Response too short ({len(response)} chars), likely truncated")
            # Try to build a basic card from the description
            return build_basic_card(description)
        
        # Parse response
        parsed = extract_json_from_response(response)
        
        # Handle nested "card" structure
        card = parsed.get("card", parsed)
        image_queries = parsed.get("image_queries", [])
        
        # Ensure required fields
        card = ensure_card_fields(card)
        
        # Fetch real images (3-5 images, uses caching)
        if image_queries and _image_service.is_enabled():
            # Use first query for main search (most relevant)
            main_query = image_queries[0] if isinstance(image_queries[0], str) else image_queries[0].get("query", description)
            logger.info(f"🔍 Fetching images for: {main_query[:50]}...")
            
            images = await _image_service.fetch_images(main_query, num_images=4)
            
            if images:
                card = await replace_placeholder_images(card, images)
                logger.info(f"🖼️ Replaced {len(images)} placeholder images")
        elif _image_service.is_enabled():
            # No image_queries from LLM, use description instead
            logger.info(f"🔍 Using description for image search...")
            images = await _image_service.fetch_images(description[:100], num_images=3)
            if images:
                # Insert hero image at start of body if we got images
                card["body"].insert(0, {
                    "type": "Image",
                    "url": images[0]["url"],
                    "size": "Large",
                    "horizontalAlignment": "Center"
                })
                logger.info(f"🖼️ Added {len(images)} images from description search")
        else:
            logger.info("⚠️ Image service not enabled")
        
        logger.info("✅ Card generated successfully")
        return card
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {e}")
        # Build a basic card when JSON parsing fails
        return await build_card_with_images(description)
    except Exception as e:
        logger.error(f"❌ Card generation error: {e}")
        return await build_card_with_images(description)


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

@mcp.tool()
async def generate_adaptive_card(
    description: str,
    tone: str = "professional"
) -> str:
    """
    Generate a custom Microsoft Adaptive Card from a natural language description.
    
    Creates beautiful cards with REAL IMAGES fetched from Google Image Search.
    
    Use this tool when users want to CREATE, DESIGN, or BUILD:
    - Dashboard cards
    - Profile cards
    - Notification cards
    - Form cards
    - Article/news cards
    - Any visual card layout
    
    Args:
        description: Natural language description of what the card should contain
                    Example: "Create a quarterly sales dashboard with revenue metrics"
        tone: Visual style - "professional", "friendly", "bold", or "minimal"
              Default: "professional"
    
    Returns:
        JSON string containing the Adaptive Card that renders in the chat
    """
    logger.info(f"🎨 generate_adaptive_card called: '{description[:80]}...'")
    
    if not description or len(description.strip()) < 5:
        return json.dumps({
            "success": False,
            "error": "Please provide a description for the card"
        })
    
    try:
        card = await generate_card(description.strip(), tone)
        
        # Return in format expected by router/frontend
        # Router checks for: data.get("type") == "adaptive_card"
        return json.dumps({
            "success": True,
            "type": "adaptive_card",
            "card": card
        })
        
    except Exception as e:
        logger.error(f"❌ Tool error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "type": "adaptive_card",
            "card": get_fallback_card(description, str(e))
        })


@mcp.tool()
async def get_card_template() -> str:
    """
    Get information about Adaptive Card structure and capabilities.
    
    Returns:
        JSON with card types and structure information
    """
    return json.dumps({
        "success": True,
        "data": {
            "card_types": [
                "dashboard - Business metrics, KPIs, performance data",
                "profile - Person/team info with contact details",
                "notification - Alerts, warnings, success messages",
                "form - Input forms with submit actions",
                "article - News, blog posts, content previews"
            ],
            "tones": ["professional", "friendly", "bold", "minimal"],
            "features": [
                "Real images from Google Image Search",
                "ColumnSets for layouts",
                "FactSets for data display",
                "Actions for interactivity"
            ]
        }
    })


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting Adaptive Card MCP Server...")
    logger.info(f"📷 Image search enabled: {_image_service.is_enabled()}")
    mcp.run()
