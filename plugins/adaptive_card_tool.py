"""
Adaptive Card Tool Plugin - Generate custom Adaptive Cards.

Creates beautiful, professional Microsoft Adaptive Cards based on
natural language descriptions using LLM.
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


# System prompt for card generation
CARD_SYSTEM_PROMPT = """You are an elite Adaptive Card designer creating STUNNING, professional Microsoft-quality cards.

🎯 CRITICAL REQUIREMENTS:

1. **STRUCTURED IMAGE REQUIREMENTS** (MANDATORY):
   Return image_queries as an array of objects:
   ```json
   "image_queries": [
     {"query": "professional business dashboard analytics modern", "purpose": "hero"},
     {"query": "team collaboration meeting office", "purpose": "content"},
     {"query": "revenue chart icon", "purpose": "icon"}
   ]
   ```

2. **Image Placeholders**:
   - Use: `"url": "https://placeholder.com/hero.jpg"` for ALL images
   - NEVER use adaptivecards.io URLs
   - Images will be auto-replaced with real URLs

3. **CRITICAL SCHEMA RULES**:
   ❌ NEVER put Actions inside Container items
   ❌ NEVER put Actions inside ColumnSets
   ✅ ALWAYS put Actions at card root level only
   ✅ OR inside Action.ShowCard cards only
   
4. **Valid Action Placement**:
   ```json
   {
     "card": {
       "body": [...],
       "actions": [
         {"type": "Action.OpenUrl", "title": "View Report", "url": "https://example.com"},
         {"type": "Action.Submit", "title": "Submit", "data": {"action": "submit"}}
       ]
     }
   }
   ```

5. **VISUAL HIERARCHY**:
   - Hero image (Large) at top
   - Emphasis container for title
   - ColumnSets for side-by-side content
   - FactSets for data
   - Separators between sections

6. **EXAMPLE: Dashboard Card (FOLLOW THIS STRUCTURE)**:
```json
{
  "card": {
    "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
    "type": "AdaptiveCard",
    "version": "1.5",
    "body": [
      {
        "type": "Image",
        "url": "https://placeholder.com/hero.jpg",
        "size": "Large",
        "horizontalAlignment": "Center",
        "altText": "Dashboard overview"
      },
      {
        "type": "Container",
        "style": "emphasis",
        "items": [
          {
            "type": "TextBlock",
            "text": "Q3 Performance Dashboard",
            "size": "ExtraLarge",
            "weight": "Bolder",
            "color": "Accent",
            "wrap": true
          },
          {
            "type": "TextBlock",
            "text": "Real-time insights and key metrics",
            "isSubtle": true,
            "wrap": true,
            "spacing": "None"
          }
        ]
      },
      {
        "type": "Container",
        "separator": true,
        "spacing": "Large",
        "items": [
          {
            "type": "ColumnSet",
            "columns": [
              {
                "type": "Column",
                "width": "auto",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/icon1.jpg",
                    "size": "Small",
                    "altText": "Revenue icon"
                  }
                ]
              },
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "TextBlock",
                    "text": "Revenue Growth",
                    "size": "Large",
                    "weight": "Bolder",
                    "wrap": true
                  },
                  {
                    "type": "FactSet",
                    "facts": [
                      {"title": "Current Quarter", "value": "$2.4M"},
                      {"title": "Growth Rate", "value": "+18.5%"},
                      {"title": "vs Target", "value": "112%"}
                    ]
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "type": "Container",
        "spacing": "Large",
        "items": [
          {
            "type": "ColumnSet",
            "columns": [
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/content1.jpg",
                    "size": "Medium",
                    "altText": "Team performance"
                  },
                  {
                    "type": "TextBlock",
                    "text": "Top Performing Team",
                    "weight": "Bolder",
                    "spacing": "Small",
                    "wrap": true
                  }
                ]
              },
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/content2.jpg",
                    "size": "Medium",
                    "altText": "Customer satisfaction"
                  },
                  {
                    "type": "TextBlock",
                    "text": "Customer Success",
                    "weight": "Bolder",
                    "spacing": "Small",
                    "wrap": true
                  }
                ]
              }
            ]
          }
        ]
      }
    ],
    "actions": [
      {
        "type": "Action.OpenUrl",
        "title": "View Full Report",
        "url": "https://example.com/report",
        "style": "positive"
      },
      {
        "type": "Action.Submit",
        "title": "Export Data",
        "data": {"action": "export"}
      }
    ]
  },
  "data": {},
  "image_queries": [
    {"query": "modern business analytics dashboard charts blue professional", "purpose": "hero"},
    {"query": "corporate team collaboration meeting", "purpose": "content"},
    {"query": "customer satisfaction business", "purpose": "content"},
    {"query": "revenue growth icon chart", "purpose": "icon"}
  ],
  "warnings": []
}
```

⚠️ **CRITICAL RULES - MUST FOLLOW**:
1. NEVER put Actions inside Container/ColumnSet items
2. ALL Actions go in card.actions array at root level
3. Use placeholder.com URLs for images (will be auto-replaced)
4. Always include 4-7 image_queries with purposes
5. Always set wrap: true for TextBlocks
6. Use FactSets for data, not tables
7. Return ONLY valid JSON

Generate a PRODUCTION-READY card following these rules exactly."""


class AdaptiveCardTool(BaseTool):
    """
    Tool for generating custom Adaptive Cards.
    
    Creates professional cards based on natural language descriptions
    using LLM to design the layout and content.
    """
    
    def __init__(self):
        self._llm_service = None
        self._image_service = None
        self._model = "llama-3.3-70b-versatile"
        # Check for Gemini availability
        self._gemini_available = False
        self._genai = None
        try:
            import google.generativeai as genai
            self._genai = genai
            if os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY"):
                self._gemini_available = True
        except ImportError:
            pass
    
    def _get_llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from shared_utils import get_service_manager
            self._llm_service = get_service_manager().get_llm_service()
        return self._llm_service
    
    def _get_image_service(self):
        """Lazy load image service."""
        if self._image_service is None:
            try:
                from shared_utils import get_service_manager
                self._image_service = get_service_manager().get_image_service()
            except Exception:
                pass
        return self._image_service
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="adaptive_card",
            description="Generate custom Adaptive Cards for rich visual displays. Creates professional cards with layouts, images, facts, and actions based on your description. Use for dashboards, forms, notifications, invitations, and any rich content display.",
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the card you want to create"
                    },
                    "tone": {
                        "type": "string",
                        "description": "Visual tone: professional, friendly, exciting, playful, urgent",
                        "default": "professional",
                        "enum": ["professional", "friendly", "exciting", "playful", "urgent", "elegant"]
                    },
                    "include_images": {
                        "type": "boolean",
                        "description": "Whether to fetch real images for placeholders",
                        "default": True
                    },
                    "card_type": {
                        "type": "string",
                        "description": "Type of card: dashboard, form, notification, profile, invitation, article",
                        "default": "auto"
                    }
                },
                "required": ["description"]
            },
            examples=[
                "Create an adaptive card for a sales dashboard",
                "Generate a registration form card",
                "Make a notification card for system alerts",
                "Design a profile card for team members",
                "Create an event invitation card",
                "Generate an article preview card",
                "Make a product showcase card",
                "Create a meeting summary card"
            ],
            input_examples=[
                {"description": "Sales performance dashboard with Q3 metrics"},
                {"description": "Event registration form", "card_type": "form"},
                {"description": "Team member profile card", "tone": "friendly"}
            ],
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        description: str,
        tone: str = "professional",
        include_images: bool = True,
        card_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a custom Adaptive Card.
        
        Args:
            description: Natural language description of the card
            tone: Visual tone for the card
            include_images: Whether to fetch real images
            card_type: Type of card to generate
            
        Returns:
            Dict with success, data containing the adaptive card
        """
        try:
            # Auto-detect card type if needed
            if card_type == "auto":
                card_type = self._detect_card_type(description)
            
            # Generate card using LLM
            result = await self._generate_card(description, tone, card_type)
            
            card = result.get("card", {})
            image_queries = result.get("image_queries", [])
            
            # Ensure card has required fields
            card = self._ensure_card_fields(card)
            
            # Replace placeholder images if requested
            if include_images and image_queries:
                card = await self._replace_images(card, image_queries)
            
            # Return in standard tool format with success and data keys
            return {
                "success": True,
                "data": {
                    "type": "adaptive_card",
                    "card": card,
                    "message": f"Here's your {card_type} card!",
                    "metadata": {
                        "template_id": "custom_adaptive_card",
                        "card_type": card_type,
                        "tone": tone
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Adaptive card generation failed: {e}")
            # Return a fallback card
            return {
                "success": True,
                "data": {
                    "type": "adaptive_card",
                    "card": self._get_fallback_card(description),
                    "message": "Here's a basic card. Try providing more details for a richer design."
                }
            }
    
    def _detect_card_type(self, description: str) -> str:
        """Auto-detect the type of card from description."""
        desc_lower = description.lower()
        
        type_keywords = {
            "dashboard": ["dashboard", "analytics", "metrics", "kpi", "performance", "report"],
            "form": ["form", "register", "signup", "input", "submit", "survey", "feedback"],
            "notification": ["alert", "notification", "warning", "error", "info", "message"],
            "profile": ["profile", "team", "member", "employee", "contact", "person"],
            "invitation": ["invite", "invitation", "event", "party", "meeting", "conference"],
            "article": ["article", "blog", "news", "story", "preview", "summary"]
        }
        
        for card_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return card_type
        
        return "general"
    
    async def _generate_card(self, description: str, tone: str, card_type: str) -> Dict[str, Any]:
        """Generate card using LLM."""
        
        # Check if form elements are needed
        form_keywords = ["form", "input", "submit", "register", "survey"]
        needs_inputs = any(kw in description.lower() for kw in form_keywords)
        
        input_reminder = ""
        if needs_inputs:
            input_reminder = "\n\n🔥 FORM DETECTED: Include Input.* elements and a Submit action!"
        
        prompt_body = f"""Brief: {description}
Tone: {tone}{input_reminder}

Create a stunning, Microsoft-quality Adaptive Card following ALL requirements above. Focus on:
1. Exceptional visual hierarchy
2. 5-7 detailed image queries with purpose tags (hero/content/icon)
3. Professional interactivity (2-4 actions)
4. Real, specific content

Return ONLY valid JSON."""

        prompt = f"{CARD_SYSTEM_PROMPT}\n\n{prompt_body}"

        messages = [
            {"role": "system", "content": "You are an expert Adaptive Card designer. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        llm = self._get_llm_service()
        response = llm.generate_text(
            messages=messages,
            model=self._model,
            max_tokens=1500,
            temperature=0.7,
            response_format={"type": "json_object"},
            timeout=45,
            trace_name="adaptive-card-generation"
        )
        
        # Parse response
        try:
            # Clean markdown if present
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Could not parse card JSON")
    
    def _ensure_card_fields(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure card has all required fields."""
        cleaned = deepcopy(card)
        cleaned["$schema"] = "https://adaptivecards.io/schemas/adaptive-card.json"
        cleaned["type"] = "AdaptiveCard"
        if "version" not in cleaned:
            cleaned["version"] = "1.5"
        if "body" not in cleaned:
            cleaned["body"] = []
        return cleaned
    
    async def _replace_images(self, card: Dict[str, Any], image_queries: List[Dict]) -> Dict[str, Any]:
        """Replace placeholder images with real ones using smart distribution."""
        image_service = self._get_image_service()
        
        if not image_service or not getattr(image_service, 'is_enabled', lambda: False)():
            return card
        
        # 1. Collect all images
        all_images = []
        for query_info in image_queries:
            query = query_info.get('query', '')
            purpose = query_info.get('purpose', 'content')
            try:
                # Fetch image using shared service
                result = image_service.fetch_image(query)
                if result and result.get('image_url'):
                    all_images.append({
                        "url": result.get('image_url'),
                        "context": query,
                        "purpose": purpose
                    })
            except Exception as e:
                logger.warning(f"Image fetch failed for '{query}': {e}")
        
        if not all_images:
            return card

        # 2. Distribute images based on purpose and size
        heroes, content_imgs, icon_imgs = self._get_image_by_purpose(all_images)
        hero_idx = content_idx = icon_idx = 0

        def pick_image(size: str) -> Optional[Dict[str, str]]:
            nonlocal hero_idx, content_idx, icon_idx
            size = (size or "").lower()
            
            # Try to match size/purpose
            if size in {"large", "extralarge"} and hero_idx < len(heroes):
                img = heroes[hero_idx]
                hero_idx += 1
                return img
            if size in {"medium", "auto", "stretch"} and content_idx < len(content_imgs):
                img = content_imgs[content_idx]
                content_idx += 1
                return img
            if size in {"small"} and icon_idx < len(icon_imgs):
                img = icon_imgs[icon_idx]
                icon_idx += 1
                return img
                
            # Fallback pools
            pools = [content_imgs, heroes, icon_imgs]
            indices = [content_idx, hero_idx, icon_idx]
            
            # Try to find any available image
            if content_idx < len(content_imgs):
                img = content_imgs[content_idx]
                content_idx += 1
                return img
            if hero_idx < len(heroes):
                img = heroes[hero_idx]
                hero_idx += 1
                return img
            if icon_idx < len(icon_imgs):
                img = icon_imgs[icon_idx]
                icon_idx += 1
                return img
            return None

        def traverse(node: Any) -> Any:
            if isinstance(node, dict):
                if node.get("type") == "Image" and node.get("url"):
                    url = node["url"]
                    is_placeholder = (
                        "placeholder" in url
                        or "adaptivecards.io" in url
                        or not url.lower().startswith("http")
                    )
                    if is_placeholder:
                        selected = pick_image(str(node.get("size", "")))
                        if selected:
                            node["url"] = selected["url"]
                            node.setdefault("altText", selected.get("context") or "Image")
                for key, value in list(node.items()):
                    node[key] = traverse(value)
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    node[idx] = traverse(item)
            return node

        # 3. Apply replacements
        replaced = traverse(deepcopy(card))
        
        # 4. Ensure hero image exists if we have one but it wasn't used
        if replaced.get("body") and heroes and hero_idx == 0:
            hero = heroes[0]
            # Check if first item is already an image
            first_item = replaced["body"][0]
            if not (isinstance(first_item, dict) and first_item.get("type") == "Image"):
                replaced["body"].insert(
                    0,
                    {
                        "type": "Image",
                        "url": hero["url"],
                        "size": "Large",
                        "horizontalAlignment": "Center",
                        "altText": hero.get("context") or "Hero image",
                    },
                )
                
        return replaced
    
    def _get_fallback_card(self, description: str) -> Dict[str, Any]:
        """Return a basic fallback card."""
        return {
            "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [
                {
                    "type": "Container",
                    "style": "emphasis",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": "Generated Card",
                            "size": "Large",
                            "weight": "Bolder",
                            "wrap": True
                        },
                        {
                            "type": "TextBlock",
                            "text": description[:200],
                            "wrap": True,
                            "isSubtle": True
                        }
                    ]
                }
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "OK",
                    "data": {"action": "acknowledge"}
                }
            ]
        }
    
    def _get_image_by_purpose(
        self,
        images: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """Categorize images by purpose."""
        heroes = [img for img in images if img.get("purpose") == "hero"]
        content = [img for img in images if img.get("purpose") == "content"]
        icons = [img for img in images if img.get("purpose") == "icon"]
        if not heroes and not content and not icons and images:
            heroes = images[:2]
            content = images[2:6]
            icons = images[6:]
        return heroes, content, icons
