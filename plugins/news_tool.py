"""
News Tool Plugin - Get news headlines and search news.

Uses NewsAPI to fetch top headlines and search for news articles.
"""

import os
import logging
from typing import Any, Dict, Optional
from datetime import datetime
import httpx

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class NewsTool(BaseTool):
    """
    Tool for fetching news headlines and searching news articles.
    
    Uses NewsAPI. Requires NEWS_API_KEY environment variable.
    Supports both top headlines and topic-specific news search.
    Also supports regional queries by mapping locations to country codes.
    """
    
    # Map common regions/cities/countries to country codes
    REGION_TO_COUNTRY = {
        # Middle East
        "saudi arabia": "sa", "saudi": "sa", "riyadh": "sa", "jeddah": "sa", "mecca": "sa",
        "uae": "ae", "dubai": "ae", "abu dhabi": "ae", "emirates": "ae",
        "qatar": "qa", "doha": "qa",
        "kuwait": "kw",
        "bahrain": "bh",
        "oman": "om",
        # Asia
        "india": "in", "delhi": "in", "mumbai": "in", "bangalore": "in", "chennai": "in",
        "china": "cn", "beijing": "cn", "shanghai": "cn",
        "japan": "jp", "tokyo": "jp",
        "singapore": "sg",
        "malaysia": "my", "kuala lumpur": "my",
        "indonesia": "id", "jakarta": "id",
        "pakistan": "pk", "karachi": "pk", "lahore": "pk",
        # Europe
        "uk": "gb", "britain": "gb", "england": "gb", "london": "gb",
        "germany": "de", "berlin": "de",
        "france": "fr", "paris": "fr",
        "italy": "it", "rome": "it",
        "spain": "es", "madrid": "es",
        # Americas
        "usa": "us", "america": "us", "united states": "us", "new york": "us", "washington": "us",
        "canada": "ca", "toronto": "ca",
        "brazil": "br", "sao paulo": "br",
        "mexico": "mx",
        # Others
        "australia": "au", "sydney": "au", "melbourne": "au",
        "south africa": "za", "johannesburg": "za",
        "russia": "ru", "moscow": "ru",
    }
    
    def __init__(self):
        self._api_key = os.getenv("NEWS_API_KEY", "")
        self._headlines_url = "https://newsapi.org/v2/top-headlines"
        self._search_url = "https://newsapi.org/v2/everything"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="news",
            description="Get latest news headlines or search for news on specific topics. Can fetch top headlines or search for news about any subject.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional search query for specific news topics. Leave empty for top headlines."
                    },
                    "category": {
                        "type": "string",
                        "description": "News category for headlines: business, entertainment, general, health, science, sports, technology",
                        "enum": ["business", "entertainment", "general", "health", "science", "sports", "technology"]
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for headlines (e.g., 'us', 'in', 'gb')",
                        "default": "us"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles (1-10)",
                        "default": 5
                    }
                },
                "required": []
            },
            examples=[
                "What's in the news today?",
                "Show me the latest headlines",
                "News about artificial intelligence",
                "Latest tech news",
                "Sports headlines",
                "What's happening in the world?",
                "News about climate change",
                "Business news today",
                "Breaking news",
                "Show me technology news"
            ],
            input_examples=[
                {"query": None, "max_results": 5},
                {"query": "artificial intelligence", "max_results": 5},
                {"category": "technology", "country": "us"},
                {"query": "climate change", "max_results": 3}
            ],
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        country: str = "us",
        max_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch news headlines or search for news.
        
        Args:
            query: Optional search query for specific topics
            category: News category for headlines
            country: Country code for headlines
            max_results: Maximum number of articles
            
        Returns:
            List of news articles with titles, descriptions, and URLs
        """
        if not self._api_key:
            return {
                "success": False,
                "error": "News API key not configured (NEWS_API_KEY)"
            }
        
        max_results = max(1, min(10, max_results))
        
        try:
            # Check if query is a region/country name and map to country code
            if query and query.strip():
                query_lower = query.lower().strip()
                
                # Check if the query is a known region
                if query_lower in self.REGION_TO_COUNTRY:
                    # Use headlines API with country code for better results
                    mapped_country = self.REGION_TO_COUNTRY[query_lower]
                    logger.info(f"📰 Mapped region '{query}' to country code '{mapped_country}'")
                    return await self._get_headlines(category, mapped_country, max_results)
                else:
                    # Search for specific news topic
                    return await self._search_news(query, max_results)
            else:
                # Get top headlines
                return await self._get_headlines(category, country, max_results)
                
        except Exception as e:
            logger.error(f"News error: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch news: {str(e)}"
            }
    
    async def _get_headlines(
        self,
        category: Optional[str],
        country: str,
        max_results: int
    ) -> Dict[str, Any]:
        """Fetch top headlines."""
        params = {
            "apiKey": self._api_key,
            "country": country,
            "pageSize": max_results
        }
        
        if category:
            params["category"] = category
        
        client = await self._get_client()
        response = await client.get(self._headlines_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = self._extract_articles(data.get("articles", []))
        
        return {
            "success": True,
            "data": {
                "type": "headlines",
                "category": category,
                "country": country,
                "articles": articles
            }
        }
    
    async def _search_news(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search for news on a specific topic."""
        params = {
            "apiKey": self._api_key,
            "q": query,
            "searchIn": "title,description",
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": max_results
        }
        
        client = await self._get_client()
        response = await client.get(self._search_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = self._extract_articles(data.get("articles", []))
        
        return {
            "success": True,
            "data": {
                "type": "search",
                "query": query,
                "articles": articles
            }
        }
    
    def _extract_articles(self, articles: list) -> list:
        """Extract relevant fields from articles."""
        extracted = []
        for article in articles:
            extracted.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", "")
            })
        return extracted
    
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format news articles for display - each article as separate message."""
        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"
        
        data = result.get("data", {})
        articles = data.get("articles", [])
        
        if not articles:
            return "No news articles found."
        
        lines = []
        
        for i, article in enumerate(articles[:5], 1):
            article_title = article.get("title", "No title")
            url = article.get("url", "#")
            description = article.get("description", "")
            source = article.get("source", "Unknown")
            published = article.get("published_at", "")
            
            # Format date
            date_str = ""
            if published:
                try:
                    dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    pass
            
            # Build article block with clickable HTML link
            article_lines = []
            article_lines.append(f"{i}. <a href='{url}' target='_blank' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{article_title}</a>")
            
            # Add description if available
            if description:
                desc_short = description[:120] + "..." if len(description) > 120 else description
                article_lines.append(desc_short)
            
            # Add source and date on same line
            meta = f"Source: {source}"
            if date_str:
                meta += f" | {date_str}"
            article_lines.append(meta)
            
            lines.append("\n".join(article_lines))
        
        # Join articles with MESSAGE_SPLIT for separate bubbles
        return "\n---MESSAGE_SPLIT---\n".join(lines)

