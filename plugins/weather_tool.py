"""
Weather Tool Plugin - Get current and forecast weather for a city.

Uses OpenWeatherMap API to fetch weather data.
Supports:
- Current weather ("What's the weather in London?")
- Forecast ("What's the weather tomorrow in London?")
- Relative dates ("today", "tomorrow", "in 3 days", etc.)
"""

import os
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import httpx

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """
    Tool for getting current and forecast weather information for a city.
    
    Uses OpenWeatherMap API. Requires WEATHER_API_KEY environment variable.
    Supports relative date queries like "tomorrow", "in 3 days", etc.
    """
    
    def __init__(self):
        self._api_key = os.getenv("WEATHER_API_KEY", "")
        self._current_url = "https://api.openweathermap.org/data/2.5/weather"
        self._forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="weather",
            description="Get current weather information for a city including temperature, conditions, and humidity. Can also get forecast for future days (today, tomorrow, in X days).",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for (e.g., 'Mumbai', 'London', 'New York')"
                    },
                    "when": {
                        "type": "string",
                        "description": "When to get weather for: 'today', 'tomorrow', 'in 2 days', 'next week', or a date. Defaults to 'today' (current weather)."
                    }
                },
                "required": ["city"]
            },
            examples=[
                "What's the weather in Mumbai?",
                "What's the weather tomorrow in London?",
                "Weather in Tokyo today",
                "Temperature in Delhi tomorrow",
                "Is it going to rain in New York tomorrow?",
                "Weather forecast for Paris in 3 days",
                "What will the weather be like next week in Berlin?",
                "How hot is it in Dubai today?"
            ],
            input_examples=[
                {"city": "Mumbai"},
                {"city": "London", "when": "tomorrow"},
                {"city": "New York", "when": "today"},
                {"city": "Tokyo", "when": "in 3 days"}
            ],
            defer_loading=False,
            always_loaded=True,
            system_instruction="For weather/temperature/forecast queries, use the 'weather' tool with city and optional when parameters.",
            code_example='result = await call_tool("weather", {"city": "Mumbai", "when": "tomorrow"})'
        )
    
    def _parse_when(self, when: Optional[str]) -> Tuple[int, str]:
        """
        Parse the 'when' parameter to determine days offset.
        
        Args:
            when: Natural language time reference (e.g., "today", "tomorrow", "in 3 days")
            
        Returns:
            Tuple of (days_offset, display_text)
            - days_offset: 0 for today/current, 1 for tomorrow, etc.
            - display_text: Human-readable text for the date
        """
        if not when:
            return (0, "current")
        
        when_lower = when.lower().strip()
        
        # Handle explicit keywords
        if when_lower in ("today", "now", "current", "currently"):
            return (0, "today")
        
        if when_lower in ("tomorrow", "tmrw", "tommorow", "tomorow"):
            return (1, "tomorrow")
        
        if when_lower in ("day after tomorrow", "day after tmrw"):
            return (2, "day after tomorrow")
        
        # Handle "in X days" pattern
        in_days_match = re.search(r'in\s+(\d+)\s*days?', when_lower)
        if in_days_match:
            days = int(in_days_match.group(1))
            if days == 0:
                return (0, "today")
            elif days == 1:
                return (1, "tomorrow")
            else:
                return (min(days, 5), f"in {days} days")  # Forecast API gives 5 days max
        
        # Handle "next X" pattern
        if "next week" in when_lower:
            return (5, "next week")  # ~5 days ahead (max forecast)
        
        # Handle specific weekday names (basic support)
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        today_weekday = datetime.now().weekday()
        
        for i, day_name in enumerate(weekdays):
            if day_name in when_lower:
                # Calculate days until that weekday
                days_ahead = i - today_weekday
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                return (min(days_ahead, 5), f"on {day_name.capitalize()}")
        
        # Handle "X days later/from now" pattern
        days_later_match = re.search(r'(\d+)\s*days?\s*(later|from now|ahead)', when_lower)
        if days_later_match:
            days = int(days_later_match.group(1))
            return (min(days, 5), f"in {days} days")
        
        # Default to current weather
        logger.warning(f"Could not parse 'when' parameter: {when}, defaulting to current")
        return (0, "current")
    
    async def execute(self, city: str, when: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get weather for a city, optionally for a future date.
        
        Args:
            city: City name
            when: When to get weather for (e.g., "today", "tomorrow", "in 3 days")
            
        Returns:
            Weather data including temperature, conditions, humidity
        """
        if not self._api_key:
            return {
                "success": False,
                "error": "Weather API key not configured (WEATHER_API_KEY)"
            }
        
        # Normalize city name
        city = self._normalize_city(city)
        
        # Parse the date/time reference
        days_offset, when_text = self._parse_when(when)
        
        logger.info(f"Weather request: city={city}, when={when}, parsed_offset={days_offset}, when_text={when_text}")
        
        params = {
            "q": city,
            "appid": self._api_key,
            "units": "metric"  # Celsius
        }
        
        try:
            client = await self._get_client()
            
            # Use current weather API for today, forecast API for future
            if days_offset == 0:
                # Current weather
                response = await client.get(self._current_url, params=params)
                
                if response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"City '{city}' not found. Please check the spelling."
                    }
                
                response.raise_for_status()
                data = response.json()
                
                # Extract relevant data
                weather_data = {
                    "city": data.get("name", city),
                    "country": data.get("sys", {}).get("country", ""),
                    "temperature": data.get("main", {}).get("temp"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "pressure": data.get("main", {}).get("pressure"),
                    "condition": data.get("weather", [{}])[0].get("main", "Unknown"),
                    "description": data.get("weather", [{}])[0].get("description", ""),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "visibility": data.get("visibility"),
                    "when": when_text,
                    "is_forecast": False
                }
            else:
                # Forecast weather
                response = await client.get(self._forecast_url, params=params)
                
                if response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"City '{city}' not found. Please check the spelling."
                    }
                
                response.raise_for_status()
                data = response.json()
                
                # Forecast API returns 3-hour intervals for 5 days
                # Find the forecast entry closest to the target date
                target_date = datetime.now() + timedelta(days=days_offset)
                target_noon = target_date.replace(hour=12, minute=0, second=0, microsecond=0)
                
                forecast_list = data.get("list", [])
                if not forecast_list:
                    return {
                        "success": False,
                        "error": "No forecast data available"
                    }
                
                # Find the closest forecast entry to noon on the target day
                best_forecast = None
                min_diff = float('inf')
                
                for forecast in forecast_list:
                    forecast_time = datetime.fromtimestamp(forecast.get("dt", 0))
                    time_diff = abs((forecast_time - target_noon).total_seconds())
                    
                    if time_diff < min_diff:
                        min_diff = time_diff
                        best_forecast = forecast
                
                if not best_forecast:
                    best_forecast = forecast_list[0]  # Fallback to first entry
                
                # Extract forecast data
                city_info = data.get("city", {})
                forecast_time = datetime.fromtimestamp(best_forecast.get("dt", 0))
                
                weather_data = {
                    "city": city_info.get("name", city),
                    "country": city_info.get("country", ""),
                    "temperature": best_forecast.get("main", {}).get("temp"),
                    "feels_like": best_forecast.get("main", {}).get("feels_like"),
                    "humidity": best_forecast.get("main", {}).get("humidity"),
                    "pressure": best_forecast.get("main", {}).get("pressure"),
                    "condition": best_forecast.get("weather", [{}])[0].get("main", "Unknown"),
                    "description": best_forecast.get("weather", [{}])[0].get("description", ""),
                    "wind_speed": best_forecast.get("wind", {}).get("speed"),
                    "pop": best_forecast.get("pop", 0),  # Probability of precipitation
                    "when": when_text,
                    "forecast_date": forecast_time.strftime("%A, %B %d"),
                    "forecast_time": forecast_time.strftime("%I:%M %p"),
                    "is_forecast": True
                }
            
            return {
                "success": True,
                "data": weather_data
            }
                    
        except httpx.HTTPError as e:
            logger.error(f"Weather API error: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch weather: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected weather error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _normalize_city(self, city: str) -> str:
        """Normalize city name for API."""
        city = city.strip().strip("?!.,;:")
        
        # Common misspellings
        fixes = {
            "londonn": "London",
            "mumbia": "Mumbai",
            "bombay": "Mumbai",
            "calcutta": "Kolkata",
            "madras": "Chennai",
            "bengaluru": "Bangalore",
            "banglore": "Bangalore",
        }
        
        lower = city.lower()
        if lower in fixes:
            return fixes[lower]
        
        return city
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format weather data for display with rich formatting."""
        if not result.get("success"):
            return f"❌ {result.get('error', 'Unknown error')}"
        
        data = result.get("data", {})
        
        # Get weather emoji
        condition = data.get("condition", "").lower()
        emoji = self._get_weather_emoji(condition)
        
        city = data.get("city", "Unknown")
        country = data.get("country", "")
        # Fix: Don't bold the location here to avoid nested bolding issues
        location = f"{city}, {country}" if country else f"{city}"
        
        temp = data.get("temperature")
        feels_like = data.get("feels_like")
        humidity = data.get("humidity")
        pressure = data.get("pressure")
        description = data.get("description", "").capitalize()
        wind = data.get("wind_speed")
        visibility = data.get("visibility")
        is_forecast = data.get("is_forecast", False)
        when_text = data.get("when", "current")
        
        lines = []
        
        # Build header based on whether it's forecast or current
        if is_forecast:
            forecast_date = data.get("forecast_date", "")
            lines.append(f"{emoji} **Weather Forecast for {location}**")
            if forecast_date:
                lines.append(f"📅 *{forecast_date}* ({when_text})")
        else:
            if when_text == "today":
                lines.append(f"{emoji} **Today's Weather in {location}**")
            else:
                lines.append(f"{emoji} **Current Weather in {location}**")
        
        lines.append("")  # Empty line for spacing
        
        # Temperature section - make it prominent
        if temp is not None:
            temp_rounded = round(temp)
            temp_f = round(temp * 9/5 + 32)  # Convert to Fahrenheit
            # Bold label, plain value
            temp_line = f"🌡️ **Temperature:** {temp_rounded}°C ({temp_f}°F)"
            lines.append(temp_line)
            
            if feels_like is not None:
                feels_rounded = round(feels_like)
                if abs(feels_like - temp) > 2:
                    lines.append(f"   ↳ *Feels like* {feels_rounded}°C")
        
        # Weather condition with description
        if description:
            lines.append(f"{emoji} **Condition:** {description}")
        
        lines.append("")  # Spacing
        
        # Details section
        lines.append("📊 **Details:**")
        
        if humidity is not None:
            humidity_indicator = self._get_humidity_indicator(humidity)
            # Bold label, plain value
            lines.append(f"   💧 **Humidity:** {humidity}% {humidity_indicator}")
        
        if wind is not None:
            wind_desc = self._get_wind_description(wind)
            # Bold label, plain value
            lines.append(f"   💨 **Wind:** {wind} m/s ({wind_desc})")
        
        if pressure is not None:
            # Bold label, plain value
            lines.append(f"   🔵 **Pressure:** {pressure} hPa")
        
        if visibility is not None:
            vis_km = round(visibility / 1000, 1)
            # Bold label, plain value
            lines.append(f"   👁️ **Visibility:** {vis_km} km")
        
        # Add precipitation probability for forecasts
        if is_forecast:
            pop = data.get("pop", 0)
            if pop is not None:
                rain_chance = int(pop * 100)
                if rain_chance > 50:
                    lines.append(f"   🌧️ **Rain chance:** {rain_chance}% ⚠️ *Bring an umbrella!*")
                elif rain_chance > 20:
                    lines.append(f"   🌦️ **Rain chance:** {rain_chance}%")
                elif rain_chance > 0:
                    lines.append(f"   🌤️ **Rain chance:** {rain_chance}% (unlikely)")
                else:
                    lines.append(f"   ☀️ **Rain chance:** 0% - *Clear skies expected!*")
        
        # Add a helpful tip based on conditions
        tip = self._get_weather_tip(temp, humidity, condition, wind)
        if tip:
            lines.append("")
            lines.append(f"💡 **Tip:** {tip}")
        
        return "\n".join(lines)
    
    def _get_humidity_indicator(self, humidity: int) -> str:
        """Get a visual indicator for humidity level."""
        if humidity >= 80:
            return "🌊 Very humid"
        elif humidity >= 60:
            return "💦 Humid"
        elif humidity >= 40:
            return "✨ Comfortable"
        elif humidity >= 20:
            return "🏜️ Dry"
        else:
            return "⚠️ Very dry"
    
    def _get_wind_description(self, wind_speed: float) -> str:
        """Get descriptive text for wind speed."""
        if wind_speed < 1:
            return "Calm"
        elif wind_speed < 5:
            return "Light breeze"
        elif wind_speed < 10:
            return "Moderate wind"
        elif wind_speed < 15:
            return "Strong wind"
        elif wind_speed < 20:
            return "Very strong"
        else:
            return "⚠️ Stormy"
    
    def _get_weather_tip(self, temp: float, humidity: int, condition: str, wind: float) -> str:
        """Get a helpful tip based on weather conditions."""
        tips = []
        
        if temp is not None:
            if temp > 35:
                tips.append("Stay hydrated and avoid direct sunlight! 🧴")
            elif temp > 30:
                tips.append("It's quite warm - stay cool and drink water.")
            elif temp < 10:
                tips.append("Bundle up - it's cold outside! 🧥")
            elif temp < 5:
                tips.append("Very cold! Wear warm layers. 🧣")
        
        if "rain" in condition or "drizzle" in condition:
            tips.append("Don't forget your umbrella! ☔")
        elif "thunder" in condition:
            tips.append("Stay indoors if possible - thunderstorms ahead! ⚡")
        elif "snow" in condition:
            tips.append("Watch out for slippery roads! 🛷")
        elif "fog" in condition or "mist" in condition:
            tips.append("Drive carefully - reduced visibility. 🚗")
        
        if wind and wind > 15:
            tips.append("Strong winds - secure loose items!")
        
        if humidity and humidity > 85 and temp and temp > 25:
            tips.append("High humidity - it may feel hotter than it is.")
        
        # Return the most relevant tip
        return tips[0] if tips else ""
    
    def _get_weather_emoji(self, condition: str) -> str:
        """Get emoji for weather condition."""
        emoji_map = {
            "clear": "☀️",
            "sunny": "☀️",
            "clouds": "☁️",
            "cloudy": "☁️",
            "rain": "🌧️",
            "drizzle": "🌦️",
            "thunderstorm": "⛈️",
            "snow": "❄️",
            "mist": "🌫️",
            "fog": "🌫️",
            "haze": "🌫️",
            "smoke": "💨",
            "dust": "💨",
            "sand": "💨",
            "tornado": "🌪️",
        }
        
        for key, emoji in emoji_map.items():
            if key in condition:
                return emoji
        
        return "🌤️"  # Default
