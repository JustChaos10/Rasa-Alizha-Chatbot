import os
from dotenv import load_dotenv
 
class ConfigManager:
    def __init__(self):
        load_dotenv()
        self.tavily_key = os.getenv("TAVILY_API_KEY", "")
        self.news_key = os.getenv("NEWS_API_KEY", "")
        self.weather_key = os.getenv("WEATHER_API_KEY", "")
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("BEDROCK_AWS_REGION", "ap-south-1")
        
        # Redis Configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis_conversation_ttl = int(os.getenv("REDIS_CONVERSATION_TTL", "3600")) # 1 hour