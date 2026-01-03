"""
Configuration constants for the RASA chatbot application.
"""
 
class Constants:
    # API Configuration
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 180   #30
    VISION_TIMEOUT = 60
   
    # File Processing
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    MAX_IMAGE_DIMENSION = 2048
    JPEG_QUALITY = 85
   
    # Storage Configuration
    MAX_STORAGE_SIZE = 1000
    TIMEOUT_SECONDS = 30
   
    # Directory Configuration
    UPLOAD_FOLDER = "uploads"
    DATA_FOLDER = "data"
    SURVEY_RESPONSES_FILE = "data/survey_responses.json"
   
    # Validation Rules
    MIN_PHONE_LENGTH = 10
    MAX_PHONE_LENGTH = 15
    MIN_NAME_LENGTH = 2
    MIN_ADDRESS_LENGTH = 3
 
class APIConfig:
    """API-specific configuration"""
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    DEFAULT_TEXT_MODEL = "gemma2-9b-it"
    DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_IMAGE_SIZE = Constants.MAX_IMAGE_SIZE_BYTES
    MAX_IMAGE_DIMENSION = Constants.MAX_IMAGE_DIMENSION
 
class MessageTemplates:
    """Common message templates"""
    GREETING = "Hello! I'm your advanced AI assistant. I can help you collect personal information, analyze images, process documents, generate surveys, or answer any questions you have. What would you like to do?"
   
    ADAPTIVE_CARD_FALLBACK = "📝 **Contact Information Form**\n\nPlease provide:\n• Your full name\n• Phone number\n• Complete address"
   
    ERROR_RASA_CONNECTION = "❌ Cannot connect to Rasa server. Make sure it's running on http://localhost:5005"
   
    SUCCESS_INFO_COLLECTED = "✅ **Information collected successfully!**"
   
    CARD_DISPLAYED = "📋 Adaptive Contact Card displayed above"