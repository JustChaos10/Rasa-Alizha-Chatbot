"""
Custom Exception Hierarchy for RASA Chatbot

Provides specific exception types for better error handling and debugging.
"""


class AppException(Exception):
    """Base exception for all application errors."""
    http_code = 500
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ============================================================================
# VALIDATION ERRORS (400 range)
# ============================================================================

class ValidationError(AppException):
    """Raised when input validation fails."""
    http_code = 400


class InputTooLongError(ValidationError):
    """Raised when input exceeds maximum length."""
    pass


class InvalidFileTypeError(ValidationError):
    """Raised when uploaded file type is not allowed."""
    pass


class InvalidQueryError(ValidationError):
    """Raised when query format is invalid."""
    pass


# ============================================================================
# AUTHENTICATION & AUTHORIZATION ERRORS (401, 403)
# ============================================================================

class AuthenticationError(AppException):
    """Raised when authentication fails."""
    http_code = 401


class AuthorizationError(AppException):
    """Raised when user lacks required permissions."""
    http_code = 403


class SessionExpiredError(AuthenticationError):
    """Raised when user session has expired."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""
    pass


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user lacks required role/permissions."""
    pass


# ============================================================================
# SECURITY ERRORS (403)
# ============================================================================

class SecurityError(AppException):
    """Raised when a security violation is detected."""
    http_code = 403


class SQLInjectionError(SecurityError):
    """Raised when SQL injection attempt is detected."""
    pass


class PromptInjectionError(SecurityError):
    """Raised when LLM prompt injection is detected."""
    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attempt is detected."""
    pass


# ============================================================================
# DATABASE ERRORS (500)
# ============================================================================

class DatabaseError(AppException):
    """Raised when database operations fail."""
    http_code = 500


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when database record is not found."""
    http_code = 404


class DuplicateRecordError(DatabaseError):
    """Raised when attempting to create duplicate record."""
    http_code = 409


# ============================================================================
# TOOL & SERVICE ERRORS (500)
# ============================================================================

class ToolExecutionError(AppException):
    """Raised when tool execution fails."""
    http_code = 500


class ToolNotFoundError(ToolExecutionError):
    """Raised when requested tool does not exist."""
    http_code = 404


class ToolTimeoutError(ToolExecutionError):
    """Raised when tool execution times out."""
    http_code = 504


class ServiceUnavailableError(AppException):
    """Raised when external service is unavailable."""
    http_code = 503


class LLMServiceError(ServiceUnavailableError):
    """Raised when LLM service fails."""
    pass


class TranslationServiceError(ServiceUnavailableError):
    """Raised when translation service fails."""
    pass


class MCPServerError(ServiceUnavailableError):
    """Raised when MCP server connection/execution fails."""
    pass


# ============================================================================
# RATE LIMITING ERRORS (429)
# ============================================================================

class RateLimitError(AppException):
    """Raised when rate limit is exceeded."""
    http_code = 429


class TooManyRequestsError(RateLimitError):
    """Raised when user exceeds request rate limit."""
    pass


# ============================================================================
# FILE & UPLOAD ERRORS (400, 413, 500)
# ============================================================================

class FileError(AppException):
    """Base exception for file-related errors."""
    http_code = 500


class FileSizeTooLargeError(FileError):
    """Raised when uploaded file exceeds size limit."""
    http_code = 413


class FileProcessingError(FileError):
    """Raised when file processing fails."""
    pass


class MalwareDetectedError(SecurityError):
    """Raised when malware is detected in uploaded file."""
    http_code = 400


# ============================================================================
# CONFIGURATION ERRORS (500)
# ============================================================================

class ConfigurationError(AppException):
    """Raised when configuration is invalid or missing."""
    http_code = 500


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when required environment variable is missing."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration value is invalid."""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_exception_details(exception: Exception) -> dict:
    """
    Extract details from exception for logging/response.
    
    Args:
        exception: The exception to extract details from
        
    Returns:
        Dictionary with exception details
    """
    if isinstance(exception, AppException):
        return {
            "type": exception.__class__.__name__,
            "message": exception.message,
            "http_code": exception.http_code,
            "details": exception.details
        }
    else:
        return {
            "type": exception.__class__.__name__,
            "message": str(exception),
            "http_code": 500,
            "details": {}
        }
