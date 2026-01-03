"""
Centralized error handling for Flask application.

Provides consistent error responses and logging for all exceptions.
"""

import logging
from flask import jsonify
from core.exceptions import AppException, get_exception_details

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """
    Register centralized error handlers for the Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(AppException)
    def handle_app_exception(error):
        """Handle all custom application exceptions."""
        details = get_exception_details(error)
        
        # Log the error
        logger.error(
            f"{details['type']}: {details['message']}",
            extra={"details": details['details']},
            exc_info=True
        )
        
        # Return JSON response
        response = {
            "success": False,
            "error": details['message'],
            "error_type": details['type']
        }
        
        # Include details in development mode only
        if app.config.get('DEBUG'):
            response["details"] = details['details']
        
        return jsonify(response), details['http_code']
    
    @app.errorhandler(404)
    def handle_404(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            "success": False,
            "error": "Resource not found",
            "error_type": "NotFound"
        }), 404
    
    @app.errorhandler(500)
    def handle_500(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error. Please try again later.",
            "error_type": "InternalServerError"
        }), 500
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle all unhandled exceptions."""
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        
        # Don't expose internal error details in production
        if app.config.get('DEBUG'):
            error_message = str(error)
        else:
            error_message = "An unexpected error occurred. Please try again later."
        
        return jsonify({
            "success": False,
            "error": error_message,
            "error_type": "UnhandledException"
        }), 500
    
    logger.info("✅ Centralized error handlers registered")
