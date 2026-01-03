from functools import wraps
from flask import jsonify, redirect, url_for, request
from flask_login import current_user

def role_required(required_role):
    """
    Decorator to require specific role for access to routes.
    Returns 403 for JSON requests, redirects to login for HTML requests.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json:
                    return jsonify({'error': 'Authentication required'}), 401
                return redirect(url_for('auth.login'))
            
            if not current_user.has_role(required_role):
                if request.is_json:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator