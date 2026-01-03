// Authentication utilities and helpers

class AuthManager {
    constructor() {
        this.currentUser = null;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            await this.getCurrentUser();
            this.initialized = true;
        } catch (error) {
            console.error('Failed to initialize auth:', error);
        }
    }

    async getCurrentUser() {
        try {
            const response = await fetch('/auth/me');
            const data = await response.json();
            
            this.currentUser = data.authenticated ? {
                email: data.email,
                role: data.role,
                authenticated: true
            } : null;
            
            return this.currentUser;
        } catch (error) {
            console.error('Error fetching current user:', error);
            this.currentUser = null;
            return null;
        }
    }

    isAuthenticated() {
        return this.currentUser && this.currentUser.authenticated;
    }

    hasRole(role) {
        return this.isAuthenticated() && this.currentUser.role === role;
    }

    isAdmin() {
        return this.hasRole('admin');
    }

    async login(email, password) {
        try {
            const response = await fetch('/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (data.ok) {
                this.currentUser = {
                    email: data.email,
                    role: data.role,
                    authenticated: true
                };
                return { success: true, data };
            } else {
                return { success: false, error: data.error };
            }
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    async logout() {
        try {
            const response = await fetch('/auth/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (data.ok) {
                this.currentUser = null;
                return { success: true };
            } else {
                return { success: false, error: data.error };
            }
        } catch (error) {
            console.error('Logout error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    async register(email, password, name = '') {
        try {
            const response = await fetch('/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password, name })
            });

            const data = await response.json();

            if (data.ok) {
                return { success: true, data };
            } else {
                return { success: false, error: data.error };
            }
        } catch (error) {
            console.error('Registration error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    // Helper function to handle authentication redirects
    handleAuthRedirect() {
        if (!this.isAuthenticated() && window.location.pathname !== '/auth/login' && window.location.pathname !== '/auth/register') {
            window.location.href = '/auth/login';
            return true;
        }
        return false;
    }

    // Helper function to show authentication-related toasts
    showAuthToast(message, type = 'info') {
        const classes = {
            success: 'green',
            error: 'red',
            info: 'blue',
            warning: 'orange'
        };
        
        if (typeof M !== 'undefined' && M.toast) {
            M.toast({
                html: message,
                classes: classes[type] || 'blue'
            });
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Create global auth manager instance
window.authManager = new AuthManager();

// Initialize on DOM ready
if (typeof $ !== 'undefined') {
    $(document).ready(function() {
        window.authManager.initialize();
    });
} else {
    document.addEventListener('DOMContentLoaded', function() {
        window.authManager.initialize();
    });
}

// Export for ES6 modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AuthManager;
}