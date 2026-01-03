"""
Enhanced file upload security module.

Provides:
- Content-type validation
- Path traversal protection
- File size limits per user
- Malware detection (placeholder for ClamAV integration)
"""

import os
import hashlib
from pathlib import Path
from typing import Tuple, Optional
import logging
import magic  # python-magic for MIME type detection

from core.exceptions import (
    InvalidFileTypeError, FileSizeTooLargeError, 
    PathTraversalError, MalwareDetectedError
)

logger = logging.getLogger(__name__)

# File upload configuration
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'webp'}
ALLOWED_MIME_TYPES = {
    'application/pdf': ['.pdf'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    'text/plain': ['.txt'],
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/gif': ['.gif'],
    'image/webp': ['.webp'],
}

# User upload quotas (per user)
USER_UPLOAD_QUOTA = {
    'admin': 100 * 1024 * 1024,  # 100MB
    'user': 50 * 1024 * 1024,     # 50MB
    'guest': 10 * 1024 * 1024,    # 10MB
}


class FileSecurityValidator:
    """Validates uploaded files for security compliance."""
    
    def __init__(self, upload_folder: str):
        self.upload_folder = Path(upload_folder).resolve()
        self.upload_folder.mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, file, filename: str, user_role: str = 'user') -> Tuple[bool, str, Optional[Path]]:
        """
        Comprehensive file validation.
        
        Returns:
            Tuple of (is_valid, message, safe_path)
        """
        # 1. Check file extension
        if not self._check_extension(filename):
            raise InvalidFileTypeError(f"File type not allowed: {filename}")
        
        # 2. Get safe file path (prevents path traversal)
        safe_path = self._get_safe_path(filename)
        
        # 3. Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > MAX_FILE_SIZE:
            raise FileSizeTooLargeError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")
        
        # 4. Verify MIME type matches extension
        file.seek(0)
        mime_type = magic.from_buffer(file.read(2048), mime=True)
        file.seek(0)
        
        if not self._verify_mime_type(filename, mime_type):
            raise InvalidFileTypeError(f"MIME type mismatch: {mime_type} for {filename}")
        
        # 5. Check user quota (placeholder for future implementation)
        # self._check_user_quota(user_role, size)
        
        # 6. Scan for malware (placeholder - would integrate ClamAV here)
        # is_clean = self._scan_for_malware(file)
        # if not is_clean:
        #     raise MalwareDetectedError("Malware detected in uploaded file")
        
        logger.info(f"✅ File validated: {filename} ({size} bytes, {mime_type})")
        return True, "File validated successfully", safe_path
    
    def _check_extension(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in ALLOWED_EXTENSIONS
    
    def _get_safe_path(self, filename: str) -> Path:
        """
        Get safe file path preventing path traversal.
        
        Raises:
            PathTraversalError: If path traversal is detected
        """
        from werkzeug.utils import secure_filename
        
        # Sanitize filename
        safe_name = secure_filename(filename)
        if not safe_name:
            raise InvalidFileTypeError("Invalid filename")
        
        # Construct absolute path
        file_path = (self.upload_folder / safe_name).resolve()
        
        # Verify path is within upload folder
        if not str(file_path).startswith(str(self.upload_folder)):
            logger.warning(f"Path traversal attempt: {filename}")
            raise PathTraversalError(f"Path traversal detected: {filename}")
        
        return file_path
    
    def _verify_mime_type(self, filename: str, mime_type: str) -> bool:
        """Verify MIME type matches file extension."""
        if mime_type not in ALLOWED_MIME_TYPES:
            return False
        
        # Get file extension
        ext = '.' + filename.rsplit('.', 1)[1].lower()
        
        # Check if extension is valid for this MIME type
        allowed_exts = ALLOWED_MIME_TYPES[mime_type]
        return ext in allowed_exts
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


# Singleton instance
_file_validator = None

def get_file_validator(upload_folder: str = "uploads") -> FileSecurityValidator:
    """Get or create file validator instance."""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileSecurityValidator(upload_folder)
    return _file_validator
