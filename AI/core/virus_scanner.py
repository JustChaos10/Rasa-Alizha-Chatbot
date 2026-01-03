"""
Virus Scanning Integration Module

Provides framework for ClamAV antivirus scanning of uploaded files.

Installation:
1. Install ClamAV: https://www.clamav.net/downloads
2. Windows: Download installer and run ClamAV service
3. Linux: sudo apt-get install clamav clamav-daemon
4. Install Python client: pip install pyclamd

Configuration:
- Ensure clamd is running: sudo systemctl start clamav-daemon
- Test: clamscan --version
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import ClamAV client
try:
    import clamd
    CLAMAV_AVAILABLE = True
except ImportError:
    CLAMAV_AVAILABLE = False
    logger.warning("pyclamd not installed. Virus scanning disabled. Install with: pip install pyclamd")


class VirusScanner:
    """Scans files for viruses using ClamAV."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize virus scanner.
        
        Args:
            enabled: Whether to enable scanning (defaults to True if ClamAV available)
        """
        self.enabled = enabled and CLAMAV_AVAILABLE
        self.clamd_client = None
        
        if self.enabled:
            try:
                # Try Unix socket first (Linux/Mac)
                try:
                    self.clamd_client = clamd.ClamdUnixSocket()
                    self.clamd_client.ping()
                except:
                    # Fallback to network socket (Windows)
                    self.clamd_client = clamd.ClamdNetworkSocket()
                    self.clamd_client.ping()
                
                logger.info("✅ ClamAV virus scanner initialized")
            except Exception as e:
                logger.warning(f"⚠️ ClamAV not available: {e}. Virus scanning disabled.")
                self.enabled = False
    
    def scan_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Scan a file for viruses.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            Tuple of (is_clean, virus_name)
            - (True, None) if file is clean
            - (False, "virus_name") if virus detected
            - (True, None) if scanning disabled (fail-open for usability)
        """
        if not self.enabled:
            logger.debug("Virus scanning disabled, allowing file")
            return True, None
        
        try:
            result = self.clamd_client.scan(str(file_path))
            
            if result is None:
                # File is clean
                logger.info(f"✅ Virus scan: {file_path.name} is clean")
                return True, None
            else:
                # Virus detected
                file_result = result.get(str(file_path))
                if file_result:
                    status, virus_name = file_result
                    if status == 'FOUND':
                        logger.error(f"❌ Virus detected in {file_path.name}: {virus_name}")
                        return False, virus_name
                
                return True, None
                
        except Exception as e:
            logger.error(f"Virus scan error for {file_path.name}: {e}")
            # Fail-open: allow file if scan fails (prevents DoS)
            # In production, you might want to fail-closed
            return True, None
    
    def scan_stream(self, file_stream) -> Tuple[bool, Optional[str]]:
        """
        Scan a file stream for viruses.
        
        Args:
            file_stream: File-like object to scan
            
        Returns:
            Tuple of (is_clean, virus_name)
        """
        if not self.enabled:
            return True, None
        
        try:
            # Read file content
            file_stream.seek(0)
            content = file_stream.read()
            file_stream.seek(0)
            
            # Scan content
            result = self.clamd_client.instream(content)
            
            if result and result.get('stream'):
                status, virus_name = result['stream']
                if status == 'FOUND':
                    logger.error(f"❌ Virus detected in stream: {virus_name}")
                    return False, virus_name
            
            logger.info("✅ Stream is clean")
            return True, None
            
        except Exception as e:
            logger.error(f"Stream scan error: {e}")
            return True, None  # Fail-open


# Singleton instance
_scanner = None

def get_virus_scanner(enabled: bool = True) -> VirusScanner:
    """Get or create virus scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = VirusScanner(enabled=enabled)
    return _scanner


def is_clamav_available() -> bool:
    """Check if ClamAV is available and working."""
    return CLAMAV_AVAILABLE
