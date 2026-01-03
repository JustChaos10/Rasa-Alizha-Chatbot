import logging
import sys
import re

class Logger:
    def __init__(self, name: str, log_file: str = 'mcp.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        # Use UTF-8 encoding for file output
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.handlers = []
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def _sanitize_message(self, message: str) -> str:
        """Sanitize message by removing problematic Unicode characters"""
        if not isinstance(message, str):
            message = str(message)

        # Remove null bytes and other control characters (except newlines and tabs)
        message = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', message)

        # Replace problematic Unicode characters with safe alternatives
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
        }

        for unicode_char, replacement in replacements.items():
            message = message.replace(unicode_char, replacement)

        # Ensure the message is valid UTF-8
        try:
            message.encode('utf-8')
        except UnicodeEncodeError:
            # If encoding fails, replace problematic characters
            message = message.encode('utf-8', errors='replace').decode('utf-8')

        return message

    def info(self, message: str):
        clean_message = self._sanitize_message(message)
        self.logger.info(clean_message)

    def error(self, message: str):
        clean_message = self._sanitize_message(message)
        self.logger.error(clean_message)

    def warning(self, message: str):
        clean_message = self._sanitize_message(message)
        self.logger.warning(clean_message)

    def debug(self, message: str):
        clean_message = self._sanitize_message(message)
        self.logger.debug(clean_message)
