"""
SQL Query Security Module

Provides SQL injection prevention through:
- Query pattern whitelisting
- Dangerous operation blocking
- SQL parsing and validation
- Audit logging
"""

import re
import logging
from typing import Tuple, List, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    import sqlparse
    from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
    from sqlparse.tokens import Keyword, DML
except ImportError:
    sqlparse = None
    print("Warning: sqlparse not installed. Install with: pip install sqlparse")

logger = logging.getLogger("sql-security")

# Allowed SQL operations - ONLY SELECT queries
ALLOWED_OPERATIONS = {'SELECT'}

# Blocked SQL keywords that indicate dangerous operations
BLOCKED_KEYWORDS = {
    'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 
    'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 
    'CALL', 'MERGE', 'REPLACE'
}

# Allowed table name patterns (schema.table format)
ALLOWED_SCHEMAS = [
    'production', 'sales', 'person', 
    'humanresources', 'purchasing'
]

# Query patterns that are explicitly allowed
ALLOWED_QUERY_PATTERNS = [
    # Basic SELECT from allowed schemas
    r"^SELECT\s+.+\s+FROM\s+(production|sales|person|humanresources|purchasing)\.\w+",
    # SELECT with JOIN
    r"^SELECT\s+.+\s+FROM\s+(production|sales|person|humanresources|purchasing)\.\w+\s+(LEFT|RIGHT|INNER|OUTER)?\s*JOIN",
    # Aggregate queries
    r"^SELECT\s+(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\(",
]


class SQLSecurityValidator:
    """Validates SQL queries for security and compliance."""
    
    def __init__(self, enable_audit_log: bool = True):
        self.enable_audit_log = enable_audit_log
        self.audit_log_path = Path("logs/sql_audit.jsonl")
        self.audit_log_path.parent.mkdir(exist_ok=True)
    
    def validate_query(self, sql: str, user_id: str = "system") -> Tuple[bool, str]:
        """
        Validate SQL query for security compliance.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql = sql.strip()
        
        # 1. Check for blocked keywords
        is_blocked, block_msg = self._check_blocked_keywords(sql)
        if is_blocked:
            self._log_blocked_query(sql, user_id, block_msg)
            return False, block_msg
        
        # 2. Verify it's a SELECT query
        if not self._is_select_only(sql):
            msg = "Only SELECT queries are allowed"
            self._log_blocked_query(sql, user_id, msg)
            return False, msg
        
        # 3. Validate table names
        is_valid_table, table_msg = self._validate_table_names(sql)
        if not is_valid_table:
            self._log_blocked_query(sql, user_id, table_msg)
            return False, table_msg
        
        # 4. Check against whitelist patterns
        if not self._matches_whitelist(sql):
            msg = "Query pattern not in whitelist"
            self._log_blocked_query(sql, user_id, msg)
            return False, msg
        
        # 5. Check for SQL injection patterns
        is_injection, injection_msg = self._detect_injection_patterns(sql)
        if is_injection:
            self._log_blocked_query(sql, user_id, injection_msg)
            return False, injection_msg
        
        # Query passed all checks
        self._log_allowed_query(sql, user_id)
        return True, "Query validated"
    
    def _check_blocked_keywords(self, sql: str) -> Tuple[bool, str]:
        """Check if query contains blocked keywords."""
        sql_upper = sql.upper()
        for keyword in BLOCKED_KEYWORDS:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return True, f"Blocked keyword detected: {keyword}"
        return False, ""
    
    def _is_select_only(self, sql: str) -> bool:
        """Verify query is SELECT only."""
        # Use simple string check - more reliable for basic validation
        sql_upper = sql.upper().strip()
        return sql_upper.startswith('SELECT')
    
    def _validate_table_names(self, sql: str) -> Tuple[bool, str]:
        """Validate that all table references use allowed schemas."""
        # Extract fully qualified table references (schema.table)
        qualified_pattern = r'\b(production|sales|person|humanresources|purchasing)\.(\w+)\b'
        qualified_tables = re.findall(qualified_pattern, sql, re.IGNORECASE)
        
        # Find all FROM/JOIN references
        from_pattern = r'\bFROM\s+(\w+(?:\.\w+)?)'
        join_pattern = r'\bJOIN\s+(\w+(?:\.\w+)?)'
        
        from_refs = re.findall(from_pattern, sql, re.IGNORECASE)
        join_refs = re.findall(join_pattern, sql, re.IGNORECASE)
        all_refs = from_refs + join_refs
        
        # Check each reference - it must be qualified (have a dot)
        unqualified = []
        for ref in all_refs:
            if '.' not in ref:
                unqualified.append(ref)
        
        if unqualified:
            return False, f"Unqualified table names not allowed: {unqualified}"
        
        # Verify all schemas are in allowed list
        for schema, table in qualified_tables:
            if schema.lower() not in ALLOWED_SCHEMAS:
                return False, f"Schema not allowed: {schema}"
        
        return True, ""
    
    def _matches_whitelist(self, sql: str) -> bool:
        """Check if query matches any whitelisted pattern."""
        for pattern in ALLOWED_QUERY_PATTERNS:
            if re.match(pattern, sql, re.IGNORECASE):
                return True
        return False
    
    def _detect_injection_patterns(self, sql: str) -> Tuple[bool, str]:
        """Detect common SQL injection patterns."""
        injection_patterns = [
            (r";\s*(DROP|DELETE|UPDATE|INSERT)", "Multiple statements detected"),
            (r"(--|\#|\/\*)", "SQL comment detected"),
            (r"'\s*(OR|AND)\s*'", "Potential injection in WHERE clause"),
            (r"(UNION\s+SELECT)", "UNION injection attempt"),
            (r"(xp_|sp_)", "Stored procedure call detected"),
            (r"(BENCHMARK|SLEEP|WAITFOR)", "Time-based attack detected"),
        ]
        
        for pattern, message in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True, message
        
        return False, ""
    
    def _log_blocked_query(self, sql: str, user_id: str, reason: str):
        """Log blocked SQL query attempt."""
        if not self.enable_audit_log:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "action": "BLOCKED",
                "reason": reason,
                "query": sql[:500],  # Truncate long queries
            }
            
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.warning(f"BLOCKED SQL [user={user_id}]: {reason}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _log_allowed_query(self, sql: str, user_id: str):
        """Log allowed SQL query execution."""
        if not self.enable_audit_log:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "action": "ALLOWED",
                "query": sql[:500],  # Truncate long queries
            }
            
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.info(f"ALLOWED SQL [user={user_id}]: {sql[:100]}...")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# Singleton instance
_validator = None

def get_sql_validator() -> SQLSecurityValidator:
    """Get or create SQL security validator instance."""
    global _validator
    if _validator is None:
        _validator = SQLSecurityValidator()
    return _validator
