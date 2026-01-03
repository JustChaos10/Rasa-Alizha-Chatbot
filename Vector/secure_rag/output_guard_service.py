#!/usr/bin/env python3
"""
Output Guard Service for Secure RAG Pipeline
Enhanced version of LLM Guard output scanning with terminal interface
"""

import os
import sys
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import AccessLevel for role-based logic
from .rbac_service import AccessLevel

# Try to import required modules with fallback handling
try:
    from llm_guard import scan_output
    from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
    from llm_guard.vault import Vault
    import ssl
    import urllib3

    # Corporate network SSL fixes (from existing code)
    def setup_corporate_ssl():
        """Setup SSL configurations for corporate networks"""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ['CURL_CA_BUNDLE'] = ""
        os.environ['REQUESTS_CA_BUNDLE'] = ""
        print("🔧 Corporate SSL configuration applied")

except ImportError as e:
    print(f"⚠️ Warning: {e}")
    print("LLM Guard not available. Using fallback mode.")

    # Fallback implementations
    class Vault:
        def __init__(self):
            pass

    def scan_output(scanners, prompt, response):
        return response, {"fallback": True}, {"fallback": 1.0}


class OutputScanResultType(Enum):
    """Output scan result enumeration"""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class OutputScanResult:
    """Result from output scanning"""
    is_valid: bool
    result_type: OutputScanResultType
    sanitized_response: str
    original_response: str
    original_prompt: str
    scanner_results: Dict[str, bool]
    scanner_scores: Dict[str, float]
    processing_time: float
    warnings: List[str]
    errors: List[str]
    quality_score: float


class SimpleContentFilter:
    """Simple content filter for fallback mode"""

    def __init__(self, filter_type: str = "general"):
        self.filter_type = filter_type

        # Define sensitive patterns
        self.sensitive_patterns = [
            # Personal information patterns
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone

            # Potentially harmful content
            r'\b(password|secret|confidential|classified)\b',
            r'\b(hack|exploit|vulnerability)\b',

            # Prompt leak attempts / system prompt
            r'(reveal|show|print).*(system\s+prompt|hidden\s+prompt|internal\s+instructions)',
            r'(ignore\s+all\s+previous\s+instructions)'
        ]

    def scan(self, prompt: str, response: str) -> tuple[str, bool, float]:
        """Simple content filtering with basic categories.

        - Blocks: PII-like patterns (SSN, CC, emails, phones) and prompt-leak attempts.
        - Sanitizes: generic sensitive words (password/secret/confidential/classified) by redacting them.
        - Leaves others intact.
        """
        import re

        text = response or ""

        # Detect prompt-leak attempts explicitly
        leak_patterns = [
            r'(reveal|show|print)\s+.*(system\s+prompt|hidden\s+prompt|internal\s+instructions)',
            r'ignore\s+all\s+previous\s+instructions',
            r'ignore\s+.*system\s+prompts?'
        ]
        for lp in leak_patterns:
            if re.search(lp, text, re.IGNORECASE):
                return text, False, 0.2

        # PII-like sensitive patterns (from self.sensitive_patterns except generic words)
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        ]
        for pp in pii_patterns:
            if re.search(pp, text, re.IGNORECASE):
                return text, False, 0.2

        # Sanitize generic sensitive words if present
        redacted = re.sub(r'\b(password|secret|confidential|classified)\b', '[redacted]', text, flags=re.IGNORECASE)

        # Simple quality check
        if len(redacted.strip()) < 10:
            quality_score = 0.3
        elif re.search(r"\b(I cannot|I'm not able)\b", redacted):
            quality_score = 0.5
        else:
            quality_score = 0.9

        return redacted, True, quality_score


class OutputGuardService:
    """Enhanced Output Guard Service with comprehensive scanning"""

    def __init__(self, corporate_mode: bool = True, verbose: bool = True):
        """
        Initialize Output Guard Service

        Args:
            corporate_mode: Enable corporate network compatibility
            verbose: Enable verbose terminal output
        """
        self.corporate_mode = corporate_mode
        self.verbose = verbose

        if corporate_mode:
            try:
                setup_corporate_ssl()
            except:
                print("⚠️ Could not setup corporate SSL")

        # Initialize vault for deanonymization
        self.vault = Vault()

        # Setup scanners
        self.scanners = self._setup_scanners()

        if self.verbose:
            print(f"✅ Output Guard Service initialized with {len(self.scanners)} scanners")

    def _setup_scanners(self) -> List:
        """Setup output scanners with corporate network compatibility"""
        scanners = []

        try:
            # Add Deanonymize scanner
            scanners.append(Deanonymize(self.vault))
            if self.verbose:
                print("✅ Deanonymize scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Deanonymize scanner failed: {e}")

        try:
            # Add NoRefusal scanner
            scanners.append(NoRefusal())
            if self.verbose:
                print("✅ NoRefusal scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ NoRefusal scanner failed: {e}")

        try:
            # Add Relevance scanner
            scanners.append(Relevance())
            if self.verbose:
                print("✅ Relevance scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Relevance scanner failed: {e}")

        try:
            sensitive_types = [
                "CREDIT_CARD",
                "CRYPTO",
                "EMAIL_ADDRESS",
                "IBAN_CODE",
                "IP_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "US_BANK_NUMBER",
                "CREDIT_CARD_RE",
                "UUID",
                "EMAIL_ADDRESS_RE",
                "US_SSN_RE"
            ]
            scanners.append(Sensitive(entity_types=sensitive_types, threshold=0.6))
            if self.verbose:
                print("? Sensitive scanner added")
        except Exception as e:
            if self.verbose:
                print(f"?? Sensitive scanner failed: {e}")

        # Add simple content filter as fallback
        scanners.append(SimpleContentFilter())
        if self.verbose:
            print("✅ SimpleContentFilter scanner added")

        return scanners

    def _print_progress(self, message: str, step: int = 0, total: int = 0):
        """Print progress message with formatting"""
        if not self.verbose:
            return

        if total > 0:
            progress = f"[{step}/{total}] "
        else:
            progress = ""
        print(f"🛡️ {progress}{message}")

    def scan_output(self, prompt: str, response: str, access_level: Optional[AccessLevel] = None) -> OutputScanResult:
        """
        Scan output response with all configured scanners

        Args:
            prompt: Original input prompt
            response: Generated response to scan
            access_level: User's access level for role-based sanitization

        Returns:
            OutputScanResult with comprehensive scanning results
        """
        start_time = time.time()

        self._print_progress("Starting output security scan...")

        sanitized_response = response
        scanner_results = {}
        scanner_scores = {}
        warnings = []
        errors = []
        quality_scores = []

        # Process each scanner
        for i, scanner in enumerate(self.scanners, 1):
            scanner_name = scanner.__class__.__name__

            self._print_progress(f"Running {scanner_name} scanner", i, len(self.scanners))

            try:
                if hasattr(scanner, 'scan'):
                    # Custom scanner (like SimpleContentFilter)
                    sanitized_response, is_valid, score = scanner.scan(prompt, sanitized_response)
                    scanner_results[scanner_name] = bool(is_valid)
                    scanner_scores[scanner_name] = float(score)
                    quality_scores.append(float(score))

                    if not is_valid:
                        warnings.append(f"{scanner_name}: Content flagged (score: {score:.2f})")
                else:
                    # Standard LLM Guard scanner
                    try:
                        sanitized_response, valid_results, score_results = scan_output(
                            [scanner], prompt, sanitized_response
                        )

                        # Convert any numpy types to native Python
                        valid_results = {k: bool(v) for k, v in valid_results.items()}
                        score_results = {k: float(v) for k, v in score_results.items()}

                        scanner_results.update(valid_results)
                        scanner_scores.update(score_results)

                        # Add scores to quality tracking
                        for score in score_results.values():
                            quality_scores.append(float(score))

                        # Check for warnings
                        for key, valid in valid_results.items():
                            if not valid:
                                score = score_results.get(key, 0.0)
                                warnings.append(f"{key}: Content flagged (score: {float(score):.2f})")

                    except Exception as e:
                        errors.append(f"{scanner_name}: {str(e)}")
                        # Continue with other scanners
                        continue

            except Exception as e:
                errors.append(f"{scanner_name}: {str(e)}")
                continue

        processing_time = time.time() - start_time

        # Calculate overall quality score
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

        # Downgrade non-critical scanner failures to warnings before computing validity
        for name, valid in list(scanner_results.items()):
            if valid:
                continue
            score = scanner_scores.get(name, 0.0)
            if name == "Relevance":
                warnings.append(f"Relevance: Response may be off-topic (score: {score:.2f})")
                scanner_results[name] = True
            elif name == "NoRefusal" and access_level == AccessLevel.FULL:
                warnings.append("NoRefusal: Non-refusal allowed for admin access")
                scanner_results[name] = True

        # Determine overall is_valid before overrides
        is_valid = all(scanner_results.values()) if scanner_results else False

        # Role-based sanitization logic
        if access_level is None:
            access_level = AccessLevel.LIMITED  # Default to restricted

        failed_scanners = [k for k, v in scanner_results.items() if not v]
        if failed_scanners:
            sensitive_failed = 'Sensitive' in failed_scanners
            simple_filter_failed = 'SimpleContentFilter' in failed_scanners
            other_failed = [s for s in failed_scanners if s not in ['Sensitive', 'SimpleContentFilter']]

            # Allow privileged users to view sensitive-but-approved content (never bypass SimpleContentFilter)
            if sensitive_failed and access_level == AccessLevel.FULL and not simple_filter_failed:
                warnings.append('Sensitive: Content allowed for admin access')
                scanner_results['Sensitive'] = True
                sensitive_failed = False
                failed_scanners = [s for s in failed_scanners if s != 'Sensitive']

            # Strict policy: if Sensitive OR SimpleContentFilter failed, refuse regardless of role
            if sensitive_failed or simple_filter_failed:
                refusal = (
                    "I can't share that content. If you're asking to reveal internal prompts or "
                    "sensitive information, please rephrase your request."
                )
                sanitized_response = refusal
                is_valid = False
                # keep scanner flags as-is for logging; do not override to True

            if not (sensitive_failed or simple_filter_failed):
                is_valid = all(scanner_results.values()) if scanner_results else False

        # Determine overall result after overrides
        if errors:
            result_type = OutputScanResultType.ERROR
        elif not is_valid:
            result_type = OutputScanResultType.BLOCKED
        elif warnings:
            result_type = OutputScanResultType.WARNING
        else:
            result_type = OutputScanResultType.SAFE

        # Create result object
        result = OutputScanResult(
            is_valid=is_valid,
            result_type=result_type,
            sanitized_response=sanitized_response,
            original_response=response,
            original_prompt=prompt,
            scanner_results=scanner_results,
            scanner_scores=scanner_scores,
            processing_time=processing_time,
            warnings=warnings,
            errors=errors,
            quality_score=overall_quality
        )

        # Log results
        self._log_scan_result(result)

        return result

    def _log_scan_result(self, result: OutputScanResult):
        """Log scan results to terminal"""
        if not self.verbose:
            return

        # Status icon and color
        status_icons = {
            OutputScanResultType.SAFE: "✅",
            OutputScanResultType.WARNING: "⚠️",
            OutputScanResultType.BLOCKED: "❌",
            OutputScanResultType.ERROR: "🔥"
        }

        status_icon = status_icons.get(result.result_type, "❓")

        print(f"\n{status_icon} OUTPUT SCAN COMPLETE: {result.result_type.value.upper()}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Scanners Run: {len(result.scanner_results)}")
        print(f"   Overall Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"   Quality Score: {result.quality_score:.2f}")

        # Show scanner details
        if result.scanner_results:
            print("   Scanner Results:")
            for scanner, valid in result.scanner_results.items():
                score = result.scanner_scores.get(scanner, 0.0)
                status = "✅ PASS" if valid else "❌ FAIL"
                print(f"     {scanner}: {status} (score: {score:.2f})")

        # Show warnings
        if result.warnings:
            print("   Warnings:")
            for warning in result.warnings:
                print(f"     ⚠️ {warning}")

        # Show errors
        if result.errors:
            print("   Errors:")
            for error in result.errors:
                print(f"     🔥 {error}")

        # Show content changes
        if result.sanitized_response != result.original_response:
            print("   🛡️ Content was sanitized during scanning")
            print(f"   Original length: {len(result.original_response)} chars")
            print(f"   Sanitized length: {len(result.sanitized_response)} chars")


def test_output_guard():
    """Test function for Output Guard Service"""
    print("🧪 Testing Output Guard Service")
    print("="*60)

    try:
        guard = OutputGuardService(verbose=True)

        # Test cases
        test_cases = [
            {
                "name": "Safe Response",
                "prompt": "What is AI?",
                "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
                "expected_safe": True
            },
            {
                "name": "Response with Personal Info",
                "prompt": "Tell me about yourself",
                "response": "My email is john@example.com and my phone is 555-1234. I can help you with AI questions.",
                "expected_safe": False
            },
            {
                "name": "Refusal Response",
                "prompt": "How to hack systems?",
                "response": "I cannot and will not provide information on hacking systems as it's illegal and unethical.",
                "expected_safe": True
            },
            {
                "name": "Technical Response",
                "prompt": "Write Python code",
                "response": "Here's a simple Python function:\n\n\ndef hello_world():\n    print('Hello, World!')\n    return True",
                "expected_safe": True
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['name']}")
            print("-" * 50)

            result = guard.scan_output(test_case["prompt"], test_case["response"])

            print(f"Prompt: {test_case['prompt']}")
            print(f"Response: {test_case['response'][:100]}{'...' if len(test_case['response']) > 100 else ''}")
            print(f"Expected Safe: {test_case['expected_safe']}")
            print(f"Actual Safe: {result.is_valid}")
            print(f"Result Type: {result.result_type.value}")
            print(f"Quality Score: {result.quality_score:.2f}")

            # Simple pass/fail check
            if result.is_valid == test_case['expected_safe']:
                print("✅ Test PASSED")
            else:
                print("❌ Test FAILED")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python output_guard_service.py test                           # Run tests")
        print("  python output_guard_service.py scan 'prompt' 'response'      # Scan output")
        print("  python output_guard_service.py scan-quiet 'prompt' 'response' # Scan without verbose output")
        sys.exit(1)

    command = sys.argv[1]

    if command == "test":
        test_output_guard()
    elif command == "scan" and len(sys.argv) > 3:
        guard = OutputGuardService(verbose=True)
        result = guard.scan_output(sys.argv[2], sys.argv[3])
        print(f"\nFinal Result: {result.result_type.value.upper()}")
        print(f"Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Sanitized Response: {result.sanitized_response}")
    elif command == "scan-quiet" and len(sys.argv) > 3:
        guard = OutputGuardService(verbose=False)
        result = guard.scan_output(sys.argv[2], sys.argv[3])
        print(f"Result: {result.result_type.value}")
        print(f"Valid: {result.is_valid}")
        print(f"Quality: {result.quality_score:.2f}")
        if result.sanitized_response != sys.argv[3]:
            print("Content was modified during scanning")
    else:
        print("Invalid command or missing arguments")
        sys.exit(1)


if __name__ == "__main__":
    main()


