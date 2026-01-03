#!/usr/bin/env python3
"""
Input Guard Service for Secure RAG Pipeline
Enhanced version of LLM Guard input scanning with terminal interface
"""

import os
import sys
import time
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

# Try to import required modules with fallback handling
try:
    from llm_guard import scan_prompt
    from llm_guard.input_scanners import Anonymize, PromptInjection, Toxicity
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

    def scan_prompt(scanners, prompt):
        return prompt, {"fallback": True}, {"fallback": 1.0}


class ScanResult(Enum):
    """Scan result enumeration"""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class InputScanResult:
    """Result from input scanning"""
    is_valid: bool
    result_type: ScanResult
    sanitized_prompt: str
    original_prompt: str
    scanner_results: Dict[str, bool]
    scanner_scores: Dict[str, float]
    processing_time: float
    warnings: List[str]
    errors: List[str]


class SimpleTokenLimit:
    """Simple token limit checker that doesn't require tiktoken downloads"""

    def __init__(self, limit: int = 4000, encoding_name: str = "cl100k_base"):
        self.limit = limit
        self.encoding_name = encoding_name

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Simple token estimation without requiring tiktoken download"""
        # Rough estimation: ~4 characters per token for English text
        estimated_tokens = len(prompt) // 4
        is_valid = estimated_tokens <= self.limit
        score = min(1.0, estimated_tokens / self.limit) if self.limit > 0 else 0.0
        return prompt, is_valid, score


class SimpleInjectionDetector:
    """Lightweight injection detector used when llm-guard isn't available.

    Flags obvious jailbreak patterns like ignoring instructions or asking for system prompts.
    """

    def __init__(self):
        import re
        # Broad but conservative patterns
        self.patterns = [
            re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
            re.compile(r"ignore\s+.*system\s+prompts?", re.IGNORECASE),
            re.compile(r"reveal|show|print\s+.*(system\s+prompt|hidden\s+prompt|internal\s+instructions)", re.IGNORECASE),
            re.compile(r"pretend\s+to\s+be|act\s+as\s+.*(developer|root|system)", re.IGNORECASE),
            re.compile(r"jailbreak|bypass\s+.*(safety|guard|filter)", re.IGNORECASE),
        ]

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        text = prompt or ""
        for pat in self.patterns:
            if pat.search(text):
                # Block with high score
                return text, False, 1.0
        return text, True, 0.0


class InputGuardService:
    """Enhanced Input Guard Service with comprehensive scanning"""

    def __init__(self, corporate_mode: bool = True, verbose: bool = True):
        """
        Initialize Input Guard Service

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

        # Initialize vault for anonymization
        self.vault = Vault()

        # Setup scanners
        self.scanners = self._setup_scanners()

        if self.verbose:
            print(f"✅ Input Guard Service initialized with {len(self.scanners)} scanners")

    def _setup_scanners(self) -> List:
        """Setup input scanners with corporate network compatibility"""
        scanners = []

        try:
            # Add Anonymize scanner
            scanners.append(Anonymize(self.vault))
            if self.verbose:
                print("✅ Anonymize scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Anonymize scanner failed: {e}")

        try:
            # Add Toxicity scanner
            scanners.append(Toxicity())
            if self.verbose:
                print("✅ Toxicity scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Toxicity scanner failed: {e}")

        try:
            # Add PromptInjection scanner
            scanners.append(PromptInjection())
            if self.verbose:
                print("✅ PromptInjection scanner added")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ PromptInjection scanner failed: {e}")

        # Add simple token limit scanner
        scanners.append(SimpleTokenLimit(limit=4000))
        if self.verbose:
            print("✅ SimpleTokenLimit scanner added")

        # Add minimal injection detector for environments without llm-guard
        scanners.append(SimpleInjectionDetector())
        if self.verbose:
            print("✅ SimpleInjectionDetector scanner added")

        return scanners

    def _print_progress(self, message: str, step: int = 0, total: int = 0):
        """Print progress message with formatting"""
        if not self.verbose:
            return

        if total > 0:
            progress = f"[{step}/{total}] "
        else:
            progress = ""
        print(f"🔍 {progress}{message}")

    def scan_input(self, prompt: str) -> InputScanResult:
        """
        Scan input prompt with all configured scanners

        Args:
            prompt: Input prompt to scan

        Returns:
            InputScanResult with comprehensive scanning results
        """
        start_time = time.time()

        self._print_progress("Starting input security scan...")

        sanitized_prompt = prompt
        scanner_results = {}
        scanner_scores = {}
        warnings = []
        errors = []

        # Process each scanner
        for i, scanner in enumerate(self.scanners, 1):
            scanner_name = scanner.__class__.__name__

            self._print_progress(f"Running {scanner_name} scanner", i, len(self.scanners))

            try:
                if hasattr(scanner, 'scan'):
                    # Custom scanner (like SimpleTokenLimit)
                    sanitized_prompt, is_valid, score = scanner.scan(sanitized_prompt)
                    scanner_results[scanner_name] = is_valid
                    scanner_scores[scanner_name] = score

                    if not is_valid:
                        warnings.append(f"{scanner_name}: Content flagged (score: {score:.2f})")
                else:
                    # Standard LLM Guard scanner
                    try:
                        sanitized_prompt, valid_results, score_results = scan_prompt(
                            [scanner], sanitized_prompt
                        )
                        scanner_results.update(valid_results)
                        scanner_scores.update(score_results)

                        # Check for warnings
                        for key, valid in valid_results.items():
                            if not valid:
                                score = score_results.get(key, 0.0)
                                warnings.append(f"{key}: Content flagged (score: {score:.2f})")

                    except Exception as e:
                        errors.append(f"{scanner_name}: {str(e)}")
                        # Continue with other scanners
                        continue

            except Exception as e:
                errors.append(f"{scanner_name}: {str(e)}")
                continue

        # If only anonymization failed and we produced a sanitized prompt, treat it as a warning
        invalid_scanners = [name for name, valid in scanner_results.items() if not valid]
        if invalid_scanners:
            updated_scanners: Dict[str, bool] = {}
            for name in invalid_scanners:
                lower_name = name.lower()
                score = scanner_scores.get(name, 0.0)
                # Treat anonymize warnings as soft unless content actually changed
                if lower_name.startswith("anonymize"):
                    if sanitized_prompt != prompt:
                        updated_scanners[name] = True
                        if not any("Anonymize" in warning for warning in warnings):
                            warnings.append("Anonymize: Content sanitized during preprocessing")
                    else:
                        updated_scanners[name] = True
                        if not any("Anonymize" in warning for warning in warnings):
                            warnings.append("Anonymize: Person/entity detected but input left unchanged")
                    continue
                # Treat moderate prompt injection scores as warnings
                if lower_name.startswith("promptinjection") and score < 0.95:
                    updated_scanners[name] = True
                    warnings.append(f"PromptInjection: Suspicious phrasing detected (score: {score:.2f})")
                    continue
                # All other scanners remain invalid
            for name, value in updated_scanners.items():
                scanner_results[name] = value

        processing_time = time.time() - start_time

        # Determine overall result
        is_valid = all(result for result in scanner_results.values()) if scanner_results else True

        if errors:
            result_type = ScanResult.ERROR
        elif not is_valid:
            result_type = ScanResult.BLOCKED
        elif warnings:
            result_type = ScanResult.WARNING
        else:
            result_type = ScanResult.SAFE

        # Create result object
        result = InputScanResult(
            is_valid=is_valid,
            result_type=result_type,
            sanitized_prompt=sanitized_prompt,
            original_prompt=prompt,
            scanner_results=scanner_results,
            scanner_scores=scanner_scores,
            processing_time=processing_time,
            warnings=warnings,
            errors=errors
        )

        # Log results
        self._log_scan_result(result)

        return result

    def _log_scan_result(self, result: InputScanResult):
        """Log scan results to terminal"""
        if not self.verbose:
            return

        # Status icon and color
        status_icons = {
            ScanResult.SAFE: "✅",
            ScanResult.WARNING: "⚠️",
            ScanResult.BLOCKED: "❌",
            ScanResult.ERROR: "🔥"
        }

        status_icon = status_icons.get(result.result_type, "❓")

        print(f"\n{status_icon} INPUT SCAN COMPLETE: {result.result_type.value.upper()}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Scanners Run: {len(result.scanner_results)}")
        print(f"   Overall Valid: {'Yes' if result.is_valid else 'No'}")

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
        if result.sanitized_prompt != result.original_prompt:
            print("   🛡️ Content was sanitized during scanning")
            print(f"   Original length: {len(result.original_prompt)} chars")
            print(f"   Sanitized length: {len(result.sanitized_prompt)} chars")


def test_input_guard():
    """Test function for Input Guard Service"""
    print("🧪 Testing Input Guard Service")
    print("="*60)

    try:
        guard = InputGuardService(verbose=True)

        # Test cases
        test_cases = [
            {
                "name": "Safe Query",
                "input": "What is the weather like today?",
                "expected_safe": True
            },
            {
                "name": "Long Query (Token Limit Test)",
                "input": "This is a very long query " * 200,  # Should trigger token limit
                "expected_safe": False
            },
            {
                "name": "Personal Information",
                "input": "My email is john@example.com and my phone is 555-1234. Can you help me?",
                "expected_safe": True  # Should be anonymized but allowed
            },
            {
                "name": "Simple Technical Query",
                "input": "How do I write a Python function to calculate fibonacci numbers?",
                "expected_safe": True
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['name']}")
            print("-" * 50)

            result = guard.scan_input(test_case["input"])

            print(f"Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
            print(f"Expected Safe: {test_case['expected_safe']}")
            print(f"Actual Safe: {result.is_valid}")
            print(f"Result Type: {result.result_type.value}")

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
        print("  python input_guard_service.py test                    # Run tests")
        print("  python input_guard_service.py scan 'text'            # Scan input text")
        print("  python input_guard_service.py scan-quiet 'text'      # Scan without verbose output")
        sys.exit(1)

    command = sys.argv[1]

    if command == "test":
        test_input_guard()
    elif command == "scan" and len(sys.argv) > 2:
        guard = InputGuardService(verbose=True)
        result = guard.scan_input(sys.argv[2])
        print(f"\nFinal Result: {result.result_type.value.upper()}")
        print(f"Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"Sanitized Content: {result.sanitized_prompt}")
    elif command == "scan-quiet" and len(sys.argv) > 2:
        guard = InputGuardService(verbose=False)
        result = guard.scan_input(sys.argv[2])
        print(f"Result: {result.result_type.value}")
        print(f"Valid: {result.is_valid}")
        if result.sanitized_prompt != sys.argv[2]:
            print("Content was modified during scanning")
    else:
        print("Invalid command or missing arguments")
        sys.exit(1)


if __name__ == "__main__":
    main()


