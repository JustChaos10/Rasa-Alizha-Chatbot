#!/usr/bin/env python3
import sys
import importlib
import json


def check_imports() -> bool:
    print("== Import checks ==")
    ok = True
    mods = [
        'secure_rag',
        'secure_rag.groq_service',
        'secure_rag.input_guard_service',
        'secure_rag.output_guard_service',
        'secure_rag.rbac_service',
        'secure_rag.secure_rag_pipeline',
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            print("OK import:", m)
        except Exception as e:
            ok = False
            print("FAIL import:", m, "\n ", e)
    return ok


def run_pipeline_once() -> bool:
    print("\n== Pipeline smoke test ==")
    try:
        from secure_rag.secure_rag_pipeline import SecureRAGPipeline
        p = SecureRAGPipeline(verbose=True, debug_mode=False)
        res = p.process_query("What are the company policies?", "hr_admin")
        print("Pipeline success:", res.success)
        print("Stages:", len(res.stage_results))
        print("Result type:", res.result_type.value)
        print("Final response preview:", (res.final_response or "")[:200])
        # Emit a compact JSON summary for downstream automation
        print("JSON_SUMMARY:", json.dumps({
            "success": res.success,
            "result_type": res.result_type.value,
            "stages": len(res.stage_results),
            "security_passed": res.security_summary.get('security_checks_passed', 0),
            "security_total": res.security_summary.get('total_security_checks', 0),
        }))
        return bool(res.success)
    except Exception as e:
        print("Pipeline run failed:\n ", e)
        return False


def main() -> int:
    ok = check_imports()
    if not ok:
        return 2
    ok = run_pipeline_once() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


