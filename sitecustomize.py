"""Site customization to maintain compatibility across dependencies.

This project needs both modern packages (LangChain) and legacy Rasa 3.6,
which still imports ``packaging.version.LegacyVersion``. Recent versions
of ``packaging`` (>21) removed that symbol.  We provide a lightweight
shim so the import succeeds without downgrading ``packaging`` and
breaking LangChain.
"""

import importlib
import sys
import types

try:
    from packaging import version as _packaging_version  # type: ignore

    if not hasattr(_packaging_version, "LegacyVersion"):

        class LegacyVersion(_packaging_version.Version):  # type: ignore
            """Minimal stand-in for the removed LegacyVersion.

            Rasa only checks ``isinstance(parsed_version, LegacyVersion)``
            to detect legacy semantics.  Versions produced by modern
            ``packaging`` will never be instances of this class, so the
            behaviour remains the same.
            """

            def __init__(self, version: str) -> None:  # pragma: no cover - fallback only
                super().__init__(version)

        _packaging_version.LegacyVersion = LegacyVersion  # type: ignore[attr-defined]
        if hasattr(_packaging_version, "__all__"):
            _packaging_version.__all__ = list(_packaging_version.__all__) + ["LegacyVersion"]  # type: ignore[attr-defined]
except Exception:
    # Silence any startup errors – the action server should still boot.
    pass

try:
    import pydantic as _pydantic  # type: ignore[attr-defined]

    if "pydantic.v1" not in sys.modules:

        class _PydanticV1(types.ModuleType):
            """Proxy module exposing the legacy pydantic.v1 namespace."""

            def __getattr__(self, item: str):
                return getattr(_pydantic, item)

        _pydantic_v1 = _PydanticV1("pydantic.v1")
        _pydantic_v1.__dict__.update(_pydantic.__dict__)
        sys.modules["pydantic.v1"] = _pydantic_v1

        for suffix in ("dataclasses", "typing", "tools"):
            try:
                sys.modules[f"pydantic.v1.{suffix}"] = importlib.import_module(f"pydantic.{suffix}")
            except Exception:
                continue
except Exception:
    pass
