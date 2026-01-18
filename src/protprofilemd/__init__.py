from .version import __version__

try:
    import protprofilemdrust as _rust
except Exception as e:
    raise ImportError("Failed to import compiled extension 'protprofilemdrust'") from e
