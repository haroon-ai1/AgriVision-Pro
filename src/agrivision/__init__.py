"""AgriVision Pro -- plant disease classification via transfer learning."""

__version__ = "2.0.0"

__all__ = [
    "ModelConfig",
    "Predictor",
    "build_model",
    "load_checkpoint",
    "prettify",
    "save_checkpoint",
    "__version__",
]

# Lazy re-exports. Importing the submodules eagerly here would pull torch in on
# every `import agrivision`, and would emit a RuntimeWarning when a submodule is
# then run via `python -m agrivision.<name>`.
_LAZY = {
    "ModelConfig": "model",
    "build_model": "model",
    "load_checkpoint": "model",
    "save_checkpoint": "model",
    "Predictor": "predict",
    "prettify": "predict",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
