from importlib import import_module

__all__ = ["VLMBaseModule", "Qwen2VLModule", "InvernVLModule", "Qwen3VLModule"]

_MODULE_EXPORTS = {
    "VLMBaseModule": (".vlm_module", "VLMBaseModule"),
    "Qwen2VLModule": (".qwen_module", "Qwen2VLModule"),
    "InvernVLModule": (".internvl_module", "InvernVLModule"),
    "Qwen3VLModule": (".qwen3_module", "Qwen3VLModule"),
}


def __getattr__(name):
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MODULE_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
