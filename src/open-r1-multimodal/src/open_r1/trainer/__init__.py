from importlib import import_module

__all__ = ["VLMGRPOTrainer", "GRPOConfig", "VISTATrainer", "VISTAConfig"]

_MODULE_EXPORTS = {
    "VLMGRPOTrainer": (".grpo_trainer", "VLMGRPOTrainer"),
    "GRPOConfig": (".grpo_config", "GRPOConfig"),
    "VISTATrainer": (".vista_trainer", "VISTATrainer"),
    "VISTAConfig": (".vista_config", "VISTAConfig"),
}


def __getattr__(name):
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MODULE_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
