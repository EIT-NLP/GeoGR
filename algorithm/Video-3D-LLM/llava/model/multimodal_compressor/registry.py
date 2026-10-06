from typing import Dict, List, Optional, Type

from .base import BaseCompressor


COMPRESSOR_REGISTRY: Dict[str, Type[BaseCompressor]] = {}


def register_compressor(name: str):
    def decorator(cls: Type[BaseCompressor]) -> Type[BaseCompressor]:
        if name in COMPRESSOR_REGISTRY:
            raise ValueError(f"Compressor '{name}' is already registered.")
        if not issubclass(cls, BaseCompressor):
            raise TypeError(f"{cls.__name__} must inherit from BaseCompressor.")
        COMPRESSOR_REGISTRY[name] = cls
        cls._compressor_name = name
        return cls

    return decorator


def get_compressor(name: str) -> Type[BaseCompressor]:
    if name not in COMPRESSOR_REGISTRY:
        raise KeyError(f"Unknown compressor '{name}'. Available: {list_compressors()}")
    return COMPRESSOR_REGISTRY[name]


def list_compressors() -> List[str]:
    return list(COMPRESSOR_REGISTRY.keys())


def is_registered(name: str) -> bool:
    return name in COMPRESSOR_REGISTRY


def unregister_compressor(name: str) -> Optional[Type[BaseCompressor]]:
    return COMPRESSOR_REGISTRY.pop(name, None)
