from typing import Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

DatasetPair = Tuple[Dataset, Optional[Dataset]]
DatasetFactory = Callable[..., DatasetPair]

_DATASET_REGISTRY: Dict[str, DatasetFactory] = {}


def register_dataset(name: str):
    """
    Decorator to register a new dataset factory function. The decorated function should
    return a tuple of (train_ds, val_ds).
    """

    def _decorator(func):
        assert name not in _DATASET_REGISTRY, f"dataset {name} already registered"
        _DATASET_REGISTRY[name] = func
        return func

    return _decorator


def create_dataset(name: str, **kwargs) -> DatasetPair:
    """
    Create a dataset by name. Returns a tuple of (train_ds, val_ds).
    """
    assert name in _DATASET_REGISTRY, f"dataset {name} not registered"
    dataset = _DATASET_REGISTRY[name](**kwargs)
    return dataset


def list_datasets() -> List[str]:
    """
    List all registered datasets.
    """
    return list(_DATASET_REGISTRY.keys())
