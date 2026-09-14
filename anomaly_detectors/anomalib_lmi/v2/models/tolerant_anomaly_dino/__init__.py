# TolerantAnomalyDINO (Lightning wrapper) is imported lazily to avoid pulling in
# lightning/torchmetrics/scipy at inference time — only torch_model is needed then.
from .torch_model import TolerantAnomalyDINOModel


def __getattr__(name: str):
    if name == "TolerantAnomalyDINO":
        from .lightning_model import TolerantAnomalyDINO
        return TolerantAnomalyDINO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TolerantAnomalyDINO", "TolerantAnomalyDINOModel"]
