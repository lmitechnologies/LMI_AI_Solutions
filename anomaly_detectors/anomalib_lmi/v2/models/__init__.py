from .tolerant_anomaly_dino import TolerantAnomalyDINOModel


def __getattr__(name: str):
    if name == "TolerantAnomalyDINO":
        from .tolerant_anomaly_dino import TolerantAnomalyDINO
        return TolerantAnomalyDINO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TolerantAnomalyDINO", "TolerantAnomalyDINOModel"]
