"""The rfdetr payload embedded in an exported .onnx or .engine.

One place where the key names and the fallbacks live, so the exporter in ``convert.py`` and the engine
backends in ``model.py`` cannot drift apart.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_NUM_SELECT = 300  # rfdetr's own default, for a model exported before num_select was embedded


@dataclass(frozen=True)
class RfdetrMetadata:
    """What an exported rfdetr graph cannot tell us about itself, recorded at export time.

    num_select varies by variant and is not recoverable from the graph, and neither are the class names.
    """

    class_names: Optional[List[str]] = None
    num_select: int = DEFAULT_NUM_SELECT

    @classmethod
    def from_model(cls, model) -> "RfdetrMetadata":
        """Read the pair off a live rfdetr model, at export time."""
        return cls(class_names=list(model.class_names), num_select=int(model.model.postprocess.num_select))

    @classmethod
    def from_engine(cls, metadata: Mapping[str, Any], model_path: str) -> "RfdetrMetadata":
        """Decode what an engine wrapper read off the model file, naming anything it had to fall back on.

        Args:
            metadata: The engine wrapper's ``metadata``; {} for a model carrying none.
            model_path: Path to the model file, for the warning.
        """
        class_names = metadata.get("class_names")
        num_select = metadata.get("num_select")
        if num_select is None:
            logger.warning(f"No num_select embedded in {model_path}; postprocessing with rfdetr's default {DEFAULT_NUM_SELECT}")
        return cls(
            class_names=list(class_names) if class_names else None,
            num_select=DEFAULT_NUM_SELECT if num_select is None else int(num_select),
        )

    def as_payload(self) -> Dict[str, Any]:
        """The dict to embed in the exported file; fields with nothing to say are left out."""
        return {key: value for key, value in asdict(self).items() if value is not None}
