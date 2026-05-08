import logging
import os

import cv2
import pytest
import torch

from lmi_utils.pipeline_base.pipeline_base import PipelineBase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "tests/outputs/pipeline_base"

logger = logging.getLogger(__name__)


class PipelineOD(PipelineBase):
    def load(self, model_roles: dict, configs: dict):
        self.load_models(model_roles, configs, device=DEVICE)

    def warm_up(self, configs: dict):
        pass

    def predict(self, configs: dict, inputs: dict):
        """
        Mock predict:
        1. Preprocess inputs
        2. Inference
        3. Reconstruct
        """
        images = inputs.get("images", [])
        model_role = "mock-model"
        assert model_role in self.models and len(self.models) == 1

        # 1. Preprocess
        preprocessed_images, ops_list = self.preprocess(model_role, images)

        # 2. Mock inference
        results, _ = self.models[model_role].predict(preprocessed_images, 0.5)

        results2 = self.revert_preprocess(results, ops_list)  # test revert preprocess can run without error

        # 3. annotate
        annots = []
        for i, im0 in enumerate(images):
            r = {k: v[i] for k, v in results2.items()}
            annot = self.models[model_role].annotate_image(r, im0)
            annots.append(annot)

        return {
            "outputs": {
                "annotated": annots,
            },
            "ops_list": ops_list,
        }


def _build_od_model_roles(version, model_path, preprocessing_steps):
    if version == "2":
        return {
            "mock-model": {
                "format": "pt",
                "configs": {},
                "details": {
                    "training_package": "Ultralytics",
                    "training_algorithm": "Yolo",
                    "global_preprocessing": preprocessing_steps,
                },
                "artifacts": {"pt": {"image_size": [640, 640], "model_path": model_path}},
                "model_role": "mock-model",
                "model_type": "InstanceSegmentation",
                "model_version": "1",
            }
        }
    if version == "3":
        return {
            "mock-model": {
                "format": "pt",
                "configs": {"to-fail": {}, "confidence": {}},
                "details": {
                    "image_size": [640, 640],
                    "preprocessing": preprocessing_steps,
                    "training_package": "Ultralytics",
                    "training_algorithm": "Yolo",
                },
                "artifacts": {"pt": {"attributes": {}, "model_path": model_path}},
                "model_role": "mock-model",
                "model_name": "mock-model",
                "model_type": "InstanceSegmentation",
                "model_version": "1",
            }
        }
    raise ValueError(f"Unsupported schema version: {version}")


@pytest.mark.parametrize(
    "version, preprocessing_steps, expected_types",
    [
        ("2", [{"type": "resize", "configuration": {"height": 640, "width": 640}}], ["resize"]),
        ("3", [{"type": "resize", "configuration": {"height": 640, "width": 640, "preserve_aspect": True}}], ["resize"]),
    ],
)
def test_pipeline_OD(version, preprocessing_steps, expected_types):
    # Asset paths
    model_path = os.path.abspath("tests/assets/models/od/ultralytics/yolo11n-seg.pt")
    image_dir = os.path.abspath("tests/assets/images/coco")

    model_roles = _build_od_model_roles(version, model_path, preprocessing_steps)

    pipeline = PipelineOD(version=version)
    pipeline.load(model_roles, {})

    # Load images
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"

    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files]

    # Run predict
    results = pipeline.predict({}, {"images": images})
    annotated_imgs = results["outputs"]["annotated"]
    ops_list = results["ops_list"]

    # Verify operators
    actual_types = [op.get("type") for op in ops_list]
    assert actual_types == expected_types, f"Operator mismatch: {actual_types} != {expected_types}"

    # write outputs for manual inspection
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, annot in enumerate(annotated_imgs):
        cv2.imwrite(os.path.join(OUT_DIR, f"annot_od_{idx}.png"), cv2.cvtColor(annot, cv2.COLOR_RGB2BGR))


class PipelineAD(PipelineBase):
    def load(self, model_roles: dict, configs: dict):
        self.load_models(model_roles, configs, device=DEVICE)

    def warm_up(self, configs: dict):
        pass

    def predict(self, configs: dict, inputs: dict):
        """
        Mock predict:
        1. Preprocess inputs
        2. Inference
        3. Reconstruct
        """
        images = inputs.get("images", [])
        model_role = "mock-model"
        assert model_role in self.models and len(self.models) == 1

        # 1. Preprocess
        preprocessed_images, ops_list = self.preprocess(model_role, images)

        # 2. Mock inference
        scores = self.models[model_role].predict(preprocessed_images)

        # 3. Reconstruct
        heatmap = self.revert_preprocess(scores, ops_list)

        return {
            "outputs": {
                "annotated": heatmap,
            },
            "ops_list": ops_list,
        }


def _build_ad_model_roles(version, model_path, preprocessing_steps):
    if version == "2":
        return {
            "mock-model": {
                "format": "pt",
                "configs": {},
                "details": {
                    "training_package": "Anomalib1",
                    "training_algorithm": "Patchcore",
                    "global_preprocessing": preprocessing_steps,
                },
                "artifacts": {"pt": {"image_size": [224, 224], "model_path": model_path}},
                "model_role": "mock-model",
                "model_type": "AnomalyDetection",
                "model_version": "1",
            }
        }
    if version == "3":
        return {
            "mock-model": {
                "format": "pt",
                "configs": {"min_threshold": 0.0, "max_threshold": 1.0},
                "details": {
                    "image_size": [224, 224],
                    "preprocessing": preprocessing_steps,
                    "training_package": "Anomalib1",
                    "training_algorithm": "Patchcore",
                },
                "artifacts": {"pt": {"attributes": {}, "model_path": model_path}},
                "model_role": "mock-model",
                "model_name": "mock-model",
                "model_type": "AnomalyDetection",
                "model_version": "1",
            }
        }
    raise ValueError(f"Unsupported schema version: {version}")


@pytest.mark.parametrize(
    "version, preprocessing_steps, expected_types",
    [
        ("2", [{"type": "resize", "configuration": {"height": 224, "width": 224}}], ["resize"]),
        (
            "2",
            [
                {"type": "resize", "configuration": {"height": 224, "width": 448}},
                {"type": "tile", "configuration": {"height": 224, "width": 224, "y_stride": 112, "x_stride": 112}},
            ],
            ["resize", "tile"],
        ),
        ("3", [{"type": "resize", "configuration": {"height": 224, "width": 224}}], ["resize"]),
        (
            "3",
            [
                {"type": "resize", "configuration": {"height": 224, "width": 448}},
                {"type": "tile", "configuration": {"height": 224, "width": 224, "y_stride": 112, "x_stride": 112}},
            ],
            ["resize", "tile"],
        ),
    ],
)
def test_pipeline_AD(version, preprocessing_steps, expected_types):
    # Asset paths
    model_path = os.path.abspath("tests/assets/models/ad/model_v1_trace.pt")
    image_dir = os.path.abspath("tests/assets/images/nvtec-ad")

    model_roles = _build_ad_model_roles(version, model_path, preprocessing_steps)

    pipeline = PipelineAD(version=version)
    pipeline.load(model_roles, {})

    # Load images
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"

    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files]

    # Run predict
    results = pipeline.predict({}, {"images": images})
    heatmap = results["outputs"]["annotated"]
    ops_list = results["ops_list"]

    # Verify operators
    actual_types = [op.get("type") for op in ops_list]
    assert actual_types == expected_types, f"Operator mismatch: {actual_types} != {expected_types}"

    # Verify shapes
    for original, h in zip(images, heatmap):
        orig_shape = original.shape[:2]
        h_shape = h.shape[:2]
        assert orig_shape == h_shape, f"Shape mismatch: {orig_shape} vs {h_shape}"


def test_version_1_error():
    pipeline = PipelineOD(version="1")
    model_roles = {"mock-model": {"model_role": "mock-model"}}
    with pytest.raises(ValueError, match="Gadget version 1 is no longer supported"):
        pipeline.load(model_roles, {})
