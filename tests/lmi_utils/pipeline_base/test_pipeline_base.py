import os

import cv2
import pytest

from lmi_utils.pipeline_base.pipeline_base import PipelineBase


class Pipeline(PipelineBase):
    def load(self, model_roles: dict, configs: dict):
        # We use use_tiling_in_global_preprocessing=True to trigger global tiling
        self.load_models(model_roles, configs, use_tiling_in_global_preprocessing=True)

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
        for im in preprocessed_images:
            self.models[model_role].predict(im, 0.5)

        # 3. Reconstruct
        reconstructed_images = self.reconstruct(preprocessed_images, ops_list)

        return {
            "outputs": {
                "annotated": reconstructed_images,
            },
            "ops_list": ops_list,
        }


@pytest.mark.parametrize(
    "preprocessing_steps, expected_types",
    [
        ([{"type": "resize", "configuration": {"height": 640, "width": 640}}], ["resize"]),
        (
            [
                {"type": "resize", "configuration": {"height": 640, "width": 640}},
                {"type": "tile", "configuration": {"height": 320, "width": 320, "yStride": 320, "xStride": 320}},
            ],
            ["resize", "tile"],
        ),
    ],
)
def test_pipeline_flow(preprocessing_steps, expected_types):
    # Asset paths
    model_path = os.path.abspath("tests/assets/models/od/yolo11n-seg.pt")
    image_dir = os.path.abspath("tests/assets/images/coco")

    # Mock model_roles (Schema V2)
    model_roles = {
        "mock-model": {
            "format": "pt",
            "configs": {},
            "details": {
                "deployed": "2026-01-28",
                "base_model": "yolo11n-seg.pt",
                "training_package": "Ultralytics",
                "training_algorithm": "Yolo",
                "global_preprocessing": preprocessing_steps,
            },
            "artifacts": {"pt": {"image_size": [640, 640], "model_path": model_path}},
            "model_name": "yolo",
            "model_role": "mock-model",
            "model_type": "ObjectDetection",
            "model_version": "1",
        }
    }

    pipeline = Pipeline(version="2")
    pipeline.load(model_roles, {})

    # Load images
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"

    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files]

    # Run predict
    results = pipeline.predict({}, {"images": images})
    reconstructed_images = results["outputs"]["annotated"]
    ops_list = results["ops_list"]

    # Verify operators
    actual_types = [op.get("type") for op in ops_list]
    assert actual_types == expected_types, f"Operator mismatch: {actual_types} != {expected_types}"

    # Verify shapes
    for original, reconstructed in zip(images, reconstructed_images):
        orig_shape = original.shape[:2]
        recon_shape = reconstructed.shape[:2]
        assert orig_shape == recon_shape, f"Shape mismatch: {orig_shape} vs {recon_shape}"

    print(f"Test with steps {expected_types} passed!")
