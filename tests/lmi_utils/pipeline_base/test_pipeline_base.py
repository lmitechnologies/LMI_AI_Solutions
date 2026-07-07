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
        ("3", [{"type": "resize", "id": "r1", "configuration": {"height": 640, "width": 640, "preserve_aspect": True}}], ["resize"]),
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
    actual_types = [type(op).__name__.removesuffix("Meta").lower() for op in ops_list]
    assert actual_types == expected_types, f"Operator mismatch: {actual_types} != {expected_types}"

    # write outputs for manual inspection
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, annot in enumerate(annotated_imgs):
        cv2.imwrite(os.path.join(OUT_DIR, f"annot_od_{idx}.png"), cv2.cvtColor(annot, cv2.COLOR_RGB2BGR))


def test_pipeline_OD_injects_resize_on_size_mismatch(caplog):
    """When the preprocessed image does not match the OD model's input size, a corrective resize is
    injected and recorded in history so revert_preprocess still round-trips to the original space."""
    model_path = os.path.abspath("tests/assets/models/od/ultralytics/yolo11n-seg.pt")
    image_dir = os.path.abspath("tests/assets/images/coco")

    # Configure a resize to the wrong size (320) for a 640 model: injection must correct it to 640.
    preprocessing_steps = [{"type": "resize", "configuration": {"height": 320, "width": 320, "preserve_aspect": True}}]
    model_roles = _build_od_model_roles("3", model_path, preprocessing_steps)

    pipeline = PipelineOD(version="3")
    pipeline.load(model_roles, {})

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"
    image = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2RGB)

    with caplog.at_level(logging.WARNING):
        processed, ops_list = pipeline.preprocess("mock-model", image)

    assert len(processed) == 1, f"Expected a single processed image, got {len(processed)}"
    # Configured resize + injected corrective resize, both recorded for reversion.
    actual_types = [type(op).__name__.removesuffix("Meta").lower() for op in ops_list]
    assert actual_types == ["resize", "resize"], f"Unexpected history: {actual_types}"
    assert any("injecting a resize" in r.message for r in caplog.records), "Expected an injection warning"


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
        ("3", [{"type": "resize", "id": "r1", "configuration": {"height": 224, "width": 224}}], ["resize"]),
        (
            "3",
            [
                {"type": "resize", "id": "r1", "configuration": {"height": 224, "width": 448}},
                {"type": "tile", "id": "t1", "configuration": {"height": 224, "width": 224, "y_stride": 112, "x_stride": 112}},
            ],
            ["resize", "tile"],
        ),
    ],
)
def test_pipeline_AD(version, preprocessing_steps, expected_types):
    # Asset paths
    model_path = os.path.abspath("tests/assets/models/ad/model_v1/model.ts")
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
    actual_types = [type(op).__name__.removesuffix("Meta").lower() for op in ops_list]
    assert actual_types == expected_types, f"Operator mismatch: {actual_types} != {expected_types}"

    # Verify shapes
    for original, h in zip(images, heatmap):
        orig_shape = original.shape[:2]
        h_shape = h.shape[:2]
        assert orig_shape == h_shape, f"Shape mismatch: {orig_shape} vs {h_shape}"


def test_pipeline_AD_records_inverse_resize_on_size_mismatch(caplog):
    """When AD preprocessing does not reach the model's input size, the forward images are left untouched
    (the model resizes internally) and an inverse-resize is recorded so the score maps still revert to
    the original input shapes. Uses a two-image batch to cover the batched (1:1, non-tiled) path."""
    model_path = os.path.abspath("tests/assets/models/ad/model_v1/model.ts")
    image_dir = os.path.abspath("tests/assets/images/nvtec-ad")

    # Configure a resize to the wrong size (320) for a 224 model: no resize reaches image_size.
    preprocessing_steps = [{"type": "resize", "id": "r1", "configuration": {"height": 320, "width": 320}}]
    model_roles = _build_ad_model_roles("3", model_path, preprocessing_steps)

    pipeline = PipelineAD(version="3")
    pipeline.load(model_roles, {})

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(image_files) > 0, "No images found in assets"
    image = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2RGB)
    images = [image, image[:-8, :-4]]  # different original sizes

    with caplog.at_level(logging.WARNING):
        results = pipeline.predict({}, {"images": images})

    ops_list = results["ops_list"]
    # Configured resize + recorded inverse-resize, the latter not physically applied to the forward images.
    actual_types = [type(op).__name__.removesuffix("Meta").lower() for op in ops_list]
    assert actual_types == ["resize", "resize"], f"Unexpected history: {actual_types}"
    assert any("recording an inverse-resize" in r.message for r in caplog.records), "Expected an inverse-resize warning"

    # Each reverted heatmap must still match its original input shape.
    for im, heatmap in zip(images, results["outputs"]["annotated"]):
        assert heatmap.shape[:2] == im.shape[:2], f"Shape mismatch: {heatmap.shape[:2]} vs {im.shape[:2]}"


def test_version_1_error():
    pipeline = PipelineOD(version="1")
    model_roles = {"mock-model": {"model_role": "mock-model"}}
    with pytest.raises(ValueError, match="Gadget version 1 is no longer supported"):
        pipeline.load(model_roles, {})


def test_update_results_behaviors():
    pipeline = PipelineOD(version="3")

    # append to an existing list key
    pipeline.update_results("tags", "ERROR", to_factory=True)
    assert pipeline.results["tags"] == ["ERROR"]
    assert "tags" in pipeline.results["factory_keys"]

    # overwrite replaces the list instead of appending
    pipeline.update_results("tags", ["A", "B"], overwrite=True)
    assert pipeline.results["tags"] == ["A", "B"]

    # sub_key on a missing key creates the sub dictionary
    pipeline.update_results("metrics", 0.93, sub_key="iou", to_automation=True)
    assert pipeline.results["metrics"] == {"iou": 0.93}
    assert "metrics" in pipeline.results["automation_keys"]

    # sub_key on an existing dict key updates in place
    pipeline.update_results("outputs", "img", sub_key="annotated")
    assert pipeline.results["outputs"]["annotated"] == "img"

    # plain set on a non-list key
    pipeline.update_results("should_archive", False)
    assert pipeline.results["should_archive"] is False


def test_update_results_sub_key_on_list_raises():
    pipeline = PipelineOD(version="3")
    with pytest.raises(TypeError, match="cannot set sub_key 'station_1'"):
        pipeline.update_results("tags", "NG", sub_key="station_1")
    with pytest.raises(TypeError, match="cannot set sub_key 'camera_0'"):
        pipeline.update_results("errors", "timeout", sub_key="camera_0", overwrite=True)
    assert pipeline.results["tags"] == []  # failed calls leave results untouched
    assert pipeline.results["errors"] == []


def test_update_results_rejects_unknown_kwargs():
    pipeline = PipelineOD(version="3")
    with pytest.raises(TypeError):
        pipeline.update_results("tags", "NG", to_gofactory=True)  # typo'd flag must not be silently ignored


def test_clean_up_calls_release():
    pipeline = PipelineOD(version="3")
    released = []

    class DummyModel:
        def release(self):
            released.append("dummy")

    pipeline.models["dummy"] = DummyModel()
    pipeline.clean_up()

    assert released == ["dummy"]
    assert len(pipeline.models) == 0


def test_clean_up_releases_interleaved_models_in_reversed_order():
    """clean_up pops last-in-first-out, so insertion order onnx2, trt2, onnx1, trt1
    must release as trt1, onnx1, trt2, onnx2 — interleaving engine types."""
    pipeline = PipelineOD(version="3")
    order = []

    class FakeEngine:
        def __init__(self, name):
            self.name = name

        def release(self):
            order.append(self.name)

    for name in ["onnx2", "trt2", "onnx1", "trt1"]:
        pipeline.models[name] = FakeEngine(name)

    pipeline.clean_up()

    assert order == ["trt1", "onnx1", "trt2", "onnx2"]
    assert len(pipeline.models) == 0


def test_clean_up_interleaved_trt_and_onnx_engines(tmp_path):
    """Real engines: clean_up must tear down TRT, ONNX, TRT, ONNX interleaved without
    CUDA errors, clearing each engine's resources and leaving the device usable."""
    pytest.importorskip("tensorrt")
    ort = pytest.importorskip("onnxruntime")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("onnxruntime-gpu / CUDAExecutionProvider not available")

    from lmi_common.onnx_engine import ONNXEngine
    from lmi_common.trt_engine import TRTEngine
    from tests.lmi_common.test_trt_engine import _build_engine, _export_onnx

    onnx_path = str(tmp_path / "tiny.onnx")
    engine_path = str(tmp_path / "tiny.engine")
    _export_onnx(onnx_path, dynamic=True)
    _build_engine(onnx_path, engine_path, min_b=1, opt_b=2, max_b=4, static=False)

    pipeline = PipelineOD(version="3")
    order = []

    def tracked(name, model):
        orig = model.release

        def wrapped():
            order.append(name)
            orig()

        model.release = wrapped
        return model

    # Insert in reverse so LIFO clean_up releases trt1, onnx1, trt2, onnx2.
    for name in ["onnx2", "trt2", "onnx1", "trt1"]:
        if name.startswith("trt"):
            model = TRTEngine(engine_path, device="cuda")
        else:
            model = ONNXEngine(onnx_path, device="cuda", dynamic_max_batch=4)
        pipeline.models[name] = tracked(name, model)

    # Exercise every engine so contexts and buffers are live before teardown.
    engines = dict(pipeline.models)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float32, device="cuda")
    for model in engines.values():
        assert model.infer(x)[0].shape == (2, 4)

    pipeline.clean_up()

    assert order == ["trt1", "onnx1", "trt2", "onnx2"]
    assert len(pipeline.models) == 0
    for name, model in engines.items():
        if name.startswith("trt"):
            assert model.context is None and model._engine is None, f"{name} not released"
        else:
            assert model._session is None and model._io_binding is None, f"{name} not released"

    # The device must remain usable after interleaved teardown.
    torch.cuda.synchronize()
    assert torch.ones(4, device="cuda").sum().item() == 4.0


def test_clean_up_continues_when_release_fails(caplog):
    pipeline = PipelineOD(version="3")
    released = []

    class BadModel:
        def release(self):
            raise RuntimeError("boom")

    class GoodModel:
        def release(self):
            released.append("good")

    pipeline.models["good"] = GoodModel()
    pipeline.models["bad"] = BadModel()

    with caplog.at_level(logging.ERROR):
        pipeline.clean_up()

    assert released == ["good"], "Remaining models must still be released after one release() fails"
    assert len(pipeline.models) == 0
    assert any("Failed to release 'bad'" in r.message for r in caplog.records)
