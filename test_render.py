from dataclasses import replace

import render
from render import render_scene
from scene import SCENE


def test_present_and_absent_segmentation() -> None:
    present = render_scene(SCENE)
    absent = render_scene(replace(SCENE, target_present=False))

    assert present.rgb.shape == absent.rgb.shape == (SCENE.height, SCENE.width, 3)
    assert present.ground_truth["bbox"] is not None
    assert present.ground_truth["visible_pixels"] > 0
    assert absent.ground_truth == {
        "label": "couch",
        "present": False,
        "bbox": None,
        "visible_pixels": 0,
    }
    assert not absent.target_mask.any()


def test_ground_truth_label_does_not_follow_detector_prompt() -> None:
    scene = replace(SCENE, target_label="sofa", prompt="chair")
    rendered = render_scene(scene)

    assert rendered.ground_truth["label"] == "sofa"
    assert rendered.ground_truth["bbox"] is not None


def test_absent_scene_does_not_need_couch_assets(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(render, "COUCH_OBJ", tmp_path / "missing.obj")
    monkeypatch.setattr(render, "COUCH_MTL", tmp_path / "missing.mtl")

    rendered = render_scene(replace(SCENE, target_present=False))

    assert rendered.ground_truth["bbox"] is None
    assert not rendered.target_mask.any()
