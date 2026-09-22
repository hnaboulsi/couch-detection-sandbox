from dataclasses import replace

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
