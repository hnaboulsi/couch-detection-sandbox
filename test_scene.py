import xml.etree.ElementTree as ET
from dataclasses import replace

from scene import SCENE, build_xml


def test_present_and_absent_scene_geometry() -> None:
    present = ET.fromstring(build_xml(SCENE))
    absent = ET.fromstring(build_xml(replace(SCENE, target_present=False)))
    assert [geom.get("name") for geom in present.findall(".//geom")] == [
        "floor",
        "target",
    ]
    assert [geom.get("name") for geom in absent.findall(".//geom")] == ["floor"]


def test_camera_configuration_reaches_mujoco_xml() -> None:
    scene = replace(
        SCENE,
        width=800,
        height=600,
        fx=610.0,
        fy=620.0,
        cx=401.0,
        cy=298.0,
        camera_position=(-3.0, 1.0, 0.7),
        camera_look_at=(0.0, 0.0, 0.5),
    )
    camera = ET.fromstring(build_xml(scene)).find(".//camera")
    assert camera is not None
    assert camera.get("resolution") == "800 600"
    assert camera.get("focalpixel") == "610 620"
    assert camera.get("principalpixel") == "-1.5 -1.5"
    assert camera.get("pos") == "-3 1 0.7"
