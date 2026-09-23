# Changing 3D models and environments

The repository deliberately has one prompted target and one camera. `scene.py` builds MuJoCo XML, `render.py` loads asset bytes and converts the geom named `target` into a visible-pixel mask and box, and `run.py` saves one result. There is no asset registry or automatic scene sweep. Changing the built-in couch pose or camera needs only an edit to `SCENE`; adding geometry requires a small edit to `build_xml` and, for a new mesh, the asset mapping in `render.py`.

## Move the current couch and camera

Edit the `SCENE = Scene(...)` block in `scene.py`. MuJoCo world coordinates are in metres with Z up. `couch_position` moves the mesh body; `couch_yaw_deg` turns it about the vertical axis. `camera_position` is the camera location, and `camera_look_at` is the point it faces. `width`, `height`, `fx`, `fy`, `cx`, and `cy` set the rendered image and pinhole intrinsics in pixels. These are fully editable values, not Go2 camera calibration. Run with a new output directory after every change:

```bash
MUJOCO_GL=osmesa python run.py --output output/couch-left-normal
```

`result.json` records those exact `SCENE` fields. Open `rgb.png` first to check that the couch is visible and placed as intended, then `target_mask.png` to check its visible pixels and `annotated.png` to inspect predicted boxes. A present couch outside the camera view raises an error instead of being silently scored as a detector miss.

## Replace the target couch with another OBJ

This is the smallest route for a **single different target**, such as a chair. It replaces the current target mesh; it does not add a second scored object.

1. Put the OBJ and its required companion files under `assets/`, along with its license and attribution. Check the OBJ's `mtllib` reference: the current couch OBJ names `Couch_Large1.mtl`, which is why both files are bundled. If a new OBJ references textures or material files, include them in the bytes supplied to MuJoCo or choose a mesh that does not need them.
2. In `scene.py`, change `COUCH_OBJ` and `COUCH_MTL` to the new paths and change the `file="Couch_Large1.obj"` value in `mesh_xml` to the exact new OBJ filename. Keep the mesh name `target_mesh` and geom name `target`: `render.py` looks up that geom name to derive ground truth. If the new OBJ has no MTL, remove the `COUCH_MTL` import and entry from the `assets` dictionary in `render.py` instead of pointing to a nonexistent file.
3. Set `SCENE.target_label` to the actual object, such as `"chair"`, and `SCENE.prompt` to the text sent to DINO. Ground truth uses `target_label`, so changing the query cannot change what is actually in the scene. Set `COUCH_SCALE`, `COUCH_FLOOR_OFFSET`, `COUCH_COLOR`, and the target geom's `euler="90 0 0"` rotation for the new mesh. Those values are specific to the bundled couch; do not assume a different OBJ has the same size, origin, or up axis. Rename the couch-named variables after the replacement if you intend to keep the new target.
4. Run into a fresh output directory. Inspect RGB, mask, and box before interpreting detector counts. Confirm that the model rests on the floor and that the mask covers only the intended object. A mesh may be visible but mis-scaled, rotated, or sunk below the floor while still producing a technically valid box.

The ground-truth mask comes from the MuJoCo geom named `target`, not from the prompt text or an OBJ filename. That naming rule is the critical connection to preserve.

## Add a second mesh as an environmental object

To keep the couch as the target and add a chair or table as a distractor, add its OBJ (and required companion files) under `assets/`, add their bytes to the `assets` dictionary in `render.py`, then add a second `<mesh>` under `<asset>` and a separate `<body>` under `<worldbody>` in `scene.py:build_xml`. Give the distractor's geom a unique name such as `distractor_chair`, **not** `target`. For example, after defining a mesh named `chair_mesh`:

```xml
<body name="chair_body" pos="1.2 0.5 0.0" euler="0 0 -90">
  <geom name="distractor_chair" type="mesh" mesh="chair_mesh"
        rgba="0.5 0.4 0.3 1" contype="0" conaffinity="0"/>
</body>
```

Tune that mesh's scale, rotation, and floor offset independently. Put the distractor body **outside** the conditional `target_xml` so it remains when `target_present=False`. Keep the same distractor, camera, and lights in the present and absent runs if you want target presence to be the only changed factor. A couch prediction on a chair in an absent scene is then a false positive. This evaluator scores one prompted target; if you add another real couch, its detection will be treated as an extra prediction, so multi-couch evaluation would require multi-instance ground truth and matching first.

## Change the environment around the target

The current world contains only a floor, one light, a camera, and optionally the couch. A different environment is a deliberate edit to the `<worldbody>` built in `scene.py`. You can change the floor's `rgba` to test background contrast, its size or position to change framing, or add simple wall geoms. For example, a wall behind the target can be added alongside the floor:

```xml
<geom name="back_wall" type="box" size="0.05 4 1.5"
      pos="3.5 0 1.5" rgba="0.85 0.83 0.78 1"
      contype="0" conaffinity="0"/>
```

The `lighting` value currently selects `normal`, `dim`, or `bright` values for the headlight and ceiling light. To test a different light position or direction, edit the `<light name="ceiling_light" ...>` element in `build_xml`; to add a new named lighting condition, add it to both the allowed names and the `ambient, diffuse` mapping. For a small paired experiment, first save the couch-present wall scene at `output/wall-dim-present`. Then change only `target_present` to `False` and save `output/wall-dim-absent`. The wall, floor, camera, and lights stay in both images, so any couch prediction in the absent image is an inspectable false positive. Use a different output directory for each environment and keep other settings fixed when attributing a changed detection to that environment. `target_present=False` removes only the target body, not the floor, light, camera, or any environmental objects you add.

Adding meshes, walls, or lighting changes only the synthetic experiment. The current scene has no measured Go2 mount pose, camera calibration, depth capture, or real-house validation.
