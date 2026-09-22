"""Render one couch scene, run detection, and save an inspectable result."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from detector import Detector, GroundingDINO
from evaluate import CONFIDENCE_THRESHOLD, evaluate
from render import render_scene
from scene import SCENE, Scene


def run_experiment(scene: Scene, detector: Detector, output: Path) -> dict[str, object]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")

    rendered = render_scene(scene)
    prompts = [scene.prompt]
    detections = detector.detect(rendered.rgb, prompts)
    decision = evaluate(detections, rendered.ground_truth)

    output.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rendered.rgb, mode="RGB")
    image.save(output / "rgb.png")
    Image.fromarray(rendered.target_mask, mode="L").save(output / "target_mask.png")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    if rendered.ground_truth["bbox"] is not None:
        draw.rectangle(rendered.ground_truth["bbox"], outline="cyan", width=3)
    for detection in detections:
        if detection.score >= CONFIDENCE_THRESHOLD:
            draw.rectangle(detection.bbox, outline="lime", width=3)
            draw.text(
                (detection.bbox[0], max(0, detection.bbox[1] - 14)),
                f"{detection.label} {detection.score:.2f}",
                fill="lime",
                stroke_width=2,
                stroke_fill="black",
            )
    annotated.save(output / "annotated.png")

    result = {
        "scene": scene.to_dict(),
        "prompts": prompts,
        "ground_truth": rendered.ground_truth,
        "detections": [asdict(detection) for detection in detections],
        "detector": detector.metadata(),
        "evaluation": decision,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error(f"Output directory is not empty: {args.output}")
    detector = GroundingDINO(device=args.device)
    result = run_experiment(SCENE, detector, args.output)
    decision = result["evaluation"]
    print(
        f"Saved {args.output}: TP={decision['true_positives']} "
        f"FP={decision['false_positives']} FN={decision['false_negatives']}"
    )


if __name__ == "__main__":
    main()
