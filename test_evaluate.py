from detector import Detection
from evaluate import evaluate

PRESENT = {"label": "couch", "present": True, "bbox": [10.0, 10.0, 50.0, 50.0]}
ABSENT = {"label": "couch", "present": False, "bbox": None}


def detection(score: float, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection("couch", score, bbox, "test-model")


def test_present_match_and_duplicate() -> None:
    result = evaluate(
        [detection(0.9, (10, 10, 50, 50)), detection(0.8, (10, 10, 50, 50))],
        PRESENT,
    )
    assert (
        result["true_positives"],
        result["false_positives"],
        result["false_negatives"],
    ) == (
        1,
        1,
        0,
    )


def test_present_miss_and_localization_failure() -> None:
    empty_result = evaluate([], PRESENT)
    assert empty_result["false_negatives"] == 1
    assert empty_result["matched_iou"] is None
    result = evaluate([detection(0.9, (60, 60, 90, 90))], PRESENT)
    assert (
        result["true_positives"],
        result["false_positives"],
        result["false_negatives"],
    ) == (
        0,
        1,
        1,
    )


def test_absent_scene_counts_every_qualifying_box_as_false_positive() -> None:
    result = evaluate(
        [detection(0.9, (10, 10, 50, 50)), detection(0.5, (60, 60, 90, 90))],
        ABSENT,
    )
    assert (
        result["true_positives"],
        result["false_positives"],
        result["false_negatives"],
    ) == (
        0,
        2,
        0,
    )
    assert evaluate([], ABSENT)["false_positives"] == 0
    assert result["matched_iou"] is None


def test_below_threshold_box_is_not_counted() -> None:
    result = evaluate([detection(0.34, (10, 10, 50, 50))], PRESENT)
    assert (result["false_positives"], result["false_negatives"]) == (0, 1)
