from typing import Any

from object_ctrl.dataset import filter_coco_annotation_by_labels


def coco_fixture() -> dict[str, Any]:
    """
    Return a tiny COCO fixture with mixed-label and background images.
    """
    return {
        "info": {"description": "fixture"},
        "licenses": [{"id": 1, "name": "fixture"}],
        "categories": [
            {"id": 1, "name": "ball", "supercategory": "object"},
            {"id": 2, "name": "hoop", "supercategory": "object"},
            {"id": 3, "name": "person", "supercategory": "object"},
        ],
        "images": [
            {"id": 10, "file_name": "ball.jpg"},
            {"id": 11, "file_name": "ball_person.jpg"},
            {"id": 12, "file_name": "hoop.jpg"},
            {"id": 13, "file_name": "background.jpg"},
        ],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": 1},
            {"id": 101, "image_id": 11, "category_id": 1},
            {"id": 102, "image_id": 11, "category_id": 3},
            {"id": 103, "image_id": 12, "category_id": 2},
        ],
        "extra": "preserved",
    }


def category_names(coco: dict[str, Any]) -> list[str]:
    """
    Return category names from a COCO annotation dictionary.
    """
    return [category["name"] for category in coco["categories"]]


def image_ids(coco: dict[str, Any]) -> list[int]:
    """
    Return image IDs from a COCO annotation dictionary.
    """
    return [image["id"] for image in coco["images"]]


def annotation_ids(coco: dict[str, Any]) -> list[int]:
    """
    Return annotation IDs from a COCO annotation dictionary.
    """
    return [annotation["id"] for annotation in coco["annotations"]]


def test_filter_coco_annotation_applies_include_and_exclude_labels() -> None:
    """
    Keep included labels while removing images that contain excluded labels.
    """
    trimmed = filter_coco_annotation_by_labels(
        coco_fixture(),
        labels_include={"ball", "hoop"},
        labels_exclude={"person"},
    )

    assert trimmed["info"] == {"description": "fixture"}
    assert trimmed["licenses"] == [{"id": 1, "name": "fixture"}]
    assert trimmed["extra"] == "preserved"
    assert category_names(trimmed) == ["ball", "hoop"]
    assert image_ids(trimmed) == [10, 12]
    assert annotation_ids(trimmed) == [100, 103]


def test_filter_coco_annotation_keeps_background_images_for_exclude_only() -> None:
    """
    Keep background images when only excluded labels are removed.
    """
    trimmed = filter_coco_annotation_by_labels(
        coco_fixture(),
        labels_exclude={"person"},
    )

    assert category_names(trimmed) == ["ball", "hoop"]
    assert image_ids(trimmed) == [10, 12, 13]
    assert annotation_ids(trimmed) == [100, 103]
