from pathlib import Path
import shutil
import random

# ============================================================
# DeepPCB -> YOLO11 Dataset Converter
# ============================================================

PROJECT_ROOT = Path(r"D:\PCB\AI-Powered-PCB-Inspection")

DEEPCPCB_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "DeepPCB"
    / "PCBData"
)

OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "processed"

IMAGE_SIZE = 640
TRAIN_RATIO = 0.9

# DeepPCB class ID -> YOLO class ID
CLASS_MAP = {
    1: 0,  # Open
    2: 1,  # Short
    3: 2,  # Mousebite
    4: 3,  # Spur
    5: 4,  # Pin-hole
    6: 5,  # Spurious copper
}

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "pin-hole",
    "spurious_copper",
]


def create_directories():
    """Create YOLO dataset directories."""

    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def read_split_file(filename):
    """Read DeepPCB train/test split file."""

    split_file = DEEPCPCB_ROOT / filename

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with open(split_file, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def find_test_image(image_name):
    """Find a DeepPCB _test.jpg file."""

    matches = list(DEEPCPCB_ROOT.rglob(image_name))

    if not matches:
        return None

    return matches[0]


def find_annotation(image_name):
    """
    Find annotation.

    Example:
    00041004_test.jpg
    ->
    00041004.txt
    """

    stem = Path(image_name).stem.replace("_test", "")
    annotation_name = stem + ".txt"

    matches = list(DEEPCPCB_ROOT.rglob(annotation_name))

    # Prefer annotation inside *_not directory
    for match in matches:
        if "_not" in match.parent.name:
            return match

    return matches[0] if matches else None


def convert_annotation(annotation_file):
    """
    Convert:

    x1 y1 x2 y2 class

    into:

    class x_center y_center width height
    """

    yolo_lines = []

    with open(annotation_file, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"WARNING: Invalid annotation: "
                    f"{annotation_file}"
                )
                continue

            x1, y1, x2, y2, class_id = map(
                int,
                parts
            )

            if class_id not in CLASS_MAP:
                print(
                    f"WARNING: Unknown class {class_id} "
                    f"in {annotation_file}"
                )
                continue

            # DeepPCB -> YOLO class
            yolo_class = CLASS_MAP[class_id]

            # Convert bounding box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            width = x2 - x1
            height = y2 - y1

            # Normalize for 640x640 images
            x_center /= IMAGE_SIZE
            y_center /= IMAGE_SIZE
            width /= IMAGE_SIZE
            height /= IMAGE_SIZE

            yolo_lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    return yolo_lines


def process_image(image_name, split):

    image_path = find_test_image(image_name)

    if image_path is None:
        print(f"WARNING: Image not found: {image_name}")
        return False

    annotation_path = find_annotation(image_name)

    if annotation_path is None:
        print(
            f"WARNING: Annotation not found: "
            f"{image_name}"
        )
        return False

    output_image = (
        OUTPUT_ROOT
        / "images"
        / split
        / image_name
    )

    label_name = Path(image_name).stem + ".txt"

    output_label = (
        OUTPUT_ROOT
        / "labels"
        / split
        / label_name
    )

    # Copy image
    shutil.copy2(
        image_path,
        output_image
    )

    # Convert annotation
    yolo_lines = convert_annotation(
        annotation_path
    )

    # Save YOLO annotation
    with open(
        output_label,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(yolo_lines)
        )

    return True


def main():

    print("=" * 60)
    print("DeepPCB -> YOLO11 Dataset Converter")
    print("=" * 60)

    print(f"\nDataset: {DEEPCPCB_ROOT}")
    print(f"Output:  {OUTPUT_ROOT}")

    create_directories()

    # --------------------------------------------------------
    # Read official DeepPCB splits
    # --------------------------------------------------------

    trainval = read_split_file("trainval.txt")
    test = read_split_file("test.txt")

    print(
        f"\nTrain/Validation entries: {len(trainval)}"
    )

    print(
        f"Test entries: {len(test)}"
    )

    # --------------------------------------------------------
    # Shuffle train/validation
    # --------------------------------------------------------

    random.seed(42)
    random.shuffle(trainval)

    train_count = int(
        len(trainval) * TRAIN_RATIO
    )

    train_images = trainval[:train_count]
    val_images = trainval[train_count:]

    # --------------------------------------------------------
    # Process TRAIN
    # --------------------------------------------------------

    print("\nProcessing TRAIN...")

    train_success = 0

    for entry in train_images:

        image_name = Path(entry).name

        if process_image(
            image_name,
            "train"
        ):
            train_success += 1

    # --------------------------------------------------------
    # Process VALIDATION
    # --------------------------------------------------------

    print("\nProcessing VALIDATION...")

    val_success = 0

    for entry in val_images:

        image_name = Path(entry).name

        if process_image(
            image_name,
            "val"
        ):
            val_success += 1

    # --------------------------------------------------------
    # Process TEST
    # --------------------------------------------------------

    print("\nProcessing TEST...")

    test_success = 0

    for entry in test:

        image_name = Path(entry).name

        if process_image(
            image_name,
            "test"
        ):
            test_success += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(
        f"Train images : {train_success}"
    )

    print(
        f"Val images   : {val_success}"
    )

    print(
        f"Test images  : {test_success}"
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()