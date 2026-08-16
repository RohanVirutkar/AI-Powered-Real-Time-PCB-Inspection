from pathlib import Path
import shutil
import random

# ============================================================
# DeepPCB -> YOLO11 Dataset Converter
# ============================================================

PROJECT_ROOT = Path(r"D:\PCB\AI-Powered-PCB-Inspection")

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "DeepPCB"
    / "PCBData"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

# DeepPCB class ID -> YOLO class ID
CLASS_MAP = {
    1: 0,  # Open
    2: 1,  # Short
    3: 2,  # Mousebite
    4: 3,  # Spur
    5: 4,  # Pin-hole
    6: 5,  # Spurious copper
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# READ DEEPCPCB SPLIT FILE
# ============================================================

def read_split_file(filename):
    split_file = DATASET_ROOT / filename

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    entries = []

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                print(
                    f"Skipping invalid line: {line}"
                )
                continue

            image_rel = parts[0]
            label_rel = parts[1]

            entries.append(
                (image_rel, label_rel)
            )

    return entries


# ============================================================
# FIND ACTUAL DEEPCPCB IMAGE
# ============================================================

def find_source_image(image_rel):
    """
    DeepPCB split file example:

    group20085/20085/20085000.jpg

    Actual downloaded image:

    group20085/20085/20085000_test.jpg
    """

    image_path = Path(image_rel)

    # Try original path first
    source_image = DATASET_ROOT / image_path

    if source_image.is_file():
        return source_image

    # Try the actual DeepPCB "_test.jpg" image
    if image_path.suffix.lower() == ".jpg":

        test_name = (
            image_path.stem + "_test.jpg"
        )

        source_image = (
            DATASET_ROOT
            / image_path.parent
            / test_name
        )

        if source_image.is_file():
            return source_image

    return None


# ============================================================
# CONVERT ANNOTATION TO YOLO FORMAT
# ============================================================

def convert_annotation(label_path):
    """
    DeepPCB format:

    x1 y1 x2 y2 class

    YOLO format:

    class x_center y_center width height
    """

    yolo_lines = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"Invalid annotation line: "
                    f"{line}"
                )
                continue

            try:
                x1, y1, x2, y2, class_id = map(
                    int,
                    parts
                )
            except ValueError:
                print(
                    f"Invalid numbers in: "
                    f"{label_path}"
                )
                continue

            if class_id not in CLASS_MAP:
                print(
                    f"Unknown class {class_id} "
                    f"in {label_path}"
                )
                continue

            # DeepPCB -> YOLO class
            yolo_class = CLASS_MAP[class_id]

            # Bounding box center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Bounding box size
            width = x2 - x1
            height = y2 - y1

            # Normalize to 0-1
            x_center /= IMAGE_WIDTH
            y_center /= IMAGE_HEIGHT
            width /= IMAGE_WIDTH
            height /= IMAGE_HEIGHT

            yolo_lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    return yolo_lines


# ============================================================
# PROCESS ONE IMAGE + LABEL
# ============================================================

def process_entry(
    image_rel,
    label_rel,
    split
):
    # Find actual image
    source_image = find_source_image(
        image_rel
    )

    # Find annotation
    source_label = (
        DATASET_ROOT / label_rel
    )

    # Check image
    if source_image is None:
        print(
            f"Missing image for: "
            f"{image_rel}"
        )
        return False

    # Check annotation
    if not source_label.is_file():
        print(
            f"Missing label: "
            f"{source_label}"
        )
        return False

    # Output names
    source_image_path = Path(
        source_image
    )

    image_name = (
        source_image_path.name
    )

    label_name = (
        source_image_path.stem
        + ".txt"
    )

    destination_image = (
        OUTPUT_ROOT
        / "images"
        / split
        / image_name
    )

    destination_label = (
        OUTPUT_ROOT
        / "labels"
        / split
        / label_name
    )

    # Copy image
    shutil.copy2(
        source_image,
        destination_image
    )

    # Convert annotation
    yolo_lines = convert_annotation(
        source_label
    )

    # Save YOLO label
    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(yolo_lines)
        )

    return True


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    entries,
    split
):
    success = 0
    failed = 0

    for index, (
        image_rel,
        label_rel
    ) in enumerate(
        entries,
        start=1
    ):

        result = process_entry(
            image_rel,
            label_rel,
            split
        )

        if result:
            success += 1
        else:
            failed += 1

        if index % 100 == 0:
            print(
                f"{split}: "
                f"{index}/{len(entries)} processed"
            )

    return success, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DeepPCB -> YOLO11 Dataset Converter")
    print("=" * 65)

    print(
        f"\nSource dataset:"
        f"\n{DATASET_ROOT}"
    )

    print(
        f"\nOutput dataset:"
        f"\n{OUTPUT_ROOT}"
    )

    # Create output directories
    create_directories()

    # Read official splits
    trainval_entries = read_split_file(
        "trainval.txt"
    )

    test_entries = read_split_file(
        "test.txt"
    )

    print(
        f"\nTrain/Validation entries: "
        f"{len(trainval_entries)}"
    )

    print(
        f"Test entries: "
        f"{len(test_entries)}"
    )

    # --------------------------------------------------------
    # Create reproducible train/validation split
    # --------------------------------------------------------

    random.seed(42)

    random.shuffle(
        trainval_entries
    )

    train_count = int(
        len(trainval_entries) * 0.90
    )

    train_entries = (
        trainval_entries[:train_count]
    )

    val_entries = (
        trainval_entries[train_count:]
    )

    print(
        f"\nTrain entries: "
        f"{len(train_entries)}"
    )

    print(
        f"Validation entries: "
        f"{len(val_entries)}"
    )

    print(
        f"Test entries: "
        f"{len(test_entries)}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nProcessing TRAIN...")

    train_success, train_failed = process_split(
        train_entries,
        "train"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nProcessing VALIDATION...")

    val_success, val_failed = process_split(
        val_entries,
        "val"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nProcessing TEST...")

    test_success, test_failed = process_split(
        test_entries,
        "test"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CONVERSION COMPLETE")
    print("=" * 65)

    print(
        f"Train: "
        f"{train_success} success, "
        f"{train_failed} failed"
    )

    print(
        f"Val:   "
        f"{val_success} success, "
        f"{val_failed} failed"
    )

    print(
        f"Test:  "
        f"{test_success} success, "
        f"{test_failed} failed"
    )

    print(
        f"\nDataset location:"
        f"\n{OUTPUT_ROOT}"
    )

    # Final success check
    if (
        train_success == 900
        and train_failed == 0
        and val_success == 100
        and val_failed == 0
        and test_success == 500
        and test_failed == 0
    ):
        print(
            "\nSUCCESS: "
            "All 1500 images converted correctly."
        )
    else:
        print(
            "\nWARNING: "
            "Some files were not processed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()