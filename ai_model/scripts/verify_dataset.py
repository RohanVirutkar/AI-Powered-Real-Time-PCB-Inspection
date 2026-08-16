from pathlib import Path
import cv2

PROJECT_ROOT = Path(r"D:\PCB\AI-Powered-PCB-Inspection")
DATASET = PROJECT_ROOT / "dataset" / "processed"
OUTPUT = PROJECT_ROOT / "screenshots" / "dataset_verification"

OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "pin-hole",
    5: "spurious_copper",
}

split = "train"

image_dir = DATASET / "images" / split
label_dir = DATASET / "labels" / split

images = sorted(image_dir.glob("*.jpg"))[:5]

for image_path in images:

    label_path = label_dir / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r", encoding="utf-8") as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, xc, yc, bw, bh = map(float, parts)

                class_id = int(class_id)

                # YOLO normalized -> pixel coordinates
                xc *= width
                yc *= height
                bw *= width
                bh *= height

                x1 = int(xc - bw / 2)
                y1 = int(yc - bh / 2)
                x2 = int(xc + bw / 2)
                y2 = int(yc + bh / 2)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    output_path = OUTPUT / image_path.name
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print("\nDataset verification completed.")