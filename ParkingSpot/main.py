import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "parking.jpg"

# ==========================================================
# AI PART 1 — K-MEANS (implemented manually using NumPy)
# ==========================================================
def kmeans_1d(data, k=6, iterations=15):
    data = np.array(data).reshape(-1, 1)

    # initialize centers
    centers = np.linspace(np.min(data), np.max(data), k)

    for _ in range(iterations):
        # assign to nearest center
        distances = np.abs(data - centers.reshape(1, -1))
        labels = np.argmin(distances, axis=1)

        # recompute centers
        new_centers = []
        for i in range(k):
            pts = data[labels == i]
            if len(pts) > 0:
                new_centers.append(np.mean(pts))
            else:
                new_centers.append(centers[i])
        centers = np.array(new_centers)

    return labels, centers


# ==========================================================
# AI PART 2 — Detect cars using simple CV (no YOLO)
# ==========================================================
def detect_car_pixels(gray):
    # Cars are darker; threshold picks dark pixels
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    return xs, ys, mask


# ==========================================================
# AI PART 3 — Cluster car x-positions into parking columns
# ==========================================================
def detect_columns_ai(xs, img_width):
    if len(xs) == 0:
        print("No car pixels detected!")
        return [(0, img_width)]

    # How many columns to find?
    # Try 6 because your images have ~6 columns
    k = 6

    labels, centers = kmeans_1d(xs, k=k)

    # sort centers left→right
    sorted_idx = np.argsort(centers)
    centers = centers[sorted_idx]

    # Convert centers to column boundaries
    boundaries = []
    prev = 0
    for c in centers[:-1]:
        mid = int((c + centers[np.where(centers == c)[0][0] + 1]) / 2)
        boundaries.append((prev, mid))
        prev = mid
    boundaries.append((prev, img_width))

    return boundaries


# ==========================================================
# 4. Compute emptiness for each AI column
# ==========================================================
def compute_column_emptiness(gray, columns):
    _, dark_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    emptiness_scores = []

    for (x1, x2) in columns:
        col_mask = dark_mask[:, x1:x2]
        car_fraction = np.mean(col_mask > 0)
        emptiness = 1.0 - car_fraction
        emptiness_scores.append(emptiness)

    return emptiness_scores


# ==========================================================
# 5. Draw the winner column
# ==========================================================
def draw_winner(img, columns, emptiness_scores):
    best_idx = int(np.argmax(emptiness_scores))
    x1, x2 = columns[best_idx]
    H = img.shape[0]

    out = img.copy()
    cv2.rectangle(out, (x1, 0), (x2, H), (0, 0, 255), 8)
    cv2.putText(out, f"BEST COL (AI): {best_idx+1}",
                (x1 + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4)

    print("\nAI detected columns:", len(columns))
    print("Column boundaries:", columns)
    print("\nEmptiness scores:")
    for i, e in enumerate(emptiness_scores, 1):
        print(f"  Column {i}: {e:.3f}")

    print(f"\nWinner (AI): Column {best_idx+1}")

    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Could not load image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Detecting car pixels (AI input)...")
    xs, ys, mask = detect_car_pixels(gray)

    print("Running AI clustering (K-Means)...")
    columns = detect_columns_ai(xs, gray.shape[1])

    print("Computing emptiness scores...")
    emptiness_scores = compute_column_emptiness(gray, columns)

    print("Drawing AI winner...")
    draw_winner(img, columns, emptiness_scores)


if __name__ == "__main__":
    main()
