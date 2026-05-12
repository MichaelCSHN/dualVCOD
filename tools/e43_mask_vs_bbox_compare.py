"""E-43: Compare MoCA CSV bbox annotations vs MoCA_Mask pixel-level GT for flatfish_2.

Key question: Is the MoCA CSV bbox too loose compared to the pixel-accurate mask?
"""
import os, csv, json, cv2
import numpy as np

MOCA_CSV = r"C:\datasets\MoCA\Annotations\annotations.csv"
MASK_DIR = r"C:\datasets\MoCA_Mask\TrainDataset_per_sq\flatfish_2\GT"
IMG_DIR_MASK = r"C:\datasets\MoCA_Mask\TrainDataset_per_sq\flatfish_2\Imgs"

OUT_DIR = r"D:\dualvcod\reports\e43_flatfish2_corrections"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load MoCA CSV annotations for flatfish_2
moca_bboxes = {}
with open(MOCA_CSV, "r") as f:
    for row in csv.reader(f):
        if not row or row[0].startswith("#"):
            continue
        fp = row[1].strip("/")
        if not fp.startswith("flatfish_2/"):
            continue
        fn = int(fp.replace("flatfish_2/", "").replace(".jpg", ""))
        coords = json.loads(row[4])
        if coords[0] == 2:
            x, y, w, h = float(coords[1]), float(coords[2]), float(coords[3]), float(coords[4])
            moca_bboxes[fn] = {"x": x, "y": y, "w": w, "h": h}

print(f"MoCA CSV annotations for flatfish_2: {len(moca_bboxes)} frames")
print(f"Frame range: {min(moca_bboxes.keys())} - {max(moca_bboxes.keys())}")

# 2. Load MoCA_Mask GT masks and compute tight bbox
# Determine original image size from first image
sample_img = cv2.imread(os.path.join(IMG_DIR_MASK, "00030.jpg"))
orig_h, orig_w = sample_img.shape[:2]
print(f"Original image size: {orig_w}x{orig_h}")

comparison_rows = []
for fn in sorted(moca_bboxes.keys()):
    # Check if mask exists for this frame
    mask_path = os.path.join(MASK_DIR, f"{fn:05d}.png")
    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    # Compute tight bbox from mask
    mask_bin = (mask > 0).astype(np.uint8)
    rows = np.any(mask_bin, axis=1)
    cols = np.any(mask_bin, axis=0)
    if not rows.any() or not cols.any():
        continue

    y1, y2 = np.where(rows)[0][[0, -1]].astype(np.float32)
    x1, x2 = np.where(cols)[0][[0, -1]].astype(np.float32)

    # Normalize to [0, 1]
    mask_bbox = np.array([x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h])
    mask_area = (mask_bbox[2] - mask_bbox[0]) * (mask_bbox[3] - mask_bbox[1])

    # MoCA CSV bbox (also normalize)
    mc = moca_bboxes[fn]
    moca_norm = np.array([
        mc["x"] / orig_w,
        mc["y"] / orig_h,
        (mc["x"] + mc["w"]) / orig_w,
        (mc["y"] + mc["h"]) / orig_h,
    ])
    moca_area = (moca_norm[2] - moca_norm[0]) * (moca_norm[3] - moca_norm[1])

    # Compute IoU between MoCA CSV bbox and mask-derived bbox
    inter_x1 = max(mask_bbox[0], moca_norm[0])
    inter_y1 = max(mask_bbox[1], moca_norm[1])
    inter_x2 = min(mask_bbox[2], moca_norm[2])
    inter_y2 = min(mask_bbox[3], moca_norm[3])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    mask_a = mask_area
    moca_a = moca_area
    union = mask_a + moca_a - inter
    iou = inter / union if union > 0 else 0

    # Area ratio
    area_ratio = moca_area / mask_area if mask_area > 0 else float('inf')

    # Center difference
    mask_center = np.array([(mask_bbox[0]+mask_bbox[2])/2, (mask_bbox[1]+mask_bbox[3])/2])
    moca_center = np.array([(moca_norm[0]+moca_norm[2])/2, (moca_norm[1]+moca_norm[3])/2])
    center_err = np.sqrt(np.sum((mask_center - moca_center)**2))

    comparison_rows.append({
        "frame_id": f"{fn:05d}",
        "mask_bbox_x1": round(float(mask_bbox[0]), 4),
        "mask_bbox_y1": round(float(mask_bbox[1]), 4),
        "mask_bbox_x2": round(float(mask_bbox[2]), 4),
        "mask_bbox_y2": round(float(mask_bbox[3]), 4),
        "mask_area": round(float(mask_area), 4),
        "moca_csv_x1": round(float(moca_norm[0]), 4),
        "moca_csv_y1": round(float(moca_norm[1]), 4),
        "moca_csv_x2": round(float(moca_norm[2]), 4),
        "moca_csv_y2": round(float(moca_norm[3]), 4),
        "moca_csv_area": round(float(moca_area), 4),
        "iou_moca_vs_mask": round(float(iou), 4),
        "area_ratio_csv_over_mask": round(float(area_ratio), 2),
        "center_error": round(float(center_err), 4),
        "moca_csv_w_px": round(mc["w"], 1),
        "moca_csv_h_px": round(mc["h"], 1),
        "moca_csv_y_px": round(mc["y"], 1),
        "mask_y1_px": round(float(y1), 1),
        "mask_y2_px": round(float(y2), 1),
    })

# Write comparison CSV
csv_path = os.path.join(OUT_DIR, "flatfish_2_moca_vs_mask.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"\nWrote {len(comparison_rows)} comparison rows to {csv_path}")

# Summary stats
ious = [r["iou_moca_vs_mask"] for r in comparison_rows]
area_ratios = [r["area_ratio_csv_over_mask"] for r in comparison_rows]
center_errs = [r["center_error"] for r in comparison_rows]

print(f"\n=== MoCA CSV vs MoCA_Mask GT Comparison ===")
print(f"Frames compared: {len(comparison_rows)}")
print(f"IoU (CSV vs Mask): mean={np.mean(ious):.4f}, min={np.min(ious):.4f}, max={np.max(ious):.4f}")
print(f"Area Ratio (CSV/Mask): mean={np.mean(area_ratios):.2f}, min={np.min(area_ratios):.2f}, max={np.max(area_ratios):.2f}")
print(f"Center Error: mean={np.mean(center_errs):.4f}, min={np.min(center_errs):.4f}, max={np.max(center_errs):.4f}")

# Show per-frame details
print(f"\n=== Per-Frame Details ===")
print(f"{'Frame':>7} {'IoU':>7} {'AreaR':>7} {'CtrErr':>7} {'CSV_y1':>7} {'Mask_y1':>7} {'CSV_y2':>7} {'Mask_y2':>7} {'CSV_y_px':>8} {'Mask_y1_px':>10} {'Mask_y2_px':>10}")
for r in comparison_rows:
    print(f"{r['frame_id']:>7} {r['iou_moca_vs_mask']:>7.4f} {r['area_ratio_csv_over_mask']:>7.2f} "
          f"{r['center_error']:>7.4f} {r['moca_csv_y1']:>7.4f} {r['mask_bbox_y1']:>7.4f} "
          f"{r['moca_csv_y2']:>7.4f} {r['mask_bbox_y2']:>7.4f} "
          f"{r['moca_csv_y_px']:>8.1f} {r['mask_y1_px']:>10.1f} {r['mask_y2_px']:>10.1f}")

# CONCLUSION
print(f"\n=== CONCLUSION ===")
if np.mean(area_ratios) > 1.5:
    print(f"MoCA CSV bbox is ON AVERAGE {np.mean(area_ratios):.1f}x LARGER than mask-derived tight bbox.")
    print("This CONFIRMS the GT annotations are systematically too loose.")
if np.mean(center_errs) > 0.05:
    print(f"Average center error {np.mean(center_errs):.4f} confirms significant misalignment.")
if np.mean(ious) < 0.5:
    print(f"Average IoU {np.mean(ious):.4f} between CSV bbox and mask bbox is LOW.")
    print("Mask-derived bbox is a much tighter representation of the object.")
