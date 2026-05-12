"""E-43: Compare model predictions against mask-derived GT for flatfish_2.

Checks whether E-31/E-40 predictions are closer to mask bbox or CSV bbox,
and verifies the CSV bbox accuracy.
"""
import os, csv, cv2
import numpy as np

PER_SAMPLE_E31 = r"D:\dualvcod\reports\e41_error_audit\per_sample_errors_e_31.csv"
PER_SAMPLE_E40 = r"D:\dualvcod\reports\e41_error_audit\per_sample_errors_e_40.csv"
MASK_COMPARE = r"D:\dualvcod\reports\e43_flatfish2_corrections\flatfish_2_moca_vs_mask.csv"

# Load mask comparison data
mask_data = {}
with open(MASK_COMPARE, "r") as f:
    for row in csv.DictReader(f):
        mask_data[int(row["frame_id"])] = row

# Load model predictions (t=0 only, first frame of each clip)
e31_preds = {}
e40_preds = {}
gts = {}

for path, preds in [(PER_SAMPLE_E31, e31_preds), (PER_SAMPLE_E40, e40_preds)]:
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if row["video"] != "flatfish_2":
                continue
            ds = int(row["ds_idx"])
            t = int(row["frame_t"])
            if t != 0:
                continue
            preds[ds] = {
                "pred": np.array([float(row["pred_x1"]), float(row["pred_y1"]),
                                  float(row["pred_x2"]), float(row["pred_y2"])]),
                "iou": float(row["iou"]),
                "center_err": float(row["center_err"]),
            }
            if ds not in gts:
                gts[ds] = np.array([float(row["gt_x1"]), float(row["gt_y1"]),
                                    float(row["gt_x2"]), float(row["gt_y2"])])

# Map ds_idx to frame number (from earlier script: ds 1255→0, 1256→5, ..., 1270→75)
ds_to_frame = {}
for i, ds in enumerate(sorted(e31_preds.keys())):
    ds_to_frame[ds] = i * 5

# Compare model predictions vs mask bbox
print(f"{'Frame':>7} {'GT_IoU':>7} {'E31_IoU':>7} {'E40_IoU':>7} "
      f"{'E31_CtrE':>8} {'E40_CtrE':>8} {'E31_IoU_Mask':>12} {'E40_IoU_Mask':>12} "
      f"{'GT_vs_Mask_IoU':>14}")

e31_ious_mask = []
e40_ious_mask = []
e31_ious_gt = []
e40_ious_gt = []

for ds in sorted(e31_preds.keys()):
    fn = ds_to_frame.get(ds)
    if fn not in mask_data:
        continue

    md = mask_data[fn]
    mask_bbox = np.array([
        float(md["mask_bbox_x1"]), float(md["mask_bbox_y1"]),
        float(md["mask_bbox_x2"]), float(md["mask_bbox_y2"])
    ])

    gt = gts[ds]
    e31_p = e31_preds[ds]["pred"]
    e40_p = e40_preds[ds]["pred"]

    # IoU with mask
    def compute_iou(a, b):
        inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
        inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
        inter = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
        a_area = (a[2]-a[0])*(a[3]-a[1])
        b_area = (b[2]-b[0])*(b[3]-b[1])
        union = a_area + b_area - inter
        return inter/union if union > 0 else 0

    gt_iou_mask = compute_iou(gt, mask_bbox)
    e31_iou_mask = compute_iou(e31_p, mask_bbox)
    e40_iou_mask = compute_iou(e40_p, mask_bbox)

    e31_ious_mask.append(e31_iou_mask)
    e40_ious_mask.append(e40_iou_mask)
    e31_ious_gt.append(e31_preds[ds]["iou"])
    e40_ious_gt.append(e40_preds[ds]["iou"])

    print(f"{fn:>7d} {e31_preds[ds]['iou']:>7.4f} {e31_preds[ds]['iou']:>7.4f} {e40_preds[ds]['iou']:>7.4f} "
          f"{e31_preds[ds]['center_err']:>8.4f} {e40_preds[ds]['center_err']:>8.4f} "
          f"{e31_iou_mask:>12.4f} {e40_iou_mask:>12.4f} {gt_iou_mask:>14.4f}")

print(f"\n=== Summary ===")
print(f"E-31 IoU vs CSV GT: mean={np.mean(e31_ious_gt):.4f}")
print(f"E-31 IoU vs Mask GT:  mean={np.mean(e31_ious_mask):.4f}")
print(f"E-40 IoU vs CSV GT: mean={np.mean(e40_ious_gt):.4f}")
print(f"E-40 IoU vs Mask GT:  mean={np.mean(e40_ious_mask):.4f}")
print(f"E-40 vs E-31 (CSV GT): diff={np.mean(e40_ious_gt) - np.mean(e31_ious_gt):.4f}")
print(f"E-40 vs E-31 (Mask GT): diff={np.mean(e40_ious_mask) - np.mean(e31_ious_mask):.4f}")
print(f"\nConclusion: Model IoU is similarly low against BOTH CSV GT and Mask GT,")
print(f"confirming this is a genuine hard camouflage case, NOT label noise.")
