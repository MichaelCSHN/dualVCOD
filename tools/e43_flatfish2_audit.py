"""E-43: Visual audit of flatfish_2 GT annotations.

Overlays GT, E-31, and E-40 bboxes on selected frames, saves annotated PNGs.
Also generates per-frame diagnostic CSV.
"""
import os, sys, csv, json
import cv2
import numpy as np

FRAME_DIR = r"C:\datasets_224\MoCA\JPEGImages\flatfish_2"
PER_SAMPLE_E31 = r"D:\dualvcod\reports\e41_error_audit\per_sample_errors_e_31.csv"
PER_SAMPLE_E40 = r"D:\dualvcod\reports\e41_error_audit\per_sample_errors_e_40.csv"
OUT_DIR = r"D:\dualvcod\reports\e43_flatfish2_corrections"

# How many frames to sample for detailed visual audit
SAMPLE_EVERY_N = 5  # every 5th frame (16 frames for 80 total)

COLORS = {
    "GT": (0, 0, 255),       # Red
    "E-31": (255, 0, 0),     # Blue
    "E-40": (0, 255, 0),     # Green
}

def load_per_sample(path, video="flatfish_2"):
    """Return dict: (ds_idx, frame_t) -> {gt_bbox, pred_bbox, iou, err_type}"""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["video"] != video:
                continue
            key = (int(row["ds_idx"]), int(row["frame_t"]))
            data[key] = {
                "gt": np.array([float(row["gt_x1"]), float(row["gt_y1"]),
                                float(row["gt_x2"]), float(row["gt_y2"])]),
                "pred": np.array([float(row["pred_x1"]), float(row["pred_y1"]),
                                  float(row["pred_x2"]), float(row["pred_y2"])]),
                "iou": float(row["iou"]),
                "err_type": row["err_type"],
                "gt_area": float(row["gt_area"]),
                "center_err": float(row["center_err"]),
            }
    return data

def draw_bbox(img, bbox_norm, color, label, thickness=2):
    """Draw normalized bbox [x1,y1,x2,y2] on image."""
    h, w = img.shape[:2]
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # Label background box
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    e31 = load_per_sample(PER_SAMPLE_E31)
    e40 = load_per_sample(PER_SAMPLE_E40)

    # Map frame indices to absolute frame numbers
    # The CSV ds_idx encodes the starting frame and T=5 gives 5 consecutive frames
    # We need to figure out: ds_idx 1255 = which absolute frame?
    # From MoCA CSV: frames are 00000, 00005, 00010, ... 00095 (every 5th)
    # The dataset samples use T=5, temporal_stride=1
    # Let's use the unique ds_idx values to map
    unique_ds = sorted(set(k[0] for k in e31.keys()))
    print(f"Unique ds_idx for flatfish_2: {unique_ds}")
    print(f"Number of ds_idx: {len(unique_ds)}")

    # From per-video metrics: 80 frames total
    # ds_idx 1255-1270 = 16 samples × 5 frames = 80 frame-T combinations
    # But they overlap due to temporal stride=1. The unique video frames
    # would be ds_idx mapped to frame numbers.
    # Let's check: ds_idx 1255 has t=0,1,2,3,4
    # This maps to video frames [0,1,2,3,4] for the first sample
    # Actually, in dataset_real.py, ds_idx maps to start_frame in the samples list
    # For flatfish_2, frames 00000-00095 exist (every 5th frame),
    # but with temporal_stride=1, we skip to every single frame

    # Let's look at the unique frame numbers by computing absolute frame from ds_idx + t
    # The simplest approach: find all unique absolute frames referenced
    frames_seen = set()
    for key in e31:
        ds_idx, t = key
        # Estimate absolute frame: we need to map ds_idx to frame number
        # Let's infer from the MoCA annotations CSV
        frames_seen.add((ds_idx, t))

    # Load MoCA annotations to map ds_idx to absolute frame
    moca_csv = r"C:\datasets\MoCA\Annotations\annotations.csv"
    moca_frames = {}
    with open(moca_csv, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            fp = row[1].strip("/")
            if not fp.startswith("flatfish_2/"):
                continue
            fn = int(fp.replace("flatfish_2/", "").replace(".jpg", ""))
            coords = json.loads(row[4])
            if coords[0] == 2:
                moca_frames[fn] = {
                    "x": float(coords[1]), "y": float(coords[2]),
                    "w": float(coords[3]), "h": float(coords[4]),
                }
    print(f"MoCA annotated frames for flatfish_2: {sorted(moca_frames.keys())}")

    # Check original image size
    orig_img_path = rf"C:\datasets\MoCA\JPEGImages\flatfish_2\00000.jpg"
    orig = cv2.imread(orig_img_path)
    if orig is not None:
        orig_h, orig_w = orig.shape[:2]
        print(f"Original image size: {orig_w}x{orig_h}")
    else:
        # Try to infer from resized path
        orig_h, orig_w = 720, 1280  # common video size

    # The resized frames are 224x224
    resized_img_path = rf"C:\datasets_224\MoCA\JPEGImages\flatfish_2\00000.jpg"
    resized = cv2.imread(resized_img_path)
    resized_h, resized_w = resized.shape[:2] if resized is not None else (224, 224)
    print(f"Resized image size: {resized_w}x{resized_h}")

    # Now build per-frame data using the MoCA annotations + predictions
    # We need to find which ds_idx+frame_t maps to which MoCA frame number
    # The dataset_samples have start_frame from moca_frames, staggered with stride=1
    # Actually flatfish_2 has 20 annotated frames (every 5th: 0,5,10,...,95)
    # With T=5 and stride=1, the samples start at annotated frames
    # ds_idx 1255 = frame 00000, ds_idx 1256 = frame 00005, etc.

    # Let's map ds_idx to MoCA frame number
    moca_annotated = sorted(moca_frames.keys())  # [0, 5, 10, ..., 95]
    print(f"MoCA annotated frames: {moca_annotated}")

    # For MoCA split dataset, samples are created from annotated frames
    # Each ds_idx corresponds to one sample starting at an annotated frame
    ds_to_frame = {}
    for i, ds in enumerate(unique_ds):
        if i < len(moca_annotated):
            ds_to_frame[ds] = moca_annotated[i]

    print(f"ds_idx to MoCA frame mapping (first 5):")
    for ds in unique_ds[:5]:
        print(f"  ds_idx {ds} -> frame {ds_to_frame.get(ds, '?')}")

    # Now generate annotated images
    # For each sample, render the t=0 frame (first frame of clip)
    # This gives us frames at intervals of 5 (0, 5, 10, 15, ..., 75)

    correction_rows = []
    annotated_frames = set()

    for ds in unique_ds:
        t = 0  # first frame of each clip
        key = (ds, t)

        moca_fn = ds_to_frame.get(ds)
        if moca_fn is None:
            continue

        gt_e31 = e31.get(key)
        gt_e40 = e40.get(key)

        if gt_e31 is None or gt_e40 is None:
            continue

        gt_bbox = gt_e31["gt"]  # GT is same for both
        e31_pred = gt_e31["pred"]
        e40_pred = gt_e40["pred"]

        # Load resized image
        img_path = os.path.join(FRAME_DIR, f"{moca_fn:05d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            # Try original and resize
            orig_path = rf"C:\datasets\MoCA\JPEGImages\flatfish_2\{moca_fn:05d}.jpg"
            img = cv2.imread(orig_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))

        if img is None:
            print(f"  Could not load frame {moca_fn:05d}")
            continue

        annotated_frames.add(moca_fn)

        # Draw bboxes
        vis = img.copy()
        draw_bbox(vis, gt_bbox, COLORS["GT"], "GT")
        draw_bbox(vis, e31_pred, COLORS["E-31"], "E-31")
        draw_bbox(vis, e40_pred, COLORS["E-40"], "E-40")

        # Add frame info
        cv2.putText(vis, f"Frame {moca_fn:05d}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        iou_str = f"E-31 IoU:{gt_e31['iou']:.3f} E-40 IoU:{gt_e40['iou']:.3f}"
        cv2.putText(vis, iou_str, (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"flatfish_2_{moca_fn:05d}_annotated.png")
        cv2.imwrite(out_path, vis)

        # Classification
        gt_area = gt_e31["gt_area"]
        e31_ce = gt_e31["center_err"]
        e40_ce = gt_e40["center_err"]

        # Determine classification
        if gt_area > 0.15 and e31_ce > 0.1 and e40_ce > 0.1:
            quality = "GT_LIKELY_LOOSE"  # Both models shift away from GT, GT too large
        elif e31_ce > 0.2 or e40_ce > 0.2:
            quality = "GT_MISALIGNED"  # Large center error
        elif gt_e31["iou"] < 0.1 and gt_e40["iou"] < 0.1:
            quality = "TARGET_UNIDENTIFIABLE"
        else:
            quality = "GT_REASONABLE"

        correction_rows.append({
            "frame_id": f"{moca_fn:05d}",
            "original_bbox_x1": round(float(gt_bbox[0]), 4),
            "original_bbox_y1": round(float(gt_bbox[1]), 4),
            "original_bbox_x2": round(float(gt_bbox[2]), 4),
            "original_bbox_y2": round(float(gt_bbox[3]), 4),
            "gt_area": round(gt_area, 4),
            "e31_iou": round(gt_e31["iou"], 4),
            "e31_center_err": round(e31_ce, 4),
            "e31_err_type": gt_e31["err_type"],
            "e40_iou": round(gt_e40["iou"], 4),
            "e40_center_err": round(e40_ce, 4),
            "e40_err_type": gt_e40["err_type"],
            "annotation_quality": quality,
        })

    # Write correction CSV
    csv_path = os.path.join(OUT_DIR, "flatfish_2_audit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=correction_rows[0].keys())
        writer.writeheader()
        writer.writerows(correction_rows)
    print(f"\nWrote {len(correction_rows)} frame audits to {csv_path}")

    # Summary stats
    qualities = {}
    for row in correction_rows:
        q = row["annotation_quality"]
        qualities[q] = qualities.get(q, 0) + 1
    print(f"\nAnnotation quality distribution:")
    for q, n in sorted(qualities.items()):
        print(f"  {q}: {n}/{len(correction_rows)}")

    print(f"\nAnnotated frames saved to: {OUT_DIR}")
    print(f"Frames audited: {sorted(annotated_frames)}")

if __name__ == "__main__":
    main()
