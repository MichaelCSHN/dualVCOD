"""Minimal training test — full epoch, flush after every step."""
import sys, os, time, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from src.loss import BBoxLoss
from eval.eval_video_bbox import compute_metrics, count_parameters

DEVICE = "cuda"

def _video_name_from_sample(sample):
    dir_path = sample.get("video_dir", sample["frame_dir"])
    return os.path.basename(dir_path.rstrip("/\\"))

DATASET_ROOTS = [
    r"D:\ML\COD_datasets\MoCA",
    r"D:\ML\COD_datasets\MoCA_Mask",
    r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
]

print("="*60, flush=True)
print("Loading data...", flush=True)
t0 = time.time()

moca_split_ds = RealVideoBBoxDataset([DATASET_ROOTS[0]], T=5, target_size=224, augment=False)
video_to_indices = defaultdict(list)
for i in range(len(moca_split_ds)):
    vname = _video_name_from_sample(moca_split_ds.samples[i])
    video_to_indices[vname].append(i)
videos = sorted(video_to_indices.keys())
rng = random.Random(42)
rng.shuffle(videos)
n_val = max(1, int(len(videos) * 0.2))
val_videos = set(videos[:n_val])
train_idx = [i for v in videos[n_val:] for i in video_to_indices[v]]
val_idx = [i for v in val_videos for i in video_to_indices[v]]
val_canonical_ids = {_video_name_from_sample(moca_split_ds.samples[i]) for i in val_idx}

train_sets = []
for root in DATASET_ROOTS:
    if not os.path.isdir(root):
        continue
    ds = RealVideoBBoxDataset([root], T=5, target_size=224, augment=True)
    if "MoCA" in root and "MoCA_Mask" not in root:
        ds = Subset(ds, train_idx)
    elif "MoCA_Mask" in root or "CamouflagedAnimalDataset" in root:
        valid_indices = [i for i, s in enumerate(ds.samples) if _video_name_from_sample(s) not in val_canonical_ids]
        ds = Subset(ds, valid_indices)
    train_sets.append(ds)

joint_train_ds = ConcatDataset(train_sets)
train_loader = DataLoader(joint_train_ds, batch_size=24, shuffle=True, collate_fn=collate_video_clips, num_workers=0, pin_memory=False)

val_moca = RealVideoBBoxDataset([DATASET_ROOTS[0]], T=5, target_size=224, augment=False)
val_loader = DataLoader(Subset(val_moca, val_idx), batch_size=24, shuffle=False, collate_fn=collate_video_clips, num_workers=0, pin_memory=False)

print(f"  Data ready: {len(joint_train_ds)} train, {len(val_idx)} val ({time.time()-t0:.1f}s)", flush=True)

print("\nBuilding model...", flush=True)
model = MicroVCOD(T=5, pretrained_backbone=True, backbone_name="mobilenet_v3_small", head_type="current_direct_bbox").to(DEVICE)
print(f"  Params: {count_parameters(model):,}", flush=True)

print("\nSetting up training...", flush=True)
criterion = BBoxLoss(smooth_l1_weight=1.0, giou_weight=1.0, center_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
scaler = torch.amp.GradScaler("cuda")

print("\nTraining 1 epoch (verbose every 50 batches)...", flush=True)
model.train()
total_loss = 0.0
total_miou = 0.0
n_batches = 0
t_start = time.time()
for frames, gt_bboxes in train_loader:
    frames = frames.to(DEVICE)
    gt_bboxes = gt_bboxes.to(DEVICE)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        pred = model(frames)
        losses = criterion(pred, gt_bboxes)
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    scaler.step(optimizer)
    scaler.update()
    total_loss += losses["loss"].item()
    total_miou += losses["mean_iou"].item()
    n_batches += 1
    if n_batches % 50 == 0:
        t_elapsed = time.time() - t_start
        print(f"  Batch {n_batches}: loss={total_loss/n_batches:.4f} miou={total_miou/n_batches:.4f} [{t_elapsed:.1f}s]", flush=True)

scheduler.step()
tr_loss = total_loss / n_batches
tr_miou = total_miou / n_batches
print(f"\nTrain done: loss={tr_loss:.4f} miou={tr_miou:.4f} batches={n_batches} time={time.time()-t_start:.1f}s", flush=True)

print("\nValidating...", flush=True)
t0 = time.time()
model.eval()
all_preds, all_gts = [], []
for frames, gt_bboxes in val_loader:
    with torch.amp.autocast("cuda"):
        pred = model(frames.to(DEVICE))
    all_preds.append(pred.float().cpu())
    all_gts.append(gt_bboxes)
preds = torch.cat(all_preds, dim=0)
gts = torch.cat(all_gts, dim=0)
metrics = compute_metrics(preds, gts)
print(f"  Val mIoU: {metrics['mean_iou']:.4f}  R@0.5: {metrics['recall@0.5']:.4f} ({time.time()-t0:.1f}s)", flush=True)

print("\nDONE", flush=True)
torch.cuda.empty_cache()
