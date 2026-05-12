"""Test DataLoader iteration speed."""
import sys, os, time, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips

def _video_name_from_sample(sample):
    dir_path = sample.get("video_dir", sample["frame_dir"])
    return os.path.basename(dir_path.rstrip("/\\"))

DATASET_ROOTS = [
    r"D:\ML\COD_datasets\MoCA",
    r"D:\ML\COD_datasets\MoCA_Mask",
    r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
]

print("Loading MoCA split...", flush=True)
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
print(f"  Split: {len(train_idx)} train ({time.time()-t0:.1f}s)", flush=True)

print("Loading train datasets...", flush=True)
t0 = time.time()
train_sets = []
for root in DATASET_ROOTS:
    if not os.path.isdir(root):
        continue
    ds = RealVideoBBoxDataset([root], T=5, target_size=224, augment=True)
    name = os.path.basename(root)
    if "MoCA" in root and "MoCA_Mask" not in root:
        ds = Subset(ds, train_idx)
    elif "MoCA_Mask" in root or "CamouflagedAnimalDataset" in root:
        valid_indices = [i for i, s in enumerate(ds.samples) if _video_name_from_sample(s) not in val_canonical_ids]
        ds = Subset(ds, valid_indices)
    train_sets.append(ds)
    print(f"  {name}: {len(ds)} samples", flush=True)
joint = ConcatDataset(train_sets)
print(f"  Total: {len(joint)} samples ({time.time()-t0:.1f}s)", flush=True)

print("Creating DataLoader (num_workers=0)...", flush=True)
loader = DataLoader(joint, batch_size=24, shuffle=True, collate_fn=collate_video_clips, num_workers=0)

print("Testing first 3 batches...", flush=True)
for i, (frames, gt) in enumerate(loader):
    print(f"  Batch {i+1}: {frames.shape}, {gt.shape}", flush=True)
    if i >= 2:
        break

print("DONE - DataLoader iteration works", flush=True)
