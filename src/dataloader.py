import torch
from torch.utils.data import Dataset


class SyntheticVideoDataset(Dataset):
    """Generates synthetic video clips and GT BBoxes for pipeline validation.

    Each sample is a T-frame video clip of shape (T, C, H, W) with
    T corresponding ground-truth bounding boxes in (x1, y1, x2, y2) format.
    """

    def __init__(self, num_samples=100, T=5, H=224, W=224, C=3):
        self.num_samples = num_samples
        self.T = T
        self.H = H
        self.W = W
        self.C = C

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frames = torch.randn(self.T, self.C, self.H, self.W)

        gt_bboxes = torch.zeros(self.T, 4)
        for t in range(self.T):
            x1 = float(torch.rand(1)) * 0.5
            y1 = float(torch.rand(1)) * 0.5
            x2 = x1 + 0.1 + float(torch.rand(1)) * 0.4
            y2 = y1 + 0.1 + float(torch.rand(1)) * 0.4
            gt_bboxes[t] = torch.tensor([x1, y1, min(x2, 1.0), min(y2, 1.0)])

        return frames, gt_bboxes


def collate_video_clips(batch):
    """Collate function: stacks frames→(B,T,C,H,W) and bboxes→(B,T,4)."""
    frames, bboxes = zip(*batch)
    frames = torch.stack(frames, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    return frames, bboxes
