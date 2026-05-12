"""Verify that torchvision decode_image produces RGB (not BGR) and matches cv2 RGB."""
import cv2, torch, numpy as np
from torchvision import io as tvio

src = r"C:\datasets\MoCA\JPEGImages\arabian_horn_viper\00001.jpg"
resized = r"C:\datasets_224\MoCA\JPEGImages\arabian_horn_viper\00001.jpg"

# cv2 path (reference RGB)
img_cv2 = cv2.imread(src)
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
img_cv2_rgb_224 = cv2.resize(img_cv2_rgb, (224, 224))

# torchvision path on pre-resized
data = tvio.read_file(resized)
img_tv = tvio.decode_image(data)  # (C, H, W) uint8
img_tv_np = img_tv.permute(1, 2, 0).numpy()  # (H, W, C)

print("=== Shape comparison ===")
print(f"cv2 RGB resized:      {img_cv2_rgb_224.shape} (H,W,C)")
print(f"torchvision pre-resized: {img_tv_np.shape} (H,W,C)")

print()
print("=== Color order verification ===")
h, w = img_cv2_rgb_224.shape[:2]
for (y, x) in [(50, 50), (100, 100), (150, 150)]:
    cv2_px = img_cv2_rgb_224[y, x]
    tv_px = img_tv_np[y, x]
    diff_rgb = np.abs(cv2_px.astype(float) - tv_px.astype(float))
    diff_bgr = np.abs(cv2_px.astype(float) - tv_px[::-1].astype(float))
    print(f"  Pixel ({y},{x}): cv2_RGB={cv2_px}, tv={tv_px}")
    print(f"    diff vs RGB ordering: mean={diff_rgb.mean():.1f}")
    print(f"    diff vs BGR ordering: mean={diff_bgr.mean():.1f}")

# Channel-wise mean comparison
cv2_mean = img_cv2_rgb_224.astype(float).mean(axis=(0, 1))
tv_mean = img_tv_np.astype(float).mean(axis=(0, 1))
print(f"Channel means - cv2_RGB: R={cv2_mean[0]:.1f} G={cv2_mean[1]:.1f} B={cv2_mean[2]:.1f}")
print(f"Channel means - tv:      R={tv_mean[0]:.1f} G={tv_mean[1]:.1f} B={tv_mean[2]:.1f}")

r_diff_same = abs(cv2_mean[0] - tv_mean[0]) + abs(cv2_mean[2] - tv_mean[2])
r_diff_swap = abs(cv2_mean[0] - tv_mean[2]) + abs(cv2_mean[2] - tv_mean[0])
print(f"R/B same-order diff: {r_diff_same:.1f}")
print(f"R/B swapped diff:    {r_diff_swap:.1f}")
same_order = r_diff_same < r_diff_swap
print(f"Conclusion: tv is {'RGB' if same_order else 'BGR'}")

print()
print("=== Value range ===")
print(f"cv2 range: [{img_cv2_rgb_224.min()}, {img_cv2_rgb_224.max()}]")
print(f"tv range:   [{img_tv_np.min()}, {img_tv_np.max()}]")

print()
print("=== dtype ===")
print(f"cv2 numpy: {img_cv2_rgb_224.dtype}")
print(f"tv numpy:  {img_tv_np.dtype}")
print(f"tv tensor: {img_tv.dtype}")

print()
print("=== Overall similarity ===")
mse = ((img_cv2_rgb_224.astype(float) - img_tv_np.astype(float)) ** 2).mean()
psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
print(f"MSE: {mse:.2f}, PSNR: {psnr:.1f} dB")

# Also verify: does the dataset __getitem__ produce identical results?
print()
print("=== Full pipeline comparison (3 frames) ===")
import sys; sys.path.insert(0, "src")
from dataset_real import RealVideoBBoxDataset

ds_orig = RealVideoBBoxDataset(
    [r"C:\datasets\MoCA"], T=3, target_size=224, augment=False)
ds_resized = RealVideoBBoxDataset(
    [r"C:\datasets\MoCA"], T=3, target_size=224, augment=False,
    resized_root=r"C:\datasets_224")

# Force same samples by using same index
for idx in [0, 100, 500]:
    f_orig, b_orig = ds_orig[idx]
    f_resized, b_resized = ds_resized[idx]
    diff = (f_orig - f_resized).abs()
    print(f"  idx={idx}: max pixel diff={diff.max():.6f}, mean diff={diff.mean():.6f}")
    print(f"    bbox orig={b_orig[0].tolist()}")
    print(f"    bbox resized={b_resized[0].tolist()}")

print()
print("DONE")
