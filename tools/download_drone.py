# download_drone.py
import time
from huggingface_hub import snapshot_download

local_dir = "D:/ML/COD_datasets/DroneVehicle"
max_attempts = 20

for attempt in range(1, max_attempts + 1):
    print(f"\n=== 第 {attempt} 次尝试 ===")
    try:
        snapshot_download(
            repo_id="McCheng/DroneVehicle",
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=2,          # 减少并发，更稳定
        )
        print("✅ 下载完成！")
        break
    except Exception as e:
        print(f"❌ 失败: {e}")
        if attempt < max_attempts:
            wait = min(60, 10 * attempt)
            print(f"等待 {wait} 秒后重试...")
            time.sleep(wait)
        else:
            print("已达到最大重试次数。")