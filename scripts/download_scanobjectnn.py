import os
import sys
import time
import urllib.request
import h5py

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"File already exists ({os.path.getsize(dest_path)/(1024*1024):.2f} MB): {dest_path}")
        return True

    print(f"Downloading: {url} -> {dest_path}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_size = int(resp.info().get('Content-Length', -1))
            downloaded = 0
            chunk_size = 512 * 1024
            start_time = time.time()
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        el = time.time() - start_time
                        spd = (downloaded / (1024*1024)) / (el + 1e-5)
                        if downloaded == total_size or int(pct) % 25 == 0:
                            print(f"  [{pct:5.1f}%] {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({spd:.2f} MB/s)", flush=True)
            print(f"Successfully downloaded: {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    base_url = "https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main"
    root_dir = r"c:\DLproj\datasets\real_world\ScanObjectNN"

    files_to_download = [
        # Clean Object-Only (no background) - Preferred for clean synthetic-to-real comparison
        ("main_split_nobg/training_objectdataset.h5", os.path.join(root_dir, "main_split_nobg", "training_objectdataset.h5")),
        ("main_split_nobg/test_objectdataset.h5", os.path.join(root_dir, "main_split_nobg", "test_objectdataset.h5")),
        # Standard with background (for reference/robustness)
        ("main_split/training_objectdataset.h5", os.path.join(root_dir, "main_split", "training_objectdataset.h5")),
        ("main_split/test_objectdataset.h5", os.path.join(root_dir, "main_split", "test_objectdataset.h5")),
    ]

    print("=" * 80)
    print("DOWNLOADING OFFICIAL SCANOBJECTNN DATASET")
    print(f"Target Directory: {root_dir}")
    print("=" * 80)

    for rel_url, target_path in files_to_download:
        full_url = f"{base_url}/{rel_url}"
        success = download_file(full_url, target_path)
        if not success:
            raise RuntimeError(f"Failed to download required file: {rel_url}")

    print("\nValidating downloaded H5 files integrity...")
    for _, target_path in files_to_download:
        with h5py.File(target_path, 'r') as h5f:
            keys = list(h5f.keys())
            data_shape = h5f['data'].shape if 'data' in h5f else "N/A"
            label_shape = h5f['label'].shape if 'label' in h5f else "N/A"
            print(f"  Valid H5: {os.path.basename(target_path):35s} | Keys: {keys} | data: {data_shape} | label: {label_shape}")

    print("\nScanObjectNN download and verification complete!")

if __name__ == "__main__":
    main()
