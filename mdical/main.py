# ---------- main.py ----------

import pandas as pd
import os
import time
import cv2
from compress.compact import compact_compress, compact_decompress
from utils.metrics import compute_metrics
import matplotlib.pyplot as plt

def process_dicom_compression(csv_path, output_folder):
    df = pd.read_csv(csv_path)
    results = []

    for index, row in df.iterrows():
        original_path = row.iloc[0] 
        if not os.path.isfile(original_path) or not original_path.lower().endswith('.dcm'):
            print(f"[{index+1}/{len(df)}] Skipping invalid file: {original_path}")
            continue

        print(f"[{index+1}/{len(df)}] ({(index+1)/len(df)*100:.1f}%) Processing: {original_path}")
        filename = os.path.basename(original_path)
        name = os.path.splitext(filename)[0]

        compressed_path = os.path.join(output_folder, f"{name}.bin")
        reconstructed_path = os.path.join(output_folder, f"{name}_reconstructed.png")
        diff_path = os.path.join(output_folder, f"{name}_diff.png")
        os.makedirs(output_folder, exist_ok=True)

        start = time.time()
        try:
            compact_compress(original_path, compressed_path)
            compact_decompress(compressed_path, reconstructed_path)
            compression_success = True
        except Exception as e:
            print(f"Error during compression: {e}")
            compression_success = False

        if compression_success:
            try:
                orig_kb = os.path.getsize(original_path) / 1024
                comp_kb = os.path.getsize(compressed_path) / 1024
                ratio = orig_kb / comp_kb if comp_kb > 0 else 0
                duration = time.time() - start

                psnr, ssim, diff_img = compute_metrics(original_path, reconstructed_path)
                cv2.imwrite(diff_path, diff_img)

                results.append({
                    "Index": index,
                    "Original": original_path,
                    "Compressed": compressed_path,
                    "Original_KB": round(orig_kb, 2),
                    "Compressed_KB": round(comp_kb, 2),
                    "Ratio": round(ratio, 2),
                    "Time_sec": round(duration, 3),
                    "PSNR": round(psnr, 2),
                    "SSIM": round(ssim, 4)
                })

            except Exception as e:
                print(f"Error during metrics: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(output_folder, "results.csv"), index=False)
        plot_metrics(df_results, output_folder)

    print("\n[OK] Compression + Evaluation Completed.")

def plot_metrics(results_df, output_folder):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results_df["PSNR"], 'b.-')
    plt.title("PSNR per Image")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results_df["SSIM"], 'r.-')
    plt.title("SSIM per Image")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "metrics_plot.png"))
    plt.close()

if __name__ == "__main__":
    csv_path = r"D:\dicom-compressors\mdical\dicom_paths.csv"
    output_folder = r"D:\dicom-compressors\output\compact"
    process_dicom_compression(csv_path, output_folder)
