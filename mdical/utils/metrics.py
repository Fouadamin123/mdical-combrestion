# ---------- utils/metrics.py ----------

import pydicom
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(original_dicom_path, reconstructed_image_path):
    # Load original DICOM image
    dicom = pydicom.dcmread(original_dicom_path)
    original = dicom.pixel_array.astype(np.float32)
    original = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Load decompressed image (PNG)
    reconstructed = cv2.imread(reconstructed_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize if dimensions mismatch
    if original.shape != reconstructed.shape:
        reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))

    # Calculate metrics
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=255)
    ssim = structural_similarity(original, reconstructed, data_range=255)

    # Difference image for visualization
    diff_img = cv2.absdiff(original, reconstructed)
    diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)

    return psnr, ssim, diff_img
