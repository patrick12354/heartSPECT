"""
Preprocessing and inference utilities for SPECT segmentation.
"""

import numpy as np
import pydicom
import nibabel as nib
import torch
from scipy.ndimage import zoom

TARGET_SHAPE = (64, 64, 64)


def clip_and_normalize(img: np.ndarray,
                       p_low: float = 1.0,
                       p_high: float = 99.0) -> np.ndarray:
    """Percentile clip + Z-score normalization."""
    low, high = np.percentile(img, p_low), np.percentile(img, p_high)
    img = np.clip(img, low, high).astype(np.float32)
    mean, std = img.mean(), img.std()
    return (img - mean) / (std + 1e-8)


def resample_volume(volume: np.ndarray,
                    target_shape: tuple = TARGET_SHAPE,
                    order: int = 1) -> np.ndarray:
    """Resample volume to target_shape using interpolation."""
    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, factors, order=order)


def load_dicom(dicom_path: str) -> np.ndarray:
    """Read DICOM file and return volume as numpy array (X, Y, Z)."""
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    if img.ndim == 3:
        img = np.transpose(img, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
    return img


def preprocess_for_inference(img: np.ndarray,
                             target_shape: tuple = TARGET_SHAPE) -> np.ndarray:
    """Full preprocessing pipeline for inference."""
    img = resample_volume(img, target_shape, order=1)
    img = clip_and_normalize(img)
    return img


def predict_volume(dicom_path: str,
                   model,
                   device: torch.device,
                   target_shape: tuple = TARGET_SHAPE,
                   threshold: float = 0.5):
    """
    Run segmentation inference on a DICOM file.

    Returns:
        img_preprocessed : preprocessed volume (for visualization)
        pred_binary      : binary mask (thresholded)
        prob_map         : probability map (raw sigmoid output)
    """
    model.eval()

    raw_img = load_dicom(dicom_path)
    img_preprocessed = preprocess_for_inference(raw_img, target_shape)
    img_tensor = torch.from_numpy(
        img_preprocessed[np.newaxis, np.newaxis]
    ).float().to(device)

    with torch.no_grad():
        prob_map = model(img_tensor).cpu().numpy()[0, 0]

    pred_binary = (prob_map > threshold).astype(np.uint8)

    return img_preprocessed, pred_binary, prob_map


def save_mask_as_nifti(mask: np.ndarray, output_path: str):
    """Save binary mask as NIfTI file."""
    nib.save(
        nib.Nifti1Image(mask.astype(np.float32), np.eye(4)),
        output_path
    )
