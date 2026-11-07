"""
Classical Computer Vision Post-Processing & Quality Assessment Pipeline
Mandatory Requirements:
1. Resolution optimization and high-fidelity resizing.
2. Adaptive histogram equalization (CLAHE) & color correction.
3. Edge-aware post-processing (Bilateral / Guided filtering + Unsharp Masking).
4. Artifact detection (blur, saturation clipping, edge noise).
5. No-reference image quality assessment (Sharpness, Contrast, Dynamic Range, Naturalness).
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any


class ClassicalCVEnhancer:
    """
    Intelligent Classical CV Pipeline to enhance and validate lightweight diffusion outputs.
    Compensates for low sampling steps (1-4 steps) by recovering edge sharpness,
    enhancing local contrast, correcting color balance, and filtering diffusion artifacts.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        sharpen_strength: float = 0.6,
        bilateral_sigma: float = 25.0
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.sharpen_strength = sharpen_strength
        self.bilateral_sigma = bilateral_sigma

    def pil_to_cv2(self, pil_img: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to OpenCV BGR numpy array."""
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_img: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR numpy array to PIL RGB image."""
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def adaptive_contrast_enhancement(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Enhance dynamic range and local contrast without introducing halo artifacts.
        Operates on the L channel of LAB color space to preserve color fidelity.
        """
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )
        cl = clahe.apply(l)

        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def color_vibrance_correction(
        self,
        bgr_img: np.ndarray,
        vibrance_factor: float = 1.15,
        gamma: float = 1.05
    ) -> np.ndarray:
        """
        Correct washed-out colors common in ultra-fast low-step diffusion models.
        Selectively boosts lower-saturated areas (vibrance) without clipping saturated highlights.
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Selective vibrance: boost desaturated pixels more than already saturated ones
        s = s * (1.0 + (vibrance_factor - 1.0) * (1.0 - s / 255.0))
        s = np.clip(s, 0, 255)

        # Gamma adjustment for shadows / midtone richness
        inv_gamma = 1.0 / gamma
        v = 255.0 * ((v / 255.0) ** inv_gamma)
        v = np.clip(v, 0, 255)

        hsv_corrected = cv2.merge((h, s, v)).astype(np.uint8)
        return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    def edge_aware_sharpening(
        self,
        bgr_img: np.ndarray,
        strength: float = None
    ) -> np.ndarray:
        """
        Edge-preserving detail recovery.
        Uses bilateral filtering to separate fine details from edge structures,
        preventing noise amplification while recovering crisp object boundaries.
        """
        if strength is None:
            strength = self.sharpen_strength

        if strength <= 0.0:
            return bgr_img

        # 1. Bilateral filter to smooth flat regions while retaining sharp edges
        smoothed = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=self.bilateral_sigma, sigmaSpace=self.bilateral_sigma)

        # 2. Extract edge detail map
        detail = cv2.subtract(bgr_img, smoothed)

        # 3. Add weighted edge detail back to the original image
        sharpened = cv2.addWeighted(bgr_img, 1.0, detail, strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def resolution_optimization(
        self,
        bgr_img: np.ndarray,
        target_size: Tuple[int, int] = None,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """
        Mobile-optimized high-quality resampling using Lanczos-4 interpolation
        to prevent aliasing or moiré patterns on retina/mobile displays.
        """
        h, w = bgr_img.shape[:2]
        if target_size is not None:
            new_w, new_h = target_size
        elif scale_factor != 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
        else:
            return bgr_img

        return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def assess_quality_and_artifacts(self, bgr_img: np.ndarray) -> Dict[str, float]:
        """
        Compute quantitative no-reference image quality and artifact indicators:
        - Sharpness: Variance of Laplacian (higher = crisper edges).
        - Contrast: Standard deviation of grayscale pixel intensities.
        - Dynamic Range: Percentile span [p1, p99].
        - Clipping Artifacts: Fraction of blown-out or crushed pixels.
        - Blur Indicator: True if sharpness below threshold.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 1. Edge Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = float(laplacian.var())

        # 2. Global contrast
        contrast_score = float(gray.std())

        # 3. Dynamic range
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)

        # 4. Saturation & Overexposure clipping check
        overexposed_ratio = float(np.sum(gray >= 254) / gray.size)
        underexposed_ratio = float(np.sum(gray <= 2) / gray.size)

        # 5. Composite Naturalness Score (0 - 100 normalized)
        # Calibrated based on typical photo distributions
        norm_sharpness = min(100.0, (sharpness_score / 450.0) * 100.0)
        norm_contrast = min(100.0, (contrast_score / 65.0) * 100.0)
        composite_score = float(0.5 * norm_sharpness + 0.5 * norm_contrast)

        return {
            "sharpness_laplacian": round(sharpness_score, 2),
            "contrast_std": round(contrast_score, 2),
            "dynamic_range": round(dynamic_range, 2),
            "overexposed_pct": round(overexposed_ratio * 100.0, 2),
            "underexposed_pct": round(underexposed_ratio * 100.0, 2),
            "composite_quality_score": round(composite_score, 2),
            "is_blurry": sharpness_score < 70.0,
            "has_clipping": (overexposed_ratio > 0.08) or (underexposed_ratio > 0.08)
        }

    def process(
        self,
        pil_image: Image.Image,
        enable_clahe: bool = True,
        enable_vibrance: bool = True,
        enable_sharpening: bool = True,
        target_size: Tuple[int, int] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Run the complete classical CV post-processing pipeline.
        Returns the enhanced PIL Image and a diagnostic dictionary with before/after metrics.
        """
        raw_bgr = self.pil_to_cv2(pil_image)
        metrics_before = self.assess_quality_and_artifacts(raw_bgr)

        curr = raw_bgr.copy()

        # Step 1: Adaptive CLAHE Contrast Enhancement
        if enable_clahe:
            curr = self.adaptive_contrast_enhancement(curr)

        # Step 2: Color Vibrance & Gamma Correction
        if enable_vibrance:
            curr = self.color_vibrance_correction(curr)

        # Step 3: Edge-Aware Bilateral Sharpening
        if enable_sharpening:
            curr = self.edge_aware_sharpening(curr)

        # Step 4: Resolution / Resampling Optimization
        if target_size is not None:
            curr = self.resolution_optimization(curr, target_size=target_size)

        metrics_after = self.assess_quality_and_artifacts(curr)
        enhanced_pil = self.cv2_to_pil(curr)

        report = {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "sharpness_gain_pct": round(
                ((metrics_after["sharpness_laplacian"] - metrics_before["sharpness_laplacian"])
                 / max(metrics_before["sharpness_laplacian"], 1e-4)) * 100.0,
                2
            ),
            "contrast_gain_pct": round(
                ((metrics_after["contrast_std"] - metrics_before["contrast_std"])
                 / max(metrics_before["contrast_std"], 1e-4)) * 100.0,
                2
            )
        }

        return enhanced_pil, report
