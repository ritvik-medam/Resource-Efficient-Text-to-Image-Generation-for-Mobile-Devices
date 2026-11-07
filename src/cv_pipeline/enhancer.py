"""
Advanced Classical Computer Vision Post-Processing & Adaptive Enhancement Pipeline
Mandatory Technical Requirements:
1. Resolution optimization and high-fidelity resampling (Lanczos-4).
2. Adaptive histogram equalization (CLAHE) in LAB space.
3. Edge-preserving filtering (Bilateral & Guided filtering) + Unsharp Masking.
4. Adaptive parameter selection based on image characteristics (Luminance, Dynamic Range, Edge Energy).
5. No-reference image quality and artifact assessment (Laplacian variance, contrast, clipping).
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any, Optional


class ClassicalCVEnhancer:
    """
    Intelligent, Content-Adaptive Classical Computer Vision Pipeline.
    Compensates for low sampling steps (1-4 steps) by dynamically tuning
    contrast, edge definition, color vibrance, and filtering diffusion artifacts.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        sharpen_strength: float = 0.6,
        bilateral_sigma: float = 25.0,
        filter_type: str = "guided"  # 'guided' or 'bilateral'
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.sharpen_strength = sharpen_strength
        self.bilateral_sigma = bilateral_sigma
        self.filter_type = filter_type

    def pil_to_cv2(self, pil_img: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to OpenCV BGR numpy array."""
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_img: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR numpy array to PIL RGB image."""
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def compute_adaptive_parameters(self, bgr_img: np.ndarray) -> Dict[str, float]:
        """
        Adaptive Parameter Selection:
        Analyzes image distribution (mean luminance, dynamic range, edge energy)
        to dynamically compute optimal CLAHE clip limits, gamma, and sharpening weights.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())
        contrast_std = float(gray.std())

        # Measure high-frequency edge energy
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_energy = float(laplacian.var())

        # 1. Dynamic CLAHE clipLimit:
        # If low contrast or dark image, increase clipLimit to recover details
        if contrast_std < 45.0 or mean_lum < 85.0:
            adaptive_clip = min(3.5, 2.0 + (50.0 - contrast_std) * 0.03)
        elif contrast_std > 70.0:
            adaptive_clip = max(1.2, 2.0 - (contrast_std - 70.0) * 0.02)
        else:
            adaptive_clip = 2.0

        # 2. Dynamic Sharpening Strength:
        # If image is soft (typical in 1-step diffusion), boost sharpening.
        # If image already has strong edges, tone down to prevent ringing halos.
        if edge_energy < 90.0:
            adaptive_sharpen = min(0.85, 0.55 + (90.0 - edge_energy) * 0.003)
        elif edge_energy > 300.0:
            adaptive_sharpen = max(0.25, 0.55 - (edge_energy - 300.0) * 0.001)
        else:
            adaptive_sharpen = 0.55

        # 3. Dynamic Gamma:
        # Boost shadows for underexposed generations without blowing out highlights
        if mean_lum < 90.0:
            adaptive_gamma = 1.12
        elif mean_lum > 170.0:
            adaptive_gamma = 0.95
        else:
            adaptive_gamma = 1.04

        return {
            "mean_luminance": round(mean_lum, 1),
            "contrast_std": round(contrast_std, 1),
            "edge_energy": round(edge_energy, 1),
            "adaptive_clip_limit": round(adaptive_clip, 2),
            "adaptive_sharpen_strength": round(adaptive_sharpen, 2),
            "adaptive_gamma": round(adaptive_gamma, 2)
        }

    def adaptive_contrast_enhancement(
        self,
        bgr_img: np.ndarray,
        clip_limit: Optional[float] = None
    ) -> np.ndarray:
        """
        Enhance dynamic range and local contrast without introducing halo artifacts.
        Operates on the L channel of LAB color space to preserve color fidelity.
        """
        effective_clip = clip_limit if clip_limit is not None else self.clahe_clip_limit
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=effective_clip,
            tileGridSize=self.clahe_grid_size
        )
        cl = clahe.apply(l)

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

        # Non-linear gamma adjustment for shadows / midtone richness
        inv_gamma = 1.0 / max(0.1, gamma)
        v = 255.0 * ((v / 255.0) ** inv_gamma)
        v = np.clip(v, 0, 255)

        hsv_corrected = cv2.merge((h, s, v)).astype(np.uint8)
        return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    def fast_guided_filter(
        self,
        I: np.ndarray,
        p: np.ndarray,
        radius: int = 8,
        eps: float = 0.04
    ) -> np.ndarray:
        """
        Fast Guided Filter in O(N) runtime.
        Acts as an edge-preserving smoothing filter that avoids gradient-reversal ringing.
        """
        I_f = I.astype(np.float32) / 255.0
        p_f = p.astype(np.float32) / 255.0

        mean_I = cv2.boxFilter(I_f, -1, (radius, radius))
        mean_p = cv2.boxFilter(p_f, -1, (radius, radius))
        mean_Ip = cv2.boxFilter(I_f * p_f, -1, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I_f * I_f, -1, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        q = mean_a * I_f + mean_b
        return np.clip(q * 255.0, 0, 255).astype(np.uint8)

    def edge_aware_sharpening(
        self,
        bgr_img: np.ndarray,
        strength: Optional[float] = None,
        use_guided: bool = True
    ) -> np.ndarray:
        """
        Edge-preserving detail recovery.
        Uses Guided Filtering (or Bilateral Filtering) to extract high-frequency detail
        and re-inject it with controlled gain, producing crisp, photorealistic edges.
        """
        effective_strength = strength if strength is not None else self.sharpen_strength
        if effective_strength <= 0.0:
            return bgr_img

        if use_guided or self.filter_type == "guided":
            # Fast O(N) Guided Filter base decomposition
            smoothed = self.fast_guided_filter(bgr_img, bgr_img, radius=6, eps=0.03)
        else:
            # Bilateral filter fallback
            smoothed = cv2.bilateralFilter(
                bgr_img, d=5,
                sigmaColor=self.bilateral_sigma,
                sigmaSpace=self.bilateral_sigma
            )

        # Extract high-frequency edge detail residual
        detail = cv2.subtract(bgr_img, smoothed)

        # Inject detail with gain
        sharpened = cv2.addWeighted(bgr_img, 1.0, detail, effective_strength, 0)
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
        Compute quantitative no-reference image quality and artifact indicators.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Edge Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = float(laplacian.var())

        # Global contrast
        contrast_score = float(gray.std())

        # Dynamic range
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)

        # Saturation & Overexposure clipping check
        overexposed_ratio = float(np.sum(gray >= 254) / gray.size)
        underexposed_ratio = float(np.sum(gray <= 2) / gray.size)

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
        adaptive_tuning: bool = True,
        target_size: Tuple[int, int] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Run the complete classical CV post-processing pipeline.
        Supports automated adaptive tuning based on real-time image properties.
        """
        raw_bgr = self.pil_to_cv2(pil_image)
        metrics_before = self.assess_quality_and_artifacts(raw_bgr)

        # Adaptive parameters computation
        adaptive_params = self.compute_adaptive_parameters(raw_bgr)
        clip_limit = adaptive_params["adaptive_clip_limit"] if adaptive_tuning else self.clahe_clip_limit
        sharpen_strength = adaptive_params["adaptive_sharpen_strength"] if adaptive_tuning else self.sharpen_strength
        gamma = adaptive_params["adaptive_gamma"] if adaptive_tuning else 1.05

        curr = raw_bgr.copy()

        # Step 1: Adaptive CLAHE Contrast Enhancement
        if enable_clahe:
            curr = self.adaptive_contrast_enhancement(curr, clip_limit=clip_limit)

        # Step 2: Color Vibrance & Gamma Correction
        if enable_vibrance:
            curr = self.color_vibrance_correction(curr, vibrance_factor=1.14, gamma=gamma)

        # Step 3: Edge-Aware Guided Sharpening
        if enable_sharpening:
            curr = self.edge_aware_sharpening(curr, strength=sharpen_strength, use_guided=True)

        # Step 4: Resolution / Resampling Optimization
        if target_size is not None:
            curr = self.resolution_optimization(curr, target_size=target_size)

        metrics_after = self.assess_quality_and_artifacts(curr)
        enhanced_pil = self.cv2_to_pil(curr)

        report = {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "adaptive_parameters": adaptive_params,
            "filter_applied": "Guided Filter (O(N) Edge Preserving)" if self.filter_type == "guided" else "Bilateral",
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
