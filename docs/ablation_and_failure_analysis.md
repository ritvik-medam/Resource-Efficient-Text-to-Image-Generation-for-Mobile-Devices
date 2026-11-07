# Ablation Study & Failure-Case Analysis

**Problem Statement:** P3 - Resource-Efficient Text-to-Image Generation for Mobile Devices  
**ProbID:** P3  
**Team ID:** T01  

---

## Part 1: Comprehensive Ablation Study

To evaluate the exact contribution of each architectural and algorithmic component, we conducted a systematic ablation across four configuration variants using a standardized benchmark prompt suite derived from MS-COCO captions.

### 1. Ablation Configurations

- **Config 1 (Baseline):** Conventional 50-step DDIM diffusion, FP32 precision, standard attention, no post-processing.
- **Config 2 (+Distillation):** 1-step Adversarial Diffusion Distillation (ADD), FP32 precision, no post-processing.
- **Config 3 (+Precision & Memory Optimization):** 1-step ADD, FP16 half-precision, attention slicing enabled, no post-processing.
- **Config 4 (Full Proposed Pipeline):** 1-step ADD, FP16 precision, attention slicing, plus full Classical CV post-processing (LAB CLAHE + Bilateral Sharpening + Vibrance correction).

### 2. Ablation Results Table

| Configuration Variant | Sampling Steps | Latency (s) | Latency Delta | Peak RAM (MB) | RAM Delta | CLIP Score | Sharpness (Laplacian) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1: Baseline DDIM** | 50 | 28.75 s | Reference | 3,250 MB | Reference | 0.92 | 238.1 |
| **Config 2: +Distillation** | 1 | 1.35 s | **-95.3%** | 2,850 MB | -12.3% | 0.83 | 124.3 |
| **Config 3: +FP16 & Slicing** | 1 | 0.82 s | **-97.1%** | 1,420 MB | **-56.3%** | 0.84 | 124.5 |
| **Config 4: Full Proposed** | 1 | 0.84 s | **-97.0%** | 1,420 MB | **-56.3%** | 0.84 | **184.2 (+48.2%)** |

### 3. Ablation Findings & Insights

1. **Step Reduction Drives Primary Latency Drop:** Transitioning from 50 iterative steps with CFG to a single-step ADD jump reduces latency from 28.75s to 1.35s (**21.3× speedup**).
2. **Half-Precision Unlocks Mobile RAM Compatibility:** Casting weights and activations to FP16 reduces peak memory by **56.3%** (from 3.25 GB to 1.42 GB), staying comfortably within the 2 GB headroom allocated to mobile apps.
3. **Classical CV Closes the Perceptual Quality Gap:** Single-step inference incurs a 47.7% drop in boundary sharpness (124.3 vs 238.1). Adding the classical CV pipeline recovers sharpness to **184.2 (+48.2% relative gain)** with only **20 ms computational overhead**, effectively bridging the visual gap without heavy neural compute.

---

## Part 2: Documented Failure Cases & Explainability

Transparent failure case analysis is a mandatory viva requirement. Below are three documented failure modes, their mathematical/architectural root causes, and their engineered mitigations.

---

### Failure Case 1: Legible Micro-Typography & Text Glyphs
* **Prompt Tested:** *"A cozy street coffee shop with a glowing neon sign that reads 'ESPRESSO BAR', night time, 4k"*
* **Observed Failure:** The generated storefront looks realistic, but the neon sign text contains garbled, pseudo-alphabetic strokes rather than the word "ESPRESSO".
* **Root Cause:**
  - Standard text encoders (CLIP ViT-L/14) operate at word/subword token level rather than character-level rendering.
  - In a 1-step trajectory ($N=1$), the model prioritizes coarse low-frequency image layout over high-frequency spatial coordination needed for character glyph formation.
* **Mitigation / Remedy:**
  - **Dynamic Fallback:** When the prompt contains quoted text strings, the engine automatically routes to **Profile 2 (Balanced 4-Step)** or **Profile 3 (Studio 8-Step)**.
  - **Classical Hybrid Overlay:** Future work integrates OCR-directed canvas inpainting for text regions.

---

### Failure Case 2: Attribute Swapping in Multi-Object Composition
* **Prompt Tested:** *"A small red ceramic mug sitting on top of a large green leather book on a wooden desk"*
* **Observed Failure:** The image displays a green mug on top of a red book (swapped color attributes between distinct entities).
* **Root Cause:**
  - Known as the **"binding problem"** in diffusion cross-attention maps. The cross-attention layers in single-step generation attend simultaneously to "red" and "green" without sufficient iterative refinement iterations to separate spatial activations.
* **Mitigation / Remedy:**
  - Preprocessing prompt structure with distinct entity clauses.
  - Using a slight guidance boost (CFG = 1.2 – 1.5 in Balanced Mode) to heighten attention contrast between competing tokens.

---

### Failure Case 3: Ringing / Halo Artifacts from Over-Sharpening Soft Media
* **Prompt Tested:** *"A mysterious tranquil lake in dense morning fog with distant faint pine trees, atmospheric"*
* **Observed Failure:** Subtle dark/light double-edges (ringing artifacts) appear along the silhouette of the pine trees against the foggy background.
* **Root Cause:**
  - The bilateral filtering unsharp mask assumes high edge contrast is always desirable. In a naturally soft, misty, or blurred scene, enforcing uniform bilateral gain amplifies gradient transitions unnaturally.
* **Mitigation / Remedy:**
  - Implemented **Adaptive Parameter Selection**: The classical CV pipeline first computes the image's global variance and edge density. If the image has an edge density below threshold $\tau_{\text{soft}}$, the sharpening gain $\beta$ is automatically throttled from $0.6$ down to $0.15$.
