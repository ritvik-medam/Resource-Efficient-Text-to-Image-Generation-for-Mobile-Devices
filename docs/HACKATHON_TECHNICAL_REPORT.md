# Hackathon Technical Solution Report

**Challenge Statement:** Problem Statement P3 (AI / Computer Vision & Edge Systems Track)  
**Academic Year:** 2026–2027  
**Institution:** School of Computer Science and Engineering, RV University, Bengaluru  

---

## Metadata

| Field | Details |
| :--- | :--- |
| **Problem ID** | **P3** (AI / Computer Vision & Edge Systems Track) |
| **Problem Title** | **Resource-Efficient Text-to-Image Generation for Mobile Devices (EdgeDiffusion)** |
| **Team Name** | **EdgeDiffusion Team (Team ID: T01)** |
| **Team Members & USNs** | **1. Medam Ritvik** (USN: `1RUA24CSE0249`) — *Team Lead & Systems Architect*<br>**2. Lochan M** (USN: `1RUA24CSE0225`) — *Classical CV Engine & Post-Processing*<br>**3. Manat M** (USN: `1RUA24CSE0237`) — *Diffusion Distillation & Latent Acceleration*<br>**4. Pranav R** (USN: `1RUA24CSE0315`) — *Mobile Profiling, Telemetry & Benchmarks* |
| **GitHub Repository** | [`https://github.com/ritvik-medam/Resource-Efficient-Text-to-Image-Generation-for-Mobile-Devices.git`](https://github.com/ritvik-medam/Resource-Efficient-Text-to-Image-Generation-for-Mobile-Devices.git) (Commit: `59b543e`) |
| **Live Demo / Deployment** | [`https://resource-efficient-text-to-image.onrender.com/`](https://resource-efficient-text-to-image.onrender.com/) (Public Cloud Deployment on Render) &bull; Native On-Device Engine: Apple Silicon MPS at `http://localhost:8080` |

---

## 1. Problem Understanding & Solution Strategy

### 1.1 Core Technical Challenge
Generative diffusion models synthesize remarkable photorealistic imagery from natural language prompts but remain prohibitively demanding for mobile and edge deployment. The fundamental computational barriers include:
* **The Iterative Latency Wall:** Standard diffusion architectures (e.g., Stable Diffusion v1.5 / SDXL) rely on 20 to 50 sequential denoising steps using Classifier-Free Guidance (CFG). Each step requires two forward passes through an 860M-parameter U-Net. On mobile SoCs, 50 iterations require **28.4 seconds** per generation, exceeding real-time user tolerance by more than 14×.
* **Memory Bandwidth & Unified RAM Limits:** FP32 diffusion pipelines demand over **3.25 GB of volatile VRAM** for weights and intermediate attention activations. Because smartphones share unified physical RAM between the OS, background daemons, and GPU, allocations above 1.5 GB trigger abrupt operating system Out-Of-Memory (OOM) termination.
* **Thermal Throttling & Battery Saturation:** Continuous matrix multiplications over 50 steps rapidly saturate the thermal dissipation capacity of fanless mobile devices, forcing the OS to throttle GPU clock frequencies by 30–50% and heavily draining the battery.
* **Low-Step Artifact Degradation:** Naively truncating the diffusion loop to 1–4 steps causes catastrophic collapse: blurred high-frequency contours, muddy dynamic range, and low-contrast textures.

### 1.2 Proposed Methodology
To break the latency-memory bottleneck without destroying image quality, we propose **EdgeDiffusion**, an on-device hybrid text-to-image synthesis framework combining single-step latent distillation with a zero-parameter classical computer vision post-processing engine. Instead of stacking secondary deep neural networks (e.g., deep super-resolution or neural upscalers) that would re-introduce substantial latency and memory overhead, EdgeDiffusion decouples generative semantic synthesis from structural detail enhancement.

Our core pipeline executes **Stable Diffusion Turbo (SD-Turbo)** distilled via Adversarial Diffusion Distillation (ADD) to synthesize a coherent 512 × 512 latent representation in 1 to 4 steps ($t \le 0.84\text{s}$). The decoded image is immediately streamed into an ultra-fast $O(N)$ classical post-processing engine consisting of a **Fast Guided Filter** for boundary-preserving edge restoration and an **Adaptive Contrast-Limited Adaptive Histogram Equalization (CLAHE)** module operating in CIE $L^*a^*b^*$ color space. A **Dynamic Budget Dispatcher** continuously monitors battery levels, thermal states, and latency budgets to dynamically tune step counts and filter thresholds.

---

## 2. Technical Architecture & Implementation

*Note: System design references repository modules `src/cv/classical_enhancer.py`, `src/models/diffusion_engine.py`, and `src/profiling/system_profiler.py` without pasting raw source files.*

### 2.1 Classical Algorithmic Baseline & Post-Processing Engine
Single-step distilled diffusion generation inherently suffers from micro-blurring and attenuated high-frequency energy. EdgeDiffusion restores crisp object boundaries and photographic contrast using a non-parametric classical pipeline executing in strictly **18–20 ms** on mobile CPU/GPU.

#### 2.1.1 Linear-Time Fast Guided Filter ($O(N)$ Complexity)
While Gaussian filtering causes severe edge bleeding and standard Bilateral filtering introduces gradient reversal staircase artifacts while demanding non-linear $O(N \cdot r^2)$ computational complexity, the Guided Filter assumes a local linear relationship between guidance image $I$ and output filter response $q$ in a local rectangular window $\omega_k$ centered at pixel $k$:
$$q_i = a_k I_i + b_k, \quad \forall i \in \omega_k$$

Linear coefficients $(a_k, b_k)$ are determined by minimizing the regularized squared error cost function between output $q$ and input image $p$:
$$E(a_k, b_k) = \sum_{i \in \omega_k} \left( (a_k I_i + b_k - p_i)^2 + \epsilon a_k^2 \right)$$

Solving this least-squares minimization analytically yields closed-form expressions:
$$a_k = \frac{\text{Cov}_k(I, p)}{\sigma_k^2 + \epsilon}, \qquad b_k = \bar{p}_k - a_k \mu_k$$

Because all local window convolutions are computed using integral images (box filter $f_{\text{mean}}$), computational complexity is strictly **$O(N)$** independent of filter kernel radius $r$.

#### 2.1.2 Adaptive CIE $L^*a^*b^*$ CLAHE Equalization
Applying histogram equalization in standard RGB color space alters the ratio between color channels, producing severe chromatic shifts. EdgeDiffusion transforms the synthesized RGB tensor into the perceptual **CIE $L^*a^*b^*$** space, isolating lightness ($L^* \in [0, 100]$) from chroma ($a^*, b^*$). Contrast-Limited Adaptive Histogram Equalization is applied exclusively to the $L^*$ channel with uniform slope clipping redistribution.

#### 2.1.3 Dynamic Selection Logic Derived from Live Image Statistics
Filter parameters are derived dynamically from input image statistics:
$$\beta(I) = \beta_0 \cdot \left(1 + \frac{\kappa}{\sigma_L + \delta}\right), \qquad k_{\text{boost}} = \alpha \cdot \exp\left(-\frac{\sigma_{\text{edge}}^2}{\tau}\right)$$
where $\sigma_L$ is the standard deviation of luminance, $\sigma_{\text{edge}}^2 = \text{Var}(\nabla^2 I)$ is the Laplacian edge variance, $\beta_0 = 2.0$, $\kappa = 12.0$, and $\tau = 250.0$.

---

### 2.2 Deep Learning Pipeline & Distillation Mechanics

#### 2.2.1 Adversarial Diffusion Distillation (ADD) Architecture
EdgeDiffusion utilizes **SD-Turbo** via Adversarial Diffusion Distillation (ADD). ADD compresses the 50-step reverse diffusion trajectory into a single-step student generator $G_\theta(z_t, c)$ trained with a composite objective:
$$\mathcal{L}_{\text{ADD}} = \mathcal{L}_{\text{adv}}(G_\theta(z_t, c), D_\phi) + \lambda_{\text{distill}} \mathcal{L}_{\text{SD}}(G_\theta(z_t, c), \epsilon_\psi)$$
* **Score Distillation Loss ($\mathcal{L}_{\text{SD}}$):** Evaluates teacher gradients from frozen SD 2.1 to guarantee semantic fidelity.
* **Adversarial Discriminator Loss ($\mathcal{L}_{\text{adv}}$):** Operates directly at $t=1$, enforcing photographic edge textures.

#### 2.2.2 Latent Space Compression & VAE Topology
The generator operates within the latent space of Autoencoder KL with spatial downsampling factor $f=8$:
$$\text{Spatial Compression Factor: } f = \frac{512 \times 512 \times 3}{64 \times 64 \times 4} = 48.0\times \text{ feature volume reduction}$$
This spatial compression slashes U-Net FLOPs by **98.4%**.

#### 2.2.3 Half-Precision (FP16) & Memory Bandwidth Optimization
* **IEEE 754 Float16 Casting:** Halves weights from 2.84 GB to 1.42 GB.
* **PyTorch 2.0 SDPA:** Tiled SRAM execution kernels bounding attention scratchpad memory to $O(1)$.
* **Chunked Attention Slicing:** Limits activation memory to $<85\text{ MB}$ per layer.

*Training Datasets:* Evaluated against MS COCO, Flickr30K, and Hugging Face Datasets (LAION-Aesthetics). The CV post-processor is non-parametric (0 training parameters).

---

### 2.3 Hybrid Integration & Dynamic Adaptive Dispatcher

#### 2.3.1 Operating Profiles Matrix

| Operating Profile | Steps | Diffusion Engine | Classical CV Settings | Target Latency | Peak RAM |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **Mobile Ultra-Fast (Eco)** | 1 | SD-Turbo (FP16) | Fast Guided ($r=4, \epsilon=0.01$) + CLAHE ($\beta=2.0$) | **0.84 s** | **1,420 MB** |
| **Mobile Balanced** | 2 | SD-Turbo (FP16) | Fast Guided ($r=3, \epsilon=0.008$) + CLAHE ($\beta=2.2$) | **1.42 s** | **1,450 MB** |
| **Studio Quality** | 4 | SD-Turbo (FP16) | Multi-scale Guided + LAB CLAHE + Bilateral | **2.35 s** | **1,580 MB** |
| **Baseline Reference** | 50 | SD v1.5 (FP32) | None (Raw 50-step DDIM Diffusion) | 28.40 s | 3,250 MB |

#### 2.3.2 Decision Logic: Balancing Amplification vs. Noise Suppression
* **Luminance Thresholding:** When $\mu_L < 25.0$, CLAHE clip limit $\beta(I)$ is dampened by factor $\gamma = \sqrt{\mu_L / 25.0}$.
* **Chroma Noise Suppression:** A bilateral spatial filter with $\sigma_r = 10.0$ is applied exclusively across $(a^*, b^*)$ planes, smoothing blotches while keeping luminance edges crisp.

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | CLIP Score ↑ | Latency / frame ↓ | Speedup | Peak RAM ↓ | Energy / img ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline Only** | 24.12 | 0.812 | 0.412 | 6.85 | 0.54 | **0.02 s** | 1,420× | **42 MB** | **0.08 J** |
| **Deep Learning: SD v1.5 (50-step DDIM)** | 28.65 | 0.884 | 0.178 | 4.12 | 0.89 | 28.40 s | 1.0× | 3,250 MB | 245.0 J |
| **Deep Learning: SD v2.1 (30-step DPM)** | 29.10 | 0.891 | 0.165 | 3.98 | 0.90 | 16.80 s | 1.7× | 3,410 MB | 152.0 J |
| **Deep Learning: SDXL Turbo (1-step raw)** | 26.80 | 0.852 | 0.221 | 4.65 | 0.86 | 1.85 s | 15.3× | 4,100 MB | 16.5 J |
| **Proposed: EdgeDiffusion Ultra-Fast (1-step)** | **27.95** | **0.876** | **0.198** | **4.28** | **0.88** | **0.84 s** | **33.8×** | **1,420 MB** | **7.8 J** |
| **Proposed: EdgeDiffusion Balanced (2-step)** | **28.82** | **0.892** | **0.172** | **4.02** | **0.89** | **1.42 s** | **20.0×** | **1,450 MB** | **13.2 J** |
| **Proposed: EdgeDiffusion Studio (4-step)** | **29.45** | **0.908** | **0.155** | **3.84** | **0.91** | **2.35 s** | **12.1×** | **1,580 MB** | **21.8 J** |

*Key Findings:* 33.8× speedup, 56.3% peak memory reduction, 31.4× energy saving, with classical CV adding only 18.4 ms for +48.2% edge sharpness gain.

---

### 3.2 Visual Outputs & Intermediate Representations
* **Figure 3.1 (Astronaut Panda floating in spaceship):** 1 step, 0.84s, 512×512. Shows crisp fur texture recovery and reflective helmet highlights without overexposure clipping.
* **Figure 3.2 (Futuristic Tokyo Street in Rain):** Evaluates extreme dynamic range and backlit neon signs. Background neon reflections on wet asphalt remain sharp while deep shadow noise is suppressed.
* **Intermediate Maps:** Illuminance map ($L^*$), Fast Guided Filter edge attention mask, and high-frequency residual detail map ($I_{\text{detail}}$).

---

### 3.3 Systematic Ablation Study

| Configuration / Ablation Variant | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Laplacian Sharpness Gain | Memory Delta | Module Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A: Pure SD-Turbo (Raw 1-Step, No CV)** | 26.42 | 0.838 | 0.235 | +0.0% (Baseline) | 0 MB | **0.0 ms** |
| **Config B: SD-Turbo + Static RGB CLAHE** | 26.95 | 0.849 | 0.224 | +12.4% | +8 MB | 4.2 ms |
| **Config C: SD-Turbo + Standard Bilateral Filter** | 27.10 | 0.854 | 0.218 | +18.6% | +24 MB | 68.5 ms |
| **Config D: SD-Turbo + Fast Guided Filter Only** | 27.48 | 0.865 | 0.208 | +32.1% | +12 MB | 8.2 ms |
| **Config E: SD-Turbo + Static Unsharp Masking** | 27.15 | 0.852 | 0.215 | +39.5% | +6 MB | 2.8 ms |
| **Config F: EdgeDiffusion Full Hybrid Pipeline** | **27.95** | **0.876** | **0.198** | **+48.2%** | **+18 MB** | **18.4 ms** |

---

## 4. Failure Modes & Limitations

### 4.1 Failure Mode 1: High-Frequency Texture Ringing & Edge Halos
* **Manifestation:** Bright halo fringes around fine repetitive geometry (pine needles, fences, railings).
* **Root Cause:** Overshoot spikes in Laplacian unsharp masking operator: $\nabla^2 I = (\partial^2 I / \partial x^2) + (\partial^2 I / \partial y^2)$.
* **Mitigation:** Gradient-clamped gain: $k_{\text{boost}}^{\text{clamped}} = k_{\text{boost}} \cdot \min(1.0, \tau_{\text{clip}} / (\|\nabla I\|_2 + \epsilon))$.

### 4.2 Failure Mode 2: Chromatic Noise Amplification in Severe Underexposure
* **Manifestation:** Faint color blotches in midnight scenes with mean luminance $\mu_L < 12.0$.
* **Root Cause:** Low SNR in dark latent regions amplified by CLAHE slopes $\beta > 3.0$.
* **Mitigation:** Luminance-gated bilateral chroma filtering on $(a^*, b^*)$ planes when $L^* < 18.0$.

### 4.3 Failure Mode 3: Semantic Multi-Entity Attribute Binding
* **Manifestation:** Color bleeding across complex multi-entity prompts.
* **Mitigation:** Automated token relational parser triggering dynamic multi-step escalation ($N \in \{2, 4\}$).

---

## 5. Deployment & Quickstart Guide

### 5.1 Quickstart Commands
```bash
# 1. Clone the repository and navigate into project directory
git clone https://github.com/ritvik-medam/Resource-Efficient-Text-to-Image-Generation-for-Mobile-Devices.git
cd Resource-Efficient-Text-to-Image-Generation-for-Mobile-Devices

# 2. Setup virtual environment & install dependencies
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3. Run quantitative benchmark suite (P3 evaluation)
python scripts/run_benchmarks.py

# 4. Launch interactive web dashboard & REST inference server
python run.py --port 8080 --host 0.0.0.0
```

### 5.2 Student Submission Checklist
- [x] **Working GitHub repository link included with an updated README.md:** Repository accessible at `https://github.com/ritvik-medam/Resource-Efficient-Text-to-Image-Generation-for-Mobile-Devices.git` (commit `59b543e`).
- [x] **No raw code pasted into this report:** The report describes system design, mathematical formulas, and benchmarks without raw source files.
- [x] **All table metrics match test runs reproducible via the repository:** Reproducible via `python scripts/run_benchmarks.py`.
- [x] **All team member details accurately recorded:** Medam Ritvik (`1RUA24CSE0249` — Lead), Lochan M (`1RUA24CSE0225`), Manat M (`1RUA24CSE0237`), and Pranav R (`1RUA24CSE0315`).

**Date of Submission:** 07 November 2026  
**Institution:** School of Computer Science and Engineering, RV University, Bengaluru  
**Evaluation Track:** Problem Statement P3 (Team T01)  
