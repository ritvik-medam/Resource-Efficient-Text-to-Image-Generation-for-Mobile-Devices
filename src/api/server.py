"""
FastAPI Server for Resource-Efficient Text-to-Image Generation (Problem Statement 3)
Serves:
- REST API for optimized text-to-image inference with classical CV post-processing.
- Comparative benchmark runner.
- System diagnostics & hardware monitor.
- Static Single-Page Web Application.
"""

import io
import base64
import os
import psutil
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.diffusion_engine import DiffusionEngine

app = FastAPI(
    title="Resource-Efficient Text-to-Image Generation (P3)",
    description="Intelligent, Low-Latency Text-to-Image Generation with Classical CV Post-Processing for Mobile Devices",
    version="1.0.0"
)

# Enable CORS for external access / mobile testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine (lazy or on startup)
engine = DiffusionEngine()


class GenerationRequest(BaseModel):
    prompt: str = Field(..., example="A cute red panda wearing a tiny astronaut helmet, cinematic lighting, 4k")
    negative_prompt: str = Field(default="")
    mode: str = Field(default="balanced", description="balanced | auto | quality | eco | baseline")
    auto_enhance_prompt: bool = Field(default=True, description="Enrich prompt with photographic quality cues")
    custom_steps: Optional[int] = Field(default=None, ge=1, le=100)
    custom_guidance: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None)
    apply_classical_cv: bool = Field(default=True)
    enable_clahe: bool = Field(default=True)
    enable_vibrance: bool = Field(default=True)
    enable_sharpening: bool = Field(default=True)
    adaptive_tuning: bool = Field(default=True, description="Auto-adjust CLAHE/sharpening based on image characteristics")


def pil_to_base64(img) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


@app.get("/api/system-status")
def get_system_status():
    """Returns hardware acceleration state, memory consumption, and active model."""
    device = engine.device
    mps_available = False
    cuda_available = False
    if not (os.environ.get("RENDER") or os.environ.get("DEVICE") == "cpu"):
        try:
            import torch
            mps_available = hasattr(torch, "mps") and torch.backends.mps.is_available()
            cuda_available = torch.cuda.is_available()
        except Exception:
            pass

    mem = psutil.virtual_memory()

    return {
        "status": "online",
        "device": device,
        "hardware_acceleration": {
            "apple_silicon_mps": mps_available,
            "nvidia_cuda": cuda_available,
            "cpu_fallback": not (mps_available or cuda_available)
        },
        "model_id": engine.model_id,
        "is_real_weights_loaded": engine.is_real_model_loaded,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "active_profiles": list(engine.PROFILES.keys())
    }


@app.post("/api/generate")
def generate_image(req: GenerationRequest):
    """Executes on-device text-to-image generation and classical CV enhancement."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        result = engine.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            mode=req.mode,
            auto_enhance_prompt=req.auto_enhance_prompt,
            custom_steps=req.custom_steps,
            custom_guidance=req.custom_guidance,
            seed=req.seed,
            apply_classical_cv=req.apply_classical_cv,
            enable_clahe=req.enable_clahe,
            enable_vibrance=req.enable_vibrance,
            enable_sharpening=req.enable_sharpening,
            adaptive_tuning=req.adaptive_tuning
        )

        # Convert images to base64
        raw_b64 = pil_to_base64(result["raw_image"])
        enhanced_b64 = pil_to_base64(result["enhanced_image"])

        return {
            "success": True,
            "prompt": result["prompt"],
            "effective_prompt": result["effective_prompt"],
            "negative_prompt": result["negative_prompt"],
            "prompt_enhancement": result["prompt_enhancement"],
            "mode": result["mode"],
            "mode_name": result["mode_name"],
            "steps": result["steps"],
            "guidance_scale": result["guidance_scale"],
            "seed": result["seed"],
            "resolution": result["resolution"],
            "model_used": result["model_used"],
            "device": result["device"],
            "raw_image_base64": raw_b64,
            "enhanced_image_base64": enhanced_b64,
            "metrics": result["metrics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/benchmark")
def run_benchmark(sample_prompt: Optional[str] = None):
    """
    Runs automated comparative benchmark across all 4 modes
    (Eco 1-step, Balanced 4-step, Quality 8-step, Baseline 50-step).
    """
    benchmark_prompt = sample_prompt or "A cinematic view of a futuristic Tokyo street in rain with glowing neon signs, 4k"
    modes = ["eco", "balanced", "quality", "baseline"]
    results = []

    for m in modes:
        res = engine.generate(
            prompt=benchmark_prompt,
            mode=m,
            seed=42,
            apply_classical_cv=True
        )
        results.append({
            "mode": m,
            "mode_name": res["mode_name"],
            "steps": res["steps"],
            "latency_seconds": res["metrics"]["latency_seconds"],
            "ram_used_mb": res["metrics"]["ram_used_mb"],
            "vram_used_mb": res["metrics"]["vram_used_mb"],
            "clip_score": res["metrics"]["clip_score"],
            "speedup_vs_baseline": res["metrics"]["speedup_vs_baseline"],
            "sharpness": res["metrics"]["cv_enhancements"]["metrics_after"]["sharpness_laplacian"],
            "contrast": res["metrics"]["cv_enhancements"]["metrics_after"]["contrast_std"]
        })

    return {
        "prompt": benchmark_prompt,
        "results": results,
        "summary": "Benchmarking completed successfully across all 4 operational tiers."
    }


# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

