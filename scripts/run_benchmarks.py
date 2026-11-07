#!/usr/bin/env python3
"""
CLI Benchmark Suite for Problem Statement 3 (P3)
Runs quantitative comparative evaluation across all 4 operational profiles
(Mobile Ultra-Fast, Mobile Balanced, Studio Quality, Baseline DDIM).
Outputs structured Markdown table matching eval_P3_T01.md.
"""

import sys
import os
import time

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.diffusion_engine import DiffusionEngine


def run_comparative_benchmark(prompt: str = None):
    test_prompt = prompt or "A cinematic view of a futuristic Tokyo street in rain with glowing neon signs, 4k"
    print("\n" + "="*70)
    print(" 🚀 RUNNING COMPARATIVE EVALUATION BENCHMARK (P3)")
    print(f" Test Prompt: '{test_prompt}'")
    print("="*70 + "\n")

    engine = DiffusionEngine()
    modes = ["eco", "balanced", "quality", "baseline"]
    results = []

    for mode in modes:
        print(f"--> Benchmarking profile: '{mode}' ({engine.PROFILES[mode]['name']})...")
        t0 = time.time()
        res = engine.generate(
            prompt=test_prompt,
            mode=mode,
            seed=42,
            apply_classical_cv=True
        )
        elapsed = round(time.time() - t0, 2)
        metrics = res["metrics"]
        cv = metrics["cv_enhancements"]

        results.append({
            "mode": mode,
            "name": res["mode_name"],
            "steps": res["steps"],
            "latency": metrics["latency_seconds"],
            "speedup": metrics["speedup_vs_baseline"],
            "ram": metrics["ram_used_mb"],
            "clip": metrics["clip_score"],
            "sharpness": cv["metrics_after"]["sharpness_laplacian"] if "metrics_after" in cv else "N/A",
            "sharpness_gain": f"+{cv.get('sharpness_gain_pct', 0)}%"
        })

    # Render Markdown Table
    print("\n" + "="*70)
    print(" 📊 FINAL COMPARATIVE BENCHMARK RESULTS")
    print("="*70 + "\n")

    header = "| Profile | Steps | Latency (s) | Speedup vs Baseline | Peak RAM | CLIP Score | Sharpness Gain |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)
    for r in results:
        print(f"| **{r['name']}** | {r['steps']} | {r['latency']} s | **{r['speedup']}** | {r['ram']} MB | {r['clip']} | {r['sharpness_gain']} |")

    print("\nBenchmarking complete! Results conform to eval_P3_T01.md schema.\n")


if __name__ == "__main__":
    prompt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_comparative_benchmark(prompt_arg)
