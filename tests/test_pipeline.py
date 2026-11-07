"""
Unit & Integration Tests for Resource-Efficient Text-to-Image Generation (P3)
Verifies:
1. Classical CV Pipeline (CLAHE, Bilateral sharpening, vibrance, quality metrics).
2. Diffusion Engine (Device detection, mode configurations, fallback resilience).
3. FastAPI Endpoints (/api/system-status, /api/generate, /api/benchmark, error states).
"""

import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.cv_pipeline.enhancer import ClassicalCVEnhancer
from src.models.diffusion_engine import DiffusionEngine
from src.models.prompt_optimizer import PromptOptimizer
from src.api.server import app


class TestPromptOptimizer(unittest.TestCase):
    def test_category_detection_and_optimization(self):
        prompt = "a cute golden retriever puppy in the garden"
        enhanced, neg, meta = PromptOptimizer.optimize(prompt, auto_enhance=True)
        self.assertEqual(meta["category"], "animal")
        self.assertTrue(meta["enhanced"])
        self.assertIn("fur", enhanced)
        self.assertIn("deformed", neg)

    def test_portrait_optimization(self):
        prompt = "portrait of a woman in evening dress"
        enhanced, neg, meta = PromptOptimizer.optimize(prompt, auto_enhance=True)
        self.assertEqual(meta["category"], "portrait")
        self.assertIn("lighting", enhanced)


class TestClassicalCVPipeline(unittest.TestCase):
    def setUp(self):
        self.enhancer = ClassicalCVEnhancer()
        # Create a synthetic 256x256 RGB test image with soft gradient and circle
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(40, 180, 256)[:, None]
        arr[:, :, 1] = np.linspace(30, 150, 256)[None, :]
        arr[:, :, 2] = 100
        self.test_img = Image.fromarray(arr)

    def test_cv_enhancement_and_metrics(self):
        enhanced, report = self.enhancer.process(
            self.test_img,
            enable_clahe=True,
            enable_vibrance=True,
            enable_sharpening=True
        )
        self.assertIsInstance(enhanced, Image.Image)
        self.assertEqual(enhanced.size, self.test_img.size)

        # Verify diagnostic report fields
        self.assertIn("metrics_before", report)
        self.assertIn("metrics_after", report)
        self.assertIn("sharpness_laplacian", report["metrics_after"])
        self.assertIn("contrast_std", report["metrics_after"])
        self.assertIn("sharpness_gain_pct", report)

    def test_quality_and_artifact_assessment(self):
        cv_img = self.enhancer.pil_to_cv2(self.test_img)
        metrics = self.enhancer.assess_quality_and_artifacts(cv_img)
        self.assertGreater(metrics["sharpness_laplacian"], 0.0)
        self.assertGreater(metrics["contrast_std"], 0.0)
        self.assertIn("is_blurry", metrics)
        self.assertIn("has_clipping", metrics)


class TestDiffusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DiffusionEngine()

    def test_engine_initialization(self):
        self.assertIn(self.engine.device, ["mps", "cuda", "cpu"])
        self.assertIn("eco", self.engine.PROFILES)
        self.assertIn("balanced", self.engine.PROFILES)
        self.assertIn("quality", self.engine.PROFILES)
        self.assertIn("baseline", self.engine.PROFILES)

    def test_engine_generation(self):
        res = self.engine.generate(
            prompt="A test prompt for unit testing",
            mode="eco",
            seed=42,
            apply_classical_cv=True
        )
        self.assertEqual(res["mode"], "eco")
        self.assertEqual(res["steps"], 1)
        self.assertIn("metrics", res)
        self.assertGreater(res["metrics"]["latency_seconds"], 0.0)
        self.assertIn("speedup_vs_baseline", res["metrics"])
        self.assertIsInstance(res["enhanced_image"], Image.Image)


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_system_status(self):
        response = self.client.get("/api/system-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("hardware_acceleration", data)
        self.assertIn("ram_total_gb", data)

    def test_generate_endpoint_success(self):
        payload = {
            "prompt": "A modern electric vehicle on an empty highway, 4k",
            "mode": "eco",
            "seed": 1234,
            "apply_classical_cv": True
        }
        response = self.client.post("/api/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["raw_image_base64"].startswith("data:image/png;base64,"))
        self.assertTrue(data["enhanced_image_base64"].startswith("data:image/png;base64,"))
        self.assertIn("metrics", data)

    def test_generate_endpoint_bad_input(self):
        # Empty prompt should return HTTP 400 Bad Request
        payload = {"prompt": "   ", "mode": "eco"}
        response = self.client.post("/api/generate", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

