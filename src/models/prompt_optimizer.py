"""
Intelligent Semantic Prompt Optimizer for Distilled Diffusion Models
Enhances user prompts with category-aware photographic modifiers, lighting,
and detail cues to maximize image quality in 1-4 step generation.
"""

import re
from typing import Tuple, Dict


class PromptOptimizer:
    """
    Intelligent Prompt Expander and Negative Prompt Builder.
    Distilled models (SD-Turbo, LCM) benefit significantly from explicit
    photographic descriptors and negative guidance to avoid plain or distorted renders.
    """

    CATEGORIES = {
        "portrait": {
            "keywords": ["woman", "man", "person", "girl", "boy", "face", "portrait", "model", "human"],
            "positive": "detailed skin texture, realistic eyes, soft rim lighting, shallow depth of field, 85mm portrait photography, 8k resolution, masterpiece",
            "negative": "deformed eyes, bad teeth, bad anatomy, bad hands, mutated fingers, plastic skin, asymmetry"
        },
        "animal": {
            "keywords": ["dog", "cat", "panda", "tiger", "lion", "bird", "puppy", "kitten", "wolf", "fox", "bear"],
            "positive": "detailed fur texture, sharp whiskers, expressive eyes, natural outdoor lighting, macro wildlife photography, 4k",
            "negative": "extra legs, mutated paws, deformed snout, unnatural posture, blurry fur"
        },
        "vehicle": {
            "keywords": ["car", "sports car", "automobile", "truck", "motorcycle", "bike", "airplane", "spaceship"],
            "positive": "aerodynamic body reflections, metallic paint finish, raytraced reflections, cinematic automotive lighting, 8k",
            "negative": "asymmetrical wheels, distorted chassis, warped windshield, missing tires"
        },
        "landscape": {
            "keywords": ["mountain", "lake", "forest", "city", "street", "building", "sunset", "ocean", "river", "sky"],
            "positive": "atmospheric depth, golden hour volumetric sunlight, ultra-detailed horizon, wide-angle landscape photography, crisp focus",
            "negative": "washed out sky, blurry horizon, low contrast, flat lighting, artifacts"
        },
        "scifi": {
            "keywords": ["cyberpunk", "futuristic", "neon", "robot", "cyborg", "sci-fi", "alien", "hologram"],
            "positive": "glowing neon reflections, intricate mechanical seams, volumetric atmospheric fog, Unreal Engine 5 render style, 8k",
            "negative": "blurry glow, flat textures, low resolution, noisy artifacts"
        }
    }

    DEFAULT_POSITIVE_SUFFIX = "cinematic lighting, sharp focus, rich colors, intricate details, 4k resolution, high quality"
    DEFAULT_NEGATIVE = "ugly, blurry, low quality, artifacts, watermark, distorted, duplicate, oversaturated, deformed"

    @classmethod
    def detect_category(cls, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for cat, data in cls.CATEGORIES.items():
            for kw in data["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower):
                    return cat
        return "general"

    @classmethod
    def optimize(
        cls,
        prompt: str,
        user_negative: str = "",
        auto_enhance: bool = True
    ) -> Tuple[str, str, Dict[str, str]]:
        """
        Takes raw user prompt and returns (enhanced_prompt, effective_negative, metadata).
        """
        clean_prompt = prompt.strip()
        cat = cls.detect_category(clean_prompt)

        if not auto_enhance:
            effective_negative = user_negative.strip() or cls.DEFAULT_NEGATIVE
            return clean_prompt, effective_negative, {"category": cat, "enhanced": False}

        # Select category enhancements
        if cat in cls.CATEGORIES:
            addon = cls.CATEGORIES[cat]["positive"]
            cat_neg = cls.CATEGORIES[cat]["negative"]
        else:
            addon = cls.DEFAULT_POSITIVE_SUFFIX
            cat_neg = ""

        # Avoid redundant additions if user already specified lighting or resolution
        enhanced_prompt = clean_prompt
        if not any(w in clean_prompt.lower() for w in ["4k", "8k", "lighting", "photography", "cinematic"]):
            enhanced_prompt = f"{clean_prompt}, {addon}"

        # Combine negative prompt
        combined_neg_parts = [cls.DEFAULT_NEGATIVE]
        if cat_neg:
            combined_neg_parts.append(cat_neg)
        if user_negative.strip():
            combined_neg_parts.append(user_negative.strip())

        effective_negative = ", ".join(dict.fromkeys(", ".join(combined_neg_parts).split(", ")))

        return enhanced_prompt, effective_negative, {
            "category": cat,
            "enhanced": True,
            "original_prompt": clean_prompt,
            "added_descriptors": addon
        }
