#!/usr/bin/env python3
"""
One-command entry point for EdgeDiffusion P3.
Usage:
    python run.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run EdgeDiffusion P3 Application Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live code reloading")
    args = parser.parse_args()

    print("\n" + "="*65)
    print(" 🚀 EdgeDiffusion: Resource-Efficient Text-to-Image Generation")
    print(" Problem Statement 3 (P3) | Mobile Optimized Engine")
    print(f" URL: http://{args.host}:{args.port}")
    print("="*65 + "\n")

    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

