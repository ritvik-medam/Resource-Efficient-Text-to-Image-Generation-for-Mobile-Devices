#!/usr/bin/env python3
"""
One-command entry point for EdgeDiffusion P3.
Usage:
    python run.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import os
import sys
import socket
import argparse
import uvicorn


def find_available_port(start_port: int, host: str = "127.0.0.1") -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
        port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(description="Run EdgeDiffusion P3 Application Server")
    default_port = int(os.environ.get("PORT", 8000))
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default=os.environ.get("DEVICE", "auto"), help="Hardware acceleration device")
    parser.add_argument("--reload", action="store_true", help="Enable live code reloading")
    args = parser.parse_args()

    if args.device != "auto":
        os.environ["DEVICE"] = args.device

    active_port = args.port
    # Check if port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", active_port)) == 0:
            fallback_port = find_available_port(active_port + 1)
            print(f"⚠️ Port {active_port} is already in use. Automatically switching to port {fallback_port}...")
            active_port = fallback_port

    print("\n" + "="*65)
    print(" 🚀 EdgeDiffusion: Resource-Efficient Text-to-Image Generation")
    print(" Problem Statement 3 (P3) | Mobile Optimized Engine")
    print(f" URL: http://{args.host}:{active_port}")
    print(f" Target Device: {os.environ.get('DEVICE', 'auto')}")
    print("="*65 + "\n")

    uvicorn.run("src.api.server:app", host=args.host, port=active_port, reload=args.reload)


if __name__ == "__main__":
    main()

