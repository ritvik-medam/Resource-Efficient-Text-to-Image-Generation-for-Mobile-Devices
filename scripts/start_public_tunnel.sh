#!/bin/bash
# Starts a secure public HTTPS URL for live mobile demoing
echo "=========================================================="
echo " 🌐 Generating Public HTTPS URL for EdgeDiffusion Demo"
echo "=========================================================="
echo "This will create a public URL accessible from your phone and laptop."
echo "Make sure 'python run.py' is already running in another terminal!"
echo ""
npx --yes localtunnel --port 8000
