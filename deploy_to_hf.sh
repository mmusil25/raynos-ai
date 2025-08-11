#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Quick deployment script for Hugging Face Spaces

set -e  # Exit on error

echo "🚀 Raynos AI - Hugging Face Spaces Deployment Script"
echo "=================================================="

# Check if HF username and space name are provided
if [ $# -lt 2 ]; then
    echo "Usage: ./deploy_to_hf.sh <HF_USERNAME> <SPACE_NAME> [--cpu-only]"
    echo "Example: ./deploy_to_hf.sh myusername raynos-ai-demo"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
CPU_ONLY=${3:-""}

echo "📦 Preparing deployment for: $HF_USERNAME/$SPACE_NAME"

# Create temporary deployment directory
DEPLOY_DIR="hf_space_deploy_$(date +%s)"
mkdir -p $DEPLOY_DIR

echo "📂 Copying required files..."

# Copy main files
cp app.py $DEPLOY_DIR/
cp -r src/ $DEPLOY_DIR/

# Choose appropriate requirements file
if [ "$CPU_ONLY" == "--cpu-only" ]; then
    echo "📝 Using CPU-only requirements..."
    cp requirements_cpu_only.txt $DEPLOY_DIR/requirements.txt
else
    echo "📝 Using standard requirements..."
    cp requirements_hf_spaces.txt $DEPLOY_DIR/requirements.txt
fi

# Copy README for HF Space
cp README_HF_SPACE.md $DEPLOY_DIR/README.md

# Optional: Copy example audio
if [ -d "example_audio" ]; then
    echo "🎵 Including example audio files..."
    cp -r example_audio/ $DEPLOY_DIR/
fi

# Create .gitignore
cat > $DEPLOY_DIR/.gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
.gradio/
*.log
.env
venv/
.venv/
*.swp
*.swo
*~
.DS_Store
EOF

# Initialize git repository
cd $DEPLOY_DIR
git init
git lfs install

# Track large files with Git LFS
git lfs track "*.aac" "*.mp3" "*.wav" "*.pt" "*.bin"
git add .gitattributes

# Add all files
git add .
git commit -m "Initial deployment of Raynos AI transcription app"

# Add HF Spaces remote
echo "🔗 Connecting to Hugging Face Space..."
git remote add origin https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME

echo ""
echo "✅ Deployment package ready!"
echo ""
echo "📋 Next steps:"
echo "1. Create your Space on Hugging Face if not already done:"
echo "   https://huggingface.co/new-space"
echo ""
echo "2. Push to your Space:"
echo "   cd $DEPLOY_DIR"
echo "   git push origin main"
echo ""
echo "3. If authentication is needed:"
echo "   - Use your HF username: $HF_USERNAME"
echo "   - Use an access token from: https://huggingface.co/settings/tokens"
echo ""
echo "4. Monitor your Space at:"
echo "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""

# Optional: Automatic push (commented out for safety)
# echo "🚀 Pushing to Hugging Face..."
# git push origin main

echo "💡 Tip: For CPU-only deployment, use: ./deploy_to_hf.sh $HF_USERNAME $SPACE_NAME --cpu-only"