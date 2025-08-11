# 🚀 Deploy to Hugging Face Spaces - Step-by-Step Guide

## Prerequisites
1. **Hugging Face Account**: Create one at [huggingface.co](https://huggingface.co)
2. **Git**: Installed on your local machine
3. **Git LFS**: Required for large files (models, audio)
   ```bash
   git lfs install
   ```

## Deployment Steps

### Step 1: Create a New Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure your Space:
   - **Space name**: Choose a unique name (e.g., `raynos-ai-transcription`)
   - **License**: Apache 2.0
   - **Select SDK**: Gradio
   - **SDK version**: 4.12.0
   - **Hardware**: 
     - Start with **CPU basic** (free tier)
     - Upgrade to **GPU** later if needed for better performance
   - **Space visibility**: Public or Private

### Step 2: Clone Your New Space Repository

```bash
# Replace YOUR_USERNAME with your HF username
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

### Step 3: Copy Project Files

Copy the following files and directories from this project to your Space repository:

```bash
# Required files
cp /path/to/raynos-ai/app.py .
cp -r /path/to/raynos-ai/src/ .
cp /path/to/raynos-ai/requirements_hf_spaces.txt requirements.txt
cp /path/to/raynos-ai/README_HF_SPACE.md README.md

# Optional: Include example audio
cp -r /path/to/raynos-ai/example_audio/ .

# Optional: Include unsloth cache if using Gemma extraction
# cp -r /path/to/raynos-ai/unsloth_compiled_cache/ .
```

### Step 4: Optimize for Hugging Face Spaces

Create or modify `.gitignore`:

```bash
cat > .gitignore << 'EOF'
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
```

### Step 5: Configure Space Settings (Optional)

Create or update `app.py` with environment-specific settings:

```python
import os

# For CPU-only spaces, limit model size
if not torch.cuda.is_available():
    os.environ["WHISPER_MODEL_SIZE"] = "base"  # Use smaller model on CPU
    
# Optional: Add your Deepgram API key as a Space secret
# Go to Space Settings > Repository secrets
# Add DEEPGRAM_API_KEY = your_api_key
```

### Step 6: Commit and Push

```bash
# Add all files
git add .
git lfs track "*.aac" "*.mp3" "*.wav"  # Track audio files with LFS
git add .gitattributes

# Commit
git commit -m "Initial deployment of Raynos AI transcription app"

# Push to Hugging Face
git push origin main
```

### Step 7: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Check the **"App"** tab to see if it's running
3. Check the **"Logs"** tab for any errors

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **Import Errors**
```python
# If modules aren't found, ensure app.py has:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

#### 2. **Memory Issues (CPU Spaces)**
- Use smaller Whisper models (`tiny` or `base`)
- Disable Gemma extraction if not needed
- Consider upgrading to GPU Space

#### 3. **Slow Performance**
- Upgrade to GPU hardware (T4 small is usually sufficient)
- Use smaller models
- Implement caching for repeated operations

#### 4. **Audio Issues**
- Ensure `sounddevice` is optional for Spaces:
```python
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
```

### Step 8: Add Secrets (Optional)

For Deepgram API or other services:

1. Go to Space Settings
2. Click on "Repository secrets"
3. Add secrets:
   - Name: `DEEPGRAM_API_KEY`
   - Value: `your-api-key-here`

## 🎯 Optimization Tips

### For Free Tier (CPU)
```python
# In app.py or gradio_app_integrated.py
DEFAULT_SETTINGS = {
    "whisper_model": "base",  # Smaller model
    "batch_size": 1,
    "enable_gemma": False,  # Disable heavy extraction
}
```

### For GPU Spaces
```python
DEFAULT_SETTINGS = {
    "whisper_model": "medium",  # Better accuracy
    "batch_size": 4,
    "enable_gemma": True,  # Enable extraction
}
```

## 📊 Resource Requirements

| Hardware | Whisper Model | Performance | Cost |
|----------|--------------|-------------|------|
| CPU basic | tiny/base | Slow but functional | Free |
| CPU upgrade | small/medium | Moderate | $0.03/hour |
| T4 small | medium/large | Good | $0.60/hour |
| T4 medium | large | Excellent | $0.90/hour |

## 🔗 Useful Links

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Spaces Hardware Options](https://huggingface.co/pricing#spaces)
- [Spaces Secrets Management](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)

## ✅ Final Checklist

- [ ] Created HF Space
- [ ] Cloned repository
- [ ] Copied all necessary files
- [ ] Updated requirements.txt
- [ ] Configured app.py
- [ ] Added .gitignore
- [ ] Committed and pushed
- [ ] Space is running
- [ ] Tested basic functionality
- [ ] (Optional) Added API keys as secrets
- [ ] (Optional) Upgraded hardware if needed

## 💡 Pro Tips

1. **Start with CPU**: Test on free tier first, then upgrade if needed
2. **Use Secrets**: Never commit API keys - use HF Spaces secrets
3. **Monitor Logs**: Check logs regularly during initial deployment
4. **Gradio Version**: Stick to tested version (4.12.0) for stability
5. **Model Caching**: HF Spaces cache downloaded models between restarts

---

**Need help?** Check the [Hugging Face Forums](https://discuss.huggingface.co/) or open an issue in the repository!