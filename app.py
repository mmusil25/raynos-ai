#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Hugging Face Spaces entry point for Raynos AI Audio Transcription App
"""

import os
import sys
import torch
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the main Gradio app
from gradio_app_integrated import create_interface

# Optional: Set environment variables if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available

def main():
    """Main entry point for Hugging Face Spaces"""
    
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("💻 Running on CPU (may be slower)")
    
    # Check for Deepgram API key (optional)
    if os.environ.get("DEEPGRAM_API_KEY"):
        print("✓ Deepgram API key detected - cloud transcription enabled")
    else:
        print("ℹ️ No DEEPGRAM_API_KEY found - using local Whisper model only")
    
    # Create the Gradio interface
    print("🎯 Initializing Raynos AI Audio Transcription...")
    app = create_interface()
    
    # Launch with HF Spaces compatible settings
    # Note: HF Spaces automatically sets the correct server_name and port
    app.launch(
        server_name="0.0.0.0",  # Required for HF Spaces
        server_port=7860,        # Default HF Spaces port
        share=False,             # Sharing is handled by HF Spaces
        show_error=True,         # Show detailed errors
        quiet=False,             # Show startup logs
    )

if __name__ == "__main__":
    main()