#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Verify that all files are ready for Hugging Face Spaces deployment
"""

import os
import sys
from pathlib import Path

def check_file(filepath, required=True):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (REQUIRED)" if required and not exists else ""
    print(f"{status} {filepath}{req_text}")
    return exists or not required

def main():
    print("🔍 Verifying Hugging Face Spaces Deployment Files")
    print("=" * 50)
    
    all_good = True
    
    print("\n📄 Core Files:")
    all_good &= check_file("app.py", required=True)
    all_good &= check_file("src/gradio_app_integrated.py", required=True)
    all_good &= check_file("src/gemma_3n_json_extractor.py", required=True)
    
    print("\n📦 Requirements Files:")
    all_good &= check_file("requirements_hf_spaces.txt", required=True)
    all_good &= check_file("requirements_cpu_only.txt", required=False)
    all_good &= check_file("requirements.txt", required=False)
    
    print("\n📖 Documentation:")
    all_good &= check_file("README_HF_SPACE.md", required=True)
    all_good &= check_file("DEPLOY_TO_HF_SPACES.md", required=False)
    
    print("\n🔧 Deployment Tools:")
    all_good &= check_file("deploy_to_hf.sh", required=False)
    all_good &= check_file("verify_deployment.py", required=False)
    
    print("\n🎵 Optional Files:")
    check_file("example_audio/", required=False)
    check_file("src/gemma_3n_json_extractor_cpu.py", required=False)
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✅ All required files are present!")
        print("\n🚀 Ready to deploy to Hugging Face Spaces!")
        print("\nNext steps:")
        print("1. Run: ./deploy_to_hf.sh <YOUR_HF_USERNAME> <SPACE_NAME>")
        print("2. Or follow manual instructions in DEPLOY_TO_HF_SPACES.md")
        return 0
    else:
        print("❌ Some required files are missing!")
        print("\nPlease ensure all required files are present before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())