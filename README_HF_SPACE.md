---
title: Raynos AI Audio Transcription
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - openai/whisper-base
  - google/gemma-2b-it
---

# Raynos AI - Real-time Audio Transcription & JSON Extraction

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Gradio-4.12.0-orange.svg" alt="Gradio">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
</div>

## 🎯 Overview

Raynos AI is an advanced audio transcription application that combines OpenAI's Whisper model with Google's Gemma model to provide:

- **Real-time Audio Transcription**: Convert speech to text using state-of-the-art Whisper models
- **Structured JSON Extraction**: Automatically extract key information (names, locations, dates, events) from transcriptions
- **Multiple Input Methods**: Support for microphone recording, file upload, and streaming
- **Flexible Transcription Engines**: Choose between local Whisper or cloud-based Deepgram

## 🚀 Features

### Audio Processing
- 🎤 **Live Microphone Recording**: Real-time audio capture and transcription
- 📁 **File Upload**: Process pre-recorded audio files (MP3, WAV, AAC, etc.)
- 🔄 **Streaming Mode**: Continuous transcription for long recordings
- 📱 **Mobile Support**: Optimized for mobile device audio input

### Transcription Options
- **Whisper Models**: Choose from tiny, base, small, medium, or large models
- **Deepgram Integration**: Optional cloud-based transcription (requires API key)
- **Language Support**: Auto-detect or specify language
- **Buffer Control**: Adjustable buffer duration for optimal performance

### JSON Extraction
- **Smart Information Extraction**: Automatically identifies and structures:
  - Person names
  - Locations (cities, countries, addresses)
  - Dates and times
  - Events and activities
  - Key topics and themes
- **Temporal Context**: Links extracted information to timestamps

## 🛠️ Configuration

### Environment Variables (Optional)
- `DEEPGRAM_API_KEY`: Enable Deepgram cloud transcription
- `CUDA_VISIBLE_DEVICES`: Control GPU usage

### Model Selection
The app automatically selects appropriate models based on available hardware:
- **GPU Available**: Uses larger, more accurate models
- **CPU Only**: Falls back to smaller, faster models

## 📊 Technical Details

### Models Used
- **Transcription**: OpenAI Whisper (various sizes)
- **Extraction**: Google Gemma-2B-IT (optional, for JSON extraction)

### Audio Processing
- Sample Rate: 16kHz
- Format: Mono channel
- Chunk Size: 1024 samples

## 🎮 Usage

1. **Select Input Method**:
   - Desktop: Use microphone or upload file
   - Mobile: Use mobile audio streaming

2. **Configure Settings**:
   - Choose transcription engine (Whisper/Deepgram)
   - Select model size (accuracy vs speed trade-off)
   - Set language (auto-detect or specific)

3. **Start Transcription**:
   - Click "Start Streaming" for live audio
   - Or "Process File" for uploaded audio

4. **View Results**:
   - Real-time transcription display
   - Structured JSON output with extracted information

## 📝 Notes

- First run may take time to download models
- GPU recommended for best performance
- Larger models provide better accuracy but require more resources

## 🤝 Contributing

This is an open-source project. Contributions are welcome!

## 📄 License

Apache License 2.0

---

**Built with ❤️ using Gradio and Hugging Face**