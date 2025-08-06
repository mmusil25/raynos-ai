# Raynos – Gemma 3-N / Unsloth On-Device Pipeline

A privacy-first AI pipeline that processes audio locally using Gemma 3-N with Unsloth optimization. This implementation showcases real-time audio transcription with Whisper and structured data extraction, all running completely on-device with no cloud dependencies.

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Gemma-3N-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Unsloth-Enabled-lightgrey?style=for-the-badge">
  <img src="https://img.shields.io/badge/Privacy-100%25%20On--Device-red?style=for-the-badge">
</p>

![Raynos Architecture](docs/architecture.drawio.png)


## 🚨 Important Disclosure

**This implementation demonstrates the core AI pipeline.** The BLE audio streaming functionality uses a mock implementation that simulates the expected behavior. When real BLE devices become available, the mock BLE source can be replaced with actual device communication.

**🛠️ Current Top Priority:** Replace the mock BLE audio stream with a real-time connection to physical BLE devices. Contributions in this area are highly welcomed and will have the highest impact.

## 🎯 Features

- **BLE Audio Streaming** with auto-reconnect functionality (mock and real implementations)
- **Multiple Audio Sources**: BLE devices, local microphone, or audio files
- **Real-time Transcription** using OpenAI Whisper (Tiny model for speed)
- **Intent & Entity Extraction** using mock or Gemma 3n models
- **Web Interface** via Gradio with live streaming display
- **GPU Acceleration** support for faster processing
- **Structured JSON Output** following the required schema
- **Unsloth Optimization** for 2x faster Gemma 3n inference

## 🚀 Unsloth Integration (Competition Requirement)

This project uses **Unsloth** for optimized Gemma 3n inference, which is a requirement for the $10k side prize. Unsloth provides:

- **2x Faster Inference** compared to standard transformers
- **60% Less Memory Usage** enabling larger batch sizes
- **Drop-in Replacement** requiring minimal code changes
- **Automatic Optimization** for Gemma 3n-E4B-it model

### Installation

```bash
# Install Unsloth with torch support
pip install "unsloth[torch]"
```

The integration is automatic - when Unsloth is installed, the system will automatically use the optimized `unsloth/gemma-3n-e4b-it` model for faster inference.

### Testing Unsloth

```bash
# Test Unsloth integration
python test_unsloth_gemma.py
```

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional but recommended for fast inference)
- Ubuntu/Linux (recommended) or Windows with WSL2
- For actual BLE: Bluetooth adapter
- For microphone: Audio input device (not available in WSL)
- Unsloth for 2x faster Gemma 3n inference (automatically installed)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/mmusil25/raynos-ai.git
cd python-ble-audio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Additional Dependencies

```bash
# Core dependencies
pip install torch transformers openai-whisper bleak gradio

# Audio processing
pip install numpy scipy sounddevice

# Optional: for better audio support
sudo apt-get install ffmpeg  # Required for Whisper
```

### 3. Run the Demo

#### Option A: Web Interface (Gradio)
```bash
python gradio_app.py --share
```
This will start a web server and provide a public URL if `--share` is used.

#### Option B: Command Line
```bash
# Process a single file
python demo.py file --input samples/harvard.wav

# Process all files in a directory
python demo.py directory --input samples/

# Live streaming (mock BLE)
python demo.py stream --audio-source mock

# Live streaming (microphone - requires native OS)
python demo.py stream --audio-source mic
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Source   │────▶│  Transcription   │────▶│ JSON Extraction │
│  (BLE/Mic/File) │     │  (Whisper/API)   │     │ (Mock/Gemma)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                         ┌───────▼────────┐
                         │  JSON Output   │
                         │ {transcript,   │
                         │  timestamp_ms, │
                         │  intent,       │
                         │  entities}     │
                         └────────────────┘
```

## 📁 Project Structure

```
python-ble-audio/
├── ble_listener.py              # BLE/audio source management
├── whisper_tiny_transcription.py # Transcription engine
├── gemma_3n_json_extractor.py   # JSON extraction from transcripts
├── demo.py                      # Command-line interface
├── gradio_app.py               # Web interface
├── simulate_live_mic.py        # Microphone simulation for testing
├── samples/                    # Test audio files
│   └── harvard.wav            # Sample speech file
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# For Deepgram API (alternative to Whisper)
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY  # Replace with your own key; do NOT commit secrets

# For GPU acceleration (automatic if available)
export CUDA_VISIBLE_DEVICES=0
```

### Audio Settings

- **Sample Rate**: 16kHz (configurable)
- **Channels**: Mono
- **Bit Depth**: 16-bit
- **Buffer Duration**: 2 seconds for streaming

## 📖 Usage Examples

### 1. Basic File Processing
```python
from demo import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline(
    transcription_engine="whisper",
    extraction_model="mock"
)

# Process file
result = await pipeline.process_file("audio.wav")
print(result)
```

### 2. Streaming with Callbacks
```python
from ble_listener import AudioStreamManager, MockBLESource

# Create audio source
source = MockBLESource()
manager = AudioStreamManager(source)

# Add callback
def on_audio(chunk):
    print(f"Received {len(chunk)} bytes of audio")

manager.add_callback(on_audio)

# Start streaming
await manager.start_streaming()
```

### 3. Custom JSON Extraction
```python
from gemma_3n_json_extractor import ExtractionManager

# Initialize extractor
extractor = ExtractionManager(extractor_type="mock")

# Extract from transcript
result = extractor.extract_from_transcript(
    "Hello, my name is John and I need help with my order",
    timestamp_ms=1234567890
) 
print(result)
# Output: {"transcript": "...", "intent": "request", "entities": ["John"], ...}
```

## 🐛 Known Limitations

1. **No Real Omi Device**: BLE functionality is simulated
2. **WSL Audio Issues**: Microphone/audio passthrough requires complex setup
3. **GPU Memory**: Gemma models require significant VRAM
4. **Streaming Latency**: 2-second buffer for transcription

## 🔍 Troubleshooting

### "Connection refused" for BLE
- Ensure Bluetooth is enabled
- Check that `bleak` is installed: `pip install bleak`
- Use mock source for testing: `--audio-source mock`

### "PortAudio not found" 
- Install system dependencies: `sudo apt-get install portaudio19-dev`
- Or use file input instead of microphone

### "CUDA out of memory"
- Use CPU mode: `--device cpu`
- Or use smaller models/batch sizes

### Whisper fails with ffmpeg error
- Install ffmpeg: `sudo apt-get install ffmpeg`

## 🤝 Contributing

This is a proof-of-concept implementation. Key areas for improvement:

1. Real Omi device integration
2. Optimized streaming protocols
3. Better Gemma prompt engineering
4. Additional language support
5. Mobile app integration

## 📄 License

Apache 2.0 License - See LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for Whisper
- Google for Gemma models
- Gradio team for the UI framework
- Based Hardware for the BLE audio concept

---

**Note**: This implementation demonstrates the architecture and data flow for a BLE audio processing pipeline. For production use, additional error handling, security measures, and optimization would be required.