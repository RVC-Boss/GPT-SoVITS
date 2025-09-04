# TTS Only - Minimal Requirements (Korean/English only)
# Core TTS functionality for GPT-SoVITS V4/V2Pro

# Core ML frameworks (relaxed versions for compatibility)
torch>=2.0.0,<3.0.0
torchaudio>=2.0.0,<3.0.0
numpy>=1.24.0  # Compatible with both numpy 1.x and 2.x

# Audio processing (core)
librosa>=0.10.0,<0.11.0
soundfile>=0.12.0
numba>=0.56.0

# NLP and text processing (expanded compatibility)
transformers>=4.43.0,<5.0.0  # Support newer versions
tokenizers>=0.13.0,<1.0.0

# Language-specific text processing
g2p_en>=2.1.0  # English phoneme conversion
g2pk2>=2.0.0  # Korean phoneme conversion
ko_pron>=1.3.0  # Korean pronunciation

# Utility libraries
pyyaml>=5.4.0  # Configuration files
tqdm>=4.60.0  # Progress bars
psutil>=5.8.0  # System monitoring

# Additional dependencies for TTS model
ffmpeg-python>=0.2.0  # Audio processing
requests>=2.25.0  # Model downloading
