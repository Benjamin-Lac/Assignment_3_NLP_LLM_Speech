import whisper
import sounddevice as sd
import numpy as np

# install the following packages if not already installed:
# pip install -U openai-whisper
# pip install sounddevice
# pip install scipy
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# -------------------------
# 1. Record audio from microphone
# -------------------------
duration = 5  # seconds
sample_rate = 16000  # Whisper works best with 16kHz

print("Recording for", duration, "seconds...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
print("Recording complete!")

# Flatten the array and convert to float32
audio = audio.flatten().astype(np.float32)

# Normalize between -1 and 1 if necessary
audio /= np.max(np.abs(audio))

# -------------------------
# 2. Load Whisper model
# -------------------------
model = whisper.load_model("base")

# -------------------------
# 3. Transcribe in-memory audio
# -------------------------
result = model.transcribe(audio, fp16=False, language="en")  # fp16=False avoids GPU-only issues
print("Recognized Text:", result["text"])
