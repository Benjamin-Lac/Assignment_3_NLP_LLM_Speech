import pyttsx3

# install the following package if not already installed:
# pip install pyttsx3

engine = pyttsx3.init()

# List available voices
voices = engine.getProperty('voices')
for idx, voice in enumerate(voices):
    print(f"{idx}: {voice.name} - {voice.id}")

# Choose a voice by index
engine.setProperty('voice', voices[1].id)  # e.g., 0=male, 1=female

# Optional: set speech rate
engine.setProperty('rate', 150)

text = "Hello, this is a Python text-to-speech demo."

# Speak and save to WAV
engine.say(text)
engine.save_to_file(text, "pyttsx3_output.wav")
engine.runAndWait()
