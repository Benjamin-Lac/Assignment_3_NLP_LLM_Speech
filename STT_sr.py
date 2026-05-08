# using Google speech recognition
import speech_recognition as sr

# install the following Google speech recognition packages if not already installed:
# pip install SpeechRecognition
# pip install PyAudio

# Initialize the recognizer
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Please speak something...")
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=1)
    # Listen for the first phrase and extract it into audio data
    audio_data = recognizer.listen(source)
    print("Recognizing...")

    try:
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
