import tempfile, os
import pyttsx3
import speech_recognition as sr

def tts_say(text: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"TTS Error: {e}")

def stt_from_file(audio_path: str):
    # Uses speech_recognition + local recognizer (Sphinx or Google if internet available)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        # If you have internet and want better ASR use r.recognize_google(audio)
        return r.recognize_google(audio)
    except Exception as e:
        print("STT error:", e)
        return ""
