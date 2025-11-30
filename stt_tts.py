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
        # Convert to WAV using pydub (handles WebM/Ogg from browser)
        from pydub import AudioSegment
        
        # Load audio (pydub auto-detects format)
        sound = AudioSegment.from_file(audio_path)
        
        # Export as WAV to a new temp file
        wav_path = audio_path + ".converted.wav"
        sound.export(wav_path, format="wav")
        
        with sr.AudioFile(wav_path) as src:
            audio = r.record(src)
            
        # Cleanup converted file
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        # If you have internet and want better ASR use r.recognize_google(audio)
        return r.recognize_google(audio)
    except Exception as e:
        print(f"STT Conversion Error (pydub/ffmpeg): {e}")
        
        # Fallback 1: Try soundfile (often works for some formats without system ffmpeg)
        try:
            import soundfile as sf
            wav_path = audio_path + ".sf.wav"
            data, samplerate = sf.read(audio_path)
            sf.write(wav_path, data, samplerate)
            
            with sr.AudioFile(wav_path) as src:
                audio = r.record(src)
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
            return r.recognize_google(audio)
        except Exception as e2:
            print(f"STT Soundfile Error: {e2}")

        # Fallback 2: Try direct load (works if file is already WAV)
        try:
            with sr.AudioFile(audio_path) as src:
                audio = r.record(src)
            return r.recognize_google(audio)
        except Exception as e3:
            print(f"STT Direct Load Error: {e3}")
            return ""
