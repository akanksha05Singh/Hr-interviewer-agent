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
    print(f"DEBUG: stt_from_file called with {audio_path}")
    # Uses speech_recognition + local recognizer (Sphinx or Google if internet available)
    r = sr.Recognizer()
    try:
        print("DEBUG: Attempting pydub conversion...")
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
        res = r.recognize_google(audio)
        print(f"DEBUG: pydub success: {res}")
        return res
    except Exception as e:
        print(f"DEBUG: STT Conversion Error (pydub/ffmpeg): {e}")
        
        # Fallback 1: Try soundfile (often works for some formats without system ffmpeg)
        try:
            print("DEBUG: Attempting soundfile fallback...")
            import soundfile as sf
            wav_path = audio_path + ".sf.wav"
            data, samplerate = sf.read(audio_path)
            sf.write(wav_path, data, samplerate)
            
            with sr.AudioFile(wav_path) as src:
                audio = r.record(src)
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
            res = r.recognize_google(audio)
            print(f"DEBUG: soundfile success: {res}")
            return res
        except Exception as e2:
            print(f"DEBUG: STT Soundfile Error: {e2}")

        # Fallback 2: Try direct load (works if file is already WAV)
        try:
            print("DEBUG: Attempting direct load fallback...")
            with sr.AudioFile(audio_path) as src:
                audio = r.record(src)
            res = r.recognize_google(audio)
            print(f"DEBUG: direct load success: {res}")
            return res
        except Exception as e3:
            print(f"DEBUG: STT Direct Load Error: {e3}")
            
            # Fallback 3: Gemini API (Universal Fallback)
            # This bypasses local ffmpeg entirely by sending the file to Google.
            try:
                print("DEBUG: Attempting Gemini STT Fallback...")
                import google.generativeai as genai
                
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    print("DEBUG: No API Key found for Gemini STT")
                    return ""
                    
                genai.configure(api_key=api_key)
                
                # Upload file
                print(f"DEBUG: Uploading file to Gemini: {audio_path}")
                myfile = genai.upload_file(audio_path)
                
                # Generate content
                print("DEBUG: Generating content with Gemini...")
                model = genai.GenerativeModel("gemini-1.5-flash")
                result = model.generate_content(["Transcribe this audio exactly.", myfile])
                
                print(f"DEBUG: Gemini success: {result.text}")
                return result.text.strip()
            except Exception as e4:
                print(f"DEBUG: Gemini STT Error: {e4}")
                return ""
