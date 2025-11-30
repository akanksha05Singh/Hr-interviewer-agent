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
    
    # 1. PRIMARY: Gemini API (Multimodal)
    # Best quality, handles all formats (WebM, WAV, MP3), no local ffmpeg needed.
    try:
        print("DEBUG: Attempting Gemini STT (Primary)...")
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
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
        else:
            print("DEBUG: No API Key for Gemini, falling back to local.")
    except Exception as e4:
        print(f"DEBUG: Gemini STT Error: {e4}")

    # 2. FALLBACK: Local SpeechRecognition (requires ffmpeg for WebM)
    r = sr.Recognizer()
    try:
        print("DEBUG: Attempting pydub conversion (Fallback)...")
        from pydub import AudioSegment
        sound = AudioSegment.from_file(audio_path)
        wav_path = audio_path + ".converted.wav"
        sound.export(wav_path, format="wav")
        
        with sr.AudioFile(wav_path) as src:
            audio = r.record(src)
            
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        res = r.recognize_google(audio)
        print(f"DEBUG: pydub success: {res}")
        return res
    except Exception as e:
        print(f"DEBUG: STT Conversion Error (pydub/ffmpeg): {e}")
        
        # 3. LAST RESORT: Soundfile / Direct Load
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
            return res
        except Exception as e2:
            print(f"DEBUG: STT Soundfile Error: {e2}")
            return ""
