import sounddevice as sd
import numpy as np
import wave

def capture_audio(output_filename="output.wav", duration=5, rate=44100):
    print("Recording...")

    
    audio_data = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype=np.int16)
    sd.wait()  

    print("Recording finished.")

    
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  
        wf.setframerate(rate)
        wf.writeframes(audio_data.tobytes())
