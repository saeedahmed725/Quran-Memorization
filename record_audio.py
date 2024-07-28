import tkinter as tk
import sounddevice as sd
import wavio
import threading
import numpy as np

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.sample_rate = 44100  # Sample rate in Hz
        self.filename = "recorded_audio.wav"  # Output file name
        self.audio_data = None
        self.recording_thread = None

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_data = []
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            print("Recording started...")

    def record_audio(self):
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.callback):
            while self.is_recording:
                sd.sleep(1000)

    def callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recording_thread.join()
            audio_array = np.concatenate(self.audio_data, axis=0)
            wavio.write(self.filename, audio_array, self.sample_rate, sampwidth=2)
            print(f"Recording saved to {self.filename}")

def start_button_callback():
    recorder.start_recording()

def stop_button_callback():
    recorder.stop_recording()

# Create the GUI
root = tk.Tk()
root.title("Audio Recorder")

recorder = AudioRecorder()

start_button = tk.Button(root, text="Start Recording", command=start_button_callback)
start_button.pack(pady=20)

stop_button = tk.Button(root, text="Stop Recording", command=stop_button_callback)
stop_button.pack(pady=20)

root.mainloop()
