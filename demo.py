import streamlit as st
import cv2
import numpy as np
import time

# set page config
st.set_page_config(page_title='Chatbot TLU', page_icon="🦈", layout='wide')

# Stream video from webcam
st.title("Webcam Live Feed")

# Khởi tạo các biến session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'frame_window' not in st.session_state:
    st.session_state.frame_window = st.empty()

# Cấu hình audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


class AudioVideoRecorder:
    def __init__(self):
        self.video_capture = None
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        # Khởi tạo video capture
        self.video_capture = cv2.VideoCapture(0)

        # Khởi tạo audio stream
        self.audio_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        self.is_recording = True

        # Bắt đầu thread ghi âm
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def record_audio(self):
        while self.is_recording:
            try:
                data = self.audio_stream.read(CHUNK)
                self.frames.append(data)
            except Exception as e:
                st.error(f"Lỗi ghi âm: {str(e)}")
                break

    def get_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def stop_recording(self):
        self.is_recording = False

        # Dừng và giải phóng video capture
        if self.video_capture is not None:
            self.video_capture.release()

        # Dừng và giải phóng audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        # Lưu file audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        return filename


def main():
    # sidebar
    with st.sidebar:
        st.title('🦙💬 TLU Chatbot')
        st.write('This chatbot is created using the open-source LLM model and built from Ollama. ')

    # main page
    st.markdown("# Welcome to the Chatbot TLU! :sunglasses: :sunglasses:")

    # Khởi tạo recorder nếu chưa có
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioVideoRecorder()

    # Nút điều khiển ghi hình
    if st.button("Toggle Recording"):
        if not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.recorder = AudioVideoRecorder()
            st.session_state.recorder.start_recording()
        else:
            st.session_state.recording = False
            filename = st.session_state.recorder.stop_recording()
            st.success(f"Đã lưu recording vào file: {filename}")

    # Hiển thị trạng thái
    status_text = "Đang ghi" if st.session_state.recording else "Đã dừng"
    st.text(f"Trạng thái: {status_text}")

    # Hiển thị video frame
    if st.session_state.recording:
        frame = st.session_state.recorder.get_video_frame()
        if frame is not None:
            st.session_state.frame_window.image(frame)
            time.sleep(0.033)  # Giới hạn ~30 FPS
            st.experimental_rerun()
    else:
        st.session_state.frame_window.write("Camera đã tắt")