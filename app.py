import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io

st.title("Audio Denoising App")
st.write("Generate audio or upload your own, then denoise it using Moving Average Filter.")

def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='same')

def to_wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

tab1, tab2, tab3 = st.tabs(["Generate", "Upload Clean", "Upload Noisy"])

with tab1:
    st.subheader("Generate Ode to Joy")
    noise = st.slider("Noise Level", 0.0, 0.5, 0.1, key="noise_gen")
    window = st.slider("Filter Window Size", 5, 51, 21, step=2, key="filter_gen")
    if st.button("Generate", key="btn_gen"):
        sr = 22050
        notes = [659.25, 659.25, 698.46, 783.99, 783.99, 698.46, 659.25, 587.33]
        y = np.concatenate([np.sin(2*np.pi*f*np.linspace(0, 0.5, int(sr*0.5))) for f in notes])
        noisy = y + noise * np.random.randn(len(y))
        denoised = moving_average(noisy, window)
        st.audio(to_wav_bytes(noisy, sr), format="audio/wav")
        st.write("Denoised Audio:")
        st.audio(to_wav_bytes(denoised, sr), format="audio/wav")
        st.download_button("Download Denoised Audio", to_wav_bytes(denoised, sr), "denoised.wav", key="dl_gen")

with tab2:
    st.subheader("Upload Clean Audio")
    file = st.file_uploader("Upload Clean WAV", type=["wav","mp3"], key="up_clean")
    window2 = st.slider("Filter Window Size", 5, 51, 21, step=2, key="filter_clean")
    if file is not None:
        y, sr = librosa.load(file, sr=None)
        noisy = y + 0.1 * np.random.randn(len(y))
        if st.button("Denoise", key="btn_clean"):
            denoised = moving_average(noisy, window2)
            st.audio(to_wav_bytes(denoised, sr), format="audio/wav")
            st.download_button("Download Denoised Audio", to_wav_bytes(denoised, sr), "denoised.wav", key="dl_clean")

with tab3:
    st.subheader("Upload Noisy Audio")
    file2 = st.file_uploader("Upload Noisy WAV", type=["wav","mp3"], key="up_noisy")
    window3 = st.slider("Filter Window Size", 5, 51, 21, step=2, key="filter_noisy")
    if file2 is not None:
        y, sr = librosa.load(file2, sr=None)
        if st.button("Denoise", key="btn_noisy"):
            denoised = moving_average(y, window3)
            st.audio(to_wav_bytes(denoised, sr), format="audio/wav")
            st.download_button("Download Denoised Audio", to_wav_bytes(denoised, sr), "denoised.wav", key="dl_noisy")
