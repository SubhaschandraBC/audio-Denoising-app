
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io

st.title("Audio Denoising App")
st.write("Generate audio or upload your own, then denoise it using Moving Average Filter.")

def moving_average(x, window=21):
    return np.convolve(x, np.ones(window)/window, mode='same')

option = st.radio("Choose Input:", ["Generate Default Audio (Ode to Joy)", "Upload Clean Audio", "Upload Noisy Audio"])

sr = 22050

if option == "Generate Default Audio (Ode to Joy)":
    if st.button("Generate"):
        notes = {'E': 329.63, 'F': 349.23, 'G': 392.00}
        melody = ['E','E','F','G','G','F','E','D','C','D','E','C']
        clean_list = []
        for n in melody:
            t = np.linspace(0, 0.4, int(sr*0.4))
            clean_list.extend(np.sin(2*np.pi*notes.get(n, 261.63)*t))
        clean = np.array(clean_list)
        np.random.seed(42)
        noisy = clean + np.random.randn(len(clean)) * 0.05
        st.session_state['noisy'] = noisy
        st.session_state['sr'] = sr
        st.audio(noisy, sample_rate=sr)
        st.write("Noisy audio generated")

elif option == "Upload Clean Audio":
    uploaded = st.file_uploader("Upload clean audio (wav/mp3)", type=['wav','mp3','ogg'])
    if uploaded:
        y, sr = librosa.load(uploaded, sr=22050)
        np.random.seed(42)
        noisy = y + np.random.randn(len(y)) * 0.05
        st.session_state['noisy'] = noisy
        st.session_state['sr'] = sr
        st.write("Clean audio uploaded, noise added")
        st.audio(noisy, sample_rate=sr)

elif option == "Upload Noisy Audio":
    uploaded = st.file_uploader("Upload noisy audio (wav/mp3)", type=['wav','mp3','ogg'])
    if uploaded:
        y, sr = librosa.load(uploaded, sr=22050)
        st.session_state['noisy'] = y
        st.session_state['sr'] = sr
        st.write("Noisy audio uploaded")
        st.audio(y, sample_rate=sr)

# Denoise part - same for all 3 options
if 'noisy' in st.session_state:
    window = st.slider("Filter Window Size", 5, 51, 21, step=2)
    if st.button("Denoise"):
        noisy = st.session_state['noisy']
        sr = st.session_state['sr']
        denoised = moving_average(noisy, window)
        st.write("Denoised Audio:")
        st.audio(denoised, sample_rate=sr)
        # Download button
        buf = io.BytesIO()
        sf.write(buf, denoised, sr, format='WAV')
        st.download_button("Download Denoised Audio", buf.getvalue(), "denoised.wav")

# Denoise part - same for all 3 options
if 'noisy' in st.session_state:
    window = st.slider("Filter Window Size", 5, 51, 21, step=2)
    if st.button("Denoise"):
        noisy = st.session_state['noisy']
        sr = st.session_state['sr']
        denoised = moving_average(noisy, window)
        st.write("Denoised Audio:")
        st.audio(denoised, sample_rate=sr)
        # Download button
        buf = io.BytesIO()
        sf.write(buf, denoised, sr, format='WAV')
        st.download_button("Download Denoised Audio", buf.getvalue(), "denoised.wav")
