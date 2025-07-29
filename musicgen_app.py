import streamlit as st
import torchaudio
import torch
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio
import tempfile
import os

# Title
st.title("🎵 Music Genre Transformer using MusicGen")

# Load MusicGen model
@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=10)
    return model

model = load_model()

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])

# Prompt input
prompt = st.text_input("Enter a prompt to condition the genre (e.g., 'a jazz song with saxophone')")

# Generate button
if st.button("Transform Genre"):
    if uploaded_file is None or prompt.strip() == "":
        st.warning("Please upload a file and enter a prompt.")
    else:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load audio and resample if needed
        wav, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Convert to mono and trim to 1 channel
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        st.audio(tmp_path, format="audio/wav", start_time=0)
        st.write("Generating new audio in target genre...")

        with torch.no_grad():
            output = model.generate_with_chroma(wav[None], [prompt])

        # Save the output
        out_path = os.path.join(tempfile.gettempdir(), "generated_music.wav")
        torchaudio.save(out_path, output[0].cpu(), 16000)

        st.audio(out_path, format="audio/wav")
        st.success("Transformation complete!")
