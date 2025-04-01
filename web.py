import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import subprocess
from io import BytesIO

# ========== CONFIG ==========
FPS = 24
DURATION_LIMIT = None  # in seconds (or None for full)
RESOLUTION = 512  # points per ring
FRAME_DIR = "frames_2d"
VIDEO_OUTPUT = "concentric_output.mp4"
SMOOTH_FACTOR = 0.1  # smoothing (0-1): lower = smoother

os.makedirs(FRAME_DIR, exist_ok=True)

# ========== Helper Functions ==========
def smooth(prev, new, alpha):
    return prev * (1 - alpha) + new * alpha

def extract_audio_features(chunk, sr):
    fft = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk), 1 / sr)
    mask = freqs > 0
    fft, freqs = fft[mask], freqs[mask]
    magnitude = np.abs(fft)
    amp = np.mean(np.abs(chunk))
    dominant_freq = freqs[np.argmax(magnitude)]
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    return amp, dominant_freq, centroid

def generate_frames(audio_path, pulse_intensity, colormap):
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_frame = int(sr / FPS)
    total_frames = len(y) // samples_per_frame

    amp_s, freq_s, cent_s = 0.1, 200, 500

    for i in range(total_frames):
        chunk = y[i * samples_per_frame:(i + 1) * samples_per_frame]
        amp, freq, cent = extract_audio_features(chunk, sr)

        amp_s = smooth(amp_s, amp, SMOOTH_FACTOR)
        freq_s = smooth(freq_s, freq, SMOOTH_FACTOR)
        cent_s = smooth(cent_s, cent, SMOOTH_FACTOR)

        num_lines = int(np.interp(cent_s, [200, 3000], [3, 30]))
        waviness = np.interp(freq_s, [50, 2000], [0.02, 0.2])
        bumpiness = np.interp(freq_s, [50, 2000], [3, 20])
        radius_base = np.interp(amp_s, [0.01, 0.3], [0.5, 1.2])

        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')

        theta = np.linspace(0, 2 * np.pi, RESOLUTION)
        for j in range(1, num_lines + 1):
            r = radius_base * j / num_lines
            radius = r + waviness * np.sin(bumpiness * theta + j)
            x = radius * np.cos(theta)
            y_ring = radius * np.sin(theta)
            ax.plot(x, y_ring, color=plt.get_cmap(colormap)(j / num_lines), linewidth=0.8)

        pulse_radius = pulse_intensity * amp_s * 2
        circle = plt.Circle((0, 0), pulse_radius, color='white', alpha=0.2, zorder=0)
        ax.add_artist(circle)

        filename = os.path.join(FRAME_DIR, f"frame_{i:04d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def generate_video_with_audio(audio_path, output_path, fps=FPS):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(FRAME_DIR, "frame_%04d.png"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True)

# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Concentric Audio Visualizer")
st.title("🔊 Concentric Audio Visualizer")

with st.form("controls"):
    audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
    pulse_intensity = st.slider("Pulse Intensity", 0.0, 1.0, 0.5, 0.05)
    colormap = st.selectbox("Color Map", ["plasma", "viridis", "inferno", "coolwarm"])
    submitted = st.form_submit_button("Generate Video")

if submitted and audio_file is not None:
    with st.spinner("Processing audio and generating video..."):
        # Save uploaded audio
        temp_audio_path = "temp_uploaded.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        # Create frames and render video
        generate_frames(temp_audio_path, pulse_intensity, colormap)
        output_path = VIDEO_OUTPUT
        generate_video_with_audio(temp_audio_path, output_path)

        # Show success and provide download
        st.success("✅ Video generated!")
        with open(output_path, "rb") as f:
            st.download_button("📥 Download MP4", f, file_name="visualization.mp4", mime="video/mp4")

        # Clean up
        os.remove(temp_audio_path)
        for file in os.listdir(FRAME_DIR):
            os.remove(os.path.join(FRAME_DIR, file))
