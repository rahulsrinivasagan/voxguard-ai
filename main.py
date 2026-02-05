from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
import librosa
import numpy as np
import io
import whisper
import os

# =====================================
# APPLICATION SETUP
# =====================================
app = FastAPI(title="VoxGuard AI – Voice Authenticity Checker")

app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = "my_secret_key_123"

# Reduce CPU threads (important for Render & local)
os.environ["OMP_NUM_THREADS"] = "1"

# =====================================
# WHISPER LAZY LOADING (CRITICAL)
# =====================================
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
    return whisper_model

LANGUAGE_LABELS = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam"
}

# =====================================
# ROOT ROUTE
# =====================================
@app.get("/")
def home():
    return RedirectResponse(url="/ui")

# =====================================
# FAST LANGUAGE IDENTIFICATION (FIXED)
# =====================================
def identify_language(audio, sample_rate):
    """
    FAST Whisper language detection (NO transcription)
    Works perfectly for short audio clips
    """
    try:
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        model = get_whisper_model()

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        lang_code = max(probs, key=probs.get)

        return LANGUAGE_LABELS.get(lang_code, "Unknown")

    except Exception as e:
        print("Language detection error:", e)
        return "Unknown"

# =====================================
# VOICE AUTHENTICITY ANALYSIS
# =====================================
def analyze_voice_nature(audio, sample_rate):
    spectral_smoothness = float(
        np.mean(librosa.feature.spectral_flatness(y=audio))
    )

    energy_levels = librosa.feature.rms(y=audio)[0]
    energy_variation = float(np.var(energy_levels))

    try:
        pitch_values = librosa.yin(audio, fmin=50, fmax=300)
        pitch_values = pitch_values[~np.isnan(pitch_values)]
        pitch_variation = float(np.var(pitch_values)) if len(pitch_values) else 0
    except:
        pitch_variation = 0

    rhythm_strength = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    rhythm_variation = float(np.var(rhythm_strength))

    score = 0.0

    if spectral_smoothness < 0.018:
        score += 0.35
    if energy_variation < 0.0015:
        score += 0.25
    if pitch_variation < 12:
        score += 0.25
    if rhythm_variation < 0.02:
        score += 0.15

    return min(score, 1.0)

# =====================================
# USER INTERFACE (UNCHANGED)
# =====================================
@app.get("/ui", response_class=HTMLResponse)
def user_interface():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>VoxGuard AI</title>
    <link rel="icon" href="/static/icon.png">
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            height: 100vh;
            background-image: url("/static/background_image.jpg");
            background-size: cover;
            background-position: center;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .overlay {
            background: rgba(0, 0, 0, 0.6);
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .card {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            padding: 40px;
            border-radius: 18px;
            width: 420px;
            text-align: center;
            color: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
            margin-top: 60px;
        }
        .title {
            font-size: 32px;
            font-weight: 700;
        }
        .subtitle {
            margin-top: 10px;
            margin-bottom: 30px;
            font-size: 15px;
            color: #cfd8dc;
        }
        input[type="file"] {
            display: none;
        }
        .button {
            padding: 14px 34px;
            font-size: 16px;
            border: none;
            border-radius: 30px;
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: white;
            cursor: pointer;
        }
        .result {
            margin-top: 25px;
            padding: 18px;
            border-radius: 12px;
            background: rgba(255,255,255,0.22);
            font-size: 17px;
            font-weight: bold;
        }
    </style>
</head>

<body>
<div class="overlay">
    <div class="card">
        <h1 class="title">VoxGuard – Voice Authenticity Detector</h1>
        <p class="subtitle">Verify whether a voice is Human or AI-Generated</p>

        <input type="file" id="audioInput" accept=".mp3,.wav">
        <button class="button" onclick="document.getElementById('audioInput').click()">Detect Voice</button>

        <div class="result" id="output"></div>
    </div>
</div>

<script>
document.getElementById("audioInput").addEventListener("change", async function () {
    const file = this.files[0];
    const output = document.getElementById("output");

    if (!file) return;

    output.innerHTML = "Analyzing the voice sample...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/detect-voice", {
            method: "POST",
            headers: { "x-api-key": "my_secret_key_123" },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            output.innerHTML =
                "Voice Type: " + data.classification +
                "<br>Confidence: " + data.confidence +
                "<br>Language: " + data.detected_language;
        } else {
            output.innerHTML = data.detail;
        }
    } catch {
        output.innerHTML = "Unable to reach the server.";
    }
});
</script>
</body>
</html>
"""

# =====================================
# CORE API ENDPOINT
# =====================================
@app.post("/detect-voice")
async def detect_voice(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Access denied.")

    if not file.filename.lower().endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Upload MP3 or WAV audio.")

    try:
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        audio, sample_rate = librosa.load(audio_stream, sr=None, mono=True)

        detected_language = identify_language(audio, sample_rate)

        if len(audio) < sample_rate:
            return {
                "classification": "Human Voice",
                "confidence": 0.5,
                "detected_language": detected_language
            }

        ai_probability = round(analyze_voice_nature(audio, sample_rate), 2)

        if ai_probability >= 0.8:
            classification = "AI-Generated Voice"
            confidence = ai_probability
        else:
            classification = "Human Voice"
            confidence = round(1 - ai_probability, 2)

        return {
            "classification": classification,
            "confidence": confidence,
            "detected_language": detected_language
        }

    except Exception as e:
        print("Processing error:", e)
        raise HTTPException(status_code=500, detail="Audio processing failed.")
