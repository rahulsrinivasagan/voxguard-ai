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
app = FastAPI(title="VoxGuard – Voice Authenticity Detector")

app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = "my_secret_key_123"

# Render / CPU safety
os.environ["OMP_NUM_THREADS"] = "1"

# =====================================
# WHISPER (LAZY LOAD – TINY MODEL)
# =====================================
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model

LANGUAGE_LABELS = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam"
}

MAX_AUDIO_DURATION = 15  # seconds

# =====================================
# ROOT
# =====================================
@app.get("/")
def root():
    return RedirectResponse(url="/ui")

# =====================================
# FAST LANGUAGE DETECTION (NO TRANSCRIBE)
# =====================================
def identify_language(audio, sr):
    try:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

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
# AI VOICE HEURISTIC (FAST)
# =====================================
def analyze_voice_nature(audio, sr):
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

    rms = librosa.feature.rms(y=audio)[0]
    rms_var = float(np.var(rms))

    try:
        pitch = librosa.yin(audio, fmin=50, fmax=300)
        pitch = pitch[~np.isnan(pitch)]
        pitch_var = float(np.var(pitch)) if len(pitch) else 0
    except:
        pitch_var = 0

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo_var = float(np.var(onset_env))

    score = 0.0
    if flatness < 0.018: score += 0.35
    if rms_var < 0.0015: score += 0.25
    if pitch_var < 12: score += 0.25
    if tempo_var < 0.02: score += 0.15

    return min(score, 1.0)

# =====================================
# UI
# =====================================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>VoxGuard AI</title>
<link rel="icon" href="/static/icon.png">
<style>
body {
    margin: 0;
    height: 100vh;
    background: url("/static/background_image.jpg") center/cover;
    font-family: Arial, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
}
.overlay {
    background: rgba(0,0,0,0.6);
    width: 100%;
    height: 100%;
    display:flex;
    align-items:center;
    justify-content:center;
}
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    padding: 40px;
    border-radius: 18px;
    width: 420px;
    text-align: center;
    color: white;
}
button {
    padding: 14px 34px;
    border-radius: 30px;
    border: none;
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color: white;
    font-size: 16px;
    cursor: pointer;
}
.result {
    margin-top: 25px;
    padding: 18px;
    border-radius: 12px;
    background: rgba(255,255,255,0.22);
    font-weight: bold;
}
input { display:none; }
</style>
</head>

<body>
<div class="overlay">
<div class="card">
<h2>VoxGuard – Voice Authenticity Detector</h2>
<p>Detect Human vs AI-Generated Voices</p>

<input type="file" id="audio" accept=".mp3,.wav">
<button onclick="document.getElementById('audio').click()">Detect Voice</button>

<div class="result" id="result"></div>
</div>
</div>

<script>
document.getElementById("audio").addEventListener("change", async () => {
    const file = audio.files[0];
    const result = document.getElementById("result");
    if (!file) return;

    result.innerHTML = "Analyzing voice...";

    const data = new FormData();
    data.append("file", file);

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20000);

    try {
        const res = await fetch("/detect-voice", {
            method: "POST",
            headers: {"x-api-key":"my_secret_key_123"},
            body: data,
            signal: controller.signal
        });

        const json = await res.json();
        if (res.ok) {
            result.innerHTML =
                "Voice Type: " + json.classification +
                "<br>Confidence: " + json.confidence +
                "<br>Language: " + json.detected_language;
        } else {
            result.innerHTML = json.detail;
        }
    } catch {
        result.innerHTML = "Processing took too long. Try shorter audio.";
    }
});
</script>
</body>
</html>
"""

# =====================================
# API
# =====================================
@app.post("/detect-voice")
async def detect_voice(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if not file.filename.lower().endswith((".wav",".mp3")):
        raise HTTPException(status_code=400, detail="Upload WAV or MP3 only")

    audio_bytes = await file.read()
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

    duration = len(audio) / sr
    if duration > MAX_AUDIO_DURATION:
        raise HTTPException(
            status_code=400,
            detail="Audio too long. Max 15 seconds allowed."
        )

    language = identify_language(audio, sr)
    ai_score = round(analyze_voice_nature(audio, sr), 2)

    if ai_score >= 0.8:
        classification = "AI-Generated Voice"
        confidence = ai_score
    else:
        classification = "Human Voice"
        confidence = round(1 - ai_score, 2)

    return {
        "classification": classification,
        "confidence": confidence,
        "detected_language": language
    }
