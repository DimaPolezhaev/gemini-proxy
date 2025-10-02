import os
import logging
import requests
import base64
import tempfile
import io
from flask import Flask, request, jsonify, make_response
from pydub import AudioSegment
import tarfile
import stat

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CORS helper ---
def cors(payload, code=200):
    resp = make_response(jsonify(payload), code)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# --- Пинг ---
@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "alive"})

# --- Главная страница ---
@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "✅ Server is running"})

# --- Скачивание ffmpeg/ffprobe для Vercel ---
def ensure_ffmpeg():
    ffmpeg_dir = "/tmp/ffmpeg"
    os.makedirs(ffmpeg_dir, exist_ok=True)

    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
    ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")

    if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        r = requests.get(url)
        r.raise_for_status()
        archive_path = os.path.join(ffmpeg_dir, "ffmpeg.tar.xz")
        with open(archive_path, "wb") as f:
            f.write(r.content)

        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=ffmpeg_dir)

        # Найти распакованную папку с бинарниками
        extracted_dir = next(
            os.path.join(ffmpeg_dir, d) for d in os.listdir(ffmpeg_dir)
            if os.path.isdir(os.path.join(ffmpeg_dir, d))
        )

        os.rename(os.path.join(extracted_dir, "ffmpeg"), ffmpeg_path)
        os.rename(os.path.join(extracted_dir, "ffprobe"), ffprobe_path)

        # Сделать бинарники исполняемыми
        os.chmod(ffmpeg_path, stat.S_IRWXU)
        os.chmod(ffprobe_path, stat.S_IRWXU)

    # Указать pydub использовать эти бинарники
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    # Добавить в PATH, чтобы pydub точно нашел
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    logger.info(f"ffmpeg ready: {ffmpeg_path}, ffprobe ready: {ffprobe_path}")

# --- Инициализация ffmpeg ---
ensure_ffmpeg()

# --- Эндпоинт для конвертации аудио в WAV ---
@app.route("/convert-audio", methods=["POST", "OPTIONS"])
def convert_audio():
    if request.method == "OPTIONS":
        return cors({})

    try:
        data = request.get_json(silent=True) or {}
        audio_data = data.get("audio_data")

        if not audio_data:
            return cors({"error": "Audio data not provided"}, 400)

        if len(audio_data) > 10_000_000:
            return cors({"error": "Audio too large"}, 413)

        audio_bytes = base64.b64decode(audio_data)
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()
        wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        logger.info(f"Audio converted successfully: {len(wav_bytes)} bytes")
        return cors({
            "success": True,
            "wav_data": wav_base64,
            "message": "Audio converted successfully"
        })

    except Exception as e:
        logger.exception("Audio conversion error")
        return cors({"error": f"Conversion failed: {str(e)}"}, 500)

# --- Эндпоинт генерации изображений через Gemini ---
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate_image():
    if request.method == "OPTIONS":
        return cors({})

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    image_b64 = data.get("image_base64")

    if not prompt or not image_b64:
        return cors({"error": "Prompt or image not provided"}, 400)
    if len(image_b64) > 4_000_000:
        return cors({"error": "Image too large"}, 413)

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
            ]
        }]
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        text = (r.json()
                 .get("candidates", [{}])[0]
                 .get("content", {}).get("parts", [{}])[0]
                 .get("text", ""))
        if not text.strip():
            return cors({"error": "Empty response"}, 502)
        return cors({"response": text})
    except requests.exceptions.HTTPError as e:
        return cors({"error": "Gemini API error", "details": str(e)}, r.status_code)
    except Exception as e:
        logger.exception("Proxy failure (image)")
        return cors({"error": f"Server error: {e}"}, 500)

# --- Эндпоинт анализа BirdNET (только текст) ---
@app.route("/analyze-audio", methods=["POST", "OPTIONS"])
def analyze_audio():
    if request.method == "OPTIONS":
        return cors({})

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    birdnet_results = data.get("birdnet_results")

    if not prompt:
        return cors({"error": "Prompt not provided"}, 400)
    if not birdnet_results:
        return cors({"error": "BirdNET results not provided"}, 400)

    final_prompt = f"{prompt}\n\nРезультаты анализа BirdNET:\n{birdnet_results}"
    payload = {"contents": [{"role": "user", "parts": [{"text": final_prompt}]}]}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        text = (r.json()
                 .get("candidates", [{}])[0]
                 .get("content", {}).get("parts", [{}])[0]
                 .get("text", ""))
        if not text.strip():
            return cors({"error": "Empty response"}, 502)
        return cors({"response": text})
    except requests.exceptions.HTTPError as e:
        logger.error(f"Gemini API error: {e}")
        return cors({"error": "Gemini API error", "details": str(e)}, r.status_code if hasattr(r, 'status_code') else 500)
    except Exception as e:
        logger.exception("Proxy failure (audio analysis)")
        return cors({"error": f"Server error: {e}"}, 500)

# --- Локальный запуск ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
