import os, logging, requests, base64, tempfile
from flask import Flask, request, jsonify, make_response
from pydub import AudioSegment
import io

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def cors(payload, code=200):
    resp = make_response(jsonify(payload), code)
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def convert_audio_to_wav(audio_b64):
    """Конвертирует аудио в WAV формат"""
    try:
        audio_data = base64.b64decode(audio_b64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Автоматическое определение формата
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Конвертируем в моно, 16kHz
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Экспортируем в WAV
            audio.export(temp_file.name, format="wav")
            
            with open(temp_file.name, 'rb') as f:
                wav_data = f.read()
            
            os.unlink(temp_file.name)
            
            return base64.b64encode(wav_data).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None

@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "alive"})

@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "✅ Server is running"})

# Эндпоинт для анализа изображений
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
                {"inline_data": {"mime_type":"image/jpeg","data":image_b64}}
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
        return cors({"error":"Gemini API error","details":str(e)}, r.status_code)
    except Exception as e:
        logger.exception("Proxy failure")
        return cors({"error":f"Server error: {e}"}, 500)

# Эндпоинт для анализа аудио
@app.route("/analyze-audio", methods=["POST", "OPTIONS"])
def analyze_audio():
    if request.method == "OPTIONS":
        return cors({})

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    audio_b64 = data.get("audio_base64")

    if not prompt or not audio_b64:
        return cors({"error": "Prompt or audio not provided"}, 400)

    # Конвертируем аудио в WAV
    processed_audio_b64 = convert_audio_to_wav(audio_b64)
    if not processed_audio_b64:
        return cors({"error": "Audio conversion failed"}, 400)

    # Отправляем аудио в Gemini
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": processed_audio_b64}}
            ]
        }]
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        text = (r.json()
                 .get("candidates", [{}])[0]
                 .get("content", {}).get("parts", [{}])[0]
                 .get("text", ""))
        if not text.strip():
            return cors({"error": "Empty response"}, 502)
        return cors({"response": text})
    except Exception as e:
        logger.exception("Audio analysis error")
        return cors({"error": f"Server error: {e}"}, 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)