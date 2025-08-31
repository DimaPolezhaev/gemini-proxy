import os, logging, requests
from flask import Flask, request, jsonify, make_response

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


# Пинг
@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "alive"})


# Фото (как у тебя было)
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return cors({})

    data      = request.get_json(silent=True) or {}
    prompt    = data.get("prompt")
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


# Аудио (новое)
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return cors({})

    if "file" not in request.files:
        return cors({"error": "Audio file missing"}, 400)

    audio_file = request.files["file"]

    # 1. BirdNET API
    try:
        birdnet_url = "https://birdnet.cornell.edu/api/upload"
        files = {"file": (audio_file.filename, audio_file, "audio/mpeg")}
        r = requests.post(birdnet_url, files=files, timeout=60)
        r.raise_for_status()
        birdnet_json = r.json()
    except Exception as e:
        logger.exception("BirdNET request failed")
        return cors({"error": f"BirdNET error: {e}"}, 502)

    # 2. Gemini (красивое объяснение)
    prompt = (
        "Ты — эксперт по птицам. Вот результат анализа BirdNET:\n\n"
        f"{birdnet_json}\n\n"
        "Объясни простыми словами, какие птицы распознаны и с какой вероятностью."
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    try:
        g = requests.post(url, json=payload, timeout=60)
        g.raise_for_status()
        text = (g.json()
                 .get("candidates", [{}])[0]
                 .get("content", {}).get("parts", [{}])[0]
                 .get("text", ""))
        return cors({"raw": birdnet_json, "summary": text})
    except Exception as e:
        logger.exception("Gemini request failed")
        return cors({"error": f"Gemini error: {e}"}, 502)
