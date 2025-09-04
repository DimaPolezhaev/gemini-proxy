import os
import logging
import requests
import json
import sys
import tempfile
import subprocess
import uuid
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def cors(payload, code=200):
    resp = make_response(jsonify(payload), code)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "alive"})


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return cors({})

    try:
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

        # 🔧 Исправлено: убраны лишние пробелы в URL
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        text = (
            r.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text.strip():
            return cors({"error": "Empty response from Gemini"}, 502)
        return cors({"response": text})
    except Exception as e:
        logger.exception("Proxy failure (generate)")
        return cors({"error": f"Server error: {e}"}, 500)


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return cors({})

    if "file" not in request.files:
        return cors({"error": "Audio file missing (field name must be 'file')"}, 400)

    audio_file = request.files["file"]
    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, f"{uuid.uuid4()}.wav")
    audio_file.save(input_path)
    output_path = os.path.join(tmpdir, "birdnet_output.json")

    # --- BirdNET-Analyzer ---
    try:
        logger.info(f"Running BirdNET on {input_path}")

        # ✅ Правильный вызов: использует __main__.py через -m birdnet_analyzer.analyze
        result = subprocess.run([
            sys.executable, "-m", "birdnet_analyzer.analyze",
            input_path,              # позиционный аргумент INPUT
            "-o", output_path,       # выходной путь
            "--min_conf", "0.1",     # минимальная уверенность
            "--threads", "1",        # меньше нагрузки на Vercel
            "--rtype", "table",      # формат вывода
            "--locale", "en"         # язык (по желанию)
        ], capture_output=True, text=True)

        logger.info("BirdNET stdout:\n" + result.stdout)
        if result.stderr:
            logger.error("BirdNET stderr:\n" + result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"BirdNET exited with code {result.returncode}")

        # Убедимся, что файл создан
        if not os.path.exists(output_path):
            raise FileNotFoundError("BirdNET did not create output file")

        with open(output_path, "r", encoding="utf-8") as f:
            birdnet_json = json.load(f)

    except Exception as e:
        logger.exception("BirdNET-Analyzer failed")
        return cors({"error": f"BirdNET-Analyzer error: {str(e)}"}, 502)

    # --- Готовим сводку для Gemini ---
    preds = birdnet_json.get("predictions", [])
    if preds:
        items = [f"{p.get('confidence', 0):.3f} — {p.get('species', '?')}" for p in preds[:5]]
        top_summary = "Top predictions:\n" + "\n".join(items)
    else:
        top_summary = json.dumps(birdnet_json, ensure_ascii=False, indent=2)

    prompt = (
        "Ты — эксперт-орнитолог. Вот результат BirdNET:\n\n"
        f"{top_summary}\n\n"
        "Коротко и понятно объясни пользователю: "
        "1) какая птица наиболее вероятна, 2) степень уверенности, 3) рекомендация."
    )

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )
        g = requests.post(url, json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]}, timeout=120)
        g.raise_for_status()
        text = (
            g.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return cors({"raw": birdnet_json, "summary": text})
    except Exception as e:
        logger.exception("Gemini request failed")
        return cors({"error": f"Gemini error: {str(e)}"}, 502)


@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return cors({})
    return cors({
        "status": "✅ Server is running",
        "endpoints": ["/ping", "/generate", "/analyze"]
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))