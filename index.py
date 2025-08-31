import os
import logging
import requests
import json
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')  # Логи в файл для диагностики
    ]
)
logger = logging.getLogger(__name__)

# Получение API-ключа из переменной окружения
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY не установлен")
    raise EnvironmentError("GEMINI_API_KEY environment variable is not set")

# Разрешённые расширения аудиофайлов
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a'}

# Функция для CORS
def cors(payload, code=200):
    resp = make_response(jsonify(payload), code)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return resp

# Проверка расширения файла
def allowed_audio_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    logger.info("Ping запрос получен")
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
            logger.error("Отсутствует prompt или image_base64")
            return cors({"error": "Prompt or image not provided"}, 400)
        
        if len(image_b64) > 4_000_000:
            logger.error("Изображение слишком большое: %d байт", len(image_b64))
            return cors({"error": "Image too large"}, 413)

        logger.info("Получен запрос на обработку изображения, размер base64: %d байт", len(image_b64))

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

        logger.info("Отправка запроса в Gemini для обработки изображения")
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        response_json = r.json()
        text = (
            response_json
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        
        if not text.strip():
            logger.error("Пустой ответ от Gemini")
            return cors({"error": "Empty response from Gemini"}, 502)
        
        logger.info("Успешный ответ от Gemini: %s...", text[:100])
        return cors({"response": text})
    
    except requests.exceptions.RequestException as e:
        logger.exception("Ошибка запроса к Gemini: %s", str(e))
        return cors({"error": f"Gemini error: {str(e)}"}, 502)
    except Exception as e:
        logger.exception("Ошибка обработки /generate: %s", str(e))
        return cors({"error": f"Server error: {str(e)}"}, 500)

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return cors({})
    
    if "file" not in request.files:
        logger.error("Отсутствует файл в поле 'file'")
        return cors({"error": "Audio file missing (field name must be 'file')"}, 400)

    audio_file = request.files["file"]
    filename = secure_filename(audio_file.filename or "audio.m4a")
    
    # Проверяем расширение файла
    if not allowed_audio_file(filename):
        logger.error("Неподдерживаемый формат файла: %s", filename)
        return cors({"error": f"Unsupported file format: {os.path.splitext(filename)[1]}. Use .mp3, .wav, or .m4a"}, 400)

    # Определяем MIME-тип
    mime_type = audio_file.mimetype or "audio/mp4"
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".mp3":
        mime_type = "audio/mpeg"
    elif extension == ".wav":
        mime_type = "audio/wav"

    # Логируем информацию о файле
    audio_file.stream.seek(0, os.SEEK_END)
    file_size = audio_file.stream.tell()
    audio_file.stream.seek(0)
    logger.info("Получен аудиофайл: %s, MIME-тип: %s, Размер: %d байт", filename, mime_type, file_size)

    # Проверка размера файла (например, не больше 10 МБ)
    if file_size > 10 * 1024 * 1024:
        logger.error("Файл слишком большой: %d байт", file_size)
        return cors({"error": "Audio file too large (>10 MB)"}, 413)

    # --- BirdNET ---
    try:
        birdnet_url = "https://birdnet.cornell.edu/api/upload"
        files = {
            "file": (
                filename,
                audio_file.stream,
                mime_type
            )
        }
        logger.info("Отправка файла на BirdNET: %s, MIME: %s", filename, mime_type)
        r = requests.post(birdnet_url, files=files, timeout=60)
        r.raise_for_status()
        birdnet_json = r.json()
        logger.info("Ответ BirdNET: %s", json.dumps(birdnet_json, ensure_ascii=False)[:200])
    except requests.exceptions.RequestException as e:
        logger.exception("Ошибка запроса к BirdNET: %s", str(e))
        return cors({"error": f"BirdNET error: {str(e)}"}, 502)
    except Exception as e:
        logger.exception("Ошибка обработки BirdNET: %s", str(e))
        return cors({"error": f"BirdNET error: {str(e)}"}, 500)

    # --- Готовим сводку для Gemini ---
    top_summary = ""
    try:
        preds = birdnet_json.get("prediction", {})
        if preds:
            items = []
            for k, v in list(preds.items())[:5]:
                items.append(f"{v.get('score', 0):.3f} — {v.get('species')}")
            top_summary = "Top predictions:\n" + "\n".join(items)
        else:
            top_summary = json.dumps(birdnet_json, ensure_ascii=False)
        logger.info("Сводка для Gemini: %s", top_summary[:200])
    except Exception as e:
        logger.exception("Ошибка обработки ответа BirdNET: %s", str(e))
        top_summary = json.dumps(birdnet_json, ensure_ascii=False)

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
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        }
        logger.info("Отправка запроса в Gemini: %s...", prompt[:100])
        g = requests.post(url, json=payload, timeout=60)
        g.raise_for_status()
        text = (
            g.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        logger.info("Ответ Gemini: %s...", text[:100])
        return cors({"raw": birdnet_json, "summary": text})
    except requests.exceptions.RequestException as e:
        logger.exception("Ошибка запроса к Gemini: %s", str(e))
        return cors({"error": f"Gemini error: {str(e)}"}, 502)
    except Exception as e:
        logger.exception("Ошибка обработки Gemini: %s", str(e))
        return cors({"error": f"Gemini error: {str(e)}"}, 500)

@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return cors({})
    logger.info("Запрос к корневому эндпоинту")
    return cors({"status": "✅ Server is running", "endpoints": ["/ping", "/generate", "/analyze"]})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info("Запуск сервера на порту %d", port)
    app.run(debug=True, host="0.0.0.0", port=port)