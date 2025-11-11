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
import time
import subprocess
import sys

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Глобальные переменные для кэширования
_ffmpeg_initialized = False
_ffmpeg_path = None
_ffprobe_path = None

def cors(payload, code=200):
    resp = make_response(jsonify(payload), code)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

def ensure_ffmpeg():
    global _ffmpeg_initialized, _ffmpeg_path, _ffprobe_path
    
    if _ffmpeg_initialized:
        logger.info("✅ FFmpeg already initialized")
        return True
        
    logger.info("🔄 Initializing FFmpeg...")
    start_time = time.time()
    
    # Пробуем разные пути для надежности
    possible_paths = [
        "/var/task/ffmpeg",  # Сохраняется между запросами
        "/tmp/ffmpeg",       # Временный путь
        "./ffmpeg"           # Текущая директория
    ]
    
    for ffmpeg_dir in possible_paths:
        try:
            os.makedirs(ffmpeg_dir, exist_ok=True)
            _ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
            _ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")
            
            # Проверяем существующие бинарники
            if os.path.exists(_ffmpeg_path) and os.path.exists(_ffprobe_path):
                logger.info(f"✅ Found existing FFmpeg in {ffmpeg_dir}")
                break
                
            # Скачиваем если нет
            logger.info(f"📥 Downloading FFmpeg to {ffmpeg_dir}...")
            ffmpeg_url = "https://github.com/eugeneware/ffmpeg-static/releases/download/b5.0.1/linux-x64"
            response = requests.get(ffmpeg_url, timeout=60)
            response.raise_for_status()
            
            with open(_ffmpeg_path, "wb") as f:
                f.write(response.content)
            
            # Создаем ffprobe как симлинк
            if not os.path.exists(_ffprobe_path):
                os.symlink(_ffmpeg_path, _ffprobe_path)
            
            # Права на выполнение
            os.chmod(_ffmpeg_path, 0o755)
            os.chmod(_ffprobe_path, 0o755)
            
            logger.info(f"✅ FFmpeg downloaded to {ffmpeg_dir}")
            break
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup FFmpeg in {ffmpeg_dir}: {e}")
            continue
    
    if not _ffmpeg_path or not os.path.exists(_ffmpeg_path):
        logger.error("❌ All FFmpeg setup attempts failed")
        return False

    try:
        # Настраиваем pydub
        AudioSegment.converter = _ffmpeg_path
        AudioSegment.ffprobe = _ffprobe_path
        
        # Добавляем в PATH
        ffmpeg_dir = os.path.dirname(_ffmpeg_path)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Тестируем
        result = subprocess.run(
            [_ffmpeg_path, "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            version = result.stdout.split('\n')[0] if result.stdout else "unknown"
            logger.info(f"✅ FFmpeg ready: {version}")
            _ffmpeg_initialized = True
            return True
        else:
            logger.error(f"❌ FFmpeg test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FFmpeg configuration failed: {e}")
        return False

# Предварительная инициализация
logger.info("🚀 Server starting, initializing FFmpeg...")
ffmpeg_ready = ensure_ffmpeg()
if ffmpeg_ready:
    logger.info("🎉 FFmpeg initialized successfully")
else:
    logger.warning("⚠️ FFmpeg initialization failed, audio features will not work")

# --- Пинг ---
@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({
        "status": "alive", 
        "timestamp": time.time(),
        "ffmpeg_ready": _ffmpeg_initialized
    })

# --- Главная страница ---
@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return cors({})
    return cors({
        "status": "✅ Server is running", 
        "ffmpeg_ready": _ffmpeg_initialized,
        "timestamp": time.time()
    })

# --- Эндпоинт для конвертации аудио в WAV ---
@app.route("/convert-audio", methods=["POST", "OPTIONS"])
def convert_audio():
    if request.method == "OPTIONS":
        return cors({})

    try:
        if not ensure_ffmpeg():
            return cors({
                "error": "FFmpeg not available", 
                "message": "Audio conversion temporarily unavailable"
            }, 503)

        data = request.get_json(silent=True) or {}
        audio_data = data.get("audio_data")
        filename = data.get("filename", "audio")

        if not audio_data:
            return cors({"error": "Audio data not provided"}, 400)

        if len(audio_data) > 8_000_000:
            return cors({"error": "Audio file too large (max 8MB)"}, 413)

        logger.info(f"🔄 Converting audio: {filename}, size: {len(audio_data)} bytes")
        
        audio_bytes = base64.b64decode(audio_data)
        
        # Конвертируем в WAV
        audio_file = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()
        wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        logger.info(f"✅ Audio converted successfully: {len(wav_bytes)} bytes")
        
        return cors({
            "success": True,
            "wav_data": wav_base64,
            "original_size": len(audio_bytes),
            "converted_size": len(wav_bytes),
            "message": "Audio converted to WAV successfully"
        })

    except Exception as e:
        logger.exception(f"❌ Audio conversion error: {e}")
        return cors({
            "error": f"Conversion failed: {str(e)}",
            "message": "Please try with a different audio format"
        }, 500)

# --- Эндпоинт генерации изображений через Gemini ---
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate_image():
    if request.method == "OPTIONS":
        return cors({})

    start_time = time.time()
    
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt")
        image_b64 = data.get("image_base64")

        if not prompt:
            return cors({"error": "Prompt not provided"}, 400)
        if not image_b64:
            return cors({"error": "Image not provided"}, 400)
            
        if len(image_b64) > 3_500_000:
            return cors({"error": "Image too large (max 3.5MB)"}, 413)

        logger.info(f"🔄 Processing image analysis, image size: {len(image_b64)} bytes")

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-1.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        
        result = response.json()
        text = (result
                 .get("candidates", [{}])[0]
                 .get("content", {})
                 .get("parts", [{}])[0]
                 .get("text", ""))
                 
        if not text.strip():
            logger.warning("⚠️ Empty response from Gemini API")
            return cors({"error": "Empty response from AI service"}, 502)
            
        processing_time = time.time() - start_time
        logger.info(f"✅ Image analysis completed in {processing_time:.2f}s")
        
        return cors({
            "response": text,
            "processing_time": processing_time
        })
        
    except requests.exceptions.Timeout:
        logger.error("⏰ Gemini API timeout")
        return cors({"error": "AI service timeout - try again"}, 504)
    except requests.exceptions.HTTPError as e:
        logger.error(f"🔴 Gemini API HTTP error: {e}")
        status_code = e.response.status_code if e.response else 500
        
        if status_code == 429:
            return cors({"error": "Rate limit exceeded - try again later"}, 429)
        elif status_code == 403:
            return cors({"error": "API key invalid or quota exceeded"}, 403)
        else:
            return cors({"error": "AI service temporarily unavailable"}, 503)
            
    except Exception as e:
        logger.exception(f"❌ Image analysis error: {e}")
        return cors({
            "error": "Service temporarily unavailable - try again"
        }, 503)

# --- Эндпоинт анализа BirdNET (только текст) ---
@app.route("/analyze-audio", methods=["POST", "OPTIONS"])
def analyze_audio():
    if request.method == "OPTIONS":
        return cors({})

    start_time = time.time()
    
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt")
        birdnet_results = data.get("birdnet_results")

        if not prompt:
            return cors({"error": "Prompt not provided"}, 400)
        if not birdnet_results:
            return cors({"error": "BirdNET results not provided"}, 400)

        logger.info(f"🔄 Processing audio analysis")

        final_prompt = f"{prompt}\n\nРезультаты анализа BirdNET:\n{birdnet_results}"
        
        payload = {
            "contents": [{
                "role": "user", 
                "parts": [{"text": final_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 800,
            }
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-1.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        response = requests.post(url, json=payload, timeout=25)
        response.raise_for_status()
        
        result = response.json()
        text = (result
                 .get("candidates", [{}])[0]
                 .get("content", {})
                 .get("parts", [{}])[0]
                 .get("text", ""))
                 
        if not text.strip():
            logger.warning("⚠️ Empty response from Gemini API for audio analysis")
            return cors({"error": "Empty response from AI service"}, 502)
            
        processing_time = time.time() - start_time
        logger.info(f"✅ Audio analysis completed in {processing_time:.2f}s")
        
        return cors({
            "response": text,
            "processing_time": processing_time
        })
        
    except requests.exceptions.Timeout:
        logger.error("⏰ Gemini API timeout for audio analysis")
        return cors({"error": "AI service timeout - try again"}, 504)
    except requests.exceptions.HTTPError as e:
        logger.error(f"🔴 Gemini API HTTP error for audio analysis: {e}")
        return cors({"error": "AI service temporarily unavailable"}, 503)
    except Exception as e:
        logger.exception(f"❌ Audio analysis error: {e}")
        return cors({"error": "Service temporarily unavailable - try again"}, 503)

# --- Health check ---
@app.route("/health", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return cors({})
    
    return cors({
        "status": "healthy",
        "timestamp": time.time(),
        "ffmpeg_ready": _ffmpeg_initialized,
        "service": "bird_identifier_api"
    })

# --- Локальный запуск ---
if __name__ == "__main__":
    logger.info("🚀 Starting optimized server...")
    app.run(host="0.0.0.0", port=5000, debug=False)