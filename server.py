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
        return True
        
    logger.info("🔄 Initializing FFmpeg...")
    start_time = time.time()
    
    ffmpeg_dir = "/tmp/ffmpeg"
    os.makedirs(ffmpeg_dir, exist_ok=True)

    _ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
    _ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")

    # Проверяем, существуют ли уже бинарники
    if os.path.exists(_ffmpeg_path) and os.path.exists(_ffprobe_path):
        logger.info("✅ FFmpeg binaries already exist, reusing...")
    else:
        try:
            logger.info("📥 Downloading FFmpeg...")
            # Используем более быстрый источник
            url = "https://github.com/eugeneware/ffmpeg-static/releases/download/b5.0.1/linux-x64"
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            # Сохраняем ffmpeg
            with open(_ffmpeg_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Создаем симлинк для ffprobe (используем тот же бинарник)
            os.symlink(_ffmpeg_path, _ffprobe_path)
            
            # Делаем исполняемыми
            os.chmod(_ffmpeg_path, stat.S_IRWXU)
            os.chmod(_ffprobe_path, stat.S_IRWXU)
            
            logger.info("✅ FFmpeg downloaded and configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to download FFmpeg: {e}")
            # Создаем заглушки чтобы не падать
            with open(_ffmpeg_path, "wb") as f:
                f.write(b"#!/bin/bash\necho 'FFmpeg not available'")
            with open(_ffprobe_path, "wb") as f:
                f.write(b"#!/bin/bash\necho 'FFprobe not available'")
            os.chmod(_ffmpeg_path, stat.S_IRWXU)
            os.chmod(_ffprobe_path, stat.S_IRWXU)

    # Настраиваем pydub
    try:
        AudioSegment.converter = _ffmpeg_path
        AudioSegment.ffprobe = _ffprobe_path
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Тестируем ffmpeg
        test_result = os.system(f"{_ffmpeg_path} -version > /dev/null 2>&1")
        if test_result == 0:
            logger.info(f"✅ FFmpeg initialized successfully in {time.time() - start_time:.2f}s")
            _ffmpeg_initialized = True
            return True
        else:
            logger.warning("⚠️ FFmpeg test failed, audio conversion may not work")
            _ffmpeg_initialized = True
            return False
            
    except Exception as e:
        logger.error(f"❌ FFmpeg configuration failed: {e}")
        _ffmpeg_initialized = True
        return False

# --- Пинг ---
@app.route("/ping", methods=["GET", "OPTIONS"])
def ping():
    if request.method == "OPTIONS":
        return cors({})
    return cors({"status": "alive", "timestamp": time.time()})

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
        # Ленивая инициализация ffmpeg
        ffmpeg_ready = ensure_ffmpeg()
        if not ffmpeg_ready:
            return cors({
                "error": "FFmpeg not available", 
                "message": "Audio conversion temporarily unavailable"
            }, 503)

        data = request.get_json(silent=True) or {}
        audio_data = data.get("audio_data")

        if not audio_data:
            return cors({"error": "Audio data not provided"}, 400)

        # Проверка размера
        if len(audio_data) > 10_000_000:  # ~10MB
            return cors({"error": "Audio file too large (max 10MB)"}, 413)

        logger.info(f"🔄 Converting audio, size: {len(audio_data)} bytes")
        
        # Декодируем base64
        audio_bytes = base64.b64decode(audio_data)
        
        # Определяем формат по расширению или заголовкам
        audio_file = io.BytesIO(audio_bytes)
        
        # Конвертируем в WAV
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
            
        # Проверка размера
        if len(image_b64) > 4_000_000:
            return cors({"error": "Image too large (max 4MB)"}, 413)

        logger.info(f"🔄 Processing image analysis, prompt length: {len(prompt)}, image size: {len(image_b64)} bytes")

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
                "maxOutputTokens": 2048,
            }
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        response = requests.post(url, json=payload, timeout=60)
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
        return cors({"error": "AI service timeout"}, 504)
    except requests.exceptions.HTTPError as e:
        logger.error(f"🔴 Gemini API HTTP error: {e}")
        status_code = e.response.status_code if e.response else 500
        return cors({
            "error": "AI service error", 
            "details": str(e)
        }, status_code)
    except Exception as e:
        logger.exception(f"❌ Image analysis error: {e}")
        return cors({
            "error": f"Server error: {str(e)}"
        }, 500)

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

        logger.info(f"🔄 Processing audio analysis, prompt length: {len(prompt)}")

        final_prompt = f"{prompt}\n\nРезультаты анализа BirdNET:\n{birdnet_results}"
        
        payload = {
            "contents": [{
                "role": "user", 
                "parts": [{"text": final_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        response = requests.post(url, json=payload, timeout=30)
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
        return cors({"error": "AI service timeout"}, 504)
    except requests.exceptions.HTTPError as e:
        logger.error(f"🔴 Gemini API HTTP error for audio analysis: {e}")
        status_code = e.response.status_code if e.response else 500
        return cors({
            "error": "AI service error", 
            "details": str(e)
        }, status_code)
    except Exception as e:
        logger.exception(f"❌ Audio analysis error: {e}")
        return cors({
            "error": f"Server error: {str(e)}"
        }, 500)

# --- Health check с информацией о ffmpeg ---
@app.route("/health", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return cors({})
    
    ffmpeg_status = "ready" if _ffmpeg_initialized else "not_initialized"
    return cors({
        "status": "healthy",
        "timestamp": time.time(),
        "ffmpeg": ffmpeg_status,
        "gemini_api_key": "configured" if GEMINI_API_KEY else "missing"
    })

# --- Локальный запуск ---
if __name__ == "__main__":
    logger.info("🚀 Starting server...")
    # Предварительная инициализация ffmpeg при локальном запуске
    ensure_ffmpeg()
    app.run(host="0.0.0.0", port=5000, debug=True)