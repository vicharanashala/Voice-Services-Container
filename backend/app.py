#!/usr/bin/env python3
"""
Voice Recommendation System - FastAPI Backend
Based on the notebook implementation for multilingual voice processing and semantic Q&A search.
"""

import os
import time
import logging
from typing import Optional, List
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import login
import uvicorn
import torch
# Import configuration and services
from config import config
# from services.audio_service import AudioService  # Removed - using external transcription API
from services.punctuation_service import PunctuationService
from services.transcription_service import TranscriptionService

# Custom formatter for IST timezone
class ISTFormatter(logging.Formatter):
    """Custom formatter to show logs in Indian Standard Time (IST)"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo('Asia/Kolkata'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            s = f"{s},{int(record.msecs):03d}"
        return s

# Configure logging
log_file = "/app/backend/app_performance.log"
formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger()  # Use root logger to ensure it works
logger.info("✅ Logging is working with IST timezone")

# Initialize FastAPI app
def create_app(config_name='development'):
    app = FastAPI(
        title="Voice Recommendation System API",
        description="API for multilingual voice processing and semantic Q&A search",
        version="1.0.0"
    )
    
    # Load configuration
    app_config = config[config_name]
    app.state.config = app_config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

app = create_app()

# Global service instances

mms_lid_language_detector = None
punctuation_service = None
transcription_service = None



async def normalize_transcription_result(result: dict) -> dict:
    return {
        "success": result.get("success", False),
        "transcription": result.get("transcription", ""),
        "detected_language": result.get("detected_language", "unknown"),
        "language_code": result.get("language_code", "auto"),
        "error": result.get("error")
    }

def detect_language_mms_lid(audio_file_path: str) -> dict:
    """Use MMS-LID language detection to detect language before transcription.
    
    Args:
        audio_file_path: Local path to the audio file
        
    Returns:
        Dict with detected language information
    """
    global mms_lid_language_detector
    
    try:
        # Use pre-loaded global hybrid detector instance
        if mms_lid_language_detector is None:
            logger.warning("MMS-LID language detector not initialized - falling back to auto-detect")
            return {
                "success": False,
                "detected_language": "Auto-detect",
                "language_code": None,
                "confidence": 0.0,
                "detection_method": "not_initialized",
                "error": "MMS-LID language detector not initialized"
            }
        
        # Use MMS-LID detection with pre-loaded models
        result = mms_lid_language_detector.detect_language(audio_file_path)
        logger.info(f"MMS-LID language detection result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in MMS-LID language detection: {e}")
        return {
            "success": False,
            "detected_language": "Auto-detect",
            "language_code": None,
            "confidence": 0.0,
            "detection_method": "error_fallback",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize services on FastAPI startup."""
    logger.info("FastAPI startup: Initializing services...")
    if not initialize_services():
        logger.error("Failed to initialize services during startup")
    else:
        logger.info("Services initialized successfully during startup")

def initialize_services():
    """Initialize services for hybrid language detection only."""
    global mms_lid_language_detector, punctuation_service, transcription_service
    
    try:
        app_config = app.state.config
        
        # HF Login (optional, as we pass token directly to models)
        # if app_config.HF_TOKEN:
        #    login(token=app_config.HF_TOKEN)
        
        # Initialize MMS-LID language detection models at startup
        logger.info("Initializing MMS-LID language detection models...")
        try:
            from services.mms_lid_language_detection import MmsLidLanguageDetection
            mms_lid_language_detector = MmsLidLanguageDetection(app_config)
            if not mms_lid_language_detector.initialize_models():
                logger.warning("Failed to initialize MMS-LID language detection models - continuing without it")
                mms_lid_language_detector = None
            else:
                logger.info("✅ MMS-LID language detection models loaded successfully")
        except Exception as e:
            logger.warning(f"MMS-LID language detection initialization failed: {e} - continuing without it")
            mms_lid_language_detector = None
        

        
        # Initialize punctuation service
        logger.info("Initializing punctuation service...")
        try:
            punctuation_service = PunctuationService(app_config)
            if not punctuation_service.initialize_model():
                logger.warning("Failed to initialize punctuation service - continuing without it")
                punctuation_service = None
            else:
                logger.info("✅ Punctuation service loaded successfully")
        except Exception as e:
            logger.warning(f"Punctuation service initialization failed: {e} - continuing without it")
            punctuation_service = None
            
        # Initialize transcription service
        logger.info("Initializing transcription service...")
        try:
            transcription_service = TranscriptionService(app_config)
            if not transcription_service.initialize_models():
                logger.warning("Failed to initialize transcription service models")
                transcription_service = None
            else:
                logger.info("✅ Transcription service loaded successfully")
        except Exception as e:
            logger.error(f"Transcription service initialization failed: {e}")
            transcription_service = None
        

        
        logger.info("All services initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return False



def allowed_file(filename: str) -> bool:
    """Check if uploaded file has allowed extension."""
    app_config = app.state.config
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app_config.ALLOWED_EXTENSIONS

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "services": {
                "mms_lid": mms_lid_language_detector is not None,
                "punctuation": punctuation_service is not None,
                "transcription": transcription_service is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/debug-audio")
async def debug_audio(audio: UploadFile = File(...)):
    """
    Debug endpoint to analyze audio file issues.
    
    Expected: multipart/form-data with 'audio' file field
    Returns: JSON with detailed audio file information
    """
    try:
        # audio_service is optional - we use external transcription API
        
        # Create uploads directory if it doesn't exist
        app_config = app.state.config
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        filename = audio.filename
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        try:
            # Simple file validation - just check if file exists and has size
            import os
            file_size = os.path.getsize(filepath)
            is_valid = file_size > 0
            
            return {
                "success": True,
                "filename": filename,
                "file_info": {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                },
                "validation": {
                    "is_valid": is_valid
                },
                "preprocessing_test": {
                    "success": True,
                    "note": "Using external transcription API - no local preprocessing needed"
                }
            }
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Error in debug-audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

@app.post("/punctuate")
async def punctuate_text(request: TextRequest):
    """
    Add punctuation to text.
    
    Expected: JSON with 'text' field
    Returns: JSON with punctuation results
    """
    try:
        if not punctuation_service: # Changed from audio_service to punctuation_service
            raise HTTPException(status_code=500, detail="Punctuation service not initialized")
        
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text provided")
            
        # Simple language detection for punctuation endpoint
        def is_english_text(text: str) -> bool:
            """Simple heuristic to detect if text is primarily English."""
            if not text:
                return True
            # Check if text contains mostly Latin characters
            latin_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
            total_chars = sum(1 for c in text if c.isalpha())
            if total_chars == 0:
                return True
            return (latin_chars / total_chars) > 0.9
        
        # Use punctuation service (skip for English)
        if punctuation_service and not is_english_text(text):
            logger.info("Applying punctuation for non-English text")
            result = punctuation_service.punctuate_text(text)
        else:
            if is_english_text(text):
                logger.info("Skipping punctuation for English text")
            result = {'punctuated_text': text, 'success': True}  # Skip punctuation for English or fallback
        
        return {
            "success": result['success'],
            "original_text": result.get('original_text', text),
            "punctuated_text": result.get('punctuated_text', text),
            "character_count": result.get('character_count', len(text)),
            "word_count": result.get('word_count', len(text.split())),
            "punctuation_added": result.get('punctuation_added', 0),
            "error": result.get('error', '') if not result['success'] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in punctuate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# OpenAI-compatible API endpoints

class OpenAIAudioTranscriptionRequest(BaseModel):

    """OpenAI-compatible request model for audio transcription."""

    model: str = "whisper-1"  # Compatible model name

    language: Optional[str] = None  # ISO-639-1 language code

    response_format: str = "json"  # json, text, srt, verbose_json, vtt

    temperature: float = 0.0



@app.post("/v1/audio/transcriptions")

async def openai_audio_transcriptions(

    file: UploadFile = File(...),

    model: str = Form(default="whisper-1"),

    language: Optional[str] = Form(default=None),

    response_format: str = Form(default="json"),

    temperature: float = Form(default=0.0)

):

    """

    OpenAI-compatible audio transcription endpoint.

    

    Transcribes audio into the input language.

    Compatible with OpenAI's /v1/audio/transcriptions API.

    

    Expected: multipart/form-data with:

    - 'file' (required): The audio file to transcribe

    - 'model' (optional): Model to use (default: whisper-1)

    - 'language' (optional): ISO-639-1 language code (e.g., 'en', 'hi')

    - 'response_format' (optional): Format of response (json, text, verbose_json)

    - 'temperature' (optional): Sampling temperature (0-1)

    

    Returns: JSON response in OpenAI format

    """

    start_time = time.time()

    

    try:

        # Create uploads directory if it doesn't exist

        app_config = app.state.config

        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)

        

        # Save uploaded file

        filename = file.filename

        if not allowed_file(filename):

            raise HTTPException(

                status_code=400,

                detail=f"File type not supported. Allowed types: {', '.join(app_config.ALLOWED_EXTENSIONS)}"

            )

        

        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)

        

        with open(filepath, "wb") as buffer:

            content = await file.read()

            buffer.write(content)

        

        try:

            # Step 1: Language Detection (if not provided)

            if language:

                # Use provided language code

                detected_language_code = language

                detected_language_name = language.upper()

                confidence = 1.0

                detection_method = "user_provided"

            else:

                # Detect language using MMS-LID

                lang_result = detect_language_mms_lid(filepath)

                

                if not lang_result['success']:

                    raise HTTPException(

                        status_code=500,

                        detail=f"Language detection failed: {lang_result.get('error', 'Unknown error')}"

                    )

                

                detected_language_code = lang_result['language_code']

                detected_language_name = lang_result['detected_language']

                confidence = lang_result.get('confidence', 0.0)

                detection_method = lang_result.get('detection_method', 'auto')

            

            # Validate supported languages

            supported_indian_languages = {

                'as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'ta', 'te', 'ur'

            }

            

            if detected_language_code != 'en' and detected_language_code not in supported_indian_languages:

                logger.warning(f"Unsupported language detected: {detected_language_name} ({detected_language_code})")

                raise HTTPException(

                    status_code=400,

                    detail="Unsupported language. Only English and Indian languages are supported."

                )

            

            logger.info(f"Language validation passed: {detected_language_name} ({detected_language_code})")

            
            # Step 2: Transcribe via internal service
            if not transcription_service:
                 raise HTTPException(status_code=500, detail="Transcription service not available")

            transcription_result = await transcription_service.process_transcription(filepath, language_code=detected_language_code)
            
            # Normalize result
            transcription_result = await normalize_transcription_result(transcription_result)

            

            if not transcription_result.get('success', False):

                raise HTTPException(

                    status_code=500,

                    detail=f"Transcription failed: {transcription_result.get('error', 'unknown')}"

                )

            

            # Validate transcription quality

            original_text = transcription_result.get('transcription', '').strip()

            

            if not original_text or len(original_text) < 2:

                raise HTTPException(

                    status_code=400,

                    detail="Audio file is not transcribable. Please provide clear audio."

                )

            

            # Quality check for English

            if detected_language_code == 'en':

                latin_chars = sum(1 for c in original_text if c.isalpha() and ord(c) < 128)

                total_alpha_chars = sum(1 for c in original_text if c.isalpha())

                

                if total_alpha_chars > 0:

                    english_ratio = latin_chars / total_alpha_chars

                    if english_ratio < 0.5:

                        logger.warning(f"Low English character ratio: {english_ratio:.2f}")

                        raise HTTPException(

                            status_code=400,

                            detail="Audio quality is too low. Please provide clear audio."

                        )

            

            # Step 3: Add punctuation (if not English)

            if punctuation_service and detected_language_code != 'en':

                logger.info(f"Applying punctuation for {detected_language_name}")

                punctuation_result = punctuation_service.punctuate_text(original_text)

                final_text = punctuation_result.get('punctuated_text', original_text)

            else:

                final_text = original_text

            

            # Calculate duration and processing time

            duration = time.time() - start_time

            

            # Return response based on format

            if response_format == "text":

                # Plain text response

                return final_text

            

            elif response_format == "verbose_json":

                # Verbose JSON response (OpenAI format with additional details)

                return {

                    "task": "transcribe",

                    "language": detected_language_code,

                    "duration": round(duration, 2),

                    "text": final_text,

                    "segments": [

                        {

                            "id": 0,

                            "seek": 0,

                            "start": 0.0,

                            "end": round(duration, 2),

                            "text": final_text,

                            "tokens": [],

                            "temperature": temperature,

                            "avg_logprob": -0.3,

                            "compression_ratio": 1.0,

                            "no_speech_prob": 0.0

                        }

                    ],

                    "words": []

                }

            

            else:  # default: "json"

                # Standard JSON response (OpenAI format)

                return {

                    "text": final_text

                }

        

        finally:

            # Clean up uploaded file

            if os.path.exists(filepath):

                os.remove(filepath)

    

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Error in OpenAI-compatible transcription endpoint: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





@app.exception_handler(413)
async def too_large_handler(request, exc):
    """Handle file too large error."""
    return JSONResponse(
        status_code=413,
        content={"success": False, "error": "File too large. Maximum size is 50MB."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error occurred."}
    )

if __name__ == '__main__':
    logger.info("Initializing Voice Recommendation System API...")
    
    # Initialize services
    if not initialize_services():
        logger.error("Failed to initialize services. Exiting.")
        exit(1)
    
    logger.info("All systems initialized successfully!")
    
    # Run the FastAPI app with uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5020,
        reload=True,
        log_level="info",
        workers=1  
    )

    #uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2
