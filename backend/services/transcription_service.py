
"""
Transcription Service
Service for audio transcription (Whisper + IndicConformer) directly integrated into the backend.
"""

import os
import logging
import torch
import librosa
import numpy as np
import tempfile
import asyncio
from typing import Optional, Dict, Any, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel
import soundfile as sf
import audioread

logger = logging.getLogger(__name__)

class TranscriptionService:
    """Service for handling transcription requests using local models."""
    
    def __init__(self, config=None):
        self.config = config
        self.whisper_model = None
        self.whisper_processor = None
        self.indic_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False

    def initialize_models(self):
        """Initialize Whisper and IndicConformer models."""
        try:
            logger.info(f"Initializing Transcription Service models on device: {self.device}")
            
            # Read HF auth token
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN not set. Access to gated Hugging Face models might fail.")
            
            # Load Whisper (English)
            whisper_model_name = "openai/whisper-small"
            logger.info(f"Loading Whisper model: {whisper_model_name}")
            self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(self.device)
            self.whisper_model.eval()
            
            # Load IndicConformer (Indian Languages)
            indic_model_name = "ai4bharat/indic-conformer-600m-multilingual"
            logger.info(f"Loading IndicConformer model: {indic_model_name}")
            indic_model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if hf_token:
                indic_model_kwargs["use_auth_token"] = hf_token
            
            self.indic_model = AutoModel.from_pretrained(
                indic_model_name,
                **indic_model_kwargs,
            ).to(self.device)
            self.indic_model.eval()
            
            self.initialized = True
            logger.info("✅ Transcription models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize transcription models: {e}")
            self.initialized = False
            return False

    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file: convert to mono and resample to target sampling rate.
        """
        # Try to fix recorded audio issues first
        fixed_audio_path = self.fix_recorded_audio_issues(audio_path)
        use_fixed_file = fixed_audio_path != audio_path
        
        try:
            waveform, orig_sr = None, None
            file_to_load = fixed_audio_path if use_fixed_file else audio_path
            
            # Try loading with librosa (robust fallback)
            try:
                # Load with librosa
                audio_data, orig_sr = librosa.load(file_to_load, sr=target_sr, mono=True)
                
                # Normalize
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.95
                
                # Convert to torch
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
                
            except Exception as e:
                logger.debug(f"Librosa load failed: {e}. Trying torchaudio.")
                import torchaudio
                waveform, orig_sr = torchaudio.load(file_to_load)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample
                if orig_sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)
                    orig_sr = target_sr

            return waveform, target_sr
            
        except Exception as e:
            raise Exception(f"Error in audio preprocessing: {str(e)}")
        finally:
            if use_fixed_file and os.path.exists(fixed_audio_path):
                try:
                    os.remove(fixed_audio_path)
                except:
                    pass

    def fix_recorded_audio_issues(self, audio_path: str) -> str:
        """
        Fix common issues with recorded audio files using librosa/soundfile.
        Returns the path to the fixed audio file.
        """
        try:
            # Check if file needs fixing (e.g. valid load)
            with sf.SoundFile(audio_path) as f:
                return audio_path # Seems valid
        except Exception:
            pass # Continue to fix
            
        try:
            # Use librosa to decode and rewrite
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            base_name = os.path.splitext(audio_path)[0]
            fixed_path = f"{base_name}_fixed.wav"
            
            sf.write(fixed_path, y, sr, subtype='PCM_16')
            return fixed_path
        except Exception as e:
            logger.debug(f"Could not fix audio: {e}")
            return audio_path

    def transcribe_with_indic_conformer(self, waveform, language_code: str) -> str:
        """Transcribe audio using IndicConformer."""
        try:
            with torch.no_grad():
                waveform_device = waveform.to(self.device)
                transcription = self.indic_model(waveform_device, language_code, "rnnt")
            return transcription.strip() if transcription else ""
        except Exception as e:
            logger.error(f"Error in IndicConformer transcription: {e}")
            raise

    def transcribe_with_whisper(self, waveform, language_code: Optional[str]) -> str:
        """Transcribe audio using Whisper."""
        try:
            with torch.no_grad():
                input_features = self.whisper_processor(
                    waveform.squeeze(), 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate tokens
                # language_code can be passed to generate if needed, but Whisper usually auto-detects or we can force
                if language_code and language_code in ["en", "eng"]:
                     predicted_ids = self.whisper_model.generate(input_features, language="english")
                else:
                     predicted_ids = self.whisper_model.generate(input_features)

                transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            raise

    def detect_language_simple(self, audio_path: str) -> str:
        """Simple language detection fallback."""
        # This should ideally use MMS-LID from the other service, but here we can return a default
        # or implement a simple heuristic.
        # Check if we can use the main app's detection.
        # But this service might be used independently? No, it's integrated.
        # Current logic in old API was just "return 'hi'".
        return 'hi' # Default to Hindi to trigger IndicConformer if unknown

    async def process_transcription(self, audio_path: str, language_code: Optional[str] = None) -> Dict:
        """
        Process transcription request.
        """
        if not self.initialized:
            raise Exception("Transcription service not initialized")

        try:
            # Preprocess
            waveform, sr = self.preprocess_audio(audio_path)
            
            detected_language = language_code
            if not detected_language:
                detected_language = self.detect_language_simple(audio_path) # Fallback if not provided
            
            # Select model
            # Note: "en" or "eng" -> Whisper
            # Others -> IndicConformer
            if detected_language and detected_language.lower() in ["en", "eng", "english"]:
                transcription = self.transcribe_with_whisper(waveform, language_code="en")
            else:
                # IndicConformer expects language code like "hi", "ta", etc.
                # If language is not supported or unknown, might default or fail.
                # The old API defaulted to IndicConformer for anything not English.
                transcription = self.transcribe_with_indic_conformer(waveform, detected_language or "hi")
                
            return {
                "success": True,
                "transcription": transcription,
                "detected_language": detected_language,
                "language_code": detected_language
            }
            
        except Exception as e:
            logger.error(f"Transcription processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
