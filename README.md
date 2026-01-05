# KCC Voice Services Backend

This repository hosts a high-performance, containerized backend for voice processing services, specializing in Indian languages. It provides APIs for speech-to-text (transcription), language identification, and punctuation restoration.

## Features

*   **Multilingual Transcription**: 
    *   **English**: Uses OpenAI's **Whisper** model (Small).
    *   **Indian Languages**: Uses AI4Bharat's **IndicConformer** (600M) for state-of-the-art accuracy in 22+ Indian languages.
*   **Language Identification (LID)**: Powered by Facebook's **MMS-LID** model to automatically detect the spoken language before transcription.
*   **Punctuation Restoration**: Integated punctuation service to add proper punctuation to transcribed text, enhancing readability.
*   **OpenAI-Compatible API**: exposing an interface compatible with `POST /v1/audio/transcriptions`.

## Architecture

The system runs as a single Docker container service (`kcc_voice_system`) managing the FastAPI backend.

*   **Backend**: FastAPI application (`app.py`).
*   **Services**: Modularized services in `backend/services/`:
    *   `TranscriptionService`: Handles audio loading, preprocessing, and model inference (Whisper/IndicConformer).
    *   `MmsLidLanguageDetection`: Handles language detection.
    *   `PunctuationService`: Handles monitoring restoration.

## getting Started

### Prerequisites

*   Docker & Docker Compose
*   NVIDIA GPU with drivers installed (Container uses `nvidia` runtime).

### Installation & Run

1.  **Set Environment Variables**:
    Create a `.env` file or ensure `HF_TOKEN` is set for accessing gated Hugging Face models (like IndicConformer).

    ```bash
    HF_TOKEN=hf_your_huggingface_token
    ```

2.  **Build and Start**:

    ```bash
    docker-compose up --build -d
    ```

    The service will be available at `http://localhost:5020`.

### Logs

To view application logs:

```bash
docker-compose logs -f
```

Or access logs inside the container:
```bash
docker exec -it kcc_voice_system tail -f /app/backend/logs/app.log
```

## API Reference

### 1. Transcribe Audio (OpenAI Compatible)

**Endpoint**: `POST /v1/audio/transcriptions`

Transcribes the uploaded audio file. Automatically detects language if not provided.

**Parameters**:
*   `file`: Audio file (wav, mp3, etc.)
*   `model`: (Optional) "whisper-1" (default)
*   `language`: (Optional) ISO language code (e.g., "en", "hi", "ta"). If omitted, LID is performed.
*   `response_format`: "json" (default), "text", "verbose_json".

**Example**:
```bash
curl http://localhost:5020/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=hi"
```

### 2. Punctuate Text

**Endpoint**: `POST /punctuate`

Adds punctuation to the provided text.

**Body**:
```json
{
  "text": "kya haal hai bhai kaise ho"
}
```

**Response**:
```json
{
  "success": true,
  "punctuated_text": "Kya haal hai bhai? Kaise ho?"
}
```

### 3. Health Check

**Endpoint**: `GET /health`

Returns the health status of the service and initialized components.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "mms_lid": true,
    "punctuation": true,
    "transcription": true
  }
}
```
