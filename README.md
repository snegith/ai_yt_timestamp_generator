# YouTube Timestamp Generator

A Chrome extension + FastAPI backend that generates AI-powered chapter timestamps for YouTube videos and lets users jump directly to any chapter from the YouTube page.

## Overview

YouTube Timestamp Generator helps users turn long videos into structured, clickable chapters. The workflow is:

1. The Chrome extension injects a sidebar into YouTube watch pages.
2. The user clicks **Generate Timestamps**.
3. The extension extracts the current video ID and sends it to the backend.
4. The backend fetches or transcribes the video transcript, preprocesses it, detects topic boundaries, and generates concise chapter titles using Groq.
5. The extension renders the returned timestamps as clickable items that seek the video player.

This project is designed to be easy to run locally and later deploy the backend to Railway.

## Features

### Chrome Extension
- Injects a sidebar on YouTube watch pages.
- Extracts the current YouTube video ID from the URL.
- Sends `{ video_id }` to the backend.
- Renders timestamps as clickable buttons.
- Seeks the current video to the selected timestamp.
- Supports YouTube navigation changes using the YouTube SPA lifecycle.

### FastAPI Backend
- Exposes a single `POST /generate` endpoint.
- Validates `video_id` using Pydantic.
- Uses an in-memory cache keyed by `video_id` to avoid repeated work.
- Runs the pipeline:
  1. Transcript retrieval
  2. Transcript preprocessing
  3. Topic boundary detection
  4. Groq title generation
- Returns timestamps in chronological order.

### Pipeline Details
- **Transcript retrieval**:
  - Prefer `youtube-transcript-api`
  - Fall back to `yt-dlp` + `faster-whisper`
- **Preprocessing**:
  - Removes filler words
  - Uses `nltk.sent_tokenize`
  - Groups transcript text into approximate `~200` word windows
- **Boundary detection**:
  - Uses `sentence-transformers` with `all-MiniLM-L6-v2`
  - Performs cosine similarity between consecutive windows
  - Marks boundaries when similarity is below `0.35`
  - Always includes the first window
- **Title generation**:
  - Sends a single batched prompt to Groq
  - Parses the response as a JSON array
  - Returns chapter titles aligned with the boundary windows

## Project Structure

```text
.
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── cache.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── railway.toml
│   └── pipeline/
│       ├── __init__.py
│       ├── transcript.py
│       ├── preprocess.py
│       ├── boundaries.py
│       └── titles.py
├── extension/
│   ├── content.js
│   ├── manifest.json
│   ├── sidebar.css
│   ├── package.json
│   └── tests/
└── README.md
```

## Prerequisites

### Backend
- Python 3.10+
- `pip`
- A valid Groq API key
- FFmpeg installed if you want the fallback transcript path to work

### Frontend
- Google Chrome
- The extension loaded as an unpacked extension

## Local Setup

### 1. Install backend dependencies

```powershell
cd C:\Users\My PC\Desktop\mini projects\yt_timestamps\backend
python -m pip install -r requirements.txt
```

### 2. Set the Groq API key

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

### 3. Start the backend

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Load the extension in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `extension/` folder

### 5. Test on a real YouTube video

Open a YouTube watch page, make sure the sidebar appears, and click **Generate Timestamps**.

## Current Frontend Configuration

The extension is currently configured to call the local backend at:

- `http://127.0.0.1:8000`

Before deploying, update the backend URL in `extension/content.js` and the `host_permissions` in `extension/manifest.json` to your production backend URL.

## Notes for Deployment

For Railway deployment:
- Set `GROQ_API_KEY` as an environment variable in the Railway service.
- Update `extension/content.js` to point to the deployed Railway URL.
- Update `extension/manifest.json` `host_permissions` to include the deployed backend domain.

## Testing

The repository includes property-based and example tests under `backend/tests/` and `extension/tests/`.

To run backend tests:

```powershell
cd C:\Users\My PC\Desktop\mini projects\yt_timestamps\backend
pytest
```

If you are testing the extension locally, use Chrome DevTools to inspect console errors and network requests to `/generate`.

## Troubleshooting

### Backend fails to start
- Make sure `GROQ_API_KEY` is set.
- Ensure dependencies are installed with `python -m pip install -r requirements.txt`.

### Generate button hangs
- Check that the backend is running on `http://127.0.0.1:8000`.
- Confirm the extension is pointing to the correct backend URL.
- Check Chrome DevTools for network errors.

### Transcript fallback fails
- Install FFmpeg and confirm `ffmpeg` and `ffprobe` are available in `PATH`.

## Future Improvements

- Add a production-ready backend deployment configuration for Railway.
- Improve extension styling and UX.
- Add more automated tests for the frontend.
- Add support for additional video sources or transcript providers.

## License

This project is for personal and educational use unless you add a specific license file.
