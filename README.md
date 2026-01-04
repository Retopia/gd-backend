# GD Rhythm Trainer - Backend

FastAPI backend for the Geometry Dash Rhythm Trainer application.

## Tech Stack

- **FastAPI 0.115.0** - Web framework
- **Uvicorn 0.30.0** - ASGI server
- **Pydantic 2.9.2** - Data validation
- **msgpack 1.0.8** - Binary serialization for .gdr files
- **python-multipart 0.0.6** - File upload support

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the backend directory (copy from `.env.example`):

```bash
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=*
STORAGE_LIMIT_MB=500
MAX_FILE_SIZE_MB=100
```

### Development

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000` (or your configured PORT)

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Maps
- `GET /api/maps` - List all maps
- `POST /api/maps/upload` - Upload .gdr files
- `POST /api/maps/{map_name}/load` - Load map for gameplay
- `GET /api/maps/{map_name}/download` - Download a map file
- `POST /api/maps/download-zip` - Download multiple maps as zip
- `DELETE /api/maps/{map_name}` - Delete a map and its music

### Music
- `GET /api/music/{map_name}` - Get music info for a map
- `POST /api/music/upload` - Upload music files
- `DELETE /api/music/{filename}` - Delete a music file

### Results
- `POST /api/results/evaluate` - Evaluate gameplay results
- `POST /api/results/export` - Export results to text file

### Storage
- `GET /api/storage` - Get storage usage information

## Storage

- Maps are stored in `./maps/`
- Music files are stored in `./music/`
- Results are stored in `./results/`
- Storage limit: 500 MB (configurable in `main.py`)

## Supported File Formats

- Maps: `.gdr` (msgpack format)
- Music: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`
