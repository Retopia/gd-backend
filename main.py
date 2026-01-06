from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from pydantic import BaseModel
import shutil
import zipfile
import io
import os
import json
from dotenv import load_dotenv

from game_logic import (
    load_gdr, infer_fps, build_expected_events, build_visual_notes,
    compute_compact_stats, export_results_text, ResultRow
)

# Load environment variables
load_dotenv()

# Configuration from environment
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = CORS_ORIGINS_STR.split(",") if CORS_ORIGINS_STR != "*" else ["*"]
STORAGE_LIMIT_MB = int(os.getenv("STORAGE_LIMIT_MB", "500"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

app = FastAPI(title="GD Rhythm Trainer API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAPS_DIR = Path(__file__).parent / "maps"
RESULTS_DIR = Path(__file__).parent / "results"
MUSIC_DIR = Path(__file__).parent / "music"
LENIENCY_DIR = Path(__file__).parent / "leniency"

MAPS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)
LENIENCY_DIR.mkdir(exist_ok=True)

# Storage limit in bytes
STORAGE_LIMIT_BYTES = STORAGE_LIMIT_MB * 1024 * 1024
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def calculate_storage_usage():
    """Calculate total storage used by maps and music files in bytes"""
    total = 0
    for file_path in MAPS_DIR.glob("*.gdr"):
        total += file_path.stat().st_size
    for file_path in MUSIC_DIR.glob("*"):
        if file_path.suffix.lower() in [".mp3", ".wav", ".ogg"]:
            total += file_path.stat().st_size
    return total


# Pydantic models for request/response
class EventLeniency(BaseModel):
    early_ms: Optional[float] = None
    late_ms: Optional[float] = None

    class Config:
        exclude_none = True


class LeniencyConfig(BaseModel):
    default_early_ms: float = 20.833  # 5 ticks at 240 FPS
    default_late_ms: float = 20.833   # 5 ticks at 240 FPS
    custom: Dict[str, EventLeniency] = {}


class MapInfo(BaseModel):
    name: str
    events: int
    fps: float
    duration: float
    has_music: bool = False


class LoadedMap(BaseModel):
    name: str
    fps: float
    duration: float
    events: List[Dict[str, Any]]
    notes: List[Dict[str, Any]]
    leniency: Optional[LeniencyConfig] = None


class InputEvent(BaseModel):
    kind: str
    actual_t: float


class EvaluateRequest(BaseModel):
    map_name: str
    btn: int = 1
    second_player: bool = False
    hit_window_ms: float = 18.0
    hold_min_frames: int = 3
    input_events: List[InputEvent]
    end_time: Optional[float] = None
    press_only_mode: bool = False


class DetailedResult(BaseModel):
    idx: int
    kind: str
    expected_t: Optional[float]
    expected_frame: int
    actual_t: Optional[float]
    offset_ms: Optional[float]
    verdict: str


class StatsResponse(BaseModel):
    hits: float
    misses: float
    extras: float
    mean_early: float
    mean_late: float
    mae: float
    completion: float
    detailed_results: List[DetailedResult] = []


class ExportResponse(BaseModel):
    filename: str
    content: str


class StorageInfo(BaseModel):
    used_bytes: int
    limit_bytes: int
    used_mb: float
    limit_mb: float
    percentage: float


# Cache for map list to avoid re-parsing all files every time
_maps_cache = None
_maps_cache_time = 0

# Routes

@app.get("/health")
async def health():
    """Health check endpoint for Coolify and monitoring"""
    try:
        # Check if critical directories are accessible (results dir is not critical)
        maps_accessible = MAPS_DIR.exists() and MAPS_DIR.is_dir()
        music_accessible = MUSIC_DIR.exists() and MUSIC_DIR.is_dir()

        if not all([maps_accessible, music_accessible]):
            raise HTTPException(status_code=503, detail="Critical directories not accessible")

        return {
            "status": "healthy",
            "service": "GD Rhythm Trainer Backend",
            "timestamp": time.time(),
            "directories": {
                "maps": maps_accessible,
                "music": music_accessible
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/api/storage", response_model=StorageInfo)
async def get_storage():
    """Get current storage usage"""
    used = calculate_storage_usage()
    return StorageInfo(
        used_bytes=used,
        limit_bytes=STORAGE_LIMIT_BYTES,
        used_mb=round(used / 1024 / 1024, 2),
        limit_mb=round(STORAGE_LIMIT_BYTES / 1024 / 1024, 2),
        percentage=round((used / STORAGE_LIMIT_BYTES) * 100, 2) if STORAGE_LIMIT_BYTES > 0 else 0,
    )


@app.get("/api/maps", response_model=List[MapInfo])
async def list_maps(refresh: bool = False):
    """List all available .gdr maps"""
    global _maps_cache, _maps_cache_time

    # Use cache if available and not forcing refresh
    if not refresh and _maps_cache is not None:
        return _maps_cache

    if not MAPS_DIR.exists():
        return []

    maps = []
    for map_path in sorted(MAPS_DIR.glob("*.gdr"), key=lambda p: p.name.lower()):
        try:
            gdr = load_gdr(str(map_path))
            fps = infer_fps(gdr)
            expected = build_expected_events(gdr, fps=fps, btn=1, second_player=False)
            duration = float(gdr.get("duration") or 0.0)
            if duration <= 0.0 and expected:
                duration = expected[-1].t

            # Check if music file exists
            map_stem = map_path.stem
            has_music = any(
                (MUSIC_DIR / f"{map_stem}{ext}").exists()
                for ext in [".mp3", ".wav", ".ogg"]
            )

            maps.append(MapInfo(
                name=map_path.name,
                events=len(expected),
                fps=fps,
                duration=duration,
                has_music=has_music,
            ))
        except Exception as e:
            print(f"Error loading map {map_path.name}: {e}")
            continue

    _maps_cache = maps
    _maps_cache_time = time.time()
    return maps


@app.post("/api/maps/{map_name}/load", response_model=LoadedMap)
async def load_map(
    map_name: str,
    btn: int = 1,
    second_player: bool = False,
    hold_min_frames: int = 3,
):
    """Load map data for gameplay"""
    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    try:
        gdr = load_gdr(str(map_path))
        fps = infer_fps(gdr)
        expected = build_expected_events(gdr, fps=fps, btn=btn, second_player=second_player)

        if not expected:
            raise HTTPException(status_code=400, detail="No inputs found for this button/player")

        notes = build_visual_notes(expected, hold_min_frames=hold_min_frames)
        duration = float(gdr.get("duration") or 0.0)
        if duration <= 0.0 and expected:
            duration = expected[-1].t

        # Load leniency configuration
        leniency_config = load_leniency_config(map_name)

        return LoadedMap(
            name=map_path.name,
            fps=fps,
            duration=duration,
            events=[
                {
                    "idx": e.idx,
                    "frame": e.frame,
                    "t": e.t,
                    "kind": e.kind,
                }
                for e in expected
            ],
            notes=[
                {
                    "start_t": n.start_t,
                    "end_t": n.end_t,
                    "is_hold": n.is_hold,
                }
                for n in notes
            ],
            leniency=leniency_config,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load map: {str(e)}")


@app.get("/api/maps/{map_name}/download")
async def download_map(map_name: str):
    """Download a map file"""
    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    return FileResponse(
        path=str(map_path),
        media_type="application/octet-stream",
        filename=map_path.name
    )


class DownloadMapsRequest(BaseModel):
    map_names: List[str]


@app.post("/api/maps/download-zip")
async def download_maps_zip(request: DownloadMapsRequest):
    """Download multiple maps and their music as a zip file"""
    if not request.map_names:
        raise HTTPException(status_code=400, detail="No maps specified")

    # Create zip file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for map_name in request.map_names:
            # Add map file
            map_path = MAPS_DIR / map_name
            if map_path.exists():
                zip_file.write(map_path, f"maps/{map_path.name}")

            # Add music file if available
            base_name = map_path.stem if map_path.exists() else map_name.replace('.gdr', '')
            for ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']:
                music_path = MUSIC_DIR / f"{base_name}{ext}"
                if music_path.exists():
                    zip_file.write(music_path, f"music/{music_path.name}")
                    break

    # Prepare response
    zip_buffer.seek(0)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=gd-maps.zip"}
    )


@app.post("/api/results/evaluate", response_model=StatsResponse)
async def evaluate_results(request: EvaluateRequest):
    """Evaluate gameplay results"""
    try:
        map_path = MAPS_DIR / request.map_name
        if not map_path.exists():
            raise HTTPException(status_code=404, detail="Map not found")

        gdr = load_gdr(str(map_path))
        fps = infer_fps(gdr)
        expected = build_expected_events(gdr, fps=fps, btn=request.btn, second_player=request.second_player)

        if not expected:
            raise HTTPException(status_code=400, detail="No inputs found for this button/player")

        # Load leniency configuration
        leniency_config = load_leniency_config(request.map_name)

        def get_leniency_window(event_idx: int) -> tuple[float, float]:
            """Get early and late leniency windows for an event (in seconds)"""
            idx_str = str(event_idx)
            early_ms = leniency_config.default_early_ms
            late_ms = leniency_config.default_late_ms

            if idx_str in leniency_config.custom:
                leniency = leniency_config.custom[idx_str]
                if leniency.early_ms is not None:
                    early_ms = leniency.early_ms
                if leniency.late_ms is not None:
                    late_ms = leniency.late_ms

            return early_ms / 1000.0, late_ms / 1000.0

        results = []
        next_idx = 0

        # Process input events
        for input_evt in request.input_events:
            actual_t = input_evt.actual_t
            kind = input_evt.kind

            # Skip release events in press-only mode
            if request.press_only_mode and kind == "up":
                continue

            # Mark misses for events that have passed (using late window since that's the positive offset)
            while next_idx < len(expected):
                ev = expected[next_idx]

                # Skip release events in press-only mode
                if request.press_only_mode and ev.kind == "up":
                    next_idx += 1
                    continue

                _, late_win_s = get_leniency_window(ev.idx)
                if actual_t > ev.t + late_win_s:
                    results.append(ResultRow(
                        idx=ev.idx,
                        kind=ev.kind,
                        expected_t=ev.t,
                        expected_frame=ev.frame,
                        actual_t=None,
                        offset_ms=None,
                        verdict="miss",
                    ))
                    next_idx += 1
                else:
                    break

            if next_idx >= len(expected):
                results.append(ResultRow(
                    idx=len(results),
                    kind=kind,
                    expected_t=float("nan"),
                    expected_frame=-1,
                    actual_t=actual_t,
                    offset_ms=None,
                    verdict="miss",
                ))
                continue

            ev = expected[next_idx]
            if ev.kind != kind:
                results.append(ResultRow(
                    idx=len(results),
                    kind=kind,
                    expected_t=float("nan"),
                    expected_frame=-1,
                    actual_t=actual_t,
                    offset_ms=None,
                    verdict="miss",
                ))
                continue

            dt = actual_t - ev.t
            early_win_s, late_win_s = get_leniency_window(ev.idx)

            # Check if within leniency window (asymmetric)
            is_hit = (dt < 0 and abs(dt) <= early_win_s) or (dt >= 0 and dt <= late_win_s)

            if is_hit:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=actual_t,
                    offset_ms=dt * 1000.0,
                    verdict="hit",
                ))
            else:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=actual_t,
                    offset_ms=dt * 1000.0,
                    verdict="miss",
                ))
            next_idx += 1

        # Mark remaining events as misses
        end_t = request.end_time if request.end_time is not None else float('inf')
        while next_idx < len(expected):
            ev = expected[next_idx]

            # Skip release events in press-only mode
            if request.press_only_mode and ev.kind == "up":
                next_idx += 1
                continue

            _, late_win_s = get_leniency_window(ev.idx)
            if end_t > ev.t + late_win_s:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=None,
                    offset_ms=None,
                    verdict="miss",
                ))
                next_idx += 1
            else:
                break

        stats = compute_compact_stats(results, expected_count=len(expected))

        # Convert results to DetailedResult format
        # Replace NaN values with None for JSON compliance
        import math
        detailed_results = [
            DetailedResult(
                idx=r.idx,
                kind=r.kind,
                expected_t=None if math.isnan(r.expected_t) else r.expected_t,
                expected_frame=r.expected_frame,
                actual_t=r.actual_t,
                offset_ms=r.offset_ms,
                verdict=r.verdict
            )
            for r in results
        ]

        return StatsResponse(**stats, detailed_results=detailed_results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to evaluate results: {str(e)}")


@app.post("/api/results/export", response_model=ExportResponse)
async def export_results(request: EvaluateRequest):
    """Export results to a text file"""
    try:
        map_path = MAPS_DIR / request.map_name
        if not map_path.exists():
            raise HTTPException(status_code=404, detail="Map not found")

        # Evaluate the results
        gdr = load_gdr(str(map_path))
        fps = infer_fps(gdr)
        expected = build_expected_events(gdr, fps=fps, btn=request.btn, second_player=request.second_player)

        if not expected:
            raise HTTPException(status_code=400, detail="No inputs found for this button/player")

        # Load leniency configuration
        leniency_config = load_leniency_config(request.map_name)

        def get_leniency_window(event_idx: int) -> tuple[float, float]:
            """Get early and late leniency windows for an event (in seconds)"""
            idx_str = str(event_idx)
            early_ms = leniency_config.default_early_ms
            late_ms = leniency_config.default_late_ms

            if idx_str in leniency_config.custom:
                leniency = leniency_config.custom[idx_str]
                if leniency.early_ms is not None:
                    early_ms = leniency.early_ms
                if leniency.late_ms is not None:
                    late_ms = leniency.late_ms

            return early_ms / 1000.0, late_ms / 1000.0

        results = []
        next_idx = 0

        for input_evt in request.input_events:
            actual_t = input_evt.actual_t
            kind = input_evt.kind

            # Skip release events in press-only mode
            if request.press_only_mode and kind == "up":
                continue

            while next_idx < len(expected):
                ev = expected[next_idx]

                # Skip release events in press-only mode
                if request.press_only_mode and ev.kind == "up":
                    next_idx += 1
                    continue

                _, late_win_s = get_leniency_window(ev.idx)
                if actual_t > ev.t + late_win_s:
                    results.append(ResultRow(
                        idx=ev.idx,
                        kind=ev.kind,
                        expected_t=ev.t,
                        expected_frame=ev.frame,
                        actual_t=None,
                        offset_ms=None,
                        verdict="miss",
                    ))
                    next_idx += 1
                else:
                    break

            if next_idx >= len(expected):
                results.append(ResultRow(
                    idx=len(results),
                    kind=kind,
                    expected_t=float("nan"),
                    expected_frame=-1,
                    actual_t=actual_t,
                    offset_ms=None,
                    verdict="miss",
                ))
                continue

            ev = expected[next_idx]
            if ev.kind != kind:
                results.append(ResultRow(
                    idx=len(results),
                    kind=kind,
                    expected_t=float("nan"),
                    expected_frame=-1,
                    actual_t=actual_t,
                    offset_ms=None,
                    verdict="miss",
                ))
                continue

            dt = actual_t - ev.t
            early_win_s, late_win_s = get_leniency_window(ev.idx)
            is_hit = (dt < 0 and abs(dt) <= early_win_s) or (dt >= 0 and dt <= late_win_s)

            if is_hit:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=actual_t,
                    offset_ms=dt * 1000.0,
                    verdict="hit",
                ))
            else:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=actual_t,
                    offset_ms=dt * 1000.0,
                    verdict="miss",
                ))
            next_idx += 1

        # Mark remaining events as misses
        end_t = request.end_time if request.end_time is not None else float('inf')
        while next_idx < len(expected):
            ev = expected[next_idx]

            # Skip release events in press-only mode
            if request.press_only_mode and ev.kind == "up":
                next_idx += 1
                continue

            _, late_win_s = get_leniency_window(ev.idx)
            if end_t > ev.t + late_win_s:
                results.append(ResultRow(
                    idx=ev.idx,
                    kind=ev.kind,
                    expected_t=ev.t,
                    expected_frame=ev.frame,
                    actual_t=None,
                    offset_ms=None,
                    verdict="miss",
                ))
                next_idx += 1
            else:
                break

        # Generate export content
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        map_stem = Path(request.map_name).stem
        filename = f"{map_stem}_results_{timestamp}.txt"

        meta = {
            "macro_file": request.map_name,
            "fps": f"{fps:.6f}",
            "window_ms": f"{request.hit_window_ms:.2f}",
            "expected_events": str(len(expected)),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_x_frac": "0.333",
            "inputs_captured": str(len([r for r in results if r.actual_t is not None])),
        }

        content = export_results_text(results, None, meta)

        return ExportResponse(
            filename=filename,
            content=content,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to export results: {str(e)}")


@app.get("/api/music/{map_name}")
async def get_music(map_name: str):
    """Check if music exists for a map"""
    map_stem = Path(map_name).stem
    for ext in [".mp3", ".wav", ".ogg"]:
        music_path = MUSIC_DIR / f"{map_stem}{ext}"
        if music_path.exists():
            return {
                "available": True,
                "filename": music_path.name,
                "url": f"/api/music/file/{music_path.name}",
            }
    return {"available": False}


@app.get("/api/music/file/{filename}")
async def get_music_file(filename: str):
    """Serve a music file"""
    from fastapi.responses import FileResponse

    music_path = MUSIC_DIR / filename
    if not music_path.exists():
        raise HTTPException(status_code=404, detail="Music file not found")

    # Determine content type
    ext = music_path.suffix.lower()
    content_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    return FileResponse(music_path, media_type=content_type)


@app.post("/api/maps/upload")
async def upload_map(file: UploadFile = File(...)):
    """Upload a .gdr map file"""
    global _maps_cache

    if not file.filename.endswith(".gdr"):
        raise HTTPException(status_code=400, detail="Only .gdr files are allowed")

    file_path = MAPS_DIR / file.filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"Map '{file.filename}' already exists")
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    # Check storage limit
    current_usage = calculate_storage_usage()
    if current_usage + file_size > STORAGE_LIMIT_BYTES:
        used_mb = round(current_usage / 1024 / 1024, 2)
        limit_mb = round(STORAGE_LIMIT_BYTES / 1024 / 1024, 2)
        raise HTTPException(
            status_code=507,
            detail=f"Storage limit exceeded. Using {used_mb}MB of {limit_mb}MB. Cannot upload {round(file_size / 1024 / 1024, 2)}MB file."
        )

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Invalidate cache
        _maps_cache = None

        return {"message": f"Map {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload map: {str(e)}")


@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)):
    """Upload a music file (.mp3, .wav, .ogg)"""
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp3", ".wav", ".ogg"]:
        raise HTTPException(status_code=400, detail="Only .mp3, .wav, and .ogg files are allowed")

    # Check file size
    # Read first chunk to check size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Seek back to start

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' is too large ({size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB."
        )
    
    # Check storage limit
    current_usage = calculate_storage_usage()
    if current_usage + file_size > STORAGE_LIMIT_BYTES:
        used_mb = round(current_usage / 1024 / 1024, 2)
        limit_mb = round(STORAGE_LIMIT_BYTES / 1024 / 1024, 2)
        raise HTTPException(
            status_code=507,
            detail=f"Storage limit exceeded. Using {used_mb}MB of {limit_mb}MB. Cannot upload {round(file_size / 1024 / 1024, 2)}MB file."
        )

    # Check if corresponding .gdr file exists
    file_stem = Path(file.filename).stem
    gdr_path = MAPS_DIR / f"{file_stem}.gdr"
    if not gdr_path.exists():
        raise HTTPException(
            status_code=400, 
            detail=f"No corresponding .gdr file found for '{file.filename}'. Please upload the .gdr file first."
        )

    file_path = MUSIC_DIR / file.filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"Music file '{file.filename}' already exists")

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"message": f"Music file {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload music: {str(e)}")


@app.delete("/api/maps/{map_name}")
async def delete_map(map_name: str):
    """Delete a map file and its corresponding music file (if it exists)"""
    global _maps_cache

    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    try:
        # Delete the .gdr file
        map_path.unlink()

        # Delete corresponding music file if it exists
        map_stem = Path(map_name).stem
        for ext in [".mp3", ".wav", ".ogg"]:
            music_path = MUSIC_DIR / f"{map_stem}{ext}"
            if music_path.exists():
                music_path.unlink()
                break

        # Invalidate cache
        _maps_cache = None

        return {"message": f"Map {map_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete map: {str(e)}")


@app.delete("/api/music/{filename}")
async def delete_music(filename: str):
    """Delete a music file"""
    music_path = MUSIC_DIR / filename
    if not music_path.exists():
        raise HTTPException(status_code=404, detail="Music file not found")

    try:
        music_path.unlink()
        return {"message": f"Music file {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete music: {str(e)}")


def get_leniency_path(map_name: str) -> Path:
    """Get the path to the leniency file for a map"""
    map_stem = Path(map_name).stem
    return LENIENCY_DIR / f"{map_stem}.json"


def load_leniency_config(map_name: str) -> LeniencyConfig:
    """Load leniency configuration for a map, or return default"""
    leniency_path = get_leniency_path(map_name)
    if not leniency_path.exists():
        return LeniencyConfig()

    try:
        with leniency_path.open("r") as f:
            data = json.load(f)
            # Convert custom dict values to EventLeniency objects
            custom = {}
            for idx, leniency in data.get("custom", {}).items():
                custom[idx] = EventLeniency(**leniency)
            return LeniencyConfig(
                default_early_ms=data.get("default_early_ms", 20.833),
                default_late_ms=data.get("default_late_ms", 20.833),
                custom=custom
            )
    except Exception as e:
        print(f"Error loading leniency config for {map_name}: {e}")
        return LeniencyConfig()


def save_leniency_config(map_name: str, config: LeniencyConfig):
    """Save leniency configuration for a map"""
    leniency_path = get_leniency_path(map_name)

    try:
        # Convert to JSON-serializable format
        data = {
            "default_early_ms": config.default_early_ms,
            "default_late_ms": config.default_late_ms,
            "custom": {
                idx: {k: v for k, v in {"early_ms": leniency.early_ms, "late_ms": leniency.late_ms}.items() if v is not None}
                for idx, leniency in config.custom.items()
            }
        }
        with leniency_path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save leniency config: {str(e)}")


@app.get("/api/maps/{map_name}/leniency", response_model=LeniencyConfig, response_model_exclude_none=True)
async def get_leniency(map_name: str):
    """Get leniency configuration for a map"""
    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    return load_leniency_config(map_name)


@app.put("/api/maps/{map_name}/leniency", response_model=LeniencyConfig)
async def update_leniency(map_name: str, config: LeniencyConfig):
    """Update leniency configuration for a map"""
    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    save_leniency_config(map_name, config)
    return config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
