import os
import re
import hashlib
import cv2
import numpy as np
import pytesseract

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Election Result Collation API", version="1.0.0")


# =========================================================
# Canonical candidates (row index -> candidate)
# =========================================================
CANDIDATES = [
    {"id": "kennedy_agyapong", "name": "Kennedy Agyapong"},
    {"id": "bryan_acheampong", "name": "Dr. Bryan Acheampong"},
    {"id": "mahamudu_bawumia", "name": "Dr. Mahamudu Bawumia"},
    {"id": "yaw_adutwum", "name": "Dr. Yaw Osei Adutwum"},
    {"id": "kwabena_agyepong", "name": "Kwabena Agyei Agyepong"},
]
CANDIDATE_IDS = [c["id"] for c in CANDIDATES]


# =========================================================
# In-memory database (simple + demo-friendly)
# =========================================================
# constituency_id -> { "name": str, "stations": set(station_id) }
CONSTITUENCIES: Dict[str, Dict[str, Any]] = {}

# station_id -> { "name": str, "constituency_id": str }
POLLING_STATIONS: Dict[str, Dict[str, Any]] = {}

# station_id -> stored sheet result
# {
#   "sheet_hash": str,
#   "votes": {candidate_id: int, ...},
#   "raw_results": [...]
# }
STATION_SHEETS: Dict[str, Dict[str, Any]] = {}


# =========================================================
# Layout detection (row slicing)
# =========================================================
def _find_vote_panels(img_bgr: np.ndarray):
    """
    Split the image into horizontal rows and extract vote panels.
    NOTE: used for row alignment / debug; not strictly required for collation.
    """
    h, w, _ = img_bgr.shape
    rows = []
    header_offset = int(h * 0.18)
    row_height = int(h * 0.13)

    for i in range(5):
        y = header_offset + i * row_height
        if y + row_height > h:
            break

        x = int(w * 0.62)
        panel_w = int(w * 0.33)
        panel_h = int(row_height * 0.70)

        rows.append((x, y + 10, panel_w, panel_h))

    return rows


# =========================================================
# (Optional) Vision template mode hooks (kept for later)
# =========================================================
def _load_digit_templates(template_dir="templates"):
    templates = {}
    for d in range(10):
        path = os.path.join(template_dir, f"{d}.png")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            templates[d] = img
    return templates


DIGIT_TEMPLATES = _load_digit_templates()


def _binarize_for_matching(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _read_votes_by_templates(panel_bgr: np.ndarray) -> int | None:
    if panel_bgr is None or panel_bgr.size == 0:
        return None
    if not DIGIT_TEMPLATES:
        return None

    bw = _binarize_for_matching(panel_bgr)

    detections = []  # (x, digit, score)
    for digit, tmpl in DIGIT_TEMPLATES.items():
        th, tw = tmpl.shape[:2]
        if bw.shape[0] < th or bw.shape[1] < tw:
            continue

        res = cv2.matchTemplate(bw, tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.65)
        for y, x in zip(ys, xs):
            detections.append((x, digit, float(res[y, x])))

    if not detections:
        return None

    detections.sort(key=lambda t: (t[0], -t[2]))
    filtered = []
    min_dx = 12
    for x, d, s in detections:
        if not filtered or abs(x - filtered[-1][0]) > min_dx:
            filtered.append((x, d, s))

    number_str = "".join(str(d) for _, d, _ in filtered)
    return int(number_str) if number_str else None


# =========================================================
# Models
# =========================================================
class ConstituencyCreate(BaseModel):
    constituency_id: str = Field(..., description="Unique ID, e.g. 'effutu'")
    name: str = Field(..., description="Display name, e.g. 'Effutu Constituency'")


class PollingStationCreate(BaseModel):
    station_id: str = Field(..., description="Unique ID, e.g. 'effutu_ps_001'")
    name: str = Field(..., description="Display name, e.g. 'Winneba Presby JHS'")
    constituency_id: str = Field(..., description="Parent constituency ID")


class VotesPayload(BaseModel):
    # votes by candidate_id; useful for mode=data demo
    votes: Dict[str, int]


# =========================================================
# Helpers: aggregation
# =========================================================
def _empty_vote_dict() -> Dict[str, int]:
    return {cid: 0 for cid in CANDIDATE_IDS}


def _sum_votes(vote_dicts: List[Dict[str, int]]) -> Dict[str, int]:
    out = _empty_vote_dict()
    for vd in vote_dicts:
        for cid in CANDIDATE_IDS:
            out[cid] += int(vd.get(cid, 0))
    return out


def _winner_from_votes(votes: Dict[str, int]) -> Optional[Dict[str, Any]]:
    if not votes:
        return None
    best_id = max(votes.keys(), key=lambda k: votes.get(k, 0))
    best_name = next((c["name"] for c in CANDIDATES if c["id"] == best_id), best_id)
    return {"candidate_id": best_id, "name": best_name, "votes": votes.get(best_id, 0)}


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =========================================================
# API: Setup (Constituencies + Polling Stations)
# =========================================================
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Election Result Collation API",
        "setup": [
            "POST /constituencies",
            "POST /polling-stations",
            "POST /pink-sheet/upload (auto-collates)",
            "GET /aggregate/national",
        ],
    }


@app.post("/constituencies")
def create_constituency(payload: ConstituencyCreate):
    cid = payload.constituency_id.strip().lower()
    CONSTITUENCIES.setdefault(cid, {"name": payload.name, "stations": set()})
    CONSTITUENCIES[cid]["name"] = payload.name
    return {"status": "ok", "constituency": {"constituency_id": cid, "name": payload.name}}


@app.get("/constituencies")
def list_constituencies():
    return {
        "count": len(CONSTITUENCIES),
        "constituencies": [
            {"constituency_id": cid, "name": meta["name"], "stations": len(meta["stations"])}
            for cid, meta in CONSTITUENCIES.items()
        ],
    }


@app.post("/polling-stations")
def create_polling_station(payload: PollingStationCreate):
    sid = payload.station_id.strip().lower()
    cid = payload.constituency_id.strip().lower()

    if cid not in CONSTITUENCIES:
        return JSONResponse({"error": "unknown_constituency", "constituency_id": cid}, status_code=404)

    POLLING_STATIONS[sid] = {"name": payload.name, "constituency_id": cid}
    CONSTITUENCIES[cid]["stations"].add(sid)
    return {"status": "ok", "station": {"station_id": sid, "name": payload.name, "constituency_id": cid}}


@app.get("/polling-stations")
def list_polling_stations(constituency_id: Optional[str] = None):
    if constituency_id:
        cid = constituency_id.strip().lower()
        stations = [
            {"station_id": sid, **meta}
            for sid, meta in POLLING_STATIONS.items()
            if meta["constituency_id"] == cid
        ]
    else:
        stations = [{"station_id": sid, **meta} for sid, meta in POLLING_STATIONS.items()]

    return {"count": len(stations), "stations": stations}


# =========================================================
# Upload pink sheet (auto-collate)
# - mode=data: send votes in JSON payload (demo / authoritative)
# - mode=vision: attempt digit template matching from the image
# =========================================================
@app.post("/pink-sheet/upload")
async def upload_pink_sheet(
    station_id: str = Query(..., description="Polling station ID, must exist"),
    mode: str = Query("data", description="data or vision"),
    overwrite: bool = Query(False, description="Allow replacing an existing station submission"),
    file: UploadFile = File(...),
    votes_json: Optional[str] = Query(None, description="For mode=data only: JSON like {'kennedy_agyapong': 10, ...}")
):
    sid = station_id.strip().lower()
    if sid not in POLLING_STATIONS:
        return JSONResponse({"error": "unknown_station", "station_id": sid}, status_code=404)

    raw = await file.read()
    sheet_hash = _hash_bytes(raw)

    if (sid in STATION_SHEETS) and (not overwrite):
        return JSONResponse(
            {"error": "already_submitted", "station_id": sid, "hint": "use overwrite=true to replace"},
            status_code=409,
        )

    # Decode image for debug/vision usage
    np_img = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "invalid_image"}, status_code=400)

    panels = _find_vote_panels(img)

    # Build per-row results
    per_row = []
    votes_out = _empty_vote_dict()

    if mode.lower() == "vision":
        # Vision mode: read each row panel as digits (requires templates)
        for i, (x, y, ww, hh) in enumerate(panels[:len(CANDIDATES)]):
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(img.shape[1], x + ww); y2 = min(img.shape[0], y + hh)
            panel = img[y1:y2, x1:x2]

            # Save what the system sees (helps you tune templates)
            cv2.imwrite(f"debug_panel_{sid}_{i}.png", panel)

            cand = CANDIDATES[i]
            v = _read_votes_by_templates(panel)
            v = int(v) if v is not None else 0

            votes_out[cand["id"]] += v
            per_row.append({"candidate_id": cand["id"], "name": cand["name"], "votes": v})

    else:
        # mode=data: authoritative numbers supplied (demo/real DB integration pattern)
        # votes_json comes as query string; keep it simple.
        if not votes_json:
            return JSONResponse(
                {"error": "missing_votes_json", "hint": "Provide votes_json when mode=data"},
                status_code=422,
            )

        import json
        try:
            supplied = json.loads(votes_json)
        except Exception:
            return JSONResponse(
                {"error": "bad_votes_json", "hint": "votes_json must be valid JSON"},
                status_code=422,
            )

        for i, cand in enumerate(CANDIDATES):
            v = int(supplied.get(cand["id"], 0))
            votes_out[cand["id"]] += v
            per_row.append({"candidate_id": cand["id"], "name": cand["name"], "votes": v})

    # Store station submission
    STATION_SHEETS[sid] = {
        "sheet_hash": sheet_hash,
        "votes": votes_out,
        "raw_results": per_row,
    }

    constituency_id = POLLING_STATIONS[sid]["constituency_id"]

    return {
        "status": "ok",
        "station_id": sid,
        "constituency_id": constituency_id,
        "mode": mode,
        "stored_hash": sheet_hash,
        "per_row": per_row,
        "station_total_votes": int(sum(votes_out.values())),
    }


# =========================================================
# Aggregation endpoints
# =========================================================
@app.get("/aggregate/station/{station_id}")
def aggregate_station(station_id: str):
    sid = station_id.strip().lower()
    if sid not in POLLING_STATIONS:
        return JSONResponse({"error": "unknown_station", "station_id": sid}, status_code=404)

    sheet = STATION_SHEETS.get(sid)
    if not sheet:
        return JSONResponse({"error": "no_submission", "station_id": sid}, status_code=404)

    votes = sheet["votes"]
    return {
        "station_id": sid,
        "station_name": POLLING_STATIONS[sid]["name"],
        "constituency_id": POLLING_STATIONS[sid]["constituency_id"],
        "votes": votes,
        "total_votes": int(sum(votes.values())),
        "winner": _winner_from_votes(votes),
        "sheet_hash": sheet["sheet_hash"],
    }


@app.get("/aggregate/constituency/{constituency_id}")
def aggregate_constituency(constituency_id: str):
    cid = constituency_id.strip().lower()
    if cid not in CONSTITUENCIES:
        return JSONResponse({"error": "unknown_constituency", "constituency_id": cid}, status_code=404)

    stations = list(CONSTITUENCIES[cid]["stations"])
    vote_dicts = []
    submitted = 0

    for sid in stations:
        sheet = STATION_SHEETS.get(sid)
        if sheet:
            vote_dicts.append(sheet["votes"])
            submitted += 1

    votes = _sum_votes(vote_dicts)

    return {
        "constituency_id": cid,
        "constituency_name": CONSTITUENCIES[cid]["name"],
        "stations_total": len(stations),
        "stations_submitted": submitted,
        "votes": votes,
        "total_votes": int(sum(votes.values())),
        "winner": _winner_from_votes(votes),
    }


@app.get("/aggregate/national")
def aggregate_national():
    vote_dicts = []
    submitted = 0

    for sid, sheet in STATION_SHEETS.items():
        vote_dicts.append(sheet["votes"])
        submitted += 1

    votes = _sum_votes(vote_dicts)
    return {
        "polling_stations_submitted": submitted,
        "constituencies_count": len(CONSTITUENCIES),
        "votes": votes,
        "total_votes": int(sum(votes.values())),
        "winner": _winner_from_votes(votes),
    }


# =========================================================
# Admin: reset everything (handy during testing)
# =========================================================
@app.post("/admin/reset")
def admin_reset():
    CONSTITUENCIES.clear()
    POLLING_STATIONS.clear()
    STATION_SHEETS.clear()
    return {"status": "ok", "message": "All in-memory data cleared"}
