import os

# Force single-threaded BLAS/OpenMP execution per worker to prevent CPU contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # Ensures AVX2 fast-path on AMD Ryzen / Threadripper CPUs

import sys as _sys

def _resolve_runtime_home() -> Path:
    """Returns a writable user home/data directory whether running from source or a PyInstaller standalone bundle."""
    try:
        home = Path.home()
        if home.exists():
            return home
    except Exception:
        pass
    return Path.cwd()

def _is_frozen_bundle() -> bool:
    return bool(getattr(_sys, "frozen", False))


# ==============================================================================
# AUDIO ANALYSER PRO V.12.1 STANDALONE RELEASE - ACTIVE CPU TELEMETRY & PRE-BATCH CLONE QUARANTINE
# ==============================================================================

import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback
from typing import Dict, List, Optional, Tuple
import urllib.parse
import warnings

import librosa
from mutagen import File
from mutagen.aiff import AIFF
from mutagen.id3 import (
    COMM, GEOB, ID3, TALB, TBPM, TCON, TIT2, TKEY, TMOO, TPE1, TXXX, ID3NoHeaderError
)
from mutagen.wave import WAVE
import numpy as np
import soundfile as sf
from tkinterdnd2 import DND_FILES, TkinterDnD
try:
    from numba.core.errors import NumbaWarning
    warnings.filterwarnings("ignore", category=NumbaWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Pre-warm Librosa's Numba JIT compiler on the main thread before spawning workers
try:
    _dummy = np.zeros(4096, dtype=np.float32)
    _ = librosa.stft(_dummy, n_fft=512, hop_length=256)
    _ = librosa.beat.beat_track(y=_dummy, sr=22050)
except Exception:
    pass

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
TARGET_LUFS = -14.0
SPECTRAL_128K_CUTOFF_HZ = 14500
AUDIO_EXTENSIONS = ('.mp3', '.flac', '.wav', '.aiff', '.m4a')
ID3_COMPATIBLE_EXTENSIONS = ('.mp3', '.wav', '.aiff')
LONG_TRACK_THRESHOLD_SEC = 900.0
RENAME_LEDGER_FILE = Path.home() / '.vanta_audio_analyser_rename_ledger.json'
QUARANTINE_LEDGER_FILE = Path.home() / '.vanta_audio_analyser_quarantine_ledger.json'

CUE_COLORS = [
    (204, 0, 0),     # 1: Red (Beat 1 Transient)
    (255, 128, 0),   # 2: Orange (Intro phrase)
    (0, 102, 255),   # 3: Blue (Build 1)
    (0, 204, 0),     # 4: Green (Drop 1)
    (153, 51, 255),  # 5: Purple (Breakdown)
    (0, 204, 204),   # 6: Cyan (Build 2)
    (255, 204, 0),   # 7: Yellow (Drop 2)
    (255, 0, 128)    # 8: Magenta (Outro)
]

CUE_LABELS = [
    "BEAT 1",
    "INTRO",
    "BUILD 1",
    "DROP 1",
    "BREAK",
    "BUILD 2",
    "DROP 2",
    "OUTRO"
]

CAMELOT_MAP = {
    (0, False): '8B',   (0, True): '5A',
    (1, False): '3B',   (1, True): '12A',
    (2, False): '10B',  (2, True): '7A',
    (3, False): '5B',   (3, True): '2A',
    (4, False): '12B',  (4, True): '9A',
    (5, False): '7B',   (5, True): '4A',
    (6, False): '2B',   (6, True): '11A',
    (7, False): '9B',   (7, True): '6A',
    (8, False): '4B',   (8, True): '1A',
    (9, False): '11B',  (9, True): '8A',
    (10, False): '6B',  (10, True): '3A',
    (11, False): '1B',  (11, True): '10A',
}

WATERMARK_PATTERNS = [
    r'\[(zippyshare|yt-dlp|beatport|free download|official|320kbps|preview|myfreemp3|traxsource)\]',
    r'[\(\[]\s*(free download series|free download|official|preview)\s*[\)\]]',
    r'\bfree download series\b',
    r'\bfree download\b',
    r'\(downloaded from [^\)]+\)',
    r'\[downloaded from [^\]]+\]',
    r'(https?://\S+|www\.\S+|[a-z0-9_\-]+\.(?:com|net|org|ru|info|biz|co|io|me|tv|fm|uk|de|au|cc|xyz|club|site|online)\b)',
    r'@[a-z0-9_\-]+',
    r'\[[0-9a-zA-Z_\-]{11}\]'
]

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class TrackDiagnostic:
    path: Path
    raw_title: str = ''
    raw_artist: str = ''
    raw_album: str = ''
    raw_genre: str = ''
    raw_mood: str = ''
    clean_title: str = ''
    clean_artist: str = ''
    clean_genre: str = ''
    folder_category: str = ''
    has_watermark: bool = False
    has_existing_cues: bool = False
    is_duplicate: bool = False
    duplicate_reason: str = ''
    master_path: Optional[Path] = None
    is_long_track: bool = False
    is_locked: bool = False
    is_cached: bool = False
    is_fake_320k: bool = False
    is_clipped: bool = False
    proposed_filename: str = ''
    flagged: bool = False

@dataclass
class TrackAnalysisResult:
    filepath: Path
    raw_title: str = ''
    raw_artist: str = ''
    cleaned_title: str = ''
    cleaned_artist: str = ''
    detected_genre: str = ''
    folder_category: str = ''
    detected_bpm: float = 0.0
    key: str = 'Unknown'
    tuning_cents: float = 0.0
    mood: str = 'Warm'
    energy_level: int = 5
    energy_delta: float = 0.0
    crest_factor_db: float = 12.0
    true_peak: float = 0.0
    intro_bars: int = 16
    outro_bars: int = 32
    energy_summary: str = ''
    integrated_lufs: float = 0.0
    gain_offset_db: float = 0.0
    spectral_cutoff_hz: float = 0.0
    is_fake_320k: bool = False
    is_clipped: bool = False
    is_long_track: bool = False
    cues_ms: List[int] = field(default_factory=list)
    flagged: bool = False

# ==============================================================================
# SANITIZER & FILE LOCKING ENGINE
# ==============================================================================
def is_generic_artist_name(artist_str: str) -> bool:
    """Returns True if the artist tag is empty or a generic placeholder like 'Unknown Artist'."""
    if not artist_str:
        return True
    norm = artist_str.strip().lower()
    return norm in (
        'unknown artist', 'unknown', 'various artists', 'various artist',
        'va', 'v.a.', 'v/a', 'unassigned', 'none', 'null', 'artist', 'track'
    )

COPY_SUFFIX_PATTERN = re.compile(
    r'(?:\s*[-–_]\s*copy(?:\s*\(\d+\))?|\s*[\(\[](?:copy|\d{1,2})[\)\]])\s*$',
    re.IGNORECASE
)

def strip_copy_suffix(stem_str: str) -> Tuple[str, bool]:
    """Strips OS duplicate copy suffixes like ' (2)', ' (3)', ' [1]', or ' - Copy'. Returns (base_stem, had_suffix)."""
    new_stem = COPY_SUFFIX_PATTERN.sub('', stem_str).strip()
    if new_stem and new_stem != stem_str.strip():
        return new_stem, True
    return stem_str, False

def sanitize_name(raw_str: str) -> str:
    if re.fullmatch(r'\d+', raw_str):
        return ""
    # Decode URL-escaped sequences (%20 -> space) BEFORE stripping % symbols
    cleaned = urllib.parse.unquote(raw_str)
    # Strip trailing OS copy suffix e.g. 'Break on Through (BT Remix) (2)' -> 'Break on Through (BT Remix)'
    cleaned, _ = strip_copy_suffix(cleaned)
    for pattern in WATERMARK_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    # Strip empty parentheses/brackets left behind by compilation tags e.g. 'Atmospheric Progressive 025 ()'
    cleaned = re.sub(r'\(\s*\)|\[\s*\]', '', cleaned)
    cleaned = re.sub(r'^[\-_.\s]+|[\-_.\s]+$', '', cleaned)
    cleaned = re.sub(r'^\d{1,4}\s*[-._\s]+\s*', '', cleaned)
    # Strip mid-filename compilation track numbers like 'Artist - 14 Title' or 'Artist - Album - 07 Title'
    cleaned = re.sub(r'(\s[-–]\s)\d{1,3}\s+(?=[A-Za-z])', r'\1', cleaned)
    # Preserve '&' in artist/title names while stripping illegal/junk symbols
    cleaned = re.sub(r'[!%#@$^*+=<>?/\\|~`]', '', cleaned)
    cleaned = re.sub(r'[\x00-\x1f]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip('. -_')

def sanitize_name_string(raw_str: str) -> str:
    cleaned = urllib.parse.unquote(raw_str)
    for pattern in WATERMARK_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(\s*\)|\[\s*\]', '', cleaned)
    cleaned = re.sub(r'[!%#@$^*+=<>?/\\|~`]', '', cleaned)
    cleaned = re.sub(r'^\s*\d{1,4}\s*[-–.]\s*', '', cleaned)
    cleaned = re.sub(r'\s*-\s*-\s*', ' - ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' -_.')
    return cleaned

def ensure_extension(filename: str, original_suffix: str) -> str:
    ext_pattern = r'(\.(mp3|flac|wav|aiff|aif|m4a|ogg|wma))+$'
    base = re.sub(ext_pattern, '', filename, flags=re.IGNORECASE)
    return base.strip('. ') + original_suffix.lower()

def is_hash_or_numeric_filename(filename: str) -> bool:
    stem = filename.rsplit('.', 1)[0]
    stem_clean = re.sub(r'\s*\(\d+\)$', '', stem)
    numeric_hash_pattern = re.compile(r'^[0-9a-fA-F_\-\.]{6,}$')
    return bool(numeric_hash_pattern.match(stem_clean)) or bool(re.fullmatch(r'\d+', stem_clean))

class FileLockManager:
    @staticmethod
    def set_read_only(filepath: Path, read_only: bool = True):
        try:
            p_str = str(filepath.resolve())
            current_mode = os.stat(p_str).st_mode
            if read_only:
                os.chmod(p_str, current_mode & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
            else:
                os.chmod(p_str, current_mode | stat.S_IWRITE)
        except Exception:
            pass

    @staticmethod
    def is_read_only(filepath: Path) -> bool:
        try:
            p_str = str(filepath.resolve())
            mode = os.stat(p_str).st_mode
            return not bool(mode & stat.S_IWRITE)
        except Exception:
            return False

# ==============================================================================
# UNIFIED ID3 HELPER FOR MP3 / WAV / AIFF
# ==============================================================================
class UnifiedID3Handler:
    @staticmethod
    def load_tags(filepath: Path) -> Optional[ID3]:
        ext = filepath.suffix.lower()
        p_str = str(filepath.resolve())
        try:
            if ext == '.mp3':
                try:
                    return ID3(p_str)
                except ID3NoHeaderError:
                    return ID3()
            elif ext == '.wav':
                w = WAVE(p_str)
                if w.tags is None:
                    w.add_tags()
                return w.tags
            elif ext == '.aiff':
                a = AIFF(p_str)
                if a.tags is None:
                    a.add_tags()
                return a.tags
        except Exception:
            return None
        return None

    @staticmethod
    def save_tags(filepath: Path, tags: ID3):
        ext = filepath.suffix.lower()
        p_str = str(filepath.resolve())
        if ext == '.mp3':
            tags.save(p_str, v2_version=3)
        elif ext in ('.wav', '.aiff'):
            tags.save(p_str, v2_version=3)

# ==============================================================================
# TWO-STAGE DUPLICATE & CLONE AUDITOR
# ==============================================================================
class DuplicateAuditor:
    @staticmethod
    def _normalize_stem_for_clone_check(stem: str) -> str:
        """Strips copy suffixes like (1), (2), (Copy), and normalizes spacing/punctuation."""
        s = stem.lower()
        s = re.sub(r'\s*[\(\[](?:copy|\d+)[\)\]]\s*$', '', s)
        s = re.sub(r'[^a-z0-9]+', '', s)
        return s

    @staticmethod
    def _hash_audio_tail(filepath: Path, tail_bytes: int = 524288) -> Optional[str]:
        """Hashes the last 512 KB of the file (bypassing ID3v2 header padding differences)."""
        try:
            sz = filepath.stat().st_size
            if sz < 65536:
                return None
            offset = max(0, sz - tail_bytes)
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                f.seek(offset)
                data = f.read(tail_bytes)
                hasher.update(data)
            return hasher.hexdigest()
        except Exception:
            return None

    @classmethod
    def audit_duplicates(cls, diagnostics: List[TrackDiagnostic]) -> List[TrackDiagnostic]:
        tail_hash_map: Dict[str, Path] = {}
        meta_map: Dict[str, Path] = {}
        stem_map: Dict[str, Path] = {}

        # Audit clean un-suffixed files BEFORE '(2)' / 'Copy' files so the clean filename is always kept as the Master
        ordered_diags = sorted(
            diagnostics,
            key=lambda x: (1 if strip_copy_suffix(x.path.stem)[1] else 0, len(x.path.name), x.path.name.lower())
        )

        for d in ordered_diags:
            d.is_duplicate = False
            d.duplicate_reason = ''
            d.master_path = None
            d.is_locked = FileLockManager.is_read_only(d.path)

            base_stem, has_copy_suffix = strip_copy_suffix(d.path.stem)

            # Stage 0 (Disk-Level Copy Suffix Check): If file is named 'Track (2).mp3' or 'Track - Copy.mp3',
            # check if 'Track.mp3' exists on disk in the same folder (even if not in a partial drag-and-drop selection!)
            if has_copy_suffix:
                candidate_master = d.path.parent / f"{base_stem}{d.path.suffix}"
                if candidate_master.exists() and candidate_master.resolve() != d.path.resolve():
                    d.is_duplicate = True
                    d.master_path = candidate_master
                    d.duplicate_reason = f"OS Copy Suffix Clone of {candidate_master.name} (On Disk)"
                    continue

            # Stage 1: Audio-tail MD5 hash (catches identical audio even when ID3 padding differs)
            tail_hash = cls._hash_audio_tail(d.path)
            if tail_hash:
                if tail_hash in tail_hash_map:
                    d.is_duplicate = True
                    d.master_path = tail_hash_map[tail_hash]
                    d.duplicate_reason = f"Audio Stream Clone of {d.master_path.name}"
                    continue
                else:
                    tail_hash_map[tail_hash] = d.path

            # Stage 2: Metadata / Artist-Title clone check (ignore generic 'Unknown Artist')
            if d.clean_artist and not is_generic_artist_name(d.clean_artist) and d.clean_title:
                key_str = f"{d.clean_artist.lower()} - {d.clean_title.lower()}"
                if key_str in meta_map:
                    d.is_duplicate = True
                    d.master_path = meta_map[key_str]
                    d.duplicate_reason = f"Metadata Clone of {d.master_path.name}"
                    continue
                else:
                    meta_map[key_str] = d.path

            # Stage 3: Normalized Filename Stem check (catches 'Track (2).mp3' & 'Track -118' vs 'Track - 118')
            norm_stem = cls._normalize_stem_for_clone_check(d.path.stem)
            if len(norm_stem) >= 5:
                if norm_stem in stem_map:
                    d.is_duplicate = True
                    d.master_path = stem_map[norm_stem]
                    d.duplicate_reason = f"Filename Clone of {d.master_path.name}"
                else:
                    stem_map[norm_stem] = d.path

        return diagnostics

# ==============================================================================
# DIAGNOSTICS & AUDITING ENGINE
# ==============================================================================
class DiagnosticEngine:
    @staticmethod
    def audit_crate(file_list: List[Path], root_dir: Path, progress_callback=None) -> List[TrackDiagnostic]:
        diagnostics = []
        total = len(file_list)

        for idx, p in enumerate(file_list, start=1):
            if progress_callback and (idx % 15 == 0 or idx == total):
                progress_callback(idx, total, p.name)

            d = TrackDiagnostic(path=p)
            base_name = p.stem

            try:
                rel = p.relative_to(root_dir)
                if len(rel.parts) > 1:
                    d.folder_category = rel.parts[0].title()
            except Exception:
                pass

            # Fast duration check during crate scan
            try:
                info = sf.info(str(p))
                if info.duration >= LONG_TRACK_THRESHOLD_SEC:
                    d.is_long_track = True
            except Exception:
                try:
                    m_file = File(str(p))
                    if m_file and m_file.info and getattr(m_file.info, 'length', 0) >= LONG_TRACK_THRESHOLD_SEC:
                        d.is_long_track = True
                except Exception:
                    pass

            if p.suffix.lower() in ID3_COMPATIBLE_EXTENSIONS:
                tags = UnifiedID3Handler.load_tags(p)
                if tags is not None:
                    d.raw_title = str(tags.get('TIT2', ''))
                    d.raw_artist = str(tags.get('TPE1', ''))
                    d.raw_album = str(tags.get('TALB', ''))
                    d.raw_genre = str(tags.get('TCON', ''))
                    d.raw_mood = str(tags.get('TMOO', ''))

                    if 'TXXX:FurtherAnalysis' in tags:
                        d.flagged = True

                    if 'GEOB:VantaAudioMetrics' in tags or (
                        any(k.startswith('COMM') for k in tags.keys()) and 'TKEY' in tags
                    ):
                        d.is_cached = True
                        if 'GEOB:VantaAudioMetrics' in tags:
                            try:
                                payload = json.loads(tags['GEOB:VantaAudioMetrics'].data.decode('utf-8'))
                                d.is_fake_320k = bool(payload.get('is_fake_320k', False))
                                d.is_clipped = bool(payload.get('is_clipped', False))
                            except Exception:
                                pass

                    d.clean_title = strip_copy_suffix(sanitize_name_string(d.raw_title))[0]
                    d.clean_artist = sanitize_name_string(d.raw_artist)
                    d.clean_genre = sanitize_name_string(d.raw_genre)
                    d.has_existing_cues = 'GEOB:Serato Markers2' in tags

            clean_album = sanitize_name_string(d.raw_album)
            has_valid_artist = bool(d.clean_artist and not is_generic_artist_name(d.clean_artist))
            has_album_clutter = bool(clean_album and len(clean_album) >= 4 and clean_album.lower() in base_name.lower())
            has_mid_track_num = bool(re.search(r'\s[-–]\s*\d{1,3}\s+[A-Za-z]', base_name))

            if (is_hash_or_numeric_filename(base_name) or has_album_clutter or has_mid_track_num) and has_valid_artist and d.clean_title:
                # Prevent double-artist prefixing if clean_title already starts with clean_artist
                if d.clean_title.lower().startswith(d.clean_artist.lower() + " - "):
                    ideal_name = d.clean_title
                else:
                    ideal_name = f"{d.clean_artist} - {d.clean_title}"
            else:
                clean_name = sanitize_name(base_name)
                if clean_name:
                    ideal_name = clean_name
                elif has_valid_artist and d.clean_title:
                    ideal_name = f"{d.clean_artist} - {d.clean_title}"
                elif d.clean_title:
                    ideal_name = d.clean_title
                else:
                    ideal_name = base_name

            safe_filename = re.sub(r'[\\/*?:"<>!%#]', '', ideal_name).strip('. ')
            raw_proposed = f"{safe_filename}{p.suffix}" if safe_filename else p.name
            d.proposed_filename = ensure_extension(raw_proposed, p.suffix)
            diagnostics.append(d)

        diagnostics = DuplicateAuditor.audit_duplicates(diagnostics)
        return diagnostics

# ==============================================================================
# SERATO BINARY & NATIVE CRATE ENGINE
# ==============================================================================
class SeratoBinaryEngine:
    @staticmethod
    def build_serato_markers2_payload(cue_positions_ms: List[int], track_color_rgb: Tuple[int, int, int] = (255, 255, 255)) -> bytes:
        entries = []
        tr, tg, tb = track_color_rgb
        entries.append(b'COLOR\x00' + struct.pack('>I B B B B', 4, 0x00, tr, tg, tb))
        entries.append(b'BPMLOCK\x00' + struct.pack('>I B', 1, 0x00))

        for slot, ms in enumerate(cue_positions_ms[:8]):
            r, g, b = CUE_COLORS[slot % len(CUE_COLORS)]
            label_str = CUE_LABELS[slot] if slot < len(CUE_LABELS) else f'Cue {slot + 1}'
            cue_label = label_str.encode('utf-8') + b'\x00'
            cue_data = struct.pack('>B B I B B B B B B', 0x00, slot, int(ms), 0x00, r, g, b, 0x00, 0x00) + cue_label
            entries.append(b'CUE\x00' + struct.pack('>I', len(cue_data)) + cue_data)

        decoded_body = b'\x01\x01' + b''.join(entries)
        full_payload = b'\x01\x01' + base64.b64encode(decoded_body)
        if len(full_payload) < 512:
            full_payload = full_payload.ljust(512, b'\x00')
        return full_payload

    @staticmethod
    def build_autogain_payload(gain_db: float) -> bytes:
        return base64.b64encode(struct.pack('>f', float(gain_db)))

    @staticmethod
    def _pack_chunk(tag: bytes, data: bytes) -> bytes:
        return tag + struct.pack('>I', len(data)) + data

    @staticmethod
    def _format_serato_path(full_win_path: str) -> str:
        """Exact path formatter from Harmonic Arc Engine v10.7."""
        p = full_win_path.replace('\\', '/')
        p = re.sub(r'^[a-zA-Z]:', '', p)
        return p.lstrip('/')

    @staticmethod
    def resolve_serato_subcrates_dir(music_or_track_path: Path) -> Path:
        """Exact Subcrates directory resolver from Harmonic Arc Engine v10.7."""
        base_dir = music_or_track_path if music_or_track_path.is_dir() else music_or_track_path.parent
        candidates = [
            base_dir / "_Serato_" / "Subcrates",
            base_dir.parent / "_Serato_" / "Subcrates",
            Path.home() / "Music" / "_Serato_" / "Subcrates",
        ]
        if os.name == "nt" and base_dir.drive:
            candidates.insert(1, Path(base_dir.drive + "\\") / "_Serato_" / "Subcrates")
        for c in candidates:
            if c.exists():
                return c
        return candidates[-1]

    @classmethod
    def _build_default_serato_header(cls) -> bytes:
        """Builds a complete Serato ScratchLive Crate binary header with visible column definitions (ovct/tvcn/tvcw)."""
        vrsn_str = "81.0/Serato ScratchLive Crate".encode('utf-16-be')
        header = cls._pack_chunk(b'vrsn', vrsn_str)

        # Sort by song column
        osrt_body = cls._pack_chunk(b'tvcn', "song".encode('utf-16-be')) + cls._pack_chunk(b'brev', b'\x00')
        header += cls._pack_chunk(b'osrt', osrt_body)

        # Standard visible Serato DJ Pro columns so the crate table renders all rows
        for col_name in ("song", "artist", "bpm", "key", "length", "comment"):
            ovct_body = (
                cls._pack_chunk(b'tvcn', col_name.encode('utf-16-be')) +
                cls._pack_chunk(b'tvcw', "0".encode('utf-16-be'))
            )
            header += cls._pack_chunk(b'ovct', ovct_body)
        return header

    @classmethod
    def _get_serato_crate_header(cls, subcrates_dir: Path) -> bytes:
        """Clones the exact binary header (including ovct column chunks) from a native Serato .crate file, matching Harmonic Arc Engine v10.7."""
        template_path = subcrates_dir / "created manually added.crate"
        candidates = [template_path] if template_path.exists() else []
        for p in subcrates_dir.glob("*.crate"):
            if p not in candidates:
                candidates.append(p)

        for cand in candidates:
            try:
                template_data = cand.read_bytes()
                otrk_idx = template_data.find(b"otrk")
                header = template_data[:otrk_idx] if otrk_idx != -1 else template_data
                # Ensure the template actually contains Serato column definitions (ovct)
                if b"vrsn" in header and b"ovct" in header:
                    return header
            except Exception:
                continue

        return cls._build_default_serato_header()

    @staticmethod
    def _read_existing_crate_rel_paths(crate_path: Path) -> List[str]:
        """Reads existing UTF-16BE ptrk entries from a Serato .crate file using byte-search like Harmonic Arc Engine v10.7."""
        rel_paths: List[str] = []
        if not crate_path.exists():
            return rel_paths
        try:
            data = crate_path.read_bytes()
            idx = 0
            while True:
                idx = data.find(b"ptrk", idx)
                if idx == -1 or idx + 8 > len(data):
                    break
                length = struct.unpack(">I", data[idx + 4 : idx + 8])[0]
                raw_path = data[idx + 8 : idx + 8 + length].decode("utf-16-be", errors="ignore")
                if raw_path:
                    rel_paths.append(raw_path)
                idx += 8 + length
        except Exception:
            pass
        return rel_paths

    @staticmethod
    def is_serato_running() -> bool:
        """Detects if Serato DJ Pro / ScratchLive is currently running in RAM."""
        try:
            if os.name == 'nt':
                creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                out = subprocess.check_output(
                    ['tasklist', '/NH'],
                    creationflags=creation_flags,
                    stderr=subprocess.DEVNULL
                ).decode('utf-8', errors='ignore').lower()
                return ('serato dj' in out) or ('scratchlive' in out)
            else:
                out = subprocess.check_output(['ps', '-A'], stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore').lower()
                return 'serato' in out
        except Exception:
            return False

    @classmethod
    def _rel_path_exists_on_disk(cls, rel_path: str, crate_file: Path, sample_tracks: List[Path]) -> bool:
        """Checks if an existing relative ptrk entry still exists on disk (purges renamed/deleted ghost tracks)."""
        clean_rel = rel_path.lstrip('/\\')
        candidate_anchors = []
        if crate_file.drive:
            candidate_anchors.append(Path(crate_file.drive + "\\"))
        for p in sample_tracks[:3]:
            if p.drive:
                anch = Path(p.drive + "\\")
                if anch not in candidate_anchors:
                    candidate_anchors.append(anch)
        candidate_anchors.append(Path("/"))

        for anchor in candidate_anchors:
            try:
                if (anchor / clean_rel).exists():
                    return True
            except Exception:
                continue
        return False

    @classmethod
    def _write_single_crate_file(cls, crate_file: Path, header: bytes, track_paths: List[Path], append: bool = True) -> Optional[Path]:
        ordered_rel_paths: List[str] = []
        seen = set()

        if append and crate_file.exists():
            for existing_rel in cls._read_existing_crate_rel_paths(crate_file):
                norm_key = existing_rel.lower()
                # Ghost-Path Cleaner: only retain existing crate entries if the file still exists on disk
                if norm_key not in seen and cls._rel_path_exists_on_disk(existing_rel, crate_file, track_paths):
                    seen.add(norm_key)
                    ordered_rel_paths.append(existing_rel)

        for p in track_paths:
            clean_p = cls._format_serato_path(str(p))
            norm_key = clean_p.lower()
            if norm_key not in seen:
                seen.add(norm_key)
                ordered_rel_paths.append(clean_p)

        track_payload = b""
        for clean_p in ordered_rel_paths:
            ptrk_block = cls._pack_chunk(b"ptrk", clean_p.encode("utf-16-be"))
            track_payload += cls._pack_chunk(b"otrk", ptrk_block)

        with open(crate_file, "wb") as f:
            f.write(header + track_payload)
        return crate_file

    @classmethod
    def export_serato_crate(cls, crate_name: str, track_paths: List[Path], output_dir: Optional[Path] = None, append: bool = True) -> Optional[Path]:
        """
        Exports a native Serato .crate using the proven Harmonic Arc Engine v10.7 binary cloner.
        Also automatically populates any parent crates (when '%%' hierarchy is used) so parent folders are never empty in Serato!
        """
        valid_paths = [p.resolve() for p in track_paths if p.exists()]
        if not valid_paths:
            return None
        try:
            subcrates_dir = output_dir if output_dir is not None else cls.resolve_serato_subcrates_dir(valid_paths[0])
            subcrates_dir.mkdir(parents=True, exist_ok=True)

            header = cls._get_serato_crate_header(subcrates_dir)
            safe_crate = re.sub(r'[\\/*?:"<>|]', '_', crate_name).strip('. ')

            # If '%%' nested subcrates are used (e.g. Vanta%%Energy%%8-10 Peak Hour),
            # also populate parent crates (Vanta.crate and Vanta%%Energy.crate) so clicking them in Serato shows all child tracks
            if "%%" in safe_crate:
                parts = [pt for pt in safe_crate.split("%%") if pt]
                for i in range(1, len(parts)):
                    parent_name = "%%".join(parts[:i])
                    parent_file = subcrates_dir / f"{parent_name}.crate"
                    cls._write_single_crate_file(parent_file, header, valid_paths, append=True)

            crate_file = subcrates_dir / f"{safe_crate}.crate"
            return cls._write_single_crate_file(crate_file, header, valid_paths, append=append)
        except Exception:
            traceback.print_exc()
            return None

    @classmethod
    def inject_to_file(cls, filepath: Path, analysis: TrackAnalysisResult, config: Dict[str, bool]) -> bool:
        try:
            was_locked = FileLockManager.is_read_only(filepath)
            if was_locked:
                FileLockManager.set_read_only(filepath, read_only=False)

            file_str = str(filepath.resolve())
            ext = filepath.suffix.lower()

            if ext in ID3_COMPATIBLE_EXTENSIONS:
                tags = UnifiedID3Handler.load_tags(filepath)
                if tags is None:
                    return False

                if config.get('clean_tags', True):
                    if config.get('tag_title_artist', True):
                        if analysis.cleaned_title:
                            tags['TIT2'] = TIT2(encoding=3, text=[analysis.cleaned_title])
                        if analysis.cleaned_artist and not is_generic_artist_name(analysis.cleaned_artist):
                            tags['TPE1'] = TPE1(encoding=3, text=[analysis.cleaned_artist])

                    if config.get('tag_genre', True):
                        if analysis.detected_genre and analysis.detected_genre not in ('Unknown', 'Unassigned'):
                            tags['TCON'] = TCON(encoding=3, text=[analysis.detected_genre])

                if config.get('run_dsp', True):
                    if config.get('tag_bpm', True) and analysis.detected_bpm > 0:
                        tags['TBPM'] = TBPM(encoding=3, text=[f"{analysis.detected_bpm:.2f}"])

                    if config.get('tag_key', True) and analysis.key and analysis.key != 'Unknown':
                        tags.setall('TKEY', [TKEY(encoding=3, text=[analysis.key])])
                        tags.setall('TXXX:Key', [TXXX(encoding=3, desc='Key', text=analysis.key)])
                        tags.setall('TXXX:INITIALKEY', [TXXX(encoding=3, desc='INITIALKEY', text=analysis.key)])

                    if config.get('tag_mood', True) and analysis.mood:
                        tags.setall('TMOO', [TMOO(encoding=3, text=[analysis.mood])])
                        tags.setall('TXXX:Mood', [TXXX(encoding=3, desc='Mood', text=analysis.mood)])

                    if config.get('tag_energy', True):
                        tags.setall('TXXX:Energy', [TXXX(encoding=3, desc='Energy', text=str(analysis.energy_level))])
                        tags.setall('TXXX:EnergyDelta', [TXXX(encoding=3, desc='EnergyDelta', text=f"{analysis.energy_delta:+.2f}")])

                    if config.get('tag_geob_metrics', True):
                        metrics_dict = {
                            "bpm": analysis.detected_bpm,
                            "key": analysis.key,
                            "tuning_cents": round(analysis.tuning_cents, 1),
                            "energy": analysis.energy_level,
                            "energy_delta": round(analysis.energy_delta, 3),
                            "crest_factor_db": round(analysis.crest_factor_db, 2),
                            "mood": analysis.mood,
                            "lufs": round(analysis.integrated_lufs, 2),
                            "cutoff_hz": round(analysis.spectral_cutoff_hz, 1),
                            "is_fake_320k": analysis.is_fake_320k,
                            "is_clipped": analysis.is_clipped,
                            "intro_bars": analysis.intro_bars,
                            "outro_bars": analysis.outro_bars,
                            "flagged": analysis.flagged
                        }
                        geob_json = json.dumps(metrics_dict).encode('utf-8')
                        tags.delall('GEOB:VantaAudioMetrics')
                        tags.add(GEOB(encoding=3, mime='application/json', desc='VantaAudioMetrics', data=geob_json))

                    # Cleanly replace default Serato comment frames (COMM::eng and COMM:Summary)
                    tags.delall('COMM')
                    if config.get('tag_comment', True):
                        qual_flag = " | [LOW-BW]" if analysis.is_fake_320k else (" | [CLIP]" if analysis.is_clipped else "")
                        summary_text = (
                            f"Genre:{analysis.detected_genre} | Energy {analysis.energy_level}/10 "
                            f"| Key:{analysis.key} | BPM:{analysis.detected_bpm:.1f}{qual_flag}"
                        )
                        tags.add(COMM(encoding=3, lang='eng', desc='', text=summary_text))

                    if analysis.flagged:
                        tags['TXXX:FurtherAnalysis'] = TXXX(encoding=3, desc='FurtherAnalysis', text='TRUE')
                    else:
                        tags.delall('TXXX:FurtherAnalysis')

                if config.get('inject_cues', True) and analysis.cues_ms and not analysis.is_long_track:
                    tags.delall('GEOB:Serato Markers_')
                    tags.delall('GEOB:Serato Markers2')
                    tags.delall('GEOB:Serato Offsets_')
                    # Visual tint in Serato row if fake 320k or flagged
                    row_color = (255, 153, 153) if analysis.is_fake_320k else ((255, 230, 153) if analysis.flagged else (255, 255, 255))
                    marker_data = cls.build_serato_markers2_payload(analysis.cues_ms, track_color_rgb=row_color)
                    tags.add(GEOB(encoding=0, mime='application/octet-stream', desc='Serato Markers2', data=marker_data))

                if config.get('inject_autogain', True):
                    tags.delall('GEOB:Serato Autogain')
                    gain_data = cls.build_autogain_payload(analysis.gain_offset_db)
                    tags.add(GEOB(encoding=0, mime='application/octet-stream', desc='Serato Autogain', data=gain_data))

                UnifiedID3Handler.save_tags(filepath, tags)
                os.utime(file_str, None)
            else:
                audio = File(file_str)
                if audio is not None:
                    try:
                        if getattr(audio, 'tags', None) is None:
                            audio.add_tags()
                        qual_flag = " | [LOW-BW]" if analysis.is_fake_320k else (" | [CLIP]" if analysis.is_clipped else "")
                        summary_str = (
                            f"Genre:{analysis.detected_genre} | Energy {analysis.energy_level}/10 "
                            f"| Key:{analysis.key} | BPM:{analysis.detected_bpm:.1f}{qual_flag}"
                        )
                        if ext == '.flac':
                            if config.get('clean_tags', True):
                                if config.get('tag_title_artist', True):
                                    if analysis.cleaned_title:
                                        audio['title'] = [analysis.cleaned_title]
                                    if analysis.cleaned_artist:
                                        audio['artist'] = [analysis.cleaned_artist]
                                if config.get('tag_genre', True) and analysis.detected_genre:
                                    audio['genre'] = [analysis.detected_genre]
                            if config.get('run_dsp', True):
                                if config.get('tag_bpm', True) and analysis.detected_bpm > 0:
                                    audio['bpm'] = [f"{analysis.detected_bpm:.2f}"]
                                if config.get('tag_key', True) and analysis.key and analysis.key != 'Unknown':
                                    audio['initialkey'] = [analysis.key]
                                    audio['key'] = [analysis.key]
                                if config.get('tag_mood', True) and analysis.mood:
                                    audio['mood'] = [analysis.mood]
                                if config.get('tag_energy', True):
                                    audio['energy'] = [str(analysis.energy_level)]
                                if config.get('tag_comment', True):
                                    audio['comment'] = [summary_str]
                        elif ext == '.m4a':
                            if config.get('clean_tags', True):
                                if config.get('tag_title_artist', True):
                                    if analysis.cleaned_title:
                                        audio['\xa9nam'] = [analysis.cleaned_title]
                                    if analysis.cleaned_artist:
                                        audio['\xa9ART'] = [analysis.cleaned_artist]
                                if config.get('tag_genre', True) and analysis.detected_genre:
                                    audio['\xa9gen'] = [analysis.detected_genre]
                            if config.get('run_dsp', True):
                                if config.get('tag_bpm', True) and analysis.detected_bpm > 0:
                                    audio['tmpo'] = [int(round(analysis.detected_bpm))]
                                if config.get('tag_key', True) and analysis.key and analysis.key != 'Unknown':
                                    audio['----:com.apple.iTunes:initialkey'] = [analysis.key.encode('utf-8')]
                                if config.get('tag_comment', True):
                                    audio['\xa9cmt'] = [summary_str]
                        else:
                            if config.get('run_dsp', True) and config.get('tag_comment', True):
                                audio['comment'] = [summary_str]
                        audio.save()
                    except Exception:
                        pass

            if config.get('lock_files', True):
                FileLockManager.set_read_only(filepath, read_only=True)

            return True
        except Exception:
            traceback.print_exc()
            return False

# ==============================================================================
# SINGLE-PASS DSP ENGINE WITH ANTI-HALVING & TUNING CALIBRATION
# ==============================================================================
class DSPAnalyzer:
    @staticmethod
    def load_audio_fast_and_safe(filepath: Path) -> Tuple[np.ndarray, int]:
        """
        Loads audio into a mono float32 array.
        0. Pre-flight check: verifies file exists, is a regular file, and is non-empty.
        1. Primary: soundfile (libsndfile).
        2. Fallback A: In-memory MPEG frame-sync recovery (scans up to 2 MB to strip corrupt ID3v2 padding/garbage).
        3. Fallback B: audioread decode.
        4. Fallback C: librosa.load, raising a clean Corrupt/Unreadable error if all decoders fail.
        """
        if not filepath.exists() or not filepath.is_file():
            raise ValueError("Missing or non-regular audio file")
        try:
            if filepath.stat().st_size < 128:
                raise ValueError("Empty or truncated audio file (<128 bytes)")
        except OSError as e:
            raise ValueError(f"Unreadable file handle: {e}")

        p_str = str(filepath.resolve())
        try:
            data, rate = sf.read(p_str, always_2d=True, dtype='float32')
            y = np.mean(data, axis=1)
            return y, int(rate)
        except Exception:
            pass

        # Fallback A: Strip corrupt ID3v2 header/padding in-memory and feed clean MPEG stream to soundfile
        try:
            import io
            raw_bytes = filepath.read_bytes()
            start_offset = 0
            if raw_bytes[:3] == b'ID3' and len(raw_bytes) >= 10:
                tag_size = (
                    ((raw_bytes[6] & 0x7F) << 21) |
                    ((raw_bytes[7] & 0x7F) << 14) |
                    ((raw_bytes[8] & 0x7F) << 7) |
                    (raw_bytes[9] & 0x7F)
                )
                start_offset = min(len(raw_bytes) - 4, 10 + tag_size)
            # Scan forward up to 2 MB for first valid MPEG Layer III sync word
            scan_limit = min(len(raw_bytes) - 4, start_offset + (2 * 1024 * 1024))
            found_sync = -1
            for idx in range(start_offset, scan_limit):
                if raw_bytes[idx] == 0xFF and (raw_bytes[idx + 1] & 0xE0) == 0xE0 and (raw_bytes[idx + 1] & 0x06) == 0x02:
                    found_sync = idx
                    break
            if found_sync > 0:
                bio = io.BytesIO(raw_bytes[found_sync:])
                data, rate = sf.read(bio, always_2d=True, dtype='float32')
                if len(data) > 0:
                    return np.mean(data, axis=1), int(rate)
        except Exception:
            pass

        # Fallback B: Direct audioread decode (bypasses soundfile completely)
        try:
            import audioread
            chunks = []
            with audioread.audio_open(p_str) as afile:
                rate = int(afile.samplerate)
                channels = int(afile.channels)
                for buf in afile:
                    pcm = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                    if channels > 1:
                        pcm = pcm.reshape(-1, channels).mean(axis=1)
                    chunks.append(pcm)
            if chunks:
                return np.concatenate(chunks), rate
        except Exception:
            pass

        # Last-resort fallback
        try:
            y, rate = librosa.load(p_str, sr=22050, mono=True)
            if len(y) == 0:
                raise ValueError("Decoded 0 audio samples")
            return y.astype(np.float32), int(rate)
        except Exception:
            raise ValueError("Corrupt / Unreadable Audio File (all decoders failed)")

    @staticmethod
    def fast_downsample_22k(y: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
        """
        Replaces slow Kaiser-sinc librosa.resample with instant strided decimation
        after a lightweight 2-tap moving-average anti-aliasing filter.
        """
        if orig_sr >= 30000 and len(y) >= 2:
            # Ensure even length so y[0:even_len:2] and y[1:even_len:2] have identical shapes
            even_len = (len(y) // 2) * 2
            y_low = (y[0:even_len:2] + y[1:even_len:2]) * 0.5
            return y_low.astype(np.float32), orig_sr // 2
        return y.astype(np.float32), orig_sr

    @staticmethod
    def extract_filename_bpm_hint(stem: str) -> Optional[float]:
        """
        Extracts an explicit DJ BPM hint from filenames like 'Track - 8A - 132' or 'Track - 88 - 8A'.
        Ignores leading-zero compilation/episode numbers (e.g. '068', '014') and track numbers.
        """
        matches = re.findall(r'(?:^|[-_\s])([1-9]\d{1,2}(?:\.\d)?)(?=$|[-_\s])', stem)
        for m in reversed(matches):
            try:
                val = float(m)
                if 70.0 <= val <= 190.0 and val not in (1997, 1998, 1999, 2000, 2001, 2002, 2007, 2008, 2012, 2013, 2014, 2017, 2019, 2026):
                    return val
            except ValueError:
                pass
        return None

    @staticmethod
    def compute_precise_bpm(
        oenv: np.ndarray,
        beat_times: np.ndarray,
        sr: int,
        hop_length: int,
        stft_sample: np.ndarray,
        freqs: np.ndarray,
        filename_stem: str = "",
        context_hint: str = ""
    ) -> Tuple[float, bool]:
        """
        Computes sub-bin accurate BPM using Inter-Beat Interval (IBI) median filtering
        and reconciles 2x, 0.5x, 1.5x (3/2 Jungle polyrhythm), 4/3, and 3/4 ratios.
        """
        flagged = False
        hint_bpm = DSPAnalyzer.extract_filename_bpm_hint(filename_stem)

        ibi_bpm = 0.0
        ibi_cv = 0.0
        if len(beat_times) >= 8:
            ibis = np.diff(beat_times)
            valid_ibis = ibis[(ibis >= 0.25) & (ibis <= 1.1)]
            if len(valid_ibis) >= 6:
                median_ibi = float(np.median(valid_ibis))
                inliers = valid_ibis[np.abs(valid_ibis - median_ibi) <= (0.12 * median_ibi)]
                if len(inliers) >= 4:
                    mean_ibi = float(np.mean(inliers))
                    ibi_bpm = 60.0 / mean_ibi
                    ibi_cv = float(np.std(inliers) / (mean_ibi + 1e-6))
                else:
                    ibi_bpm = 60.0 / median_ibi
                    ibi_cv = float(np.std(valid_ibis) / (median_ibi + 1e-6))

        if ibi_bpm <= 0.0:
            try:
                tempi = librosa.feature.rhythm.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
            except AttributeError:
                tempi = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
            ibi_bpm = float(np.median(tempi)) if len(tempi) > 0 else 124.0

        bpm = ibi_bpm

        # Reconcile harmonic/triplet multiples against filename BPM hint (2x, 0.5x, 1.5x, 3/4, 4/3)
        if hint_bpm:
            for mult in (1.0, 2.0, 0.5, 1.5, 2.0 / 3.0, 0.75, 4.0 / 3.0):
                if abs((bpm * mult) - hint_bpm) <= 3.0:
                    bpm = bpm * mult
                    break

        ctx_lower = f"{filename_stem} {context_hint}".lower()
        is_jungle_context = any(
            kw in ctx_lower for kw in (
                "jungle", "dnb", "d&b", "drum & bass", "drum and bass",
                "ragga", "riddim", "amen", "breakcore", "stp rmx", "stp mash", "rastafarando"
            )
        )

        low_energy = float(np.mean(stft_sample[freqs <= 140, :]))
        mid_energy = float(np.mean(stft_sample[(freqs > 140) & (freqs <= 2000), :])) + 1e-6
        high_energy = float(np.mean(stft_sample[freqs > 3000, :]))
        sub_to_mid = low_energy / mid_energy
        high_to_mid = high_energy / mid_energy
        onset_density = float(np.mean(oenv > np.percentile(oenv, 75))) if len(oenv) > 0 else 0.15

        # 1. Jungle / D&B 2/3 Polyrhythmic Triplet Trap (110.0 - 124.0 BPM -> * 1.5 = 165 - 186 BPM)
        # Requires either explicit Jungle/D&B/Ragga/Riddim/160 context OR extreme Amen/breakbeat transient choppiness
        # (prevents 120-122 BPM Progressive/Deep House from being falsely scaled by 1.5x)
        flux_val = float(np.mean(np.maximum(0.0, np.diff(stft_sample, axis=1)))) if stft_sample.shape[1] > 1 else 0.0
        mean_amp = float(np.mean(stft_sample)) + 1e-6
        perc_ratio = float(flux_val / (mean_amp * 0.45 + flux_val + 1e-6))

        has_160_tag = ("160" in filename_stem.lower())
        if 110.0 <= bpm <= 124.0 and not hint_bpm:
            if (
                (is_jungle_context and bpm <= 124.0)
                or (has_160_tag and 105.0 <= bpm <= 112.0)
                or (perc_ratio > 0.56 and high_to_mid > 0.30 and onset_density > 0.26 and sub_to_mid > 1.45)
            ):
                bpm = bpm * 1.5

        # 2. Jungle / D&B Half-Time Trap (80.0 - 95.0 BPM -> * 2.0 = 160 - 190 BPM)
        if 80.0 <= bpm <= 95.0 and (not hint_bpm or hint_bpm >= 155.0):
            if is_jungle_context or (sub_to_mid > 1.70 and high_to_mid > 0.16 and onset_density > 0.17):
                bpm = bpm * 2.0
                if not is_jungle_context:
                    flagged = True

        if bpm > 186.0 and (not hint_bpm or hint_bpm < 150.0):
            bpm = bpm * 0.5
            flagged = True

        nearest_half = round(bpm * 2.0) / 2.0
        if abs(bpm - nearest_half) <= 0.22 and ibi_cv < 0.05:
            bpm = nearest_half
        else:
            bpm = round(bpm, 1)

        if ibi_cv > 0.085 and not is_jungle_context:
            flagged = True

        return float(bpm), flagged

    @staticmethod
    def compute_spectral_cutoff_hz(y: np.ndarray, sr: int) -> Tuple[float, bool]:
        """Fast 30-second mid-band referenced spectral roll-off detector."""
        try:
            slice_len = min(len(y), sr * 30)
            if slice_len < 8192:
                return 20000.0, False
            start_idx = min(sr * 30, max(0, len(y) - slice_len))
            y_seg = y[start_idx:start_idx + slice_len]

            fft_size = 2048
            S = np.abs(librosa.stft(y_seg, n_fft=fft_size, hop_length=2048))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_size)

            spec_p90 = np.percentile(S, 90, axis=1)
            spec_db = 20.0 * np.log10(np.maximum(spec_p90, 1e-7))

            mid_mask = (freqs >= 1000.0) & (freqs <= 7000.0)
            if not np.any(mid_mask):
                return 20000.0, False

            ref_mid_db = float(np.percentile(spec_db[mid_mask], 75))
            threshold_db = max(ref_mid_db - 42.0, -95.0)

            high_mask = freqs >= 8000.0
            audible_high_bins = np.where(high_mask & (spec_db >= threshold_db))[0]
            if len(audible_high_bins) > 0:
                cutoff_hz = float(freqs[audible_high_bins[-1]])
            else:
                audible_any = np.where(spec_db >= threshold_db)[0]
                cutoff_hz = float(freqs[audible_any[-1]]) if len(audible_any) > 0 else 16000.0

            is_fake = cutoff_hz < SPECTRAL_128K_CUTOFF_HZ
            return cutoff_hz, is_fake
        except Exception:
            return 20000.0, False

    @staticmethod
    def compute_dj_energy_level(integrated_lufs: float, rms_full: np.ndarray, oenv_full: np.ndarray, bpm: float) -> int:
        clamped_lufs = float(np.clip(integrated_lufs, -24.0, -3.5))
        lufs_norm = (clamped_lufs - (-24.0)) / (-3.5 - (-24.0))

        if len(oenv_full) > 0:
            active_ratio = float(np.mean(oenv_full > np.percentile(oenv_full, 65)))
            onset_norm = float(np.clip((active_ratio - 0.15) / 0.25, 0.0, 1.0))
        else:
            onset_norm = 0.5

        bpm_norm = float(np.clip((bpm - 95.0) / 80.0, 0.0, 1.0))

        combined = (lufs_norm * 0.68) + (onset_norm * 0.20) + (bpm_norm * 0.12)
        score = 1.0 + (9.0 * combined)
        return int(np.clip(np.round(score), 1, 10))

    @staticmethod
    def classify_genre_fast(stft_sample: np.ndarray, freqs: np.ndarray, oenv_sample: np.ndarray, bpm: float, folder_category: str, existing_genre: str) -> str:
        """
        Ultra-fast genre classifier using precomputed STFT band ratios & transient crest
        (eliminates slow 2D median filtering in librosa.effects.hpss).
        """
        if existing_genre and existing_genre not in ('Unknown', 'Unassigned', ''):
            return existing_genre

        try:
            low_energy = float(np.mean(stft_sample[freqs <= 150, :]))
            total_energy = float(np.mean(stft_sample)) + 1e-6
            low_ratio = low_energy / total_energy

            # Percussive transient ratio from frame-to-frame spectral flux & onset crest
            flux = np.mean(np.maximum(0.0, np.diff(stft_sample, axis=1)))
            perc_ratio = float(flux / (total_energy * 0.45 + flux + 1e-6))

            onset_density = float(np.mean(oenv_sample > np.percentile(oenv_sample, 75))) if len(oenv_sample) > 0 else 0.15

            if bpm >= 158:
                if "ragga" in (folder_category or "").lower() or "jungle" in (folder_category or "").lower():
                    return "Jungle / Drum & Bass"
                if low_ratio > 0.42 and perc_ratio > 0.58 and onset_density > 0.24:
                    return "Hardcore / Electronic"
                return "Jungle / Drum & Bass"
            elif 138 <= bpm < 158:
                if perc_ratio > 0.48 and onset_density > 0.16:
                    return "Breaks / Electronic"
                return "Trance / Psy / Electronic"
            elif 118 <= bpm < 138:
                if perc_ratio > 0.46 and onset_density > 0.15:
                    return "Breaks / Electronic"
                elif low_ratio > 0.34:
                    return "Tech House / Electronic"
                return "House / Electronic"
            elif bpm < 118 and perc_ratio < 0.35 and low_ratio < 0.25:
                return "Ambient / Downtempo"
            elif folder_category and not any(
                k in folder_category.lower() for k in (
                    "various artist", "va -", "vol.", "vol ", "sampler", "retrospective",
                    "downloads", "bootleg", " lp", " ep", "dubz"
                )
            ):
                clean_folder = re.sub(r'[\(\)].*?[\(\)]|\d{4}', '', folder_category).strip()
                if clean_folder:
                    return clean_folder

            if low_ratio > 0.32 or bpm < 118:
                return "Midtempo / Funk / Breaks"
        except Exception:
            pass

        return "Electronic"

    @staticmethod
    def detect_safe_phrasing(rms_full: np.ndarray, frame_times: np.ndarray, total_sec: float, bpm: float) -> Tuple[int, int]:
        if bpm <= 0 or len(rms_full) == 0:
            return 16, 32

        sec_per_bar = (60.0 / bpm) * 4.0

        intro_window_frames = frame_times <= min(30.0, total_sec * 0.25)
        if np.any(intro_window_frames):
            baseline_rms = np.mean(rms_full[intro_window_frames])
            jump_idx = np.where(rms_full[intro_window_frames] > baseline_rms * 1.3)[0]
            intro_sec = float(frame_times[jump_idx[0]]) if len(jump_idx) > 0 else 16.0
        else:
            intro_sec = 16.0

        intro_bars_raw = round(intro_sec / sec_per_bar)
        intro_bars = 8 if intro_bars_raw <= 12 else (16 if intro_bars_raw <= 24 else 32)

        outro_window_frames = frame_times >= (total_sec - 30.0)
        if np.any(outro_window_frames):
            outro_rms = rms_full[outro_window_frames]
            drop_idx = np.where(outro_rms < np.max(outro_rms) * 0.5)[0]
            outro_sec = float(total_sec - frame_times[outro_window_frames][drop_idx[0]]) if len(drop_idx) > 0 else 32.0
        else:
            outro_sec = 32.0

        outro_bars_raw = round(outro_sec / sec_per_bar)
        outro_bars = 8 if outro_bars_raw <= 12 else (16 if outro_bars_raw <= 24 else 32)

        return intro_bars, outro_bars

    @staticmethod
    def detect_quantized_cues(
        sub_energy: np.ndarray,
        frame_times: np.ndarray,
        beat_times: np.ndarray,
        total_sec: float,
        bpm: float,
        sr: int,
        hop_length: int = 512,
        num_cues: int = 8
    ) -> List[int]:
        if total_sec < 15.0 or len(beat_times) < 16:
            return [0]

        sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
        sec_per_bar = sec_per_beat * 4.0

        kernel_size = max(1, int(sr / hop_length * sec_per_bar))
        sub_smoothed = np.convolve(sub_energy, np.ones(kernel_size) / kernel_size, mode='same')
        sub_diff = np.append(np.diff(sub_smoothed), 0.0)

        def snap_to_bar(target_s: float) -> float:
            closest_idx = int(np.argmin(np.abs(beat_times - target_s)))
            quantized_idx = int(round(closest_idx / 4.0) * 4)
            quantized_idx = min(max(0, quantized_idx), len(beat_times) - 1)
            return float(beat_times[quantized_idx])

        cue_1 = float(beat_times[0])
        drop1_search = (frame_times >= 15.0) & (frame_times <= min(180.0, total_sec * 0.45))
        if np.any(drop1_search):
            raw_drop1 = float(frame_times[drop1_search][np.argmax(sub_diff[drop1_search])])
            cue_4 = snap_to_bar(raw_drop1)
        else:
            cue_4 = snap_to_bar(min(60.0, total_sec * 0.33))

        bars_to_drop = (cue_4 - cue_1) / sec_per_bar
        if bars_to_drop >= 32:
            cue_2 = snap_to_bar(cue_1 + (32 * sec_per_bar))
        elif bars_to_drop >= 16:
            cue_2 = snap_to_bar(cue_1 + (16 * sec_per_bar))
        else:
            cue_2 = snap_to_bar(cue_1 + (8 * sec_per_bar))

        cue_3 = snap_to_bar(max(cue_1, cue_4 - (8 * sec_per_bar)))
        raw_cues = [cue_1, cue_2, cue_3, cue_4]

        if num_cues >= 8 and total_sec > 120.0:
            break_search = (frame_times >= cue_4 + (16 * sec_per_bar)) & (frame_times <= total_sec * 0.70)
            if np.any(break_search):
                raw_break = float(frame_times[break_search][np.argmin(sub_smoothed[break_search])])
                cue_5 = snap_to_bar(raw_break)
            else:
                cue_5 = snap_to_bar(total_sec * 0.50)

            drop2_search = (frame_times >= cue_5 + (8 * sec_per_bar)) & (frame_times <= total_sec * 0.85)
            if np.any(drop2_search):
                raw_drop2 = float(frame_times[drop2_search][np.argmax(sub_diff[drop2_search])])
                cue_7 = snap_to_bar(raw_drop2)
            else:
                cue_7 = snap_to_bar(total_sec * 0.75)

            cue_6 = snap_to_bar(max(cue_5, cue_7 - (8 * sec_per_bar)))
            cue_8 = snap_to_bar(max(cue_7 + (16 * sec_per_bar), total_sec - (32 * sec_per_bar)))
            raw_cues.extend([cue_5, cue_6, cue_7, cue_8])

        sorted_cues = sorted(list(set(raw_cues)))
        buffered_cues = [sorted_cues[0]]
        for c in sorted_cues[1:]:
            if (c - buffered_cues[-1]) >= 4.0:
                buffered_cues.append(c)

        return [int(c * 1000) for c in buffered_cues][:num_cues]

    @staticmethod
    def analyze_audio(
        filepath: Path,
        root_dir: Path,
        run_cues: bool = True,
        num_cues: int = 8,
        flagged: bool = False,
        skip_long_tracks: bool = True
    ) -> TrackAnalysisResult:
        res = TrackAnalysisResult(filepath=filepath, flagged=flagged)
        raw_genre = ''

        tags = UnifiedID3Handler.load_tags(filepath)
        if tags is not None:
            res.raw_title = str(tags.get('TIT2', filepath.stem))
            res.raw_artist = str(tags.get('TPE1', 'Unknown Artist'))
            raw_genre = str(tags.get('TCON', ''))
        else:
            res.raw_title = filepath.stem
            res.raw_artist = 'Unknown Artist'

        res.cleaned_title = sanitize_name_string(res.raw_title)
        res.cleaned_artist = sanitize_name_string(res.raw_artist)

        try:
            rel = filepath.relative_to(root_dir)
            if len(rel.parts) > 1:
                res.folder_category = rel.parts[0].title()
        except Exception:
            pass

        try:
            probe_info = sf.info(str(filepath))
            if probe_info.duration >= LONG_TRACK_THRESHOLD_SEC:
                res.is_long_track = True
        except Exception:
            try:
                m_f = File(str(filepath))
                if m_f and m_f.info and getattr(m_f.info, 'length', 0) >= LONG_TRACK_THRESHOLD_SEC:
                    res.is_long_track = True
            except Exception:
                pass

        if res.is_long_track and skip_long_tracks:
            res.energy_summary = "Long Mix / Continuous Set | Bypass Active"
            res.key = "MIX"
            res.detected_bpm = 120.0
            res.mood = "Ambient"
            res.energy_level = 5
            res.energy_delta = 0.0
            res.detected_genre = raw_genre if raw_genre else (res.folder_category or "Electronic")
            return res

        # Fast, bulletproof mono loader (sf.read -> audioread -> librosa fallback)
        y, rate = DSPAnalyzer.load_audio_fast_and_safe(filepath)

        total_sec = len(y) / float(rate) if rate > 0 else 0.0
        if total_sec >= LONG_TRACK_THRESHOLD_SEC and skip_long_tracks:
            res.is_long_track = True
            res.energy_summary = "Long Mix / Continuous Set | Bypass Active"
            res.key = "MIX"
            res.detected_bpm = 120.0
            res.mood = "Ambient"
            res.energy_level = 5
            res.detected_genre = raw_genre if raw_genre else (res.folder_category or "Electronic")
            return res

        # Instant 2x decimation instead of slow sinc resampling
        y_22k, sr_dsp = DSPAnalyzer.fast_downsample_22k(y, rate)

        hop_length = 512
        stft_full = np.abs(librosa.stft(y_22k, hop_length=hop_length))
        freqs_22k = librosa.fft_frequencies(sr=sr_dsp)
        rms_full = librosa.feature.rms(S=stft_full, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr_dsp, hop_length=hop_length)
        oenv_full = librosa.onset.onset_strength(S=librosa.amplitude_to_db(stft_full, ref=np.max), sr=sr_dsp, hop_length=hop_length)

        frames_45s = min(stft_full.shape[1], int((45.0 * sr_dsp) / hop_length))
        start_f = min(int((20.0 * sr_dsp) / hop_length), max(0, stft_full.shape[1] - frames_45s))
        stft_sample = stft_full[:, start_f:start_f + frames_45s]
        oenv_sample = oenv_full[start_f:start_f + frames_45s]

        # Precision IBI Beat & Tempo tracking
        beat_times = np.array([0.0])
        try:
            _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv_full, sr=sr_dsp, hop_length=hop_length)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr_dsp, hop_length=hop_length)
            precise_bpm, bpm_flagged = DSPAnalyzer.compute_precise_bpm(
                oenv=oenv_full,
                beat_times=beat_times,
                sr=sr_dsp,
                hop_length=hop_length,
                stft_sample=stft_sample,
                freqs=freqs_22k,
                filename_stem=filepath.stem,
                context_hint=f"{res.folder_category} {raw_genre}"
            )
            res.detected_bpm = precise_bpm
            if bpm_flagged:
                res.flagged = True
        except Exception:
            res.detected_bpm = 124.0

        # Fast Genre Classification (no HPSS bottleneck)
        res.detected_genre = DSPAnalyzer.classify_genre_fast(
            stft_sample,
            freqs_22k,
            oenv_sample,
            res.detected_bpm,
            res.folder_category,
            raw_genre
        )

        # Phrasing & Fast Active-Core LUFS (90s core window at decimated rate)
        res.intro_bars, res.outro_bars = DSPAnalyzer.detect_safe_phrasing(rms_full, frame_times, total_sec, res.detected_bpm)

        # Fast, Thread-Safe Calibrated ITU-R BS.1770 LUFS from Time-Domain Audio (+0.35 dB reference lock)
        if len(y_22k) >= 2048:
            block_size = max(512, int(sr_dsp * 0.4))
            n_blocks = len(y_22k) // block_size
            blocks = y_22k[:n_blocks * block_size].reshape(n_blocks, block_size)
            block_powers = np.mean(blocks ** 2, axis=1)

            active_powers = block_powers[block_powers > 1e-7]
            if len(active_powers) > 0:
                rel_thresh = np.mean(active_powers) * 0.1
                gated_powers = active_powers[active_powers >= rel_thresh]
                mean_power = float(np.mean(gated_powers)) if len(gated_powers) > 0 else float(np.mean(active_powers))

                hf_mask = freqs_22k >= 1500.0
                hf_ratio = float(np.mean(stft_sample[hf_mask, :]) / (np.mean(stft_sample) + 1e-6)) if np.any(hf_mask) else 0.5
                k_shelf_db = float(np.clip(2.2 + (hf_ratio - 0.35) * 4.0, 0.5, 4.2))

                res.integrated_lufs = round(float(np.clip(-0.691 + 10.0 * np.log10(max(mean_power, 1e-10)) + 0.35 + k_shelf_db, -30.0, -1.0)), 1)
                res.gain_offset_db = float(TARGET_LUFS - res.integrated_lufs)
            else:
                res.integrated_lufs = -20.0
                res.gain_offset_db = float(TARGET_LUFS - (-20.0))
        else:
            res.integrated_lufs = -14.0
            res.gain_offset_db = 0.0

        # True Peak & Crest Factor
        res.true_peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        mean_rms = float(np.mean(rms_full)) if len(rms_full) > 0 else 1e-5
        res.crest_factor_db = float(20.0 * np.log10(max(res.true_peak, 1e-5) / max(mean_rms, 1e-5)))
        res.is_clipped = (res.true_peak >= 0.999 and res.crest_factor_db < 5.5)

        # High-Band Spectral Roll-Off Cutoff Check
        res.spectral_cutoff_hz, res.is_fake_320k = DSPAnalyzer.compute_spectral_cutoff_hz(y, rate)

        # Calibrated 1-10 DJ Energy Score
        res.energy_level = DSPAnalyzer.compute_dj_energy_level(res.integrated_lufs, rms_full, oenv_full, res.detected_bpm)
        chunk_size = len(rms_full) // 3
        if chunk_size > 0:
            res.energy_delta = float(np.mean(rms_full[-chunk_size:]) - np.mean(rms_full[:chunk_size]))

        # Fast STFT-reused Chroma, Tuning & Spectral Centroid (reuses stft_sample instead of 6x CQT)
        if stft_sample.shape[1] >= 16:
            try:
                S_pow = stft_sample ** 2
                t_est = float(librosa.estimate_tuning(S=stft_sample, sr=sr_dsp))
                res.tuning_cents = float(t_est * 100.0)
                if abs(res.tuning_cents) >= 38.0:
                    res.flagged = True

                chroma = librosa.feature.chroma_stft(S=S_pow, sr=sr_dsp, tuning=t_est)
                chroma_avg = np.mean(chroma, axis=1)
                major_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                minor_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                best_score = -1.0
                best_key = (0, False)
                for pitch in range(12):
                    s_maj = np.corrcoef(chroma_avg, np.roll(major_p, pitch))[0, 1]
                    s_min = np.corrcoef(chroma_avg, np.roll(minor_p, pitch))[0, 1]
                    if s_maj > best_score:
                        best_score, best_key = s_maj, (pitch, False)
                    if s_min > best_score:
                        best_score, best_key = s_min, (pitch, True)
                res.key = CAMELOT_MAP.get(best_key, 'Unknown')

                centroid_mean = float(np.mean(librosa.feature.spectral_centroid(S=stft_sample, sr=sr_dsp)))
                res.mood = 'Crisp' if centroid_mean > 2200.0 else 'Warm'
            except Exception:
                res.key = 'Unknown'

            res.energy_summary = (
                f"{res.mood} | Energy:{res.energy_level}/10 (Δ:{res.energy_delta:+.2f}) "
                f"| Key:{res.key} ({res.tuning_cents:+.0f}c) | BPM:{res.detected_bpm:.1f}"
            )
        else:
            res.energy_summary = f"Cleaned | Energy:{res.energy_level}/10 | Key:{res.key} | BPM:{res.detected_bpm:.1f}"

        if run_cues and not res.is_long_track:
            sub_energy = np.mean(stft_full[freqs_22k <= 150, :], axis=0)
            res.cues_ms = DSPAnalyzer.detect_quantized_cues(
                sub_energy=sub_energy,
                frame_times=frame_times,
                beat_times=beat_times,
                total_sec=total_sec,
                bpm=res.detected_bpm,
                sr=sr_dsp,
                hop_length=hop_length,
                num_cues=num_cues
            )

        return res


def _parallel_worker_task(
    filepath: Path,
    root_dir: Path,
    run_cues: bool,
    num_cues: int,
    flagged: bool,
    skip_long_tracks: bool,
    config: Dict[str, bool]
) -> Tuple[TrackAnalysisResult, bool, Optional[str]]:
    """Top-level multiprocessing worker function for ThreadPoolExecutor."""
    try:
        res = DSPAnalyzer.analyze_audio(
            filepath=filepath,
            root_dir=root_dir,
            run_cues=run_cues,
            num_cues=num_cues,
            flagged=flagged,
            skip_long_tracks=skip_long_tracks
        )
        ok = SeratoBinaryEngine.inject_to_file(filepath, res, config)
        return res, ok, None
    except Exception as e:
        return TrackAnalysisResult(filepath=filepath), False, str(e)

# ==============================================================================
# DYNAMIC DANCE FLOOR ENGINE
# ==============================================================================
class DanceFloorEngine:
    DANCER_POSES = [ 
        {"head": r"    `\o/,    ", "body": r"     |⁛|     ", "legs":  r"    _/ \_    "},
        {"head": r"    `\o/-    ", "body": r"     |⁙|     ", "legs":  r"    _/ \_    "},     
        {"head": r"    ,\o/'    ", "body": r"     |⁘|     ", "legs":  r"    _/ \_    "},
        {"head": r"    ,\o/-    ", "body": r"     |⁜|     ", "legs":  r"    _/ \_    "},
    ]

    DJ_FRAMES = [
        {"notes": r" ♪  * ♫ ", "head": r"   (Ϙ)   ", "body": r"   ^_/|◌|\_^   ", "decks": r"  [⨂][⁙][⨀] "},
        {"notes": r" ♫  * ♪ ", "head": r"   (Ϙ)   ", "body": r"   _/|◍|\_    ", "decks": r"  [⨁][⁐][⨀] "},
        {"notes": r" ♬ *  ♩ ", "head": r"   (Ϙ)   ", "body": r"   <_/|◌|\_    ", "decks": r"  [⨂][⁙][⨀] "},
        {"notes": r" ♩  * ♬ ", "head": r"   (Ϙ)   ", "body": r"    _/|◍|\_>   ", "decks": r"  [⨁][⁐][⨀] "},
    ]
    COP_POSES = [
        {"head": r"  ¬[■]o  ", "body": r"   -|█|/   ", "legs": r"   / \   "},
        {"head": r"  ¬[■]o  ", "body": r"   |█|=#   ", "legs": r"   < \   "},
    ]
 
    @classmethod
    def render_frame(cls, frame_idx: int, elapsed_sec: float) -> str:
        dancer_count = min(2 + int(elapsed_sec // 18.0), 20)
        left_count = dancer_count // 2
        right_count = dancer_count - left_count

        dj = cls.DJ_FRAMES[frame_idx % len(cls.DJ_FRAMES)]
        note_chars = ["♪", "♫", "♬", "♩", "·"]
        extra_notes_left = "".join(f"  {note_chars[(frame_idx + i) % len(note_chars)]}  " for i in range(left_count))
        extra_notes_right = "".join(f"  {note_chars[(frame_idx + i + 2) % len(note_chars)]}  " for i in range(right_count))

        l_heads_list, l_bodies_list, l_legs_list = [], [], []
        for i in range(left_count):
            p = cls.DANCER_POSES[(frame_idx + i) % len(cls.DANCER_POSES)]
            l_heads_list.append(p["head"])
            l_bodies_list.append(p["body"])
            l_legs_list.append(p["legs"])

        r_heads_list, r_bodies_list, r_legs_list = [], [], []
        for i in range(right_count):
            p = cls.DANCER_POSES[(frame_idx + i + 1) % len(cls.DANCER_POSES)]
            r_heads_list.append(p["head"])
            r_bodies_list.append(p["body"])
            r_legs_list.append(p["legs"])

        l_heads = "  ".join(l_heads_list)
        l_bodies = "  ".join(l_bodies_list)
        l_legs = "  ".join(l_legs_list)

        r_heads = "  ".join(r_heads_list)
        r_bodies = "  ".join(r_bodies_list)
        r_legs = "  ".join(r_legs_list)

        side_width = 35
        row1 = f"{extra_notes_left:>{side_width}}{dj['notes']}{extra_notes_right:<{side_width}}"
        row2 = f"{l_heads:>{side_width}}       {dj['head']}       {r_heads:<{side_width}}"
        row3 = f"{l_bodies:>{side_width}}     {dj['body']}    {r_bodies:<{side_width}}"
        row4 = f"{l_legs:>{side_width}}     {dj['decks']}     {r_legs:<{side_width}}"
        return f"{row1}\n{row2}\n{row3}\n{row4}"

    @classmethod
    def render_bust_frame(cls, step: int) -> Tuple[str, str]:
        siren = "🚨 [POLICE RAID,  !!EVERYONE STOP DANCING!! ,POLICE RAID] 🚨" if step % 2 == 0 else "🛑 [PARTY'S OVER] 🛑"
        color = "#ff1744" if step % 2 == 0 else "#2979ff"
        cop = cls.COP_POSES[step % len(cls.COP_POSES)]

        row1 = f"         {siren}         "
        row2 = rf" \o/    \o/        (o)    {cop['head']}"
        row3 = rf" |█|    |█|      _/|_|\_   {cop['body']}"
        row4 = rf"_/ \_  _/ \_     [■]-[■]  {cop['legs']}"

        return f"{row1}\n{row2}\n{row3}\n{row4}", color

# ==============================================================================
# ACTIVE PROCESSOR LOAD & HARDWARE TELEMETRY ENGINE (INTEL / AMD AUTO-SCALER)
# ==============================================================================
class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

class CpuTelemetryEngine:
    def __init__(self):
        self.logical_cores = max(1, os.cpu_count() or 4)
        self.cpu_model = self._detect_cpu_model()
        self.is_amd = any(k in self.cpu_model.upper() for k in ("AMD", "RYZEN", "THREADRIPPER", "EPYC"))
        self.is_intel = "INTEL" in self.cpu_model.upper()
        self.recommended_workers = self._compute_recommended_workers()

        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
            self._psutil.cpu_percent(percpu=True)
        except Exception:
            self._psutil = None

        self._prev_idle = 0
        self._prev_kernel = 0
        self._prev_user = 0
        self._last_pkg_load = 0.0
        self._per_core_loads = [0.0] * self.logical_cores
        self._poll_windows_kernel_times()

    @staticmethod
    def _detect_cpu_model() -> str:
        if os.name == "nt":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)
                if name:
                    return re.sub(r'\s+', ' ', str(name)).strip()
            except Exception:
                pass
        return platform.processor() or f"x86_64 Processor ({os.cpu_count() or 4} Threads)"

    def _compute_recommended_workers(self) -> int:
        c = self.logical_cores
        if c <= 2:
            return 1
        if c <= 4:
            return c - 1  # e.g., 4-core i5 -> 3 workers (leaves 1 core for UI)
        if self.is_amd:
            # AMD Ryzen all-big-core architecture scales cleanly up to 75% (capped at 14)
            return min(14, max(4, int(round(c * 0.75))))
        # Intel i7 / i9 hybrid P/E-core architecture scales best around 65-75% (capped at 12)
        return min(12, max(4, int(round(c * 0.70))))

    def _poll_windows_kernel_times(self) -> Optional[float]:
        if os.name != "nt":
            return None
        try:
            idle_ft = _FILETIME()
            kernel_ft = _FILETIME()
            user_ft = _FILETIME()
            res = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle_ft),
                ctypes.byref(kernel_ft),
                ctypes.byref(user_ft)
            )
            if not res:
                return None
            idle = (idle_ft.dwHighDateTime << 32) | idle_ft.dwLowDateTime
            kernel = (kernel_ft.dwHighDateTime << 32) | kernel_ft.dwLowDateTime
            user = (user_ft.dwHighDateTime << 32) | user_ft.dwLowDateTime

            d_idle = idle - self._prev_idle
            d_kernel = kernel - self._prev_kernel
            d_user = user - self._prev_user
            self._prev_idle, self._prev_kernel, self._prev_user = idle, kernel, user

            total_sys = d_kernel + d_user
            if total_sys <= 0:
                return self._last_pkg_load
            busy = max(0, total_sys - d_idle)
            pct = float(np.clip((busy / float(total_sys)) * 100.0, 0.0, 100.0))
            self._last_pkg_load = pct
            return pct
        except Exception:
            return None

    def sample_load(self, active_workers: int = 0, max_workers: int = 1) -> Tuple[float, List[float]]:
        """Returns (package_cpu_percent, per_core_percent_list)."""
        if self._psutil is not None:
            try:
                per_core = self._psutil.cpu_percent(percpu=True)
                if per_core:
                    pkg = float(np.mean(per_core))
                    self._last_pkg_load = pkg
                    self._per_core_loads = [float(x) for x in per_core]
                    return pkg, self._per_core_loads
            except Exception:
                pass

        pkg_win = self._poll_windows_kernel_times()
        if pkg_win is not None:
            pkg = pkg_win
        else:
            pkg = min(98.0, (active_workers / float(max(1, self.logical_cores))) * 92.0) if active_workers > 0 else 4.0

        # Synthesize per-core distribution around real kernel package load when psutil isn't installed
        per_core_est = []
        for idx in range(self.logical_cores):
            if active_workers > 0 and idx < active_workers:
                jitter = ((idx * 7 + int(time.time() * 10)) % 11) - 5
                c_val = float(np.clip(pkg * 1.08 + jitter, 12.0, 100.0))
            else:
                jitter = ((idx * 3 + int(time.time() * 8)) % 7) - 3
                c_val = float(np.clip(pkg * 0.65 + jitter, 2.0, 95.0))
            per_core_est.append(c_val)
        self._per_core_loads = per_core_est
        return pkg, per_core_est


class ProcessorLoadVisualizer(tk.Frame):
    """Live Hardware Telemetry & Multi-Core LED Load Visualizer for Intel i5/i7/i9 & AMD Ryzen."""
    def __init__(self, master, telemetry: CpuTelemetryEngine, on_worker_override_change=None):
        super().__init__(master, bg="#101218", bd=1, relief=tk.SOLID)
        self.telemetry = telemetry
        self.on_worker_override_change = on_worker_override_change
        self.active_workers = 0
        self.max_workers = telemetry.recommended_workers

        top_row = tk.Frame(self, bg="#101218")
        top_row.pack(fill=tk.X, padx=6, pady=(3, 1))

        vendor_color = "#ff6d00" if telemetry.is_amd else "#00d2ff"
        vendor_tag = "AMD" if telemetry.is_amd else ("INTEL" if telemetry.is_intel else "CPU")
        short_model = telemetry.cpu_model
        if len(short_model) > 36:
            short_model = short_model[:34] + ".."

        self.cpu_lbl = tk.Label(
            top_row,
            text=f"⚡ [{vendor_tag}] {short_model} ({telemetry.logical_cores}T)",
            font=("Consolas", 7, "bold"),
            fg=vendor_color,
            bg="#101218"
        )
        self.cpu_lbl.pack(side=tk.LEFT)

        # Worker Thread Override Selector (for i5 / i7 / i9 / Ryzen benchmarking)
        ctrl_frame = tk.Frame(top_row, bg="#101218")
        ctrl_frame.pack(side=tk.RIGHT)
        tk.Label(ctrl_frame, text="Workers:", font=("Segoe UI", 7, "bold"), fg="#9aa0b4", bg="#101218").pack(side=tk.LEFT, padx=(0, 3))

        auto_label = f"Auto ({telemetry.recommended_workers}T)"
        worker_opts = [auto_label, "2", "3", "4", "6", "8", "10", "12", "16"]
        self.worker_var = tk.StringVar(value=auto_label)
        self.worker_combo = ttk.Combobox(
            ctrl_frame, textvariable=self.worker_var, values=worker_opts,
            width=10, state="readonly", font=("Segoe UI", 7)
        )
        self.worker_combo.pack(side=tk.RIGHT)
        self.worker_combo.bind("<<ComboboxSelected>>", self._on_combo_change)

        self.canvas = tk.Canvas(self, bg="#08090c", height=48, highlightthickness=0)
        self.canvas.pack(fill=tk.X, padx=6, pady=(2, 4))
        self.canvas.bind("<Configure>", lambda e: self.redraw(0.0, [0.0] * self.telemetry.logical_cores))

    def get_selected_worker_count(self) -> int:
        val = self.worker_var.get()
        if val.startswith("Auto"):
            return self.telemetry.recommended_workers
        try:
            return max(1, int(val))
        except ValueError:
            return self.telemetry.recommended_workers

    def _on_combo_change(self, event=None):
        self.max_workers = self.get_selected_worker_count()
        if self.on_worker_override_change:
            self.on_worker_override_change(self.max_workers)

    def update_telemetry(self, active_workers: int, max_workers: int):
        self.active_workers = active_workers
        self.max_workers = max_workers
        pkg, per_core = self.telemetry.sample_load(active_workers=active_workers, max_workers=max_workers)
        self.redraw(pkg, per_core)

    def redraw(self, pkg_pct: float, per_core: List[float]):
        c = self.canvas
        c.delete("all")
        w = max(220, c.winfo_width())
        h = 48

        # Row 1: Overall Package Load LED Bar + Active Worker Readout
        c.create_text(6, 10, text=f"PKG LOAD: {pkg_pct:4.0f}%", anchor="w", font=("Consolas", 7, "bold"), fill="#e0e0e0")
        worker_status = f"WORKERS: {self.active_workers}/{self.max_workers} ACTIVE"
        w_color = "#00e676" if self.active_workers > 0 else "#7e869c"
        c.create_text(w - 6, 10, text=worker_status, anchor="e", font=("Consolas", 7, "bold"), fill=w_color)

        bar_x0 = 92
        bar_x1 = max(bar_x0 + 40, w - 125)
        bar_w = bar_x1 - bar_x0
        segments = 20
        seg_w = max(2.0, (bar_w - (segments - 1) * 2) / float(segments))
        lit_segs = int(round((pkg_pct / 100.0) * segments))

        for s_idx in range(segments):
            sx0 = bar_x0 + s_idx * (seg_w + 2)
            sx1 = sx0 + seg_w
            if s_idx < lit_segs:
                if s_idx < int(segments * 0.6):
                    col = "#00e676"
                elif s_idx < int(segments * 0.85):
                    col = "#ffd600"
                else:
                    col = "#ff1744"
            else:
                col = "#1b1e26"
            c.create_rectangle(sx0, 5, sx1, 15, fill=col, outline="")

        # Row 2: Per-Core / Multi-Thread Adaptive LED Grid (supports 4-core i5 up to 32-thread i9/Ryzen 9)
        n_cores = len(per_core)
        grid_y0 = 23
        grid_h = 20
        pad_x = 6
        avail_w = w - (pad_x * 2)
        gap = 3 if n_cores <= 16 else 2
        cell_w = max(6.0, (avail_w - (n_cores - 1) * gap) / float(max(1, n_cores)))

        for idx, core_pct in enumerate(per_core):
            cx0 = pad_x + idx * (cell_w + gap)
            cx1 = cx0 + cell_w
            c.create_rectangle(cx0, grid_y0, cx1, grid_y0 + grid_h, fill="#14171f", outline="#262b38")

            fill_h = int(round((core_pct / 100.0) * (grid_h - 2)))
            if fill_h > 0:
                if core_pct < 60.0:
                    f_col = "#00d2ff"
                elif core_pct < 85.0:
                    f_col = "#00e676"
                else:
                    f_col = "#ff9100"
                c.create_rectangle(cx0 + 1, grid_y0 + grid_h - 1 - fill_h, cx1 - 1, grid_y0 + grid_h - 1, fill=f_col, outline="")

            if n_cores <= 12 and cell_w >= 18:
                c.create_text((cx0 + cx1) * 0.5, grid_y0 + 10, text=f"C{idx}", font=("Consolas", 6, "bold"), fill="#ffffff")

# ==============================================================================
# MODERN GRAPHICAL USER INTERFACE (V 12.0)
# ==============================================================================
class SeratoV10App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Serato Analyzer Pro — v12.1 Beta Edition")
        self.geometry("1520x890")
        self.minsize(1100, 720)
        self.configure(bg="#121316")

        self.root_folder: Optional[Path] = None
        self.audio_files: List[Path] = []
        self.diagnostics: List[TrackDiagnostic] = []
        self.diag_by_iid: Dict[str, TrackDiagnostic] = {}
        self.is_running = False
        self.is_scanning = False
        self.is_paused = False
        self.stop_requested = False
        self.anim_idx = 0
        self.batch_start_time = 0.0

        self.cfg_clean_tags = tk.BooleanVar(value=True)
        self.cfg_rename_disk = tk.BooleanVar(value=True)
        self.cfg_inject_cues = tk.BooleanVar(value=True)
        self.cfg_protect_cues = tk.BooleanVar(value=False)
        self.cfg_inject_autogain = tk.BooleanVar(value=True)
        self.cfg_run_dsp = tk.BooleanVar(value=True)
        self.cfg_flag_analysis = tk.BooleanVar(value=False)
        self.cfg_skip_long_tracks = tk.BooleanVar(value=True)
        self.cfg_skip_cached = tk.BooleanVar(value=True)
        self.cfg_skip_clones = tk.BooleanVar(value=True)
        self.cfg_lock_files = tk.BooleanVar(value=True)
        self.cfg_export_crates = tk.BooleanVar(value=True)

        self.cfg_tag_title_artist = tk.BooleanVar(value=True)
        self.cfg_tag_genre = tk.BooleanVar(value=True)
        self.cfg_tag_key = tk.BooleanVar(value=True)
        self.cfg_tag_bpm = tk.BooleanVar(value=True)
        self.cfg_tag_mood = tk.BooleanVar(value=True)
        self.cfg_tag_energy = tk.BooleanVar(value=True)
        self.cfg_tag_geob_metrics = tk.BooleanVar(value=True)
        self.cfg_tag_comment = tk.BooleanVar(value=True)

        self.col_visibility = {
            "Flag": tk.BooleanVar(value=True),
            "Lock": tk.BooleanVar(value=True),
            "Status / Notes": tk.BooleanVar(value=True),
            "Filename": tk.BooleanVar(value=True),
            "Proposed Rename": tk.BooleanVar(value=True),
            "Folder Category": tk.BooleanVar(value=True)
        }

        self.cpu_telemetry = CpuTelemetryEngine()
        self.active_worker_threads: int = 0
        self.current_max_workers: int = self.cpu_telemetry.recommended_workers
        self.last_batch_results: Dict[str, TrackAnalysisResult] = {}
        self.last_batch_elapsed_sec: float = 0.0
        self.rename_ledger: List[Tuple[str, str]] = []
        self.quarantine_ledger: List[Tuple[str, str]] = []
        self.beta_targets_tree: Optional[ttk.Treeview] = None
        self.beta_targets_win: Optional[tk.Toplevel] = None
        self.beta_summary_lbl: Optional[tk.Label] = None

        self._build_ui()
        # Automatically launch Operational Guide & Beta Testing Targets on startup for Beta Testers
        self.after(250, self._poll_ambient_cpu_telemetry)
        self.after(350, self.open_operational_instructions_popup)

    def _build_ui(self):
        header = tk.Frame(self, bg="#1a1c23", height=42)
        header.pack(fill=tk.X, side=tk.TOP)
        sub_lbl = tk.Label(
            header,
            text="SERATO AUDIO ANALYZER PRO — v12.1 STANDALONE EDITION (CALIBRATED TURBO DSP & QA SUITE)",
            font=("Segoe UI", 11, "bold"),
            fg="#00d2ff",
            bg="#1a1c23"
        )
        sub_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        tk.Button(
            header, text="🎯 BETA TEST TARGETS", font=("Segoe UI", 8, "bold"),
            bg="#00e676", fg="#121316", activebackground="#00c853", activeforeground="#121316",
            relief=tk.FLAT, padx=10, pady=3, command=self.open_beta_targets_popup
        ).pack(side=tk.RIGHT, padx=(4, 15), pady=6)

        tk.Button(
            header, text="📖 OPERATIONAL INSTRUCTIONS", font=("Segoe UI", 8, "bold"),
            bg="#ffd600", fg="#121316", activebackground="#ffab00", activeforeground="#121316",
            relief=tk.FLAT, padx=10, pady=3, command=self.open_operational_instructions_popup
        ).pack(side=tk.RIGHT, padx=4, pady=6)

        top_bar = tk.Frame(self, bg="#121316")
        top_bar.pack(fill=tk.X, padx=15, pady=6)
        self.folder_var = tk.StringVar(value="Drag & Drop Folder here or click SELECT FOLDER...")
        folder_entry = tk.Entry(top_bar, textvariable=self.folder_var, font=("Segoe UI", 9), bg="#21242d", fg="#e0e0e0", relief=tk.FLAT)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 10))

        self.browse_btn = tk.Button(
            top_bar, text="SELECT FOLDER", font=("Segoe UI", 9, "bold"),
            bg="#0066ff", fg="#ffffff", activebackground="#0052cc", activeforeground="#ffffff",
            relief=tk.FLAT, padx=12, command=self.select_folder
        )
        self.browse_btn.pack(side=tk.RIGHT, padx=(0, 5))

        self.clear_btn = tk.Button(
            top_bar, text="🔄 CLEAR VIEWER", font=("Segoe UI", 9, "bold"),
            bg="#2d313d", fg="#00d2ff", activebackground="#3a3f4d", activeforeground="#ffffff",
            relief=tk.FLAT, padx=8, command=self.clear_track_viewer
        )
        self.clear_btn.pack(side=tk.RIGHT)

        self.master_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.master_paned.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        left_pane = tk.Frame(self.master_paned, bg="#121316")
        self.master_paned.add(left_pane, weight=5)

        list_container = tk.LabelFrame(left_pane, text=" CRATE TRACK VIEWER & QUEUE ", font=("Segoe UI", 9, "bold"), fg="#9aa0b4", bg="#181a20")
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 2))

        col_bar = tk.Frame(list_container, bg="#181a20")
        col_bar.pack(fill=tk.X, padx=8, pady=(4, 4))

        tk.Label(col_bar, text="Columns:", font=("Segoe UI", 7, "bold"), fg="#9aa0b4", bg="#181a20").pack(side=tk.LEFT, padx=(0, 4))
        for col_name, var in self.col_visibility.items():
            cb = tk.Checkbutton(
                col_bar, text=col_name, variable=var, font=("Segoe UI", 7),
                bg="#181a20", fg="#e0e0e0", selectcolor="#21242d",
                activebackground="#181a20", activeforeground="#00d2ff",
                command=self._refresh_tree_columns
            )
            cb.pack(side=tk.LEFT, padx=3)

        tk.Label(col_bar, text="★=Flag | 🔒=Lock | ⚠️=Dup/Transcode", font=("Segoe UI", 7), fg="#7e869c", bg="#181a20").pack(side=tk.RIGHT, padx=4)

        tree_frame = tk.Frame(list_container, bg="#181a20")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        all_cols = ("Flag", "Lock", "Status / Notes", "Filename", "Proposed Rename", "Folder Category")
        self.tree = ttk.Treeview(tree_frame, columns=all_cols, show="headings", selectmode="extended")

        self.tree.heading("Flag", text="★", command=lambda: self.sort_column("Flag", False))
        self.tree.heading("Lock", text="🔒", command=lambda: self.sort_column("Lock", False))
        self.tree.heading("Status / Notes", text="Status ↕", command=lambda: self.sort_column("Status / Notes", False))
        self.tree.heading("Filename", text="Original Filename ↕", command=lambda: self.sort_column("Filename", False))
        self.tree.heading("Proposed Rename", text="Proposed Smart Rename (Double-Click to Edit) ↕", command=lambda: self.sort_column("Proposed Rename", False))
        self.tree.heading("Folder Category", text="Folder Category ↕", command=lambda: self.sort_column("Folder Category", False))

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)
        self.tree.drop_target_register(DND_FILES)
        self.tree.dnd_bind('<<Drop>>', self.handle_drop)

        self._refresh_tree_columns()

        center_pane = tk.Frame(self.master_paned, bg="#121316")
        self.master_paned.add(center_pane, weight=2)

        opts_frame = tk.LabelFrame(center_pane, text=" WORKFLOW & ACTIONS ", font=("Segoe UI", 9, "bold"), fg="#9aa0b4", bg="#181a20")
        opts_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        preset_frame = tk.Frame(opts_frame, bg="#181a20")
        preset_frame.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Label(preset_frame, text="PRESETS:", font=("Segoe UI", 7, "bold"), fg="#9aa0b4", bg="#181a20").pack(side=tk.TOP, anchor="w", pady=(0, 1))

        p_btns_row = tk.Frame(preset_frame, bg="#181a20")
        p_btns_row.pack(fill=tk.X)
        tk.Button(p_btns_row, text="Full", bg="#252836", fg="#00e676", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, command=self.preset_full).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(p_btns_row, text="Safe", bg="#252836", fg="#00d2ff", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, command=self.preset_safe).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(p_btns_row, text="Cues", bg="#252836", fg="#ffd600", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, command=self.preset_cues).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(p_btns_row, text="Meta", bg="#252836", fg="#ff4081", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, command=self.preset_meta).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        actions_sub_bar = tk.Frame(opts_frame, bg="#181a20")
        actions_sub_bar.pack(fill=tk.X, padx=8, pady=(3, 1))

        tk.Button(actions_sub_bar, text="🚩 Flag", font=("Segoe UI", 7, "bold"), bg="#ff9100", fg="#121316", relief=tk.FLAT, command=self.toggle_selected_flag).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(actions_sub_bar, text="🔍 Clones", font=("Segoe UI", 7, "bold"), bg="#00d2ff", fg="#121316", relief=tk.FLAT, command=self.run_manual_clone_audit).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(actions_sub_bar, text="📦 Crates", font=("Segoe UI", 7, "bold"), bg="#b388ff", fg="#121316", relief=tk.FLAT, command=self.open_subcrate_builder).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(actions_sub_bar, text="🧹 Wipe", font=("Segoe UI", 7, "bold"), bg="#ff1744", fg="#ffffff", relief=tk.FLAT, command=self.wipe_selected_id3_tags).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        safety_sub_bar = tk.Frame(opts_frame, bg="#181a20")
        safety_sub_bar.pack(fill=tk.X, padx=8, pady=(1, 3))

        tk.Button(safety_sub_bar, text="🔓 Unlock Files", font=("Segoe UI", 7, "bold"), bg="#26a69a", fg="#121316", relief=tk.FLAT, command=self.unlock_selected_files).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(safety_sub_bar, text="↩️ Undo Renames", font=("Segoe UI", 7, "bold"), bg="#ffd600", fg="#121316", relief=tk.FLAT, command=self.undo_last_renames).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        toggles_main_frame = tk.Frame(opts_frame, bg="#181a20")
        toggles_main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        wf_sub = tk.Frame(toggles_main_frame, bg="#181a20")
        wf_sub.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))

        tk.Label(wf_sub, text="Modules", font=("Segoe UI", 7, "bold"), fg="#9aa0b4", bg="#181a20").pack(anchor="w", pady=(0, 1))

        self.workflow_tabs = {}
        self.workflow_configs = [
            ("Clean ID3", self.cfg_clean_tags, "#00d2ff"),
            ("Rename", self.cfg_rename_disk, "#00d2ff"),
            ("Named Cues", self.cfg_inject_cues, "#ffd600"),
            ("Protect Cues", self.cfg_protect_cues, "#ffd600"),
            ("LUFS (-14)", self.cfg_inject_autogain, "#00e676"),
            ("Audio DSP", self.cfg_run_dsp, "#00e676"),
            ("Skip Clones", self.cfg_skip_clones, "#00d2ff"),
            ("Skip Long", self.cfg_skip_long_tracks, "#ff9100"),
            ("Skip Cached", self.cfg_skip_cached, "#ff9100"),
            ("Lock File", self.cfg_lock_files, "#ff4081"),
            ("Force-Flag All", self.cfg_flag_analysis, "#ff4081"),
            ("Auto-Crates", self.cfg_export_crates, "#b388ff")
        ]

        def make_tab_toggle(name, var, active_color):
            def toggle():
                val = not var.get()
                var.set(val)
                update_tab_styling(name, val, active_color)
            return toggle

        def update_tab_styling(name, is_active, active_color):
            btn = self.workflow_tabs[name]
            if is_active:
                btn.config(bg=active_color, fg="#121316", relief="sunken")
            else:
                btn.config(bg="#222530", fg="#8a8d9b", relief="flat")

        for name, var, color in self.workflow_configs:
            is_active = var.get()
            b_bg = color if is_active else "#222530"
            b_fg = "#121316" if is_active else "#8a8d9b"

            tab_btn = tk.Button(
                wf_sub,
                text=name,
                font=("Segoe UI", 7, "bold"),
                bg=b_bg,
                fg=b_fg,
                relief="flat",
                anchor="w",
                padx=6, pady=2,
                command=make_tab_toggle(name, var, color)
            )
            tab_btn.pack(fill=tk.X, pady=1)
            self.workflow_tabs[name] = tab_btn

        id3_sub = tk.Frame(toggles_main_frame, bg="#181a20")
        id3_sub.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))

        tk.Label(id3_sub, text="ID3 Targets", font=("Segoe UI", 7, "bold"), fg="#00d2ff", bg="#181a20").pack(anchor="w", pady=(0, 1))

        tag_checks = [
            ("Title/Artist", self.cfg_tag_title_artist),
            ("Genre", self.cfg_tag_genre),
            ("Key (TKEY)", self.cfg_tag_key),
            ("BPM (TBPM)", self.cfg_tag_bpm),
            ("Mood", self.cfg_tag_mood),
            ("Energy", self.cfg_tag_energy),
            ("GEOB Metrics", self.cfg_tag_geob_metrics),
            ("Comment", self.cfg_tag_comment)
        ]

        for t_name, t_var in tag_checks:
            cb = tk.Checkbutton(
                id3_sub, text=t_name, variable=t_var, font=("Segoe UI", 7),
                bg="#181a20", fg="#e0e0e0", selectcolor="#21242d",
                activebackground="#181a20", activeforeground="#00d2ff"
            )
            cb.pack(anchor="w", pady=1)

        right_pane = tk.Frame(self.master_paned, bg="#121316")
        self.master_paned.add(right_pane, weight=3)

        right_container = tk.LabelFrame(right_pane, text=" EXECUTION & CONSOLE ", font=("Segoe UI", 9, "bold"), fg="#9aa0b4", bg="#181a20")
        right_container.pack(fill=tk.BOTH, expand=True, pady=(0, 2))

        action_bar = tk.Frame(right_container, bg="#181a20")
        action_bar.pack(fill=tk.X, padx=8, pady=(6, 4))

        self.run_btn = tk.Button(
            action_bar, text="EXECUTE BATCH (ALL)", font=("Segoe UI", 8, "bold"),
            bg="#0066ff", fg="#ffffff", activebackground="#0052cc", activeforeground="#ffffff",
            relief=tk.FLAT, pady=5, command=lambda: self.start_batch_thread(selected_only=False)
        )
        self.run_btn.pack(fill=tk.X, pady=2)

        self.run_selected_btn = tk.Button(
            action_bar, text="SCAN HIGHLIGHTED", font=("Segoe UI", 8, "bold"),
            bg="#00c853", fg="#ffffff", activebackground="#009624", activeforeground="#ffffff",
            relief=tk.FLAT, pady=5, command=lambda: self.start_batch_thread(selected_only=True)
        )
        self.run_selected_btn.pack(fill=tk.X, pady=2)

        sub_actions = tk.Frame(action_bar, bg="#181a20")
        sub_actions.pack(fill=tk.X, pady=2)

        self.pause_btn = tk.Button(
            sub_actions, text="⏸️ PAUSE", font=("Segoe UI", 8, "bold"),
            bg="#ff9100", fg="#121316", activebackground="#ffab40",
            relief=tk.FLAT, pady=4, state=tk.DISABLED, command=self.toggle_pause_batch
        )
        self.pause_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.stop_btn = tk.Button(
            sub_actions, text="⏹️ STOP", font=("Segoe UI", 8, "bold"),
            bg="#ff1744", fg="#ffffff", activebackground="#ff5252",
            relief=tk.FLAT, pady=4, state=tk.DISABLED, command=self.stop_batch
        )
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self.progress = ttk.Progressbar(right_container, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, padx=8, pady=4)

        self.cpu_visualizer = ProcessorLoadVisualizer(
            right_container,
            telemetry=self.cpu_telemetry,
            on_worker_override_change=self._on_worker_count_changed
        )
        self.cpu_visualizer.pack(fill=tk.X, padx=8, pady=(2, 4))

        stage_frame = tk.Frame(right_container, bg="#000000", bd=1, relief=tk.SOLID)
        stage_frame.pack(fill=tk.X, padx=8, pady=4)
        self.anim_label = tk.Label(stage_frame, text="", font=("Consolas", 7, "bold"), fg="#00e5ff", bg="#000000", justify=tk.CENTER, pady=3)
        self.anim_label.pack(fill=tk.X)

        console_frame = tk.Frame(right_container, bg="#0d0e11")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.console = tk.Text(console_frame, bg="#0a0a0d", fg="#a6accd", font=("Consolas", 8), relief=tk.FLAT, padx=6, pady=6)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, text: str):
        self.after(0, lambda: self._thread_safe_log(text))

    def _thread_safe_log(self, text: str):
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)

    def sort_column(self, col, reverse: bool):
        data_rows = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
        data_rows.sort(key=lambda x: x[0].lower(), reverse=reverse)
        for index, (val, child) in enumerate(data_rows):
            self.tree.move(child, '', index)
        self.tree.heading(col, command=lambda: self.sort_column(col, not reverse))

    def _refresh_tree_columns(self):
        active_cols = [col for col, var in self.col_visibility.items() if var.get()]
        self.tree.configure(columns=active_cols)

        for col in active_cols:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))
            if col in ("Flag", "Lock"):
                self.tree.column(col, width=30, anchor="center")
            elif col == "Status / Notes":
                self.tree.column(col, width=110, anchor="w")
            elif col == "Filename":
                self.tree.column(col, width=180, anchor="w")
            elif col == "Proposed Rename":
                self.tree.column(col, width=200, anchor="w")
            elif col == "Folder Category":
                self.tree.column(col, width=100, anchor="w")

        self._refresh_tree_display()

    def _get_selected_diagnostics(self) -> List[TrackDiagnostic]:
        selected_iids = self.tree.selection()
        return [self.diag_by_iid[iid] for iid in selected_iids if iid in self.diag_by_iid]

    def toggle_selected_flag(self):
        selected_diags = self._get_selected_diagnostics()
        if not selected_diags:
            messagebox.showinfo("Selection Required", "Please select one or more tracks from the list to flag/unflag.")
            return

        for d in selected_diags:
            d.flagged = not d.flagged
        self._refresh_tree_display()

    def wipe_selected_id3_tags(self):
        if self.is_running or self.is_scanning:
            messagebox.showinfo("Busy", "Wait for the active scan or batch operation to finish before wiping tags.")
            return

        selected_diags = self._get_selected_diagnostics()
        if not selected_diags:
            if not messagebox.askyesno("Wipe All Tags", "No tracks selected. Do you want to wipe ID3 tags across ALL loaded tracks?"):
                return
            target_diags = list(self.diagnostics)
        else:
            target_diags = list(selected_diags)

        if not target_diags:
            return

        self.is_running = True
        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")
        self.progress.configure(value=0, maximum=len(target_diags))
        self.log(f"[*] Starting background ID3 tag wipe across {len(target_diags)} track(s)...")

        def _worker():
            wiped_count = 0
            total = len(target_diags)
            for idx, d in enumerate(target_diags, start=1):
                try:
                    was_locked = FileLockManager.is_read_only(d.path)
                    if was_locked:
                        FileLockManager.set_read_only(d.path, read_only=False)
                        d.is_locked = False

                    if d.path.suffix.lower() in ID3_COMPATIBLE_EXTENSIONS:
                        tags = UnifiedID3Handler.load_tags(d.path)
                        if tags is not None:
                            tags.delete(str(d.path.resolve()))
                            d.is_cached = False
                            d.has_existing_cues = False
                            wiped_count += 1
                except Exception:
                    pass

                if idx % 15 == 0 or idx == total:
                    self.after(0, lambda c=idx: self.progress.configure(value=c))

            def _done():
                self.is_running = False
                self.progress.configure(value=0)
                self.browse_btn.config(state=tk.NORMAL, bg="#0066ff")
                self.run_btn.config(state=tk.NORMAL, bg="#0066ff")
                self.run_selected_btn.config(state=tk.NORMAL, bg="#00c853")
                self._refresh_tree_display()
                self.log(f"[*] Unlocked and wiped ID3 tags from {wiped_count} track(s).")
                messagebox.showinfo("Tags Wiped", f"Successfully cleared ID3 tags from {wiped_count} track(s).")

            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def unlock_selected_files(self):
        """Unlocks Read-Only status on selected (or all) tracks in a background thread without wiping ID3 tags."""
        if self.is_running or self.is_scanning:
            messagebox.showinfo("Busy", "Wait for the active scan or batch operation to finish before unlocking files.")
            return

        selected_diags = self._get_selected_diagnostics()
        if not selected_diags:
            if not messagebox.askyesno("Unlock All Tracks", "No tracks highlighted. Do you want to unlock (remove Read-Only) across ALL loaded tracks while keeping their ID3 tags intact?"):
                return
            target_diags = list(self.diagnostics)
        else:
            target_diags = list(selected_diags)

        if not target_diags:
            return

        self.is_running = True
        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")
        self.progress.configure(value=0, maximum=len(target_diags))
        self.log(f"[*] Starting background Read-Only unlock across {len(target_diags)} track(s)...")

        def _worker():
            unlocked_count = 0
            total = len(target_diags)
            for idx, d in enumerate(target_diags, start=1):
                try:
                    FileLockManager.set_read_only(d.path, read_only=False)
                    d.is_locked = False
                    unlocked_count += 1
                except Exception:
                    pass

                if idx % 25 == 0 or idx == total:
                    self.after(0, lambda c=idx: self.progress.configure(value=c))

            def _done():
                self.is_running = False
                self.progress.configure(value=0)
                self.browse_btn.config(state=tk.NORMAL, bg="#0066ff")
                self.run_btn.config(state=tk.NORMAL, bg="#0066ff")
                self.run_selected_btn.config(state=tk.NORMAL, bg="#00c853")
                self._refresh_tree_display()
                self.log(f"[*] Unlocked Read-Only attribute on {unlocked_count} track(s) (ID3 tags preserved).")
                messagebox.showinfo("Files Unlocked", f"Removed Read-Only lock on {unlocked_count} track(s) while preserving all ID3 tags & cues.")

            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _save_rename_ledger(self):
        try:
            with open(RENAME_LEDGER_FILE, "w", encoding="utf-8") as f:
                json.dump(self.rename_ledger, f, indent=2)
        except Exception:
            pass

    def undo_last_renames(self):
        """Rolls back disk filename changes from the most recent batch using the rename ledger."""
        ledger = list(self.rename_ledger)
        if not ledger and RENAME_LEDGER_FILE.exists():
            try:
                with open(RENAME_LEDGER_FILE, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
            except Exception:
                ledger = []

        if not ledger:
            messagebox.showinfo("No Renames to Undo", "No recorded disk renames found in the rollback ledger.")
            return

        if not messagebox.askyesno("Undo Last Renames", f"Found {len(ledger)} renamed file(s) in the rollback ledger.\n\nRestore original filenames on disk now?"):
            return

        restored = 0
        path_lookup = {str(d.path.resolve()): d for d in self.diagnostics}
        for old_str, new_str in reversed(ledger):
            old_p = Path(old_str)
            new_p = Path(new_str)
            if new_p.exists() and not old_p.exists():
                try:
                    was_ro = FileLockManager.is_read_only(new_p)
                    if was_ro:
                        FileLockManager.set_read_only(new_p, read_only=False)
                    os.rename(new_p, old_p)
                    if was_ro:
                        FileLockManager.set_read_only(old_p, read_only=True)
                    if str(new_p.resolve()) in path_lookup:
                        path_lookup[str(new_p.resolve())].path = old_p
                    restored += 1
                except Exception:
                    pass

        self.rename_ledger.clear()
        self._save_rename_ledger()
        self._refresh_tree_display()
        self.refresh_beta_targets_data()
        self.log(f"[↩️ Rollback] Restored {restored} track(s) to their original filenames.")
        messagebox.showinfo("Undo Complete", f"Restored {restored} file(s) back to their original filenames.")

    def clear_track_viewer(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.diagnostics.clear()
        self.diag_by_iid.clear()
        self.audio_files.clear()
        self.last_batch_results.clear()
        self.last_batch_elapsed_sec = 0.0
        self.root_folder = None
        self.folder_var.set("Drag & Drop Crate Folder here or click SELECT CRATE...")
        self.progress['value'] = 0
        self.is_paused = False
        self.stop_requested = True

        self.preset_full()
        self.anim_label.config(text="", fg="#00e5ff")
        self.log("[*] Viewer cleared, switches reset, and crate selection wiped.")

    def _read_cached_metrics(self, filepath: Path) -> Dict:
        if filepath.suffix.lower() in ID3_COMPATIBLE_EXTENSIONS:
            tags = UnifiedID3Handler.load_tags(filepath)
            if tags is not None and 'GEOB:VantaAudioMetrics' in tags:
                try:
                    return json.loads(tags['GEOB:VantaAudioMetrics'].data.decode('utf-8'))
                except Exception:
                    pass
        return {}

    def open_subcrate_builder(self):
        if not self.diagnostics:
            messagebox.showinfo("Crate Empty", "Please load a folder or tracks first.")
            return

        win = tk.Toplevel(self)
        win.title("Native Serato Subcrate Builder")
        win.geometry("460x490")
        win.configure(bg="#181a20")
        win.transient(self)
        win.grab_set()

        tk.Label(
            win, text="NATIVE SERATO SUBCRATE BUILDER",
            font=("Segoe UI", 10, "bold"), fg="#b388ff", bg="#181a20"
        ).pack(pady=(12, 2))
        tk.Label(
            win, text="Populates .crate files directly into _Serato_/Subcrates (uses %% nesting)",
            font=("Segoe UI", 8), fg="#9aa0b4", bg="#181a20"
        ).pack(pady=(0, 10))

        # Root prefix input
        prefix_frame = tk.Frame(win, bg="#181a20")
        prefix_frame.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(prefix_frame, text="Parent Crate Prefix:", font=("Segoe UI", 8, "bold"), fg="#e0e0e0", bg="#181a20").pack(side=tk.LEFT)
        prefix_var = tk.StringVar(value="Vanta")
        tk.Entry(prefix_frame, textvariable=prefix_var, font=("Segoe UI", 9), bg="#21242d", fg="#00d2ff", relief=tk.FLAT, width=22).pack(side=tk.RIGHT)

        # Smart subcrate checkboxes
        opts_box = tk.LabelFrame(win, text=" SMART SUBCRATE RULES (FROM CACHED DSP TAGS) ", font=("Segoe UI", 8, "bold"), fg="#9aa0b4", bg="#181a20")
        opts_box.pack(fill=tk.X, padx=16, pady=8)

        v_energy = tk.BooleanVar(value=True)
        v_genre = tk.BooleanVar(value=True)
        v_mood = tk.BooleanVar(value=False)
        v_key = tk.BooleanVar(value=False)
        v_audit = tk.BooleanVar(value=True)
        v_dups = tk.BooleanVar(value=False)

        rules = [
            ("Energy Tiers (1-3 Warmup / 4-7 Mid / 8-10 Peak)", v_energy),
            ("By Detected Genre (e.g., Genre%%Jungle / D&B)", v_genre),
            ("By Mood / Tone (Crisp vs Warm)", v_mood),
            ("By Camelot Key (1A - 12B)", v_key),
            ("Audit Crates (Flagged Review & Transcodes/Clipped)", v_audit),
            ("Duplicates & Clones Crate", v_dups),
        ]
        for label, var in rules:
            tk.Checkbutton(
                opts_box, text=label, variable=var, font=("Segoe UI", 8),
                bg="#181a20", fg="#e0e0e0", selectcolor="#21242d",
                activebackground="#181a20", activeforeground="#b388ff"
            ).pack(anchor="w", padx=8, pady=2)

        # Custom subcrate from highlighted rows
        custom_box = tk.LabelFrame(win, text=" CUSTOM SUBCRATE FROM HIGHLIGHTED TRACKS ", font=("Segoe UI", 8, "bold"), fg="#9aa0b4", bg="#181a20")
        custom_box.pack(fill=tk.X, padx=16, pady=8)

        sel_count = len(self._get_selected_diagnostics())
        tk.Label(custom_box, text=f"Highlighted tracks in viewer: {sel_count}", font=("Segoe UI", 8), fg="#00e676" if sel_count else "#7e869c", bg="#181a20").pack(anchor="w", padx=8, pady=(4, 2))

        custom_row = tk.Frame(custom_box, bg="#181a20")
        custom_row.pack(fill=tk.X, padx=8, pady=(2, 8))
        custom_name_var = tk.StringVar(value="Custom Set")
        tk.Entry(custom_row, textvariable=custom_name_var, font=("Segoe UI", 9), bg="#21242d", fg="#ffffff", relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        def export_highlighted_custom():
            sel = self._get_selected_diagnostics()
            if not sel:
                messagebox.showwarning("No Selection", "Highlight one or more tracks in the main list first.", parent=win)
                return
            p_root = prefix_var.get().strip()
            c_sub = custom_name_var.get().strip() or "Custom Set"
            full_name = f"{p_root}%%{c_sub}" if p_root else c_sub
            out = SeratoBinaryEngine.export_serato_crate(full_name, [d.path for d in sel])
            if out:
                self.log(f"[+] Created custom Serato subcrate: {out.name} ({len(sel)} tracks)")
                messagebox.showinfo("Crate Exported", f"Populated {out.name} with {len(sel)} track(s)!", parent=win)

        tk.Button(
            custom_row, text="Populate Highlighted", font=("Segoe UI", 8, "bold"),
            bg="#00c853", fg="#ffffff", relief=tk.FLAT, padx=8, command=export_highlighted_custom
        ).pack(side=tk.RIGHT)

        def generate_smart_subcrates():
            p_root = prefix_var.get().strip()
            pfx = f"{p_root}%%" if p_root else ""
            buckets: Dict[str, List[Path]] = defaultdict(list)

            for d in self.diagnostics:
                res_mem = self.last_batch_results.get(str(d.path.resolve()))
                metrics = self._read_cached_metrics(d.path)
                energy = res_mem.energy_level if res_mem else metrics.get("energy")
                mood = res_mem.mood if res_mem else metrics.get("mood")
                key = res_mem.key if res_mem else metrics.get("key")
                genre = (res_mem.detected_genre if res_mem else "") or d.clean_genre or d.raw_genre or d.folder_category

                if v_energy.get() and isinstance(energy, int) and not d.is_long_track:
                    if energy <= 5:
                        buckets[f"{pfx}Energy%%1-5 Warmup & Groove"].append(d.path)
                    elif energy <= 7:
                        buckets[f"{pfx}Energy%%6-7 Mid-Set Builder"].append(d.path)
                    else:
                        buckets[f"{pfx}Energy%%8-10 Peak Hour"].append(d.path)

                if v_genre.get() and genre and genre not in ("Unknown", "Unassigned"):
                    clean_g = genre.replace("/", "&")
                    buckets[f"{pfx}Genre%%{clean_g}"].append(d.path)

                if v_mood.get() and mood:
                    buckets[f"{pfx}Mood%%{mood}"].append(d.path)

                if v_key.get() and key and key not in ("Unknown", "MIX"):
                    buckets[f"{pfx}Key%%{key}"].append(d.path)

                if v_audit.get():
                    if d.flagged or metrics.get("flagged"):
                        buckets[f"{pfx}Audit%%Flagged Review"].append(d.path)
                    if d.is_fake_320k or metrics.get("is_fake_320k") or d.is_clipped or metrics.get("is_clipped"):
                        buckets[f"{pfx}Audit%%Transcodes & Clipped"].append(d.path)

                if v_dups.get() and d.is_duplicate:
                    buckets[f"{pfx}Audit%%Duplicates"].append(d.path)

            created = 0
            for cname, paths in buckets.items():
                if SeratoBinaryEngine.export_serato_crate(cname, paths):
                    created += 1
                    self.log(f"[+] Populated Subcrate: {cname}.crate ({len(paths)} tracks)")

            if created == 0:
                messagebox.showwarning(
                    "No Cached Metrics",
                    "No smart subcrates were populated. Run a batch analysis first so tracks have cached DSP tags, or use 'Populate Highlighted'.",
                    parent=win
                )
            else:
                messagebox.showinfo("Subcrates Built", f"Successfully populated {created} native Serato subcrate(s)!", parent=win)
                win.destroy()

        tk.Button(
            win, text="BUILD & POPULATE SMART SUBCRATES", font=("Segoe UI", 9, "bold"),
            bg="#b388ff", fg="#121316", relief=tk.FLAT, pady=6, command=generate_smart_subcrates
        ).pack(fill=tk.X, padx=16, pady=(6, 12))

    def _on_worker_count_changed(self, new_count: int):
        self.current_max_workers = new_count
        self.log(f"[*] CPU Worker Thread Pool set to: {new_count} concurrent worker(s) [{self.cpu_telemetry.cpu_model}]")

    def _poll_ambient_cpu_telemetry(self):
        """Continuously updates the Active Processor Load Visualizer every 850ms when idle, or 140ms during batch runs."""
        if hasattr(self, 'cpu_visualizer') and self.cpu_visualizer.winfo_exists():
            max_w = self.cpu_visualizer.get_selected_worker_count()
            act_w = self.active_worker_threads if getattr(self, 'is_running', False) and not getattr(self, 'is_paused', False) else 0
            self.cpu_visualizer.update_telemetry(active_workers=act_w, max_workers=max_w)
        if not getattr(self, 'is_running', False):
            self.after(850, self._poll_ambient_cpu_telemetry)

    def _save_quarantine_ledger(self):
        try:
            with open(QUARANTINE_LEDGER_FILE, "w", encoding="utf-8") as f:
                json.dump(self.quarantine_ledger, f, indent=2)
        except Exception:
            pass

    def quarantine_all_clones(self, clone_diags: Optional[List[TrackDiagnostic]] = None, parent_win=None) -> int:
        """Moves detected duplicate/clone files into a '_Clones_Quarantine' subfolder and removes them from the active batch queue."""
        if clone_diags is None:
            clone_diags = [d for d in self.diagnostics if d.is_duplicate]
        if not clone_diags or not self.root_folder:
            return 0

        q_root = self.root_folder / "_Clones_Quarantine"
        q_root.mkdir(parents=True, exist_ok=True)

        moved = 0
        moved_set = set()
        for d in list(clone_diags):
            src_p = d.path.resolve()
            if not src_p.exists():
                continue
            dest_p = q_root / src_p.name
            counter = 1
            while dest_p.exists():
                dest_p = q_root / f"{src_p.stem}__dup{counter}{src_p.suffix}"
                counter += 1
            try:
                if FileLockManager.is_read_only(src_p):
                    FileLockManager.set_read_only(src_p, read_only=False)
                shutil.move(str(src_p), str(dest_p))
                self.quarantine_ledger.append((str(src_p), str(dest_p)))
                moved_set.add(d.path)
                moved += 1
            except Exception as e:
                self.log(f"[!] Could not quarantine {src_p.name}: {e}")

        if moved > 0:
            self._save_quarantine_ledger()
            self.diagnostics = [d for d in self.diagnostics if d.path not in moved_set]
            self._refresh_tree_display()
            self.refresh_beta_targets_data()
            self.log(f"[📦 Clone Quarantine] Moved {moved} duplicate track(s) to: {q_root} (Removed from batch queue).")
        return moved

    def restore_quarantined_clones(self, parent_win=None):
        """Restores quarantined clone files back to their original crate locations."""
        ledger = list(self.quarantine_ledger)
        if not ledger and QUARANTINE_LEDGER_FILE.exists():
            try:
                with open(QUARANTINE_LEDGER_FILE, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
            except Exception:
                ledger = []

        if not ledger:
            messagebox.showinfo("Quarantine Empty", "No quarantined clone files found in the rollback ledger.", parent=parent_win or self)
            return

        restored = 0
        for orig_str, q_str in reversed(ledger):
            orig_p = Path(orig_str)
            q_p = Path(q_str)
            if q_p.exists() and not orig_p.exists():
                try:
                    orig_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(q_p), str(orig_p))
                    restored += 1
                except Exception:
                    pass

        self.quarantine_ledger.clear()
        self._save_quarantine_ledger()
        self.log(f"[↩️ Quarantine Restore] Restored {restored} clone file(s) to their original folders.")
        messagebox.showinfo("Clones Restored", f"Restored {restored} quarantined clone file(s). Click SELECT FOLDER to rescan.", parent=parent_win or self)

    def run_manual_clone_audit(self):
        if not self.diagnostics:
            messagebox.showinfo("Crate Empty", "Please load a crate folder first.")
            return
        self.diagnostics = DuplicateAuditor.audit_duplicates(self.diagnostics)
        self._refresh_tree_display()
        self.refresh_beta_targets_data()
        clones = [d for d in self.diagnostics if d.is_duplicate]
        self.log(f"[*] 3-Stage Clone Audit complete: {len(clones)} duplicate/clone track(s) identified out of {len(self.diagnostics)} tracks.")
        self.open_clone_manager_popup()

    def open_clone_manager_popup(self):
        """Interactive Pre-Batch Clone Resolution & Quarantine Manager."""
        if hasattr(self, 'clone_mgr_win') and self.clone_mgr_win and self.clone_mgr_win.winfo_exists():
            self.clone_mgr_win.lift()
            self.clone_mgr_win.focus_force()
            return

        clones = [d for d in self.diagnostics if d.is_duplicate]
        win = tk.Toplevel(self)
        self.clone_mgr_win = win
        win.title("🔍 Pre-Batch Clone & Duplicate Resolution Manager (v12.0)")
        win.geometry("980x560")
        win.configure(bg="#14161d")
        win.transient(self)

        hdr = tk.Frame(win, bg="#1f2330", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text=f"🔍 PRE-BATCH CLONE & DUPLICATE RESOLUTION MANAGER ({len(clones)} CLONES DETECTED)",
            font=("Segoe UI", 11, "bold"), fg="#00d2ff", bg="#1f2330"
        ).pack(padx=16, anchor="w")
        tk.Label(
            hdr,
            text="Resolve, quarantine, or drop duplicate files BEFORE starting the DSP batch run to save CPU time and prevent rename collisions.",
            font=("Segoe UI", 8), fg="#9aa0b4", bg="#1f2330"
        ).pack(padx=16, anchor="w")

        table_frame = tk.Frame(win, bg="#181a20")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        cols = ("Clone Filename", "Master Track Kept", "Audit Stage & Detection Reason", "Folder")
        ctree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        for c_name, c_w in [("Clone Filename", 260), ("Master Track Kept", 260), ("Audit Stage & Detection Reason", 280), ("Folder", 120)]:
            ctree.heading(c_name, text=c_name)
            ctree.column(c_name, width=c_w, anchor="w")

        sb = ttk.Scrollbar(table_frame, orient="vertical", command=ctree.yview)
        ctree.configure(yscrollcommand=sb.set)
        ctree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        clone_map: Dict[str, TrackDiagnostic] = {}
        for idx, d in enumerate(clones):
            iid = f"c_{idx}"
            clone_map[iid] = d
            master_name = d.master_path.name if d.master_path else "Original in Crate"
            ctree.insert("", tk.END, iid=iid, values=(d.path.name, master_name, d.duplicate_reason, d.folder_category or "Root"))

        btn_bar = tk.Frame(win, bg="#14161d")
        btn_bar.pack(fill=tk.X, padx=16, pady=(0, 12))

        def do_quarantine_all():
            if not clones:
                messagebox.showinfo("No Clones", "0 clones in current queue.", parent=win)
                return
            if messagebox.askyesno("Quarantine All Clones", f"Move all {len(clones)} clone file(s) into '_Clones_Quarantine' and remove them from the batch queue?\n\n(You can restore them anytime with '↩️ Restore Quarantined').", parent=win):
                cnt = self.quarantine_all_clones(clones, parent_win=win)
                messagebox.showinfo("Quarantine Complete", f"Moved {cnt} clone(s) to '_Clones_Quarantine'. Ready for clean batch run!", parent=win)
                win.destroy()

        def do_drop_from_queue():
            if not clones:
                return
            self.diagnostics = [d for d in self.diagnostics if not d.is_duplicate]
            self._refresh_tree_display()
            self.refresh_beta_targets_data()
            self.log(f"[*] Removed {len(clones)} clone(s) from the active viewer queue (files left untouched on disk).")
            messagebox.showinfo("Dropped from Queue", f"Removed {len(clones)} clone(s) from the batch queue (files untouched on disk).", parent=win)
            win.destroy()

        def do_export_dup_crate():
            if not clones:
                return
            out = SeratoBinaryEngine.export_serato_crate("Vanta%%Audit%%Duplicates", [d.path for d in clones])
            if out:
                self.log(f"[+] Exported {len(clones)} duplicate tracks to Serato crate: {out.name}")
                messagebox.showinfo("Crate Exported", f"Exported {len(clones)} clone tracks to {out.name}!", parent=win)

        tk.Button(
            btn_bar, text=f"📦 QUARANTINE ALL ({len(clones)}) TO '_Clones_Quarantine'",
            font=("Segoe UI", 8, "bold"), bg="#00e676", fg="#121316", relief=tk.FLAT, padx=10, pady=6,
            command=do_quarantine_all
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_bar, text="⏭️ DROP CLONES FROM QUEUE (KEEP ON DISK)",
            font=("Segoe UI", 8, "bold"), bg="#00d2ff", fg="#121316", relief=tk.FLAT, padx=10, pady=6,
            command=do_drop_from_queue
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="📦 EXPORT DUPLICATES CRATE",
            font=("Segoe UI", 8, "bold"), bg="#b388ff", fg="#121316", relief=tk.FLAT, padx=10, pady=6,
            command=do_export_dup_crate
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="↩️ RESTORE QUARANTINED",
            font=("Segoe UI", 8, "bold"), bg="#ffd600", fg="#121316", relief=tk.FLAT, padx=10, pady=6,
            command=lambda: self.restore_quarantined_clones(parent_win=win)
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="CLOSE",
            font=("Segoe UI", 8, "bold"), bg="#2d313d", fg="#ffffff", relief=tk.FLAT, padx=10, pady=6,
            command=win.destroy
        ).pack(side=tk.RIGHT)

    def _refresh_tree_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.diag_by_iid.clear()

        active_cols = [col for col, var in self.col_visibility.items() if var.get()]

        for idx, d in enumerate(self.diagnostics):
            iid = f"track_{idx}"
            self.diag_by_iid[iid] = d

            flag_icon = "★" if d.flagged else ""
            lock_icon = "🔒" if d.is_locked else ""

            notes = []
            if d.is_duplicate:
                notes.append("⚠️ [DUP]")
            if d.is_fake_320k:
                notes.append("⚠️ [<14.5kHz]")
            if d.is_clipped:
                notes.append("🚨 [CLIP]")
            if d.is_long_track:
                notes.append("⏱️ [LONG]")
            if d.is_cached:
                notes.append("⚡ [CACHED]")
            notes_str = " ".join(notes) if notes else "Ready"

            row_data_map = {
                "Flag": flag_icon,
                "Lock": lock_icon,
                "Status / Notes": notes_str,
                "Filename": d.path.name,
                "Proposed Rename": d.proposed_filename,
                "Folder Category": d.folder_category if d.folder_category else "Root"
            }

            row_values = tuple(row_data_map[col] for col in active_cols)
            self.tree.insert("", tk.END, iid=iid, values=row_values)

    def on_tree_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region not in ('cell', 'item'):
            return

        column = self.tree.identify_column(event.x)
        col_idx_int = int(column.replace('#', '')) - 1
        active_cols = [col for col, var in self.col_visibility.items() if var.get()]

        if col_idx_int >= len(active_cols) or active_cols[col_idx_int] != "Proposed Rename":
            return

        item_id = self.tree.identify_row(event.y)
        if not item_id or item_id not in self.diag_by_iid:
            return

        target_diag = self.diag_by_iid[item_id]
        x, y, width, height = self.tree.bbox(item_id, column)
        current_values = list(self.tree.item(item_id, 'values'))
        current_text = current_values[col_idx_int]

        self.entry_edit = tk.Entry(self.tree, font=("Segoe UI", 9), bg="#21242d", fg="#00d2ff", insertbackground="#ffffff", relief=tk.FLAT)
        self.entry_edit.place(x=x, y=y, width=width, height=height)
        self.entry_edit.insert(0, current_text)
        self.entry_edit.select_range(0, tk.END)
        self.entry_edit.focus()

        def save_edit(e=None):
            if not hasattr(self, 'entry_edit') or not self.entry_edit:
                return
            new_text = self.entry_edit.get().strip()
            self.entry_edit.destroy()
            delattr(self, 'entry_edit')

            if not new_text:
                return

            new_text = ensure_extension(new_text, target_diag.path.suffix)
            target_diag.proposed_filename = new_text
            self.log(f"[*] Manual override: '{target_diag.path.name}' -> Proposed: '{new_text}'")
            self._refresh_tree_display()

        def cancel_edit(e=None):
            if hasattr(self, 'entry_edit') and self.entry_edit:
                self.entry_edit.destroy()
                delattr(self, 'entry_edit')

        self.entry_edit.bind("<Return>", save_edit)
        self.entry_edit.bind("<FocusOut>", save_edit)
        self.entry_edit.bind("<Escape>", cancel_edit)

    @staticmethod
    def _is_quarantine_path(p: Path) -> bool:
        return any(part.lower() == "_clones_quarantine" for part in p.parts)

    def handle_drop(self, event):
        raw_data = event.data
        dropped_paths = [Path(p) for p in self.tk.splitlist(raw_data)]
        if not dropped_paths:
            return

        dirs = [p for p in dropped_paths if p.is_dir() and not self._is_quarantine_path(p)]
        files = [p for p in dropped_paths if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not self._is_quarantine_path(p)]

        if len(dirs) == 1 and not files:
            self.load_crate_folder(dirs[0])
            return

        if dirs or files:
            self.load_multiple_sources(dirs, files)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select DJ Crate Folder")
        if not folder:
            return
        self.load_crate_folder(Path(folder))

    def load_crate_folder(self, folder_path: Path):
        if self.is_scanning or self.is_running:
            return

        self.root_folder = folder_path
        self.folder_var.set(str(self.root_folder))

        self.audio_files = [
            p for p in self.root_folder.rglob('*')
            if p.suffix.lower() in AUDIO_EXTENSIONS and not self._is_quarantine_path(p)
        ]
        if not self.audio_files:
            messagebox.showwarning("Crate Empty", "No audio tracks found in this directory.")
            return

        self.is_scanning = True
        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.log(f"[*] Starting fast scan of crate: {folder_path} ({len(self.audio_files)} files found)")

        thread = threading.Thread(target=self._run_async_scan, daemon=True)
        thread.start()

    def load_multiple_sources(self, folders: List[Path], loose_files: List[Path]):
        """Loads and merges audio tracks from multiple dropped folders and/or individual files in one pass."""
        if self.is_scanning or self.is_running:
            return

        collected: List[Path] = []
        seen = set()

        for folder in folders:
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not self._is_quarantine_path(p):
                    rp = str(p.resolve()).lower()
                    if rp not in seen:
                        seen.add(rp)
                        collected.append(p)

        for f in loose_files:
            rp = str(f.resolve()).lower()
            if rp not in seen:
                seen.add(rp)
                collected.append(f)

        if not collected:
            messagebox.showwarning("Selection Empty", "No supported audio tracks found in the dropped folders/files.")
            return

        all_parents = [d.parent for d in folders] + [f.parent for f in loose_files]
        try:
            common_root = Path(os.path.commonpath([str(p.resolve()) for p in all_parents]))
        except Exception:
            common_root = folders[0].parent if folders else loose_files[0].parent

        self.root_folder = common_root
        folder_names = ", ".join(d.name for d in folders[:4])
        if len(folders) > 4:
            folder_names += f" (+{len(folders) - 4} more)"
        summary_label = f"Loaded {len(folders)} folder(s)" + (f" [{folder_names}]" if folder_names else "")
        if loose_files:
            summary_label += f" + {len(loose_files)} file(s)"
        summary_label += f" ({len(collected)} tracks total)"
        self.folder_var.set(summary_label)

        self.audio_files = collected
        self.is_scanning = True
        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.log(f"[*] Multi-Folder Drop: {summary_label} -> Common Root: {common_root}")
        thread = threading.Thread(target=self._run_async_scan, daemon=True)
        thread.start()

    def load_individual_tracks(self, file_list: List[Path], root_dir: Path):
        if self.is_scanning or self.is_running:
            return

        self.root_folder = root_dir
        self.folder_var.set(f"Loaded {len(file_list)} dropped track(s) from: {root_dir}")
        self.audio_files = file_list

        self.is_scanning = True
        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.log(f"[*] Parsing dropped tracks ({len(file_list)} files)...")

        thread = threading.Thread(target=self._run_async_scan, daemon=True)
        thread.start()

    def _run_async_scan(self):
        try:
            def scan_progress(current, total, filename):
                self.after(0, lambda c=current, t=total: self.progress.configure(value=c, maximum=t))
                if current % 10 == 0 or current == total:
                    self.log(f"    [Scanning {current}/{total}] {filename}")

            self.diagnostics = DiagnosticEngine.audit_crate(self.audio_files, self.root_folder, progress_callback=scan_progress)
            self.after(0, self._finalize_scan_ui)
        except Exception as e:
            self.log(f"[!] Scan Error: {e}")
            self.after(0, self._finalize_scan_ui)

    def _finalize_scan_ui(self):
        self._refresh_tree_display()
        dup_count = sum(1 for d in self.diagnostics if d.is_duplicate)
        long_count = sum(1 for d in self.diagnostics if d.is_long_track)
        self.log(f"[+] Async Scan Complete! Loaded {len(self.diagnostics)} tracks. Clones Detected: {dup_count} | Long Mixes: {long_count}")

        self.progress['value'] = 0
        self.is_scanning = False
        self.browse_btn.config(state=tk.NORMAL, bg="#0066ff")
        self.run_btn.config(state=tk.NORMAL, bg="#0066ff")
        self.run_selected_btn.config(state=tk.NORMAL, bg="#00c853")
        self.refresh_beta_targets_data()

    def update_all_tab_buttons(self):
        for name, var, color in self.workflow_configs:
            is_active = var.get()
            btn = self.workflow_tabs[name]
            if is_active:
                btn.config(bg=color, fg="#121316", relief="sunken")
            else:
                btn.config(bg="#222530", fg="#8a8d9b", relief="flat")

    def preset_full(self):
        self.cfg_clean_tags.set(True)
        self.cfg_rename_disk.set(True)
        self.cfg_inject_cues.set(True)
        self.cfg_protect_cues.set(False)
        self.cfg_inject_autogain.set(True)
        self.cfg_run_dsp.set(True)
        self.cfg_skip_clones.set(True)
        self.cfg_skip_long_tracks.set(True)
        self.cfg_skip_cached.set(True)
        self.cfg_lock_files.set(True)
        self.cfg_flag_analysis.set(False)
        self.cfg_export_crates.set(True)
        if hasattr(self, 'workflow_tabs'):
            self.update_all_tab_buttons()

    def preset_safe(self):
        self.preset_full()
        self.cfg_rename_disk.set(False)
        self.update_all_tab_buttons()

    def preset_cues(self):
        self.cfg_clean_tags.set(False)
        self.cfg_rename_disk.set(False)
        self.cfg_inject_cues.set(True)
        self.cfg_protect_cues.set(False)
        self.cfg_inject_autogain.set(True)
        self.cfg_run_dsp.set(True)
        self.update_all_tab_buttons()

    def preset_meta(self):
        self.cfg_clean_tags.set(True)
        self.cfg_rename_disk.set(True)
        self.cfg_inject_cues.set(False)
        self.cfg_protect_cues.set(True)
        self.cfg_inject_autogain.set(False)
        self.cfg_run_dsp.set(False)
        self.update_all_tab_buttons()

    def animate_processing(self):
        if getattr(self, 'is_running', False) and not getattr(self, 'is_paused', False):
            elapsed_sec = time.time() - self.batch_start_time
            frame_text = DanceFloorEngine.render_frame(self.anim_idx, elapsed_sec)
            self.anim_label.config(text=frame_text, fg="#00e5ff")
            if hasattr(self, 'cpu_visualizer') and self.cpu_visualizer.winfo_exists():
                self.cpu_visualizer.update_telemetry(
                    active_workers=self.active_worker_threads,
                    max_workers=self.current_max_workers
                )
            self.anim_idx += 1
            self.after(140, self.animate_processing)
        elif getattr(self, 'is_paused', False):
            self.anim_label.config(text="⏸️ [BATCH PAUSED] ⏸️", fg="#ff9100")
            if hasattr(self, 'cpu_visualizer') and self.cpu_visualizer.winfo_exists():
                self.cpu_visualizer.update_telemetry(active_workers=0, max_workers=self.current_max_workers)
            self.after(500, self.animate_processing)

    def toggle_pause_batch(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶️ RESUME", bg="#00e676")
            self.log("[*] Batch paused by user.")
        else:
            self.pause_btn.config(text="⏸️ PAUSE", bg="#ff9100")
            self.log("[*] Batch resumed.")

    def stop_batch(self):
        if not self.is_running:
            return
        self.stop_requested = True
        self.is_paused = False
        self.log("[!] Stop requested by user. Aborting batch...")

    def trigger_police_bust(self, step=0, max_steps=16, success_count=0):
        if step < max_steps:
            frame_text, fg_color = DanceFloorEngine.render_bust_frame(step)
            self.anim_label.config(text=frame_text, fg=fg_color)
            self.after(280, lambda: self.trigger_police_bust(step + 1, max_steps, success_count))
        else:
            self.anim_label.config(text="", fg="#00e5ff")
            self.browse_btn.config(state=tk.NORMAL, bg="#0066ff")
            self.run_btn.config(state=tk.NORMAL, bg="#0066ff")
            self.run_selected_btn.config(state=tk.NORMAL, bg="#00c853")
            self.pause_btn.config(state=tk.DISABLED, bg="#ff9100", text="⏸️ PAUSE")
            self.stop_btn.config(state=tk.DISABLED)
            self._refresh_tree_display()
            messagebox.showinfo(
                "Batch Complete",
                f"Successfully processed {success_count} track(s)!"
            )

    def start_batch_thread(self, selected_only: bool = False):
        if self.is_running or self.is_scanning or not self.root_folder:
            return

        if selected_only:
            target_diagnostics = self._get_selected_diagnostics()
            if not target_diagnostics:
                messagebox.showinfo("Selection Required", "Please highlight one or more tracks in the list first.")
                return
        else:
            target_diagnostics = self.diagnostics

        if not target_diagnostics:
            return

        # Pre-Batch Clone Resolution Check: allow dealing with duplicate files BEFORE starting the batch run
        active_clones = [d for d in target_diagnostics if d.is_duplicate]
        if active_clones and not selected_only:
            resp = messagebox.askyesnocancel(
                f"🔍 {len(active_clones)} Clones / Duplicates Detected Before Batch",
                f"Pre-Scan detected {len(active_clones)} duplicate/clone track(s) in this crate.\n\n"
                f"• Click [YES] to QUARANTINE all {len(active_clones)} clones to '_Clones_Quarantine' first (Recommended — prevents rename collisions & saves CPU time).\n"
                f"• Click [NO] to leave them on disk and {'SKIP them during this batch' if self.cfg_skip_clones.get() else 'process them anyway'}.\n"
                f"• Click [CANCEL] to open the Clone Resolution Manager first."
            )
            if resp is None:
                self.open_clone_manager_popup()
                return
            elif resp is True:
                self.quarantine_all_clones(active_clones)
                target_diagnostics = self.diagnostics
                if not target_diagnostics:
                    return

        if self.cfg_export_crates.get() and SeratoBinaryEngine.is_serato_running():
            proceed = messagebox.askyesno(
                "⚠️ Serato DJ Pro Detected Running",
                "Serato DJ Pro is currently open in the background!\n\n"
                "Because Serato holds crates in RAM while open, it will overwrite newly exported `.crate` files when you close Serato.\n\n"
                "Close Serato DJ Pro first for best results, or click YES to run anyway."
            )
            if not proceed:
                return

        self.is_running = True
        self.is_paused = False
        self.stop_requested = False
        self.batch_start_time = time.time()
        self.anim_idx = 0

        self.browse_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_btn.config(state=tk.DISABLED, bg="#424242")
        self.run_selected_btn.config(state=tk.DISABLED, bg="#424242")
        self.pause_btn.config(state=tk.NORMAL, bg="#ff9100", text="⏸️ PAUSE")
        self.stop_btn.config(state=tk.NORMAL)

        self.progress['value'] = 0
        self.progress['maximum'] = len(target_diagnostics)

        self.animate_processing()

        thread = threading.Thread(target=self._run_batch, args=(target_diagnostics,), daemon=True)
        thread.start()

    def _run_batch(self, target_diagnostics: List[TrackDiagnostic]):
        config = {
            'clean_tags': self.cfg_clean_tags.get(),
            'rename_disk': self.cfg_rename_disk.get(),
            'inject_cues': self.cfg_inject_cues.get(),
            'protect_cues': self.cfg_protect_cues.get(),
            'inject_autogain': self.cfg_inject_autogain.get(),
            'run_dsp': self.cfg_run_dsp.get(),
            'skip_clones': self.cfg_skip_clones.get(),
            'skip_long_tracks': self.cfg_skip_long_tracks.get(),
            'skip_cached': self.cfg_skip_cached.get(),
            'lock_files': self.cfg_lock_files.get(),
            'export_crates': self.cfg_export_crates.get(),
            'tag_title_artist': self.cfg_tag_title_artist.get(),
            'tag_genre': self.cfg_tag_genre.get(),
            'tag_key': self.cfg_tag_key.get(),
            'tag_bpm': self.cfg_tag_bpm.get(),
            'tag_mood': self.cfg_tag_mood.get(),
            'tag_energy': self.cfg_tag_energy.get(),
            'tag_geob_metrics': self.cfg_tag_geob_metrics.get(),
            'tag_comment': self.cfg_tag_comment.get()
        }

        total = len(target_diagnostics)
        success = 0
        completed_count = 0
        warmup_tracks: List[Path] = []
        midset_tracks: List[Path] = []
        peak_tracks: List[Path] = []
        flagged_tracks: List[Path] = []
        transcode_tracks: List[Path] = []

        # Step 1: Pre-process fast disk renames & cached-track skips sequentially to avoid path races
        if config['rename_disk']:
            self.rename_ledger.clear()
        pending_jobs: List[Tuple[int, TrackDiagnostic, Path, bool, bool]] = []
        for idx, d in enumerate(target_diagnostics, start=1):
            if self.stop_requested:
                break

            if config['skip_clones'] and d.is_duplicate:
                self.log(f"[{idx}/{total}] SKIPPED CLONE (⚠️ [DUP] - {d.duplicate_reason}): {d.path.name}")
                completed_count += 1
                self.after(0, lambda val=completed_count: self.progress.configure(value=val))
                continue

            if config['skip_cached'] and d.is_cached and not d.flagged:
                self.log(f"[{idx}/{total}] SKIPPED (Already Cached): {d.path.name}")
                success += 1
                completed_count += 1
                self.after(0, lambda val=completed_count: self.progress.configure(value=val))
                continue

            resolved_path = d.path.resolve()
            current_path = resolved_path

            if config['rename_disk'] and d.proposed_filename:
                safe_proposed = ensure_extension(d.proposed_filename, resolved_path.suffix)
                d.proposed_filename = safe_proposed
                new_path = (resolved_path.parent / safe_proposed).resolve()
                if new_path != resolved_path and not new_path.exists():
                    try:
                        if FileLockManager.is_read_only(resolved_path):
                            FileLockManager.set_read_only(resolved_path, read_only=False)
                        old_name = resolved_path.name
                        os.rename(resolved_path, new_path)
                        self.rename_ledger.append((str(resolved_path), str(new_path)))
                        current_path = new_path
                        d.path = current_path
                        self.log(f"[{idx}/{total}] Renamed: {old_name} -> {current_path.name}")
                    except OSError:
                        current_path = resolved_path

            skip_cues = bool(config['protect_cues'] and d.has_existing_cues)
            cues_enabled = bool(config['inject_cues'] and not skip_cues)
            should_flag = bool(d.flagged or self.cfg_flag_analysis.get())
            pending_jobs.append((idx, d, current_path, cues_enabled, should_flag))

        if self.rename_ledger:
            self._save_rename_ledger()

        # Step 2: Multi-Core Parallel DSP & Tag Injection Pool (Auto-Scaled for Intel i5/i7/i9 & AMD Ryzen)
        max_workers = self.cpu_visualizer.get_selected_worker_count() if hasattr(self, 'cpu_visualizer') else self.cpu_telemetry.recommended_workers
        self.current_max_workers = max_workers
        if pending_jobs and not self.stop_requested:
            self.log(f"[*] Launching parallel turbo pool ({max_workers} CPU workers on {self.cpu_telemetry.cpu_model}) for {len(pending_jobs)} track(s)...")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {}
                job_iter = iter(pending_jobs)

                # Submit in sliding window so Pause and Stop respond immediately
                def submit_next():
                    try:
                        idx, d, cur_path, cues_en, sh_flag = next(job_iter)
                        fut = pool.submit(
                            _parallel_worker_task,
                            cur_path,
                            self.root_folder,
                            cues_en,
                            8,
                            sh_flag,
                            config['skip_long_tracks'],
                            config
                        )
                        future_map[fut] = (idx, d, cur_path)
                        self.active_worker_threads = min(max_workers, len(future_map))
                        return True
                    except StopIteration:
                        self.active_worker_threads = min(max_workers, len(future_map))
                        return False

                for _ in range(max_workers * 2):
                    if not submit_next():
                        break

                while future_map and not self.stop_requested:
                    while self.is_paused and not self.stop_requested:
                        self.active_worker_threads = 0
                        time.sleep(0.2)
                    if self.stop_requested:
                        for fut in list(future_map.keys()):
                            fut.cancel()
                        break

                    self.active_worker_threads = min(max_workers, len(future_map))
                    done_iter = as_completed(list(future_map.keys()))
                    fut = next(done_iter)
                    idx, d, cur_path = future_map.pop(fut)
                    self.active_worker_threads = min(max_workers, len(future_map))

                    try:
                        res, ok, err_msg = fut.result()
                        if err_msg:
                            self.log(f"[{idx}/{total}] {cur_path.name} -> [!] Error: {err_msg}")
                        else:
                            self.last_batch_results[str(cur_path.resolve())] = res
                            d.flagged = res.flagged
                            d.is_fake_320k = res.is_fake_320k
                            d.is_clipped = res.is_clipped
                            d.is_long_track = res.is_long_track
                            if res.detected_genre:
                                d.clean_genre = res.detected_genre

                            if not res.is_long_track:
                                if res.energy_level <= 5:
                                    warmup_tracks.append(cur_path)
                                elif res.energy_level <= 7:
                                    midset_tracks.append(cur_path)
                                else:
                                    peak_tracks.append(cur_path)
                            if res.flagged:
                                flagged_tracks.append(cur_path)
                            if res.is_fake_320k or res.is_clipped:
                                transcode_tracks.append(cur_path)

                            flag_status = " [FLAGGED - GRID/TUNING REVIEW]" if res.flagged else ""
                            bw_status = f" [LOW-BW {res.spectral_cutoff_hz/1000:.1f}kHz]" if res.is_fake_320k else ""
                            clip_status = f" [CLIPPED Crest:{res.crest_factor_db:.1f}dB]" if res.is_clipped else ""
                            long_status = " [LONG MIX BYPASS]" if res.is_long_track else ""

                            self.log(
                                f"[{idx}/{total}] {cur_path.name}\n"
                                f"        -> Genre: [{res.detected_genre}] | {res.energy_summary} "
                                f"| LUFS: {res.integrated_lufs:.1f}{flag_status}{bw_status}{clip_status}{long_status}"
                            )
                            if ok:
                                d.is_locked = config['lock_files']
                                d.is_cached = True
                                success += 1
                    except Exception as e:
                        self.log(f"[{idx}/{total}] {cur_path.name} -> [!] Worker Error: {e}")

                    completed_count += 1
                    self.after(0, lambda val=completed_count: self.progress.configure(value=val))

                    if not self.stop_requested:
                        submit_next()

        # Optional Native Serato Subcrate Generation (using Harmonic Arc Engine v10.7 binary cloner)
        if config['export_crates'] and not self.stop_requested:
            for c_name, c_list in [
                ("Vanta%%Energy%%1-5 Warmup & Groove", warmup_tracks),
                ("Vanta%%Energy%%6-7 Mid-Set Builder", midset_tracks),
                ("Vanta%%Energy%%8-10 Peak Hour", peak_tracks),
                ("Vanta%%Audit%%Flagged Review", flagged_tracks),
                ("Vanta%%Audit%%Transcodes & Clipped", transcode_tracks)
            ]:
                c_path = SeratoBinaryEngine.export_serato_crate(c_name, c_list, append=True)
                if c_path:
                    self.log(f"[+] Updated Serato Crate: {c_path.name} ({len(c_list)} tracks written -> {c_path.parent})")

        elapsed_total = time.time() - self.batch_start_time
        self.last_batch_elapsed_sec = elapsed_total
        self.log("\n" + "=" * 70)
        if self.stop_requested:
            self.log(f"[ABORTED] Batch stopped after {elapsed_total:.1f}s. Processed before abort: {success} tracks.")
        else:
            self.log(f"[SUMMARY] Processed {total} tracks in {elapsed_total:.1f}s | Successfully Audited & Locked: {success}")
        self.log("=" * 70)

        self.active_worker_threads = 0
        self.is_running = False
        self.is_paused = False
        self.after(0, self._poll_ambient_cpu_telemetry)
        self.after(0, self.refresh_beta_targets_data)
        if self.stop_requested:
            self.after(0, lambda: self._reset_stopped_ui())
        else:
            self.after(0, lambda: self.trigger_police_bust(step=0, max_steps=16, success_count=success))


    # ==========================================================================
    # BETA TESTER POPUPS: OPERATIONAL GUIDE & SELF-POPULATING QA TARGETS
    # ==========================================================================
    def export_summarized_conversion_report(self, parent_win=None):
        """Exports a concise Summarized Conversion Report (.txt) with library energy, genre, loudness, tempo, and error totals."""
        if not self.diagnostics:
            messagebox.showwarning("No Data to Export", "Load a crate folder first before exporting a Conversion Summary Report.", parent=parent_win or self)
            return

        default_name = f"Vanta_Conversion_Summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = filedialog.asksaveasfilename(
            parent=parent_win or self,
            title="Save Summarized Conversion Report",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text Report", "*.txt"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            total_loaded = len(self.diagnostics)
            analyzed_results = list(self.last_batch_results.values())
            analyzed_count = len(analyzed_results)
            music_results = [r for r in analyzed_results if not r.is_long_track and r.integrated_lufs < -0.5]
            long_count = sum(1 for d in self.diagnostics if d.is_long_track)
            dup_count = sum(1 for d in self.diagnostics if d.is_duplicate)
            flagged_count = sum(1 for d in self.diagnostics if d.flagged)
            fake_count = sum(1 for d in self.diagnostics if d.is_fake_320k)
            clip_count = sum(1 for d in self.diagnostics if d.is_clipped)
            locked_count = sum(1 for d in self.diagnostics if FileLockManager.is_read_only(d.path))

            e_warmup = sum(1 for r in music_results if r.energy_level <= 5)
            e_mid = sum(1 for r in music_results if 6 <= r.energy_level <= 7)
            e_peak = sum(1 for r in music_results if r.energy_level >= 8)

            crisp_cnt = sum(1 for r in music_results if r.mood == 'Crisp')
            warm_cnt = sum(1 for r in music_results if r.mood == 'Warm')
            genre_counts: Dict[str, int] = defaultdict(int)
            for r in analyzed_results:
                genre_counts[r.detected_genre or "Electronic"] += 1
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

            if music_results:
                lufs_vals = [r.integrated_lufs for r in music_results]
                bpm_vals = [r.detected_bpm for r in music_results if r.detected_bpm > 0]
                avg_lufs = float(np.mean(lufs_vals))
                min_lufs = float(np.min(lufs_vals))
                max_lufs = float(np.max(lufs_vals))
                avg_gain = float(np.mean([r.gain_offset_db for r in music_results]))
                min_bpm = float(np.min(bpm_vals)) if bpm_vals else 0.0
                med_bpm = float(np.median(bpm_vals)) if bpm_vals else 0.0
                max_bpm = float(np.max(bpm_vals)) if bpm_vals else 0.0
            else:
                avg_lufs = min_lufs = max_lufs = avg_gain = min_bpm = med_bpm = max_bpm = 0.0

            lines = [
                "=" * 78,
                "SERATO AUDIO ANALYZER PRO — v12.0 SUMMARIZED CONVERSION REPORT",
                "=" * 78,
                f"Generated         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Source Root       : {self.root_folder}",
                f"CPU Hardware      : {self.cpu_telemetry.cpu_model} ({self.cpu_telemetry.logical_cores}T | Pool: {self.current_max_workers} Workers)",
                f"Batch Throughput  : {self.last_batch_elapsed_sec:.1f}s total ({(self.last_batch_elapsed_sec / max(1, analyzed_count)):.2f}s / track)" if self.last_batch_elapsed_sec > 0 else "Batch Throughput  : Pre-Scan Only",
                "-" * 78,
                "1. CONVERSION & QUEUE DISPOSITION",
                "-" * 78,
                f"  Tracks in Queue         : {total_loaded}",
                f"  Successfully Audited    : {analyzed_count}",
                f"  Disk Renames Applied    : {len(self.rename_ledger)}",
                f"  Clones Quarantined      : {len(self.quarantine_ledger)} (Active Clones in Queue: {dup_count})",
                f"  Long Mixes Bypassed     : {long_count} (>= 15 min continuous sets)",
                f"  Read-Only Locked (🔒)   : {locked_count}",
                "-" * 78,
                "2. LIBRARY ENERGY, TONE & LOUDNESS PROFILE",
                "-" * 78,
                f"  Energy 1-5 (Warmup)     : {e_warmup} tracks",
                f"  Energy 6-7 (Mid-Set)    : {e_mid} tracks",
                f"  Energy 8-10 (Peak Hour) : {e_peak} tracks",
                f"  Spectral Tone Balance   : Crisp = {crisp_cnt} | Warm = {warm_cnt}",
                f"  Integrated LUFS Range   : Avg {avg_lufs:.1f} LUFS (Min {min_lufs:.1f} / Max {max_lufs:.1f}) | Avg Autogain: {avg_gain:+.1f} dB",
                f"  Tempo (BPM) Range       : Min {min_bpm:.1f} | Median {med_bpm:.1f} | Max {max_bpm:.1f} BPM",
                "-" * 78,
                "3. DETECTED GENRE DISTRIBUTION",
                "-" * 78,
            ]
            for g_name, g_cnt in top_genres[:15]:
                pct = (g_cnt / max(1, analyzed_count)) * 100.0
                lines.append(f"  • {g_name:<38} : {g_cnt:>4} tracks ({pct:4.1f}%)")

            lines.extend([
                "-" * 78,
                "4. QUALITY ASSURANCE EXCEPTIONS SUMMARY",
                "-" * 78,
                f"  ★ Flagged Grid / Off-Tuning (>= ±38c) : {flagged_count} tracks",
                f"  ⚠️ Low-Bandwidth Transcodes (<14.5kHz) : {fake_count} tracks",
                f"  🚨 Severe Brickwall Clipping          : {clip_count} tracks",
                "=" * 78,
            ])

            Path(save_path).write_text("\n".join(lines), encoding="utf-8")
            self.log(f"[📊 Summary Report] Exported Summarized Conversion Report to: {save_path}")
            messagebox.showinfo("Summary Report Exported", f"Successfully saved Summarized Conversion Report:\n{save_path}", parent=parent_win or self)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save summary report: {e}", parent=parent_win or self)

    def export_beta_qa_report(self, parent_win=None):
        """Exports a comprehensive Beta Tester QA Report (.txt) with system specs, timing benchmarks, and all 8 QA target verifications."""
        if not self.diagnostics:
            messagebox.showwarning("No Data to Export", "Load a crate folder first before exporting a Beta QA Report.", parent=parent_win or self)
            return

        default_name = f"Vanta_Beta_QA_Report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        save_path = filedialog.asksaveasfilename(
            parent=parent_win or self,
            title="Save Beta Tester QA Report",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text Report", "*.txt"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            total_loaded = len(self.diagnostics)
            analyzed_count = len(self.last_batch_results)
            dup_count = sum(1 for d in self.diagnostics if d.is_duplicate)
            flagged_count = sum(1 for d in self.diagnostics if d.flagged)
            fake_count = sum(1 for d in self.diagnostics if d.is_fake_320k)
            clip_count = sum(1 for d in self.diagnostics if d.is_clipped)
            for d in self.diagnostics:
                d.is_locked = FileLockManager.is_read_only(d.path)
            locked_count = sum(1 for d in self.diagnostics if d.is_locked)
            serato_open = SeratoBinaryEngine.is_serato_running()

            lines = [
                "=" * 80,
                "SERATO AUDIO ANALYZER PRO — VERSION 12.0 STANDALONE QA VERIFICATION REPORT",
                "=" * 80,
                f"Timestamp         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"OS / Platform     : {platform.system()} {platform.release()} ({platform.version()})",
                f"CPU Hardware      : {self.cpu_telemetry.cpu_model} ({self.cpu_telemetry.logical_cores} Logical Threads | Pool: {self.current_max_workers} Workers)",
                f"Python Runtime    : {platform.python_version()}",
                f"Serato Running?   : {'YES (WARNING: Close Serato before exporting crates)' if serato_open else 'NO (Clean disk state)'}",
                f"Crate Folder      : {self.root_folder}",
                f"Tracks Loaded     : {total_loaded} | Analyzed in Session: {analyzed_count}",
                f"Last Batch Time   : {self.last_batch_elapsed_sec:.1f}s ({(self.last_batch_elapsed_sec / max(1, analyzed_count)):.2f}s / track)" if self.last_batch_elapsed_sec > 0 else "Last Batch Time   : N/A (Pre-Scan Only)",
                f"Audit Summary     : Clones={dup_count} | Flagged(★)={flagged_count} | Low-BW(<14.5kHz)={fake_count} | Clipped={clip_count} | Locked(🔒)={locked_count}",
                "-" * 80,
                "SELF-POPULATED 8-POINT QA TEST TARGETS",
                "-" * 80,
            ]

            if self.beta_targets_tree and self.beta_targets_tree.winfo_exists():
                for iid in self.beta_targets_tree.get_children():
                    vals = self.beta_targets_tree.item(iid, "values")
                    if len(vals) >= 4:
                        lines.append(f"[{vals[0]}]")
                        lines.append(f"  Target Track : {vals[1]}")
                        lines.append(f"  Live Metrics : {vals[2]}")
                        lines.append(f"  QA Checklist : {vals[3]}")
                        lines.append("")

            lines.append("-" * 80)
            lines.append("SAMPLE ANALYZED TRACK LOG (UP TO 50 TRACKS)")
            lines.append("-" * 80)
            for idx, (p_str, res) in enumerate(list(self.last_batch_results.items())[:50], start=1):
                lines.append(
                    f"{idx:02d}. {res.filepath.name} | Genre: {res.detected_genre} | {res.energy_summary} "
                    f"| LUFS: {res.integrated_lufs:.1f} | Cutoff: {res.spectral_cutoff_hz/1000:.1f}kHz "
                    f"| Flagged: {res.flagged} | LowBW: {res.is_fake_320k}"
                )

            Path(save_path).write_text("\n".join(lines), encoding="utf-8")
            self.log(f"[📋 QA Report] Exported Beta Tester QA Report to: {save_path}")
            messagebox.showinfo("QA Report Exported", f"Successfully saved Beta Tester QA Report:\n{save_path}", parent=parent_win or self)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save QA report: {e}", parent=parent_win or self)

    def open_operational_instructions_popup(self):
        if hasattr(self, 'ops_guide_win') and self.ops_guide_win and self.ops_guide_win.winfo_exists():
            self.ops_guide_win.lift()
            self.ops_guide_win.focus_force()
            return

        win = tk.Toplevel(self)
        self.ops_guide_win = win
        win.title("📖 Beta Tester Operational Instructions — Serato Analyzer Pro v12.0")
        win.geometry("780x670")
        win.configure(bg="#14161d")
        win.transient(self)

        hdr = tk.Frame(win, bg="#1f2330", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text="📖 OPERATIONAL INSTRUCTIONS & WORKFLOW GUIDE (BETA v12.0)",
            font=("Segoe UI", 11, "bold"), fg="#ffd600", bg="#1f2330"
        ).pack(padx=16, anchor="w")
        tk.Label(
            hdr, text="Complete operating manual for DSP auditing, safety rollbacks, and native Serato crate verification.",
            font=("Segoe UI", 8), fg="#9aa0b4", bg="#1f2330"
        ).pack(padx=16, anchor="w")

        txt_frame = tk.Frame(win, bg="#14161d")
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        txt = tk.Text(
            txt_frame, bg="#0d0f14", fg="#e0e0e0", font=("Consolas", 9),
            relief=tk.FLAT, wrap=tk.WORD, padx=12, pady=12
        )
        sb = tk.Scrollbar(txt_frame, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        guide_lines = [
            "===========================================================================",
            "1. QUICK-START OPERATING PROCEDURE (v12.0 STANDALONE)",
            "===========================================================================",
            "• IMPORTANT: Close Serato DJ Pro before running a batch so Serato does not overwrite newly generated `.crate` files from RAM.",
            "• STEP 1 (LOAD CRATE): Drag & drop one or MULTIPLE folders (and/or individual MP3, WAV, AIFF, FLAC, M4A files) simultaneously onto the table, or click [SELECT FOLDER]. ('_Clones_Quarantine' folders are automatically excluded).",
            "• STEP 2 (ASYNC PRE-SCAN): The engine automatically runs a fast diagnostic pass:",
            "    - Decodes URL-escaped filenames (%20 -> space), strips compilation/album clutter & mid-filename track numbers, and preserves '&' and dotted artist names (e.g., MI.RO, St.Ego).",
            "    - Never prepends 'Unknown Artist - ' onto filenames when artist tags are blank or generic.",
            "    - Runs the 4-Stage Clone Auditor (Disk Copy-Suffix Master Check + Audio-Tail MD5 Hash + Metadata + Stem normalization) to flag duplicates with ⚠️ [DUP], and strips '(2)'/'Copy' suffixes from orphan survivors.",
            "    - Detects continuous DJ mixes (>15 mins) and marks them with ⏱️ [LONG].",
            "    - Detects already-analyzed tracks with ⚡ [CACHED] and Read-Only locked files with 🔒.",
            "• STEP 3 (DEAL WITH CLONES FIRST — BEFORE BATCH RUN):",
            "    - Click [🔍 Clones] (or click [EXECUTE BATCH] when clones are present) to launch the Pre-Batch Clone Resolution Manager.",
            "    - Click [📦 QUARANTINE ALL TO '_Clones_Quarantine'] to safely move duplicate files out of your active crate before DSP analysis starts (prevents rename collisions & saves CPU time).",
            "    - Or leave the [Skip Clones] module active so the batch runner automatically bypasses ⚠️ [DUP] rows.",
            "• STEP 4 (CHOOSE PRESET & WORKER THREADS):",
            "    - [Full]: Runs full DSP, renames files on disk, skips clones/long mixes/cached tracks, injects 8 quantized Serato cues, writes -14.0 LUFS autogain, locks files Read-Only, and populates Serato subcrates.",
            "    - [Safe]: Same as Full, but disables disk file renaming (keeps original filenames intact).",
            "    - [Cues]: Only runs DSP, Cue Point injection, and Autogain without altering text metadata.",
            "    - [Meta]: Only cleans ID3/Vorbis/MP4 tags and renames files on disk without running audio DSP.",
            "",
            "===========================================================================",
            "2. ACTIVE PROCESSOR LOAD VISUALIZER & INTEL / AMD THREAD SCALING",
            "===========================================================================",
            "• Located directly above the ASCII Dance Floor in the right-hand Execution pane.",
            "• Auto-detects your exact CPU model (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9/Threadripper) and logical thread count.",
            "• Displays a live 20-segment Package Load LED bar, Active Worker Thread counter, and an adaptive Per-Core vertical LED matrix.",
            "• [Workers Dropdown]: Defaults to 'Auto' (3 workers on 4-core i5; 6-12 workers on Intel i7/i9; up to 14 workers on all-big-core AMD Ryzen with AVX2 fast-path). Beta testers can manually select 2 to 16 workers to benchmark throughput.",
            "",
            "===========================================================================",
            "3. SAFETY NETS: INLINE RENAME, UNLOCK FILES, UNDO RENAMES & QUARANTINE RESTORE",
            "===========================================================================",
            "• INLINE RENAME OVERRIDE: Double-click any cell in the 'Proposed Smart Rename' column to manually edit the target filename before running a batch. Press [Enter] to save or [Esc] to cancel.",
            "• [🔓 Unlock Files]: Removes the Read-Only lock (🔒) on highlighted or all tracks WITHOUT wiping your ID3 tags or Serato cues.",
            "• [↩️ Undo Renames]: Automatically rolls back the most recent batch of disk file renames to their exact original filenames.",
            "• [↩️ Restore Quarantined]: Inside the [🔍 Clones] Manager, restores any quarantined duplicate tracks back to their original folders.",
            "",
            "===========================================================================",
            "4. VISUAL ROW TINTS IN SERATO DJ PRO & NATIVE SUBCRATE BUILDER (📦 Crates)",
            "===========================================================================",
            "• WHITE ROW: Clean, verified master track with stable beatgrid and standard A440 tuning.",
            "• YELLOW ROW (★ FLAGGED): High beatgrid drift (IBI variance > 8.5%), off-center pitch tuning (>= ±38 cents from A440), or anti-halving recovery. Inspect beatgrid in Serato! (Note: 'Force-Flag All' manually flags every track in a batch; leave it OFF for normal automatic DSP flagging).",
            "• RED ROW (⚠️ [<14.5kHz] / 🚨 [CLIP]): Spectral roll-off below 14.5 kHz (fake 320kbps transcode / YouTube rip) or severe digital brickwall clipping.",
            "• NATIVE SUBCRATE CLONER: Clones the exact binary column layout (`osrt` / `ovct`) from your native Serato crates, populates parent crates (`Vanta`, `Vanta%%Energy`, `Vanta%%Audit`), excludes bypassed Long Mixes from Energy crates, and purges stale ghost paths after disk renames."
        ]
        txt.insert(tk.END, "\n".join(guide_lines))
        txt.config(state=tk.DISABLED)

        btn_bar = tk.Frame(win, bg="#14161d")
        btn_bar.pack(fill=tk.X, padx=16, pady=(0, 12))

        tk.Button(
            btn_bar, text="🎯 OPEN SELF-POPULATING BETA TEST TARGETS ->",
            font=("Segoe UI", 9, "bold"), bg="#00e676", fg="#121316",
            relief=tk.FLAT, padx=14, pady=6,
            command=lambda: [win.destroy(), self.open_beta_targets_popup()]
        ).pack(side=tk.RIGHT)

        tk.Button(
            btn_bar, text="CLOSE GUIDE",
            font=("Segoe UI", 9, "bold"), bg="#2d313d", fg="#ffffff",
            relief=tk.FLAT, padx=14, pady=6, command=win.destroy
        ).pack(side=tk.LEFT)

    def open_beta_targets_popup(self):
        if self.beta_targets_win and self.beta_targets_win.winfo_exists():
            self.beta_targets_win.lift()
            self.beta_targets_win.focus_force()
            self.refresh_beta_targets_data()
            return

        win = tk.Toplevel(self)
        self.beta_targets_win = win
        win.title("🎯 Self-Populating Beta Testing Targets & QA Verification Suite (v12.0)")
        win.geometry("1080x700")
        win.configure(bg="#14161d")
        win.transient(self)

        hdr = tk.Frame(win, bg="#1f2330", pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text="🎯 SELF-POPULATING BETA TESTING TARGETS & QA VERIFICATION CHECKLIST",
            font=("Segoe UI", 11, "bold"), fg="#00e676", bg="#1f2330"
        ).pack(padx=16, anchor="w")
        tk.Label(
            hdr,
            text="Automatically extracts real tracks from your loaded/analyzed crate into 8 specific QA test cases. Double-click any target to highlight it in the main viewer!",
            font=("Segoe UI", 8), fg="#9aa0b4", bg="#1f2330"
        ).pack(padx=16, anchor="w")

        self.beta_summary_lbl = tk.Label(
            win, text="Scanning loaded crate for QA test candidates...",
            font=("Segoe UI", 9, "bold"), fg="#00d2ff", bg="#14161d", anchor="w"
        )
        self.beta_summary_lbl.pack(fill=tk.X, padx=16, pady=(8, 4))

        table_frame = tk.Frame(win, bg="#181a20")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        cols = ("Test Category", "Target Track", "Detected Metrics", "What To Test & Verify (QA Instructions)")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        self.beta_targets_tree = tree

        tree.heading("Test Category", text="QA Test Category")
        tree.heading("Target Track", text="Auto-Selected Target Track")
        tree.heading("Detected Metrics", text="Live Metrics / Status")
        tree.heading("What To Test & Verify (QA Instructions)", text="Beta Tester Verification Instructions")

        tree.column("Test Category", width=185, anchor="w")
        tree.column("Target Track", width=230, anchor="w")
        tree.column("Detected Metrics", width=215, anchor="w")
        tree.column("What To Test & Verify (QA Instructions)", width=410, anchor="w")

        sb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.beta_target_Map: Dict[str, Path] = {}

        def on_target_double_click(event):
            sel = tree.selection()
            if not sel:
                return
            t_iid = sel[0]
            target_path = self.beta_target_Map.get(t_iid)
            if not target_path:
                return
            for main_iid, diag in self.diag_by_iid.items():
                if diag.path == target_path:
                    self.tree.selection_set(main_iid)
                    self.tree.focus(main_iid)
                    self.tree.see(main_iid)
                    self.log(f"[🎯 QA Target Selected] Highlighted '{diag.path.name}' in main viewer.")
                    break

        tree.bind("<Double-1>", on_target_double_click)

        inst_box = tk.LabelFrame(
            win, text=" HOW TO USE THIS SELF-POPULATING QA BOARD ",
            font=("Segoe UI", 8, "bold"), fg="#ffd600", bg="#181a20"
        )
        inst_box.pack(fill=tk.X, padx=16, pady=6)

        inst_lines = [
            "1. Load any folder of tracks in the main window — this board immediately self-populates with real test candidates from your crate.",
            "2. If clones (⚠️ [DUP]) are detected, click '🔍 Clones' to test the Pre-Batch Clone Quarantine Manager before running your batch.",
            "3. Watch the Active Processor Load Visualizer (Intel/AMD auto-scaler), verify your Vanta crates in Serato DJ Pro, and click '📋 EXPORT QA REPORT' to send your log!"
        ]
        tk.Label(inst_box, text="\n".join(inst_lines), font=("Segoe UI", 8), fg="#e0e0e0", bg="#181a20", justify=tk.LEFT).pack(anchor="w", padx=10, pady=6)

        btn_bar = tk.Frame(win, bg="#14161d")
        btn_bar.pack(fill=tk.X, padx=16, pady=(2, 12))

        tk.Button(
            btn_bar, text="🔄 REFRESH TARGETS", font=("Segoe UI", 8, "bold"),
            bg="#0066ff", fg="#ffffff", relief=tk.FLAT, padx=12, pady=5,
            command=self.refresh_beta_targets_data
        ).pack(side=tk.LEFT, padx=(0, 6))

        def highlight_all_targets_in_main():
            matched_iids = []
            target_paths = set(self.beta_target_Map.values())
            for main_iid, diag in self.diag_by_iid.items():
                if diag.path in target_paths:
                    matched_iids.append(main_iid)
            if matched_iids:
                self.tree.selection_set(matched_iids)
                self.tree.see(matched_iids[0])
                self.log(f"[🎯 QA Suite] Highlighted {len(matched_iids)} beta test target track(s) in main viewer. Click 'SCAN HIGHLIGHTED' to test them!")
                messagebox.showinfo(
                    "Targets Highlighted",
                    f"Highlighted {len(matched_iids)} QA target tracks in the main viewer.\n\nClick 'SCAN HIGHLIGHTED' in the main window to test just these tracks!",
                    parent=win
                )
            else:
                messagebox.showwarning("No Targets Loaded", "Load a crate folder in the main window first so QA targets can auto-populate.", parent=win)

        tk.Button(
            btn_bar, text="🎯 HIGHLIGHT ALL QA TARGETS IN MAIN VIEWER", font=("Segoe UI", 8, "bold"),
            bg="#00e676", fg="#121316", relief=tk.FLAT, padx=12, pady=5,
            command=highlight_all_targets_in_main
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="🔍 OPEN CLONE MANAGER", font=("Segoe UI", 8, "bold"),
            bg="#00d2ff", fg="#121316", relief=tk.FLAT, padx=12, pady=5,
            command=self.open_clone_manager_popup
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="📊 SUMMARY REPORT (.TXT)", font=("Segoe UI", 8, "bold"),
            bg="#b388ff", fg="#121316", relief=tk.FLAT, padx=10, pady=5,
            command=lambda: self.export_summarized_conversion_report(parent_win=win)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_bar, text="📋 FULL QA REPORT (.TXT)", font=("Segoe UI", 8, "bold"),
            bg="#ffd600", fg="#121316", relief=tk.FLAT, padx=10, pady=5,
            command=lambda: self.export_beta_qa_report(parent_win=win)
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_bar, text="CLOSE", font=("Segoe UI", 8, "bold"),
            bg="#2d313d", fg="#ffffff", relief=tk.FLAT, padx=12, pady=5,
            command=win.destroy
        ).pack(side=tk.RIGHT)

        self.refresh_beta_targets_data()

    def refresh_beta_targets_data(self):
        if not self.beta_targets_tree or not self.beta_targets_tree.winfo_exists():
            return

        for item in self.beta_targets_tree.get_children():
            self.beta_targets_tree.delete(item)
        self.beta_target_Map.clear()

        if not self.diagnostics:
            if self.beta_summary_lbl and self.beta_summary_lbl.winfo_exists():
                self.beta_summary_lbl.config(
                    text="⚠️ No crate loaded yet — Drop a folder in the main window and this table will self-populate with real test tracks!",
                    fg="#ff9100"
                )
            placeholder_rows = [
                ("1. Clone / Duplicate Audit", "[Waiting for crate...]", "N/A", "Load a crate to auto-select duplicate/clone files for Stage 1-3 verification."),
                ("2. Smart Rename & Undo Rollback", "[Waiting for crate...]", "N/A", "Load a crate to auto-select tracks with dirty tags/filenames & test '↩️ Undo Renames'."),
                ("3. Filename BPM Hint Lock", "[Waiting for crate...]", "N/A", "Load a crate to auto-select tracks with BPM numbers in their filename."),
                ("4. Peak-Hour Energy (8-10/10)", "[Waiting for DSP run...]", "N/A", "Run DSP to auto-select a loud (-4 to -7 LUFS) club track & verify Peak Hour crate."),
                ("5. Warmup / Dynamic Track (1-5/10)", "[Waiting for DSP run...]", "N/A", "Run DSP to auto-select a high-headroom (-15 to -21 LUFS) track & verify '1-5 Warmup & Groove' crate."),
                ("6. Off-Tuning / Beatgrid Flag (★)", "[Waiting for DSP run...]", "N/A", "Run DSP to auto-select a track with >=38c tuning offset or beatgrid drift (Yellow Serato row)."),
                ("7. Low-BW Transcode (<14.5kHz)", "[Waiting for DSP run...]", "N/A", "Run DSP to auto-select a low-bandwidth MP3 transcode (Red Serato row & Audit crate)."),
                ("8. Anti-Halving Midtempo / D&B", "[Waiting for DSP run...]", "N/A", "Run DSP to auto-select a 85-115 BPM or 160-175 BPM track to verify zero false 2x/0.5x halving.")
            ]
            for idx, r in enumerate(placeholder_rows):
                self.beta_targets_tree.insert("", tk.END, iid=f"qa_empty_{idx}", values=r)
            return

        def get_diag_metrics(d: TrackDiagnostic) -> Dict:
            res = self.last_batch_results.get(str(d.path.resolve()))
            if res:
                return {
                    "bpm": res.detected_bpm,
                    "key": res.key,
                    "tuning_cents": res.tuning_cents,
                    "energy": res.energy_level,
                    "lufs": res.integrated_lufs,
                    "cutoff_hz": res.spectral_cutoff_hz,
                    "is_fake_320k": res.is_fake_320k,
                    "is_clipped": res.is_clipped,
                    "flagged": res.flagged,
                    "mood": res.mood
                }
            return self._read_cached_metrics(d.path)

        dup_candidates = [d for d in self.diagnostics if d.is_duplicate]
        rename_candidates = [d for d in self.diagnostics if d.proposed_filename and d.proposed_filename != d.path.name]
        hint_candidates = [d for d in self.diagnostics if DSPAnalyzer.extract_filename_bpm_hint(d.path.stem) is not None]

        peak_candidates = []
        warmup_candidates = []
        flagged_candidates = []
        transcode_candidates = []
        tempo_edge_candidates = []

        for d in self.diagnostics:
            m = get_diag_metrics(d)
            e_lvl = m.get("energy")
            bpm = m.get("bpm", 0.0)
            lufs_val = m.get("lufs", 0.0)
            is_valid_music_track = not d.is_long_track and (lufs_val is not None and lufs_val < -0.5)

            if is_valid_music_track and isinstance(e_lvl, int) and e_lvl >= 8:
                peak_candidates.append((d, m))
            if is_valid_music_track and isinstance(e_lvl, int) and e_lvl <= 5:
                warmup_candidates.append((d, m))
            if d.flagged or m.get("flagged"):
                flagged_candidates.append((d, m))
            if d.is_fake_320k or m.get("is_fake_320k") or d.is_clipped or m.get("is_clipped"):
                transcode_candidates.append((d, m))
            if is_valid_music_track and bpm and (85.0 <= bpm <= 115.0 or bpm >= 158.0):
                tempo_edge_candidates.append((d, m))

        target_rows = []

        if dup_candidates:
            d = dup_candidates[0]
            target_rows.append((
                "1. Clone / Duplicate Audit",
                d.path.name,
                d.duplicate_reason or "⚠️ [DUP] Detected",
                "Verify ⚠️ [DUP] status in viewer & check 'Vanta%%Audit%%Duplicates.crate' via Crates builder.",
                d.path
            ))
        else:
            target_rows.append((
                "1. Clone / Duplicate Audit",
                "None in current crate (0 clones)",
                "✅ All tracks unique",
                "Copy any track as 'Track (2).mp3' and click '🔍 Clones' to test Stage 1-3 duplicate detection.",
                None
            ))

        if rename_candidates:
            d = rename_candidates[0]
            target_rows.append((
                "2. Smart Rename & Undo Rollback",
                d.path.name,
                f"-> {d.proposed_filename}",
                "Run with 'Rename' active, verify ghost-path purge in crate, then test '↩️ Undo Renames' button.",
                d.path
            ))
        else:
            d = self.diagnostics[0]
            target_rows.append((
                "2. Smart Rename & Undo Rollback",
                d.path.name,
                f"Clean ({d.proposed_filename})",
                "Double-click 'Proposed Smart Rename' cell to test manual override, then test '↩️ Undo Renames'.",
                d.path
            ))

        if hint_candidates:
            d = hint_candidates[0]
            hint_val = DSPAnalyzer.extract_filename_bpm_hint(d.path.stem)
            m = get_diag_metrics(d)
            det_bpm = f"Detected: {m['bpm']:.1f} BPM" if m.get("bpm") else "Run DSP to verify lock"
            target_rows.append((
                "3. Filename BPM Hint Lock",
                d.path.name,
                f"Hint: {hint_val} | {det_bpm}",
                "Verify that IBI tempo tracker reconciles 2x/0.5x/triplet multiples against filename hint.",
                d.path
            ))
        else:
            target_rows.append((
                "3. Filename BPM Hint Lock",
                "No filename BPM hints found",
                "N/A",
                "Rename a track with '- 128' or '- 88 - 8A' to test filename BPM hint reconciliation.",
                None
            ))

        if peak_candidates:
            d, m = peak_candidates[0]
            target_rows.append((
                "4. Peak-Hour Banger (Energy 8-10)",
                d.path.name,
                f"Energy {m.get('energy')}/10 | {m.get('lufs')} LUFS | {m.get('bpm')} BPM",
                "Verify negative Serato Autogain offset & check 'Vanta%%Energy%%8-10 Peak Hour.crate' in Serato.",
                d.path
            ))
        else:
            target_rows.append((
                "4. Peak-Hour Banger (Energy 8-10)",
                "[Run Batch DSP to populate]",
                "Pending DSP...",
                "Execute batch DSP so the engine identifies your loudest (-4.5 to -8.0 LUFS) Peak Hour tracks.",
                None
            ))

        analyzed_pairs = [
            (d, get_diag_metrics(d))
            for d in self.diagnostics
            if not d.is_long_track and get_diag_metrics(d).get("lufs") is not None and get_diag_metrics(d).get("lufs", 0.0) < -0.5
        ]

        if warmup_candidates:
            d, m = warmup_candidates[0]
            target_rows.append((
                "5. Warmup / Dynamic (Energy 1-5)",
                d.path.name,
                f"Energy {m.get('energy')}/10 | {m.get('lufs')} LUFS | {m.get('bpm')} BPM",
                "Verify positive Serato Autogain boost toward -14.0 LUFS & 'Vanta%%Energy%%1-5 Warmup & Groove.crate'.",
                d.path
            ))
        elif analyzed_pairs:
            # Fallback to the most dynamic / lowest-LUFS track in the analyzed crate
            d_low, m_low = min(analyzed_pairs, key=lambda x: x[1].get("lufs", 0.0))
            target_rows.append((
                "5. Lowest-Energy Track in Crate",
                d_low.path.name,
                f"Energy {m_low.get('energy')}/10 | {m_low.get('lufs')} LUFS | {m_low.get('bpm')} BPM",
                "Crate has 0 tracks <=5/10; verify this lowest-LUFS track in 'Vanta%%Energy%%6-7 Mid-Set Builder.crate'.",
                d_low.path
            ))
        else:
            target_rows.append((
                "5. Warmup / Dynamic (Energy 1-5)",
                "[Run Batch DSP to populate]",
                "Pending DSP...",
                "Execute batch DSP to auto-select a quieter/dynamic track (-15 to -21 LUFS) for autogain verification.",
                None
            ))

        if flagged_candidates:
            d, m = flagged_candidates[0]
            t_c = m.get("tuning_cents", 0.0)
            target_rows.append((
                "6. Flagged Grid / Tuning Review (★)",
                d.path.name,
                f"★ Flagged | Tuning: {t_c:+.0f}c | BPM: {m.get('bpm', '?')}",
                "Load in Serato DJ Pro: verify YELLOW row tint, 'Vanta%%Audit%%Flagged Review.crate', and beatgrid.",
                d.path
            ))
        elif analyzed_pairs:
            d_tun, m_tun = max(analyzed_pairs, key=lambda x: abs(x[1].get("tuning_cents", 0.0)))
            target_rows.append((
                "6. Flagged Grid / Tuning Review (★)",
                f"0 Flagged (Max drift: {d_tun.path.name[:18]}...)",
                f"✅ Clean ({m_tun.get('tuning_cents', 0.0):+.0f}c | {m_tun.get('bpm')} BPM)",
                "All tracks passed grid & A440 tuning checks (<38c drift). Click '🚩 Flag' on any row to test manual flagging.",
                d_tun.path
            ))
        else:
            target_rows.append((
                "6. Flagged Grid / Tuning Review (★)",
                "[Run Batch DSP to populate]",
                "0 Flagged so far",
                "Tracks with >= ±38c off-center pitch or >8.5% beat interval drift auto-populate here after DSP.",
                None
            ))

        if transcode_candidates:
            d, m = transcode_candidates[0]
            cut_khz = m.get("cutoff_hz", 0.0) / 1000.0 if m.get("cutoff_hz") else 0.0
            label_m = f"⚠️ Cutoff: {cut_khz:.1f} kHz" if cut_khz else "⚠️ [<14.5kHz / CLIP]"
            target_rows.append((
                "7. Low-BW Transcode (<14.5kHz)",
                d.path.name,
                label_m,
                "Load in Serato DJ Pro: verify RED row tint, '[LOW-BW]' comment tag, & 'Transcodes & Clipped' crate.",
                d.path
            ))
        elif analyzed_pairs:
            d_bw, m_bw = min(analyzed_pairs, key=lambda x: x[1].get("cutoff_hz", 20000.0))
            cut_khz = m_bw.get("cutoff_hz", 20000.0) / 1000.0
            target_rows.append((
                "7. Spectral Bandwidth Audit",
                d_bw.path.name,
                f"✅ Passed (Lowest: {cut_khz:.1f} kHz)",
                "All tracks passed >=14.5 kHz full-bandwidth check! Lowest roll-off track in crate shown here.",
                d_bw.path
            ))
        else:
            target_rows.append((
                "7. Low-BW Transcode (<14.5kHz)",
                "[Run Batch DSP to populate]",
                "0 Transcodes detected",
                "Any MP3 with a brickwall spectral roll-off below 14.5 kHz auto-populates here after DSP.",
                None
            ))

        if tempo_edge_candidates:
            d, m = tempo_edge_candidates[0]
            target_rows.append((
                "8. Anti-Halving Tempo Verification",
                d.path.name,
                f"BPM: {m.get('bpm')} | Key: {m.get('key')} | {m.get('mood')}",
                "Verify 8 quantized cues (BEAT 1 -> OUTRO) snap cleanly to 4-bar phrases without 2x/0.5x tempo error.",
                d.path
            ))
        else:
            d = self.diagnostics[min(2, len(self.diagnostics) - 1)]
            m = get_diag_metrics(d)
            b_str = f"BPM: {m.get('bpm')} | Key: {m.get('key')}" if m.get("bpm") else "Run DSP to populate"
            target_rows.append((
                "8. Cue & Phrase Quantization",
                d.path.name,
                b_str,
                "Verify 8 color-coded hotcues (BEAT 1, INTRO, BUILD 1, DROP 1, BREAK, BUILD 2, DROP 2, OUTRO) in Serato.",
                d.path
            ))

        populated_count = sum(1 for r in target_rows if r[4] is not None)
        if self.beta_summary_lbl and self.beta_summary_lbl.winfo_exists():
            self.beta_summary_lbl.config(
                text=f"✅ Auto-Populated {populated_count}/8 QA Test Categories from {len(self.diagnostics)} loaded tracks ({len(self.last_batch_results)} analyzed in session). Double-click any row to jump to track!",
                fg="#00e676"
            )

        for idx, (cat, t_name, metrics_str, instr, p_obj) in enumerate(target_rows):
            iid = f"qa_row_{idx}"
            self.beta_targets_tree.insert("", tk.END, iid=iid, values=(cat, t_name, metrics_str, instr))
            if p_obj is not None:
                self.beta_target_Map[iid] = p_obj

    def _reset_stopped_ui(self):
        self.anim_label.config(text="", fg="#00e5ff")
        self.browse_btn.config(state=tk.NORMAL, bg="#0066ff")
        self.run_btn.config(state=tk.NORMAL, bg="#0066ff")
        self.run_selected_btn.config(state=tk.NORMAL, bg="#00c853")
        self.pause_btn.config(state=tk.DISABLED, bg="#ff9100", text="⏸️ PAUSE")
        self.stop_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self._refresh_tree_display()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    app = SeratoV10App()
    app.mainloop()
