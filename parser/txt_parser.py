"""
txt_parser.py
=============
Parses the plain-text screenplay format used by Kuku's input pack.

Expected format per episode file
---------------------------------
EPISODE NUMBER: Episode N
EPISODE LOGLINE: ...
CONTINUITY NOTES: ...

--- SCRIPT START ---

SCENE 01
Scene heading: INT. LOCATION – DAY
Scene start: MM:SS.ss
Scene end:   MM:SS.ss

**CHARACTER (tone):**
Start: (MM:SS.ss) | End: (MM:SS.ss)
"Dialogue text"

**CAMERA:**
Start: (MM:SS.ss) | End: (MM:SS.ss)
Description text

--- SCRIPT END ---

The parser also handles directories containing multiple .txt files.
"""

import codecs
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from parser.timestamp_parser import ts_to_sec, extract_timestamp_pair


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DialogueLine:
    character: str        # speaker name
    tone: str             # e.g. "Furious", "Whispering"
    text: str             # spoken words
    start: float          # seconds
    end: float            # seconds


@dataclass
class ActionLine:
    description: str      # camera / action description
    start: float
    end: float


@dataclass
class Scene:
    number: int
    heading: str          # e.g. "INT. CATHEDRAL – DAY"
    start: float          # seconds
    end: float            # seconds
    dialogue: List[DialogueLine] = field(default_factory=list)
    actions:  List[ActionLine]   = field(default_factory=list)
    raw_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def all_text(self) -> str:
        """All dialogue + action text combined for keyword scanning."""
        parts = [d.text for d in self.dialogue] + [a.description for a in self.actions]
        return " ".join(parts)


@dataclass
class Episode:
    number: int
    filename: str
    logline: str
    continuity: str
    scenes: List[Scene] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    encoding: str = "utf-8"   # how the source file was decoded (reported, never silent)

    @property
    def total_duration(self) -> float:
        return max((s.end for s in self.scenes), default=0.0)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class ScreenplayReadError(Exception):
    """A screenplay file could not be read or decoded. The message says why."""


def decode_screenplay_bytes(data: bytes) -> Tuple[str, str]:
    """Decode screenplay bytes and report which encoding was used.

    Order: UTF-8 (with or without BOM) -> UTF-16 (BOM present) -> Windows-1252
    as a documented legacy fallback. Anything else raises ScreenplayReadError
    instead of silently accepting malformed input.
    """
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8):].decode("utf-8"), "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError as exc:
            raise ScreenplayReadError(f"invalid UTF-16 data ({exc.reason})") from exc
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError as exc:
        raise ScreenplayReadError(
            "not valid UTF-8, UTF-16 or Windows-1252 text; re-save the file as UTF-8") from exc


def read_screenplay_file(filepath: str) -> Tuple[str, str]:
    """Read a screenplay file -> (text, encoding). Raises ScreenplayReadError with the cause."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except OSError as exc:  # missing file, permission denied, is a directory, ...
        raise ScreenplayReadError(f"cannot read file: {type(exc).__name__}: {exc.strerror or exc}") from exc
    return decode_screenplay_bytes(data)


_EPISODE_MARKER = re.compile(r"^[ \t]*EPISODE NUMBER:", re.IGNORECASE | re.MULTILINE)


def split_episode_blocks(raw: str) -> List[str]:
    """Split text holding several episodes into one block per 'EPISODE NUMBER:' marker.

    Text with zero or one marker is returned whole (single-episode files are untouched).
    """
    starts = [m.start() for m in _EPISODE_MARKER.finditer(raw)]
    if len(starts) <= 1:
        return [raw]
    starts.append(len(raw))
    return [raw[a:b] for a, b in zip(starts[:-1], starts[1:])]


def parse_txt_file_ex(filepath: str) -> Tuple[List[Episode], Optional[str]]:
    """Parse a .txt file that holds one OR several episodes.

    Returns (episodes, error). `error` is a human-readable reason when the file
    could not be read; episodes is then empty. Episodes decoded with a fallback
    encoding carry that in Episode.encoding.
    """
    try:
        raw, encoding = read_screenplay_file(filepath)
    except ScreenplayReadError as exc:
        return [], str(exc)
    episodes = []
    for block in split_episode_blocks(raw):
        ep = _parse_raw(block, filepath)
        ep.encoding = encoding
        episodes.append(ep)
    return episodes, None


def parse_txt_file(filepath: str) -> Optional[Episode]:
    """
    Parse a single .txt screenplay file (the first episode if it holds several).
    Returns an Episode object, or None if the file is unreadable.
    Use parse_txt_file_ex() to get the reason a file could not be read.
    """
    episodes, _error = parse_txt_file_ex(filepath)
    return episodes[0] if episodes else None


def _parse_raw(raw: str, filepath: str) -> Episode:
    """Parse already-decoded screenplay text for ONE episode."""
    errors: List[str] = []

    # ── Episode metadata ──────────────────────────────────────────────────
    ep_m  = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", raw, re.IGNORECASE)
    log_m = re.search(r"EPISODE LOGLINE:\s*(.+)", raw, re.IGNORECASE)
    con_m = re.search(r"CONTINUITY NOTES:\s*(.+)", raw, re.IGNORECASE)

    if not ep_m:
        # Try to extract a number from the filename
        fn_m = re.search(r"episode[_\-\s]*(\d+)", os.path.basename(filepath), re.IGNORECASE)
        ep_num = int(fn_m.group(1)) if fn_m else 0
        errors.append("EPISODE NUMBER not found in file; inferred from filename.")
    else:
        ep_num = int(ep_m.group(1))

    episode = Episode(
        number     = ep_num,
        filename   = os.path.basename(filepath),
        logline    = log_m.group(1).strip() if log_m else "",
        continuity = con_m.group(1).strip() if con_m else "",
        parse_errors = errors,
    )

    # ── Extract script body ───────────────────────────────────────────────
    script_m = re.search(
        r"--- SCRIPT START ---(.+?)--- SCRIPT END ---",
        raw, re.DOTALL | re.IGNORECASE
    )
    if not script_m:
        episode.parse_errors.append("No SCRIPT START/END block found.")
        return episode

    script = script_m.group(1)

    # ── Split into scene blocks ───────────────────────────────────────────
    scene_blocks = re.split(r"\n(?=SCENE\s+\d+)", script, flags=re.IGNORECASE)

    for block in scene_blocks:
        block = block.strip()
        if not block:
            continue

        sn_m = re.match(r"SCENE\s+(\d+)", block, re.IGNORECASE)
        if not sn_m:
            continue

        scene_num = int(sn_m.group(1))

        ss_m = re.search(r"Scene\s+start:\s*([0-9:.]+)", block, re.IGNORECASE)
        se_m = re.search(r"Scene\s+end:\s*([0-9:.]+)",   block, re.IGNORECASE)
        hd_m = re.search(r"Scene\s+heading:\s*(.+)",     block, re.IGNORECASE)

        try:
            sc_start = ts_to_sec(ss_m.group(1)) if ss_m else 0.0
            sc_end   = ts_to_sec(se_m.group(1)) if se_m else 0.0
        except ValueError as e:
            episode.parse_errors.append(f"Scene {scene_num}: bad timestamp — {e}")
            sc_start, sc_end = 0.0, 0.0

        scene = Scene(
            number   = scene_num,
            heading  = hd_m.group(1).strip() if hd_m else "",
            start    = sc_start,
            end      = sc_end,
            raw_text = block,
        )

        # ── Dialogue lines ────────────────────────────────────────────────
        # Pattern: **CHARACTER (tone):** followed by timestamps + quoted text
        dlg_pat = re.compile(
            r"\*\*([A-Z][A-Z\s'\-\(\)\.]+?)\s*(?:\(([^)]*)\))?:?\*\*\s*\n"
            r"(?:Start:\s*\(([0-9:.]+)\)\s*\|\s*End:\s*\(([0-9:.]+)\)\s*\n)?"
            r"\"?([^*\n\"]{3,})\"?",
            re.MULTILINE
        )
        for dm in dlg_pat.finditer(block):
            char = dm.group(1).strip()
            # Skip camera/action labels
            if re.match(r"^(CAMERA|ACTION|SOUND FX|SOUND|ACTIONS AND DIALOGUE"
                        r"|ACTION AND DIALOGUE)$", char, re.IGNORECASE):
                continue
            tone = (dm.group(2) or "").strip()
            try:
                ts_s = ts_to_sec(dm.group(3)) if dm.group(3) else sc_start
                ts_e = ts_to_sec(dm.group(4)) if dm.group(4) else sc_end
            except ValueError:
                ts_s, ts_e = sc_start, sc_end
            text = (dm.group(5) or "").strip()
            if text:
                scene.dialogue.append(DialogueLine(
                    character=char, tone=tone, text=text,
                    start=ts_s, end=ts_e
                ))

        # ── Action / camera lines ─────────────────────────────────────────
        act_pat = re.compile(
            r"\*\*(?:CAMERA|ACTION):\*\*\s*\n"
            r"Start:\s*\(([0-9:.]+)\)\s*\|\s*End:\s*\(([0-9:.]+)\)\s*\n"
            r"(.+?)(?=\n\*\*|\Z)",
            re.DOTALL | re.MULTILINE
        )
        for am in act_pat.finditer(block):
            try:
                ts_s = ts_to_sec(am.group(1))
                ts_e = ts_to_sec(am.group(2))
            except ValueError:
                ts_s, ts_e = sc_start, sc_end
            desc = am.group(3).strip().replace("\n", " ")
            if desc:
                scene.actions.append(ActionLine(
                    description=desc, start=ts_s, end=ts_e
                ))

        episode.scenes.append(scene)

    return episode


def parse_txt_content(content: str, filename: str = "upload.txt") -> Optional[Episode]:
    """
    Parse screenplay from an already-decoded string (used by the Streamlit uploader).
    """
    return _parse_raw(content, filename)


def parse_directory(folder: str) -> List[Episode]:
    """
    Parse all .txt files in a folder, sorted by filename.
    Returns a list of Episode objects (a file holding several episodes yields several).
    """
    episodes = []
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))
    for fname in files:
        eps, _error = parse_txt_file_ex(os.path.join(folder, fname))
        episodes.extend(eps)
    return episodes
