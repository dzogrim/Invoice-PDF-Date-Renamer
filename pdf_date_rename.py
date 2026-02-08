#!/usr/bin/env python3
"""Rename PDFs based on a date found in their text content.

Depends on `pdftotext` (poppler). On macOS:
  ```brew install poppler```
  or MacPorts:
  ```sudo port install poppler```

Behavior summary:
- Scans only the first page of each PDF (invoice title and date is usually there).
- Extracts candidate dates (numeric and French month formats, including abbreviations).
- Scores candidates using positive/negative context rules.
- Chooses the best candidate and renames to YYYY-MM-DD[_GAZ|_ELEC].pdf.
- Uses a checksum manifest to avoid reprocessing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Numeric date patterns (day-first by default).
DATE_PATTERNS = [
    # dd/mm/yy or dd/mm/yyyy or dd-mm-yy
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
    # yyyy-mm-dd or yyyy/mm/dd
    re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"),
]

# French month mapping (full names and abbreviations, accented and unaccented).
MONTHS_FR = {
    "janvier": 1,
    "janv": 1,
    "janv.": 1,
    "fevrier": 2,
    "février": 2,
    "fev": 2,
    "fev.": 2,
    "fevr": 2,
    "fevr.": 2,
    "févr": 2,
    "févr.": 2,
    "fév": 2,
    "fév.": 2,
    "mars": 3,
    "avril": 4,
    "avr": 4,
    "avr.": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "juil": 7,
    "juil.": 7,
    "aout": 8,
    "août": 8,
    "aou": 8,
    "aou.": 8,
    "aoû": 8,
    "aoû.": 8,
    "septembre": 9,
    "sept": 9,
    "sept.": 9,
    "octobre": 10,
    "oct": 10,
    "oct.": 10,
    "novembre": 11,
    "nov": 11,
    "nov.": 11,
    "decembre": 12,
    "décembre": 12,
    "dec": 12,
    "dec.": 12,
    "déc": 12,
    "déc.": 12,
}

# Regex token for all French month variants.
MONTH_TOKEN = (
    r"(janvier|janv\.?|février|fevrier|fév\.?|fev\.?|fevr\.?|févr\.?|mars|avril|avr\.?|mai|juin|juillet|juil\.?"
    r"|août|aout|aoû\.?|aou\.?|septembre|sept\.?|octobre|oct\.?|novembre|nov\.?|décembre|decembre|déc\.?|dec\.?)"
)

# "20 févr. 2024" style (with optional separators).
MONTH_NAME_PATTERN = re.compile(
    rf"\b(\d{{1,2}})\s*{MONTH_TOKEN}\s*[\.\-/]?\s*(\d{{2,4}})\b",
    re.IGNORECASE,
)

# Same as above but on whitespace-stripped text to handle spaced letters.
MONTH_NAME_PATTERN_COMPACT = re.compile(
    rf"(\d{{1,2}}){MONTH_TOKEN}[\.\-/]?(\d{{2,4}})",
    re.IGNORECASE,
)

# Month-year token detector for "févr-24 avr-24 ..." style lists.
MONTH_YEAR_TOKEN_RE = re.compile(
    rf"{MONTH_TOKEN}[\.\-/]\d{{2}}",
    re.IGNORECASE,
)

# "au 25 mai 2025" style for Carte(s) b+ statements.
BPLUS_AU_PATTERN = re.compile(
    rf"\bau\s+(\d{{1,2}})\s*{MONTH_TOKEN}\s*[\.\-/]?\s*(\d{{2,4}})\b",
    re.IGNORECASE,
)

# Positive keywords: boost candidates near invoice-specific labels.
KEYWORDS = [
    "facture",
    "facture de",
    "invoice",
    "date",
    "date de facture",
    "date d'émission",
    "date d'emission",
]

# Negative keywords: penalize non-invoice dates (future invoice notice, meter reading, payment dates).
NEGATIVE_KEYWORDS = [
    "prochaine facture",
    "facture vous sera adressée",
    "facture vous sera adressee",
    "vous sera adressée vers",
    "vous sera adressee vers",
    "votre compteur est prévue",
    "votre compteur est prevue",
    "prévue vers le",
    "prevue vers le",
    "montant preleve le",
    "montant prélevé le",
    "sera preleve le",
    "sera prélevé le",
    "rental for",
    "mon avantage offre",
    "fin d'engagement",
]

# Negative regex patterns: structured sequences that look like billing periods or lists.
NEGATIVE_REGEX = [
    re.compile(
        r"electricit[eé]\s+et\s+gaz\s+du\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+au\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"p[ée]riode\s+du\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+au\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        re.IGNORECASE,
    ),
    # Lists of month-year spans (e.g., "avr-24 juin-24 ... févr-25 mon avantage offre")
    re.compile(
        r"(janv|janvier|f[ée]vr|fevr|fev|mars|avr|avril|mai|juin|juil|juillet|ao[uû]|aout|sept|septembre|oct|octobre|nov|novembre|d[ée]c|dec|d[ée]cembre|decembre)[-./]\d{2}\\s+"
        r"(janv|janvier|f[ée]vr|fevr|fev|mars|avr|avril|mai|juin|juil|juillet|ao[uû]|aout|sept|septembre|oct|octobre|nov|novembre|d[ée]c|dec|d[ée]cembre|decembre)[-./]\d{2}.*?mon\\s+avantage\\s+offre",
        re.IGNORECASE,
    ),
    # Any line with multiple month-year tokens like "févr-24 avr-24 juin-24 ..."
    re.compile(
        r"(?:janv|janvier|f[ée]vr|fevr|fev|mars|avr|avril|mai|juin|juil|juillet|ao[uû]|aout|sept|septembre|oct|octobre|nov|novembre|d[ée]c|dec|d[ée]cembre|decembre)[-./]\d{2}"
        r"(?:\\s+|\\s*[/,-]\\s*)(?:janv|janvier|f[ée]vr|fevr|fev|mars|avr|avril|mai|juin|juil|juillet|ao[uû]|aout|sept|septembre|oct|octobre|nov|novembre|d[ée]c|dec|d[ée]cembre|decembre)[-./]\d{2}"
        r"(?:\\s+|\\s*[/,-]\\s*)(?:janv|janvier|f[ée]vr|fevr|fev|mars|avr|avril|mai|juin|juil|juillet|ao[uû]|aout|sept|septembre|oct|octobre|nov|novembre|d[ée]c|dec|d[ée]cembre|decembre)[-./]\d{2}",
        re.IGNORECASE,
    ),
]

# Strong keywords: highly likely to be the invoice title line.
STRONG_KEYWORDS = [
    "facture de",
    "facture du",
]


@dataclass
class Match:
    date: dt.date
    source: str
    context: str


def run_pdftotext(pdf_path: Path, first_page_only: bool = False) -> str:
    """Extract text from a PDF using pdftotext."""
    try:
        cmd = ["pdftotext", "-layout"]
        if first_page_only:
            cmd += ["-f", "1", "-l", "1"]
        cmd += [str(pdf_path), "-"]
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("pdftotext not found. Install poppler (brew install poppler).")

    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {pdf_path.name}: {result.stderr.strip()}")
    return result.stdout


def normalize_year(y: int) -> int:
    """Convert 2-digit years to 2000+."""
    if y < 100:
        return 2000 + y
    return y


def build_date(y: int, m: int, d: int) -> Optional[dt.date]:
    """Safely build a date, returning None if invalid."""
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str, day_first: bool) -> Iterable[Match]:
    """Extract candidate dates from text, including compacted variants."""
    text_num = normalize_numeric_text(text)
    text_compact = compact_text(text)
    # Numeric dates
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text_num):
            parts = [int(m.group(i)) for i in range(1, m.lastindex + 1)]
            if len(parts) != 3:
                continue

            if pat is DATE_PATTERNS[1]:
                y, mo, d = parts
            else:
                if day_first:
                    d, mo, y = parts
                else:
                    mo, d, y = parts
            y = normalize_year(y)
            date_obj = build_date(y, mo, d)
            if date_obj:
                yield Match(date_obj, "numeric", _context(text_num, m.start(), m.end()))

    # French month names
    for m in MONTH_NAME_PATTERN.finditer(text):
        d = int(m.group(1))
        mo = MONTHS_FR[m.group(2).lower()]
        y = normalize_year(int(m.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(date_obj, "month_name", _context(text, m.start(), m.end()))

    # Month names in compacted text (handles spaced letters like "d at e")
    for m in MONTH_NAME_PATTERN_COMPACT.finditer(text_compact):
        d = int(m.group(1))
        mo = MONTHS_FR[m.group(2).lower()]
        y = normalize_year(int(m.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(date_obj, "month_name_compact", _context(text_compact, m.start(), m.end()))


def _context(text: str, start: int, end: int, window: int = 35) -> str:
    """Return a short, normalized context snippet around a match."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ")
    return re.sub(r"\s+", " ", snippet).strip()

def compact_text(text: str) -> str:
    """Lowercase and remove all whitespace for robust matching."""
    return re.sub(r"\s+", "", text.lower())

def normalize_numeric_text(text: str) -> str:
    """Normalize spaced numeric dates like '0 6 / 1 1 / 2 5' -> '06/11/25'."""
    t = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    t = re.sub(r"(?<=\d)\s*([/.\-])\s*(?=\d)", r"\1", t)
    return t


def score_match(m: Match) -> int:
    """Score a date candidate based on context and plausibility."""
    score = 0
    ctx = m.context.lower()
    ctx_compact = compact_text(m.context)
    for kw in NEGATIVE_KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score -= 5
    for rx in NEGATIVE_REGEX:
        if rx.search(ctx):
            score -= 5
    # Penalize lines that look like month-year lists (e.g., "févr-24 avr-24 juin-24 ...")
    month_year_hits = len(list(MONTH_YEAR_TOKEN_RE.finditer(ctx_compact)))
    if month_year_hits >= 3:
        score -= 10
    for kw in KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score += 3
    for kw in STRONG_KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score += 6
    # Prefer plausible invoice dates (not future, not too old)
    today = dt.date.today()
    if m.date <= today:
        score += 1
    if m.date >= today.replace(year=today.year - 5):
        score += 1
    return score


def pick_best(matches: Iterable[Match]) -> Optional[Match]:
    """Pick the best-scoring candidate, preferring positive scores."""
    matches = list(matches)
    if not matches:
        return None
    filtered = [m for m in matches if score_match(m) > 0]
    if filtered:
        matches = filtered
    matches.sort(key=lambda m: (score_match(m), m.date), reverse=True)
    return matches[0]

def has_carte_bplus(text: str) -> bool:
    """Detect Carte(s) b+ statements."""
    return re.search(r"carte[s]?\s*b\+", text, re.IGNORECASE) is not None

def find_bplus_au_dates(text: str) -> Iterable[Match]:
    """Extract dates from 'au XX mois YYYY' patterns for Carte b+ statements."""
    for m in BPLUS_AU_PATTERN.finditer(text):
        d = int(m.group(1))
        mo = MONTHS_FR[m.group(2).lower()]
        y = normalize_year(int(m.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(date_obj, "bplus_au", _context(text, m.start(), m.end()))


def safe_target_path(pdf_path: Path, target_date: dt.date, target_dir: Optional[Path]) -> Path:
    """Resolve a collision-safe target name using YYYY-MM-DD."""
    base_dir = target_dir if target_dir else pdf_path.parent
    base_name = target_date.isoformat()
    candidate = base_dir / f"{base_name}.pdf"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = base_dir / f"{base_name}-{i}.pdf"
        if not candidate.exists():
            return candidate
        i += 1

def detect_category(context: str, text: str) -> str:
    """Detect invoice category for suffix (_GAZ/_ELEC) from context then full text."""
    ctx = context.lower()
    ctx_compact = compact_text(context)
    # Prefer the matched line/context first.
    if re.search(r"facture d[’']?\\s*électricité|facture d[’']?\\s*electricite", ctx):
        return "_ELEC"
    if re.search(r"facture de\\s+gaz|facture du\\s+gaz|facture d[’']?\\s*gaz", ctx):
        return "_GAZ"
    if "électricité" in ctx or "electricite" in ctx:
        return "_ELEC"
    if "gaz" in ctx:
        return "_GAZ"
    if re.search(r"factured[’']?électricité|factured[’']?electricite", ctx_compact):
        return "_ELEC"
    if re.search(r"facturedegas|facturedugas|factured[’']?gaz", ctx_compact):
        return "_GAZ"
    if "électricité" in ctx_compact or "electricite" in ctx_compact:
        return "_ELEC"
    if "gaz" in ctx_compact:
        return "_GAZ"

    # Fallback to full text if context doesn't mention it.
    t = text.lower()
    if re.search(r"facture d[’']?\\s*électricité|facture d[’']?\\s*electricite", t):
        return "_ELEC"
    if re.search(r"facture de\\s+gaz|facture du\\s+gaz|facture d[’']?\\s*gaz", t):
        return "_GAZ"
    return ""

def safe_target_path_with_suffix(
    pdf_path: Path, target_date: dt.date, suffix: str, target_dir: Optional[Path]
) -> Path:
    """Resolve a collision-safe target name using YYYY-MM-DD + suffix."""
    base_dir = target_dir if target_dir else pdf_path.parent
    base_name = f"{target_date.isoformat()}{suffix}"
    candidate = base_dir / f"{base_name}.pdf"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = base_dir / f"{base_name}-{i}.pdf"
        if not candidate.exists():
            return candidate
        i += 1


def iter_pdfs(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield PDF paths from a folder."""
    if recursive:
        yield from root.rglob("*.pdf")
    else:
        yield from root.glob("*.pdf")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI argument parsing."""
    p = argparse.ArgumentParser(description="Rename PDFs based on date found in their text.")
    p.add_argument("path", help="Folder containing PDFs to scan")
    p.add_argument("--apply", action="store_true", help="Actually rename files (default is dry-run)")
    p.add_argument("--recursive", action="store_true", help="Scan subfolders")
    p.add_argument("--day-first", action="store_true", default=True, help="Parse numeric dates as DD/MM/YY")
    p.add_argument("--month-first", action="store_true", help="Parse numeric dates as MM/DD/YY")
    p.add_argument("--output-dir", help="Optional folder to write renamed files")
    p.add_argument("--min-year", type=int, default=2000, help="Ignore dates earlier than this year")
    p.add_argument("--max-year", type=int, default=2100, help="Ignore dates later than this year")
    p.add_argument(
        "--manifest",
        help="Path to manifest JSON (default: .pdf_date_rename_manifest.json in scan root)",
    )
    p.add_argument(
        "--trash-duplicates",
        action="store_true",
        help="If a duplicate content is detected, ask to move the source to Trash",
    )
    p.add_argument(
        "--trash-dupes-no-ask",
        action="store_true",
        help="Move duplicate content to Trash without confirmation (implies --trash-duplicates)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose output for debugging")
    return p.parse_args(argv)

def sha256_file(path: Path) -> str:
    """Compute SHA-256 for content-based deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(path: Path) -> dict:
    """Load manifest JSON (checksums) or return empty structure."""
    if not path.exists():
        return {"checksums": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"checksums": {}}

def save_manifest(path: Path, data: dict) -> None:
    """Atomic manifest write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def move_to_trash(path: Path) -> Path:
    """Move a file to macOS Trash with collision-safe naming."""
    trash_dir = Path.home() / ".Trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    candidate = trash_dir / path.name
    if not candidate.exists():
        os.replace(path, candidate)
        return candidate
    i = 1
    while True:
        candidate = trash_dir / f"{path.stem}-{i}{path.suffix}"
        if not candidate.exists():
            os.replace(path, candidate)
            return candidate
        i += 1


def main(argv: list[str]) -> int:
    """Main program entry point."""
    args = parse_args(argv)
    root = Path(args.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Path not found or not a directory: {root}", file=sys.stderr)
        return 2

    day_first = True
    if args.month_first:
        day_first = False

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir and not output_dir.exists():
        print(f"Output dir not found: {output_dir}", file=sys.stderr)
        return 2

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else (root / ".pdf_date_rename_manifest.json")
    )
    if args.verbose and manifest_path.exists():
        print(f"- manifest found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    known_checksums = manifest.get("checksums", {})

    rows = []
    for pdf in iter_pdfs(root, args.recursive):
        checksum = None
        try:
            checksum = sha256_file(pdf)
        except OSError as e:
            rows.append((pdf, None, None, None, f"error: {e}"))
            continue

        if checksum in known_checksums:
            if args.verbose:
                print(f"- {pdf.name}: already processed (checksum={checksum[:12]})")
            rows.append((pdf, None, None, checksum, "already processed"))
            continue

        try:
            text = run_pdftotext(pdf, first_page_only=True)
        except RuntimeError as e:
            rows.append((pdf, None, None, checksum, f"error: {e}"))
            continue

        best = None
        if has_carte_bplus(text):
            bplus_matches = [
                m for m in find_bplus_au_dates(text)
                if args.min_year <= m.date.year <= args.max_year
            ]
            if args.verbose:
                print(f"- {pdf.name}: Carte b+ detected, found {len(bplus_matches)} 'au' candidates")
                for m in bplus_matches[:10]:
                    print(f"  - {m.date.isoformat()} | source={m.source} | {m.context}")
            if bplus_matches:
                best = max(bplus_matches, key=lambda m: m.date)

        matches = [m for m in find_dates(text, day_first) if args.min_year <= m.date.year <= args.max_year]
        if args.verbose:
            print(f"- {pdf.name}: found {len(matches)} date candidates")
            for m in matches[:10]:
                print(f"  - {m.date.isoformat()} | score={score_match(m)} | {m.context}")
        if not best:
            best = pick_best(matches)
        if not best:
            rows.append((pdf, None, None, checksum, "no date found"))
            continue

        suffix = detect_category(best.context, text)
        target = safe_target_path_with_suffix(pdf, best.date, suffix, output_dir)
        rows.append((pdf, target, best, checksum, "ok"))

    # Preview
    for pdf, target, best, checksum, status in rows:
        if status != "ok":
            print(f"- {pdf.name}: {status}")
            continue
        print(f"- {pdf.name} -> {target.name} | {best.date.isoformat()} | {best.context}")

    if not args.apply:
        print("\nDry-run only. Use --apply to rename.")
        return 0

    # Apply renames
    for pdf, target, best, checksum, status in rows:
        if status != "ok":
            continue
        if pdf.resolve() == target.resolve():
            continue
        if target.exists():
            try:
                target_checksum = sha256_file(target)
            except OSError as e:
                print(f"Failed to checksum existing target {target.name}: {e}", file=sys.stderr)
                continue
            if checksum == target_checksum:
                if args.trash_dupes_no_ask:
                    trashed = move_to_trash(pdf)
                    print(f"Moved to Trash: {trashed.name}")
                    known_checksums[checksum] = str(target)
                    continue
                if args.trash_duplicates:
                    resp = input(
                        f"Duplicate content for {pdf.name} and {target.name}. Move source to Trash? [y/N] "
                    )
                    if resp.strip().lower() == "y":
                        trashed = move_to_trash(pdf)
                        print(f"Moved to Trash: {trashed.name}")
                        known_checksums[checksum] = str(target)
                        continue
                print(f"Duplicate content detected; skipping {pdf.name}")
                known_checksums[checksum] = str(target)
                continue

        try:
            os.rename(pdf, target)
            known_checksums[checksum] = str(target)
        except OSError as e:
            print(f"Failed to rename {pdf.name}: {e}", file=sys.stderr)

    save_manifest(manifest_path, {"checksums": known_checksums})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
