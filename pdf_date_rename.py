#!/usr/bin/env python3
"""
Rename PDFs based on a date found in their text content.

Depends on `pdftotext` (poppler).

Behavior summary:
- Scan first page
- Extract candidate dates
- Score candidates
- Rename using YYYY-MM-DD[_GAZ|_ELEC].pdf
- Use checksum manifest to avoid reprocessing
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
from typing import Iterable, Optional

# -------------------------
# DATA STRUCTURES
# -------------------------

@dataclass
class Match:
    """Represents a detected date candidate."""
    date: dt.date
    source: str
    context: str


# Numeric date patterns.
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
    re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"),
]

# French month mapping (full names and common abbreviations).
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

MONTH_TOKEN = (
    r"(janvier|janv\.?|février|fevrier|fév\.?|fev\.?|fevr\.?|févr\.?|mars|"
    r"avril|avr\.?|mai|juin|juillet|juil\.?|août|aout|aoû\.?|aou\.?|"
    r"septembre|sept\.?|octobre|oct\.?|novembre|nov\.?|décembre|decembre|"
    r"déc\.?|dec\.?)"
)

MONTH_NAME_PATTERN = re.compile(
    rf"\b(\d{{1,2}})\s*{MONTH_TOKEN}\s*[\.\-/]?\s*(\d{{2,4}})\b",
    re.IGNORECASE,
)
MONTH_NAME_PATTERN_COMPACT = re.compile(
    rf"(\d{{1,2}}){MONTH_TOKEN}[\.\-/]?(\d{{2,4}})",
    re.IGNORECASE,
)
MONTH_YEAR_TOKEN_RE = re.compile(rf"{MONTH_TOKEN}[\.\-/]\d{{2}}", re.IGNORECASE)
BPLUS_AU_PATTERN = re.compile(
    rf"\bau\s+(\d{{1,2}})\s*{MONTH_TOKEN}\s*[\.\-/]?\s*(\d{{2,4}})\b",
    re.IGNORECASE,
)

KEYWORDS = [
    "facture",
    "facture de",
    "invoice",
    "date",
    "date de facture",
    "date d'emission",
    "date d'émission",
]

NEGATIVE_KEYWORDS = [
    "prochaine facture",
    "facture vous sera adressée",
    "facture vous sera adressee",
    "vous sera adressée vers",
    "vous sera adressee vers",
    "montant preleve le",
    "montant prélevé le",
    "sera preleve le",
    "sera prélevé le",
    "fin d'engagement",
]

NEGATIVE_REGEX = [
    re.compile(
        r"p[ée]riode\s+du\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+au\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"electricit[eé]\s+et\s+gaz\s+du\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+au\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        re.IGNORECASE,
    ),
]

STRONG_KEYWORDS = [
    "facture de",
    "facture du",
]


def normalize_year(y: int) -> int:
    """Convert 2-digit years to 2000+."""
    return 2000 + y if y < 100 else y


def build_date(y: int, m: int, d: int) -> Optional[dt.date]:
    """Safely build a date object."""
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def compact_text(text: str) -> str:
    """Lowercase and remove whitespace for fuzzy matching."""
    return re.sub(r"\s+", "", text.lower())


def normalize_numeric_text(text: str) -> str:
    """Normalize spaced numeric dates like '0 6 / 1 1 / 2 5'."""
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return re.sub(r"(?<=\d)\s*([/.\-])\s*(?=\d)", r"\1", text)


def _context(text: str, start: int, end: int, window: int = 35) -> str:
    """Return a normalized context snippet around a match."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right].replace("\n", " ")).strip()


def find_dates(text: str, day_first: bool) -> Iterable[Match]:
    """Extract candidate dates from a PDF text payload."""
    text_num = normalize_numeric_text(text)
    text_compact = compact_text(text)

    for pat in DATE_PATTERNS:
        for match in pat.finditer(text_num):
            parts = [int(match.group(i)) for i in range(1, match.lastindex + 1)]
            if pat is DATE_PATTERNS[1]:
                y, mo, d = parts
            elif day_first:
                d, mo, y = parts
            else:
                mo, d, y = parts
            date_obj = build_date(normalize_year(y), mo, d)
            if date_obj:
                yield Match(date_obj, "numeric", _context(text_num, match.start(), match.end()))

    for match in MONTH_NAME_PATTERN.finditer(text):
        d = int(match.group(1))
        mo = MONTHS_FR[match.group(2).lower()]
        y = normalize_year(int(match.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(date_obj, "month_name", _context(text, match.start(), match.end()))

    for match in MONTH_NAME_PATTERN_COMPACT.finditer(text_compact):
        d = int(match.group(1))
        mo = MONTHS_FR[match.group(2).lower()]
        y = normalize_year(int(match.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(
                date_obj,
                "month_name_compact",
                _context(text_compact, match.start(), match.end()),
            )


def score_match(match: Match) -> int:
    """Score a date candidate using nearby text context."""
    score = 0
    ctx = match.context.lower()
    ctx_compact = compact_text(match.context)

    for kw in NEGATIVE_KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score -= 5
    for rx in NEGATIVE_REGEX:
        if rx.search(ctx):
            score -= 5

    if len(list(MONTH_YEAR_TOKEN_RE.finditer(ctx_compact))) >= 3:
        score -= 10

    for kw in KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score += 3
    for kw in STRONG_KEYWORDS:
        if kw in ctx or compact_text(kw) in ctx_compact:
            score += 6

    today = dt.date.today()
    if match.date <= today:
        score += 1
    if match.date >= today.replace(year=today.year - 5):
        score += 1
    return score


def pick_best(matches: Iterable[Match]) -> Optional[Match]:
    """Pick the highest-confidence candidate date."""
    matches = list(matches)
    if not matches:
        return None

    by_date = {}
    for match in matches:
        score = score_match(match)
        entry = by_date.setdefault(match.date, {"total": 0, "best": match, "best_score": score})
        entry["total"] += score
        if score > entry["best_score"]:
            entry["best"] = match
            entry["best_score"] = score

    ranked = list(by_date.items())
    positive = [item for item in ranked if item[1]["total"] > 0]
    if positive:
        ranked = positive
    ranked.sort(key=lambda item: (item[1]["total"], item[1]["best_score"], item[0]), reverse=True)
    return ranked[0][1]["best"]


def has_carte_bplus(text: str) -> bool:
    """Detect Carte b+ statements."""
    return re.search(r"carte[s]?\s*b\+", text, re.IGNORECASE) is not None


def find_bplus_au_dates(text: str) -> Iterable[Match]:
    """Extract 'au XX month year' dates for Carte b+ statements."""
    for match in BPLUS_AU_PATTERN.finditer(text):
        d = int(match.group(1))
        mo = MONTHS_FR[match.group(2).lower()]
        y = normalize_year(int(match.group(3)))
        date_obj = build_date(y, mo, d)
        if date_obj:
            yield Match(date_obj, "bplus_au", _context(text, match.start(), match.end()))


def detect_category(context: str, text: str) -> str:
    """Detect _GAZ or _ELEC suffixes."""
    ctx = context.lower()
    ctx_compact = compact_text(context)

    if re.search(r"facture d[’']?\s*électricité|facture d[’']?\s*electricite", ctx):
        return "_ELEC"
    if re.search(r"facture de\s+gaz|facture du\s+gaz|facture d[’']?\s*gaz", ctx):
        return "_GAZ"
    if "électricité" in ctx or "electricite" in ctx:
        return "_ELEC"
    if "gaz" in ctx:
        return "_GAZ"
    if re.search(r"factured[’']?électricité|factured[’']?electricite", ctx_compact):
        return "_ELEC"
    if re.search(r"facturedegaz|facturedugaz|factured[’']?gaz", ctx_compact):
        return "_GAZ"

    text = text.lower()
    if re.search(r"facture d[’']?\s*électricité|facture d[’']?\s*electricite", text):
        return "_ELEC"
    if re.search(r"facture de\s+gaz|facture du\s+gaz|facture d[’']?\s*gaz", text):
        return "_GAZ"
    return ""


def safe_target_path_with_suffix(
    pdf_path: Path,
    target_date: dt.date,
    suffix: str,
    target_dir: Optional[Path],
) -> Path:
    """Build a collision-safe target path."""
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


# -------------------------
# CORE CLASS
# -------------------------

class PDFRenamer:
    """
    Main application class.

    Encapsulates:
    - argument handling
    - scanning logic
    - processing pipeline
    - renaming logic

    This allows reuse as a module or CLI tool.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Normalize paths early
        self.root = Path(args.path).expanduser().resolve()
        self.output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir else None
        )

        # Manifest (idempotency)
        self.manifest_path = (
            Path(args.manifest).expanduser().resolve()
            if args.manifest
            else (self.root / ".pdf_date_rename_manifest.json")
        )

        self.manifest = load_manifest(self.manifest_path)
        self.known_checksums = self.manifest.get("checksums", {})

    # -------------------------
    # ENTRYPOINT
    # -------------------------

    def run(self) -> int:
        """Execute the full workflow."""

        # --- Safety checks ---
        if not self.root.exists() or not self.root.is_dir():
            print(f"Path not found or not a directory: {self.root}", file=sys.stderr)
            return 2

        if self.output_dir and not self.output_dir.exists():
            print(f"Output dir not found: {self.output_dir}", file=sys.stderr)
            return 2

        day_first = not self.args.month_first

        if self.args.verbose and self.manifest_path.exists():
            print(f"- manifest found: {self.manifest_path}")

        # --- Processing ---
        rows = []
        for pdf in iter_pdfs(self.root, self.args.recursive):
            rows.append(self._process_pdf(pdf, day_first))

        # --- Preview ---
        self._preview(rows)

        if not self.args.apply:
            print("\nDry-run only. Use --apply to rename.")
            return 0

        # --- Apply ---
        self._apply(rows)

        save_manifest(self.manifest_path, {"checksums": self.known_checksums})
        return 0

    # -------------------------
    # PROCESSING
    # -------------------------

    def _process_pdf(self, pdf: Path, day_first: bool):
        """
        Process a single PDF file:
        - checksum
        - extract text
        - find best date
        - compute target filename
        """

        # --- Checksum ---
        try:
            checksum = sha256_file(pdf)
        except OSError as e:
            return (pdf, None, None, None, f"error: {e}")

        if checksum in self.known_checksums:
            if self.args.verbose:
                print(f"- {pdf.name}: already processed")
            return (pdf, None, None, checksum, "already processed")

        # --- Extract text ---
        try:
            text = run_pdftotext(pdf, first_page_only=True)
        except RuntimeError as e:
            return (pdf, None, None, checksum, f"error: {e}")

        # --- Find best date ---
        best = None

        if has_carte_bplus(text):
            matches = [
                m for m in find_bplus_au_dates(text)
                if self.args.min_year <= m.date.year <= self.args.max_year
            ]
            if matches:
                best = max(matches, key=lambda m: m.date)

        matches = [
            m for m in find_dates(text, day_first)
            if self.args.min_year <= m.date.year <= self.args.max_year
        ]

        if not best:
            best = pick_best(matches)

        if not best:
            return (pdf, None, None, checksum, "no date found")

        # --- Target ---
        suffix = detect_category(best.context, text)
        target = safe_target_path_with_suffix(pdf, best.date, suffix, self.output_dir)

        return (pdf, target, best, checksum, "ok")

    # -------------------------
    # OUTPUT
    # -------------------------

    def _preview(self, rows):
        """Display dry-run output."""
        for pdf, target, best, _, status in rows:
            if status != "ok":
                print(f"- {pdf.name}: {status}")
                continue
            print(f"- {pdf.name} -> {target.name} | {best.date.isoformat()}")

    # -------------------------
    # APPLY
    # -------------------------

    def _apply(self, rows):
        """Apply renaming operations."""
        for pdf, target, best, checksum, status in rows:
            if status != "ok":
                continue

            if pdf.resolve() == target.resolve():
                continue

            if target.exists():
                try:
                    target_checksum = sha256_file(target)
                except OSError:
                    print(f"Failed to checksum existing target {target.name}", file=sys.stderr)
                    continue

                if checksum == target_checksum:
                    self._handle_duplicate(pdf, target, checksum)
                    continue

                suffix = target.stem[len(best.date.isoformat()):]
                target = safe_target_path_with_suffix(pdf, best.date, suffix, self.output_dir)

            try:
                os.rename(pdf, target)
                self.known_checksums[checksum] = str(target)
            except OSError as e:
                print(f"Failed to rename {pdf.name}: {e}", file=sys.stderr)

    def _handle_duplicate(self, pdf: Path, target: Path, checksum: str):
        """Handle duplicate file detection."""
        if self.args.trash_dupes_no_ask:
            move_to_trash(pdf)
            self.known_checksums[checksum] = str(target)
            return

        if self.args.trash_duplicates:
            resp = input(f"Duplicate {pdf.name}. Trash? [y/N] ")
            if resp.lower() == "y":
                move_to_trash(pdf)
                self.known_checksums[checksum] = str(target)
                return

        print(f"Duplicate content detected; skipping {pdf.name}")
        self.known_checksums[checksum] = str(target)


# -------------------------
# UTILITIES
# -------------------------

def run_pdftotext(pdf_path: Path, first_page_only: bool = False) -> str:
    """Extract text from PDF using pdftotext."""
    cmd = ["pdftotext", "-layout"]
    if first_page_only:
        cmd += ["-f", "1", "-l", "1"]
    cmd += [str(pdf_path), "-"]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("pdftotext not found")

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_pdfs(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield PDF files."""
    yield from (root.rglob("*.pdf") if recursive else root.glob("*.pdf"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI argument parsing."""
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--month-first", action="store_true")
    p.add_argument("--output-dir")
    p.add_argument("--min-year", type=int, default=2000)
    p.add_argument("--max-year", type=int, default=2100)
    p.add_argument("--manifest")
    p.add_argument("--trash-duplicates", action="store_true")
    p.add_argument("--trash-dupes-no-ask", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def load_manifest(path: Path) -> dict:
    """Load manifest safely."""
    if not path.exists():
        return {"checksums": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"checksums": {}}


def save_manifest(path: Path, data: dict) -> None:
    """Atomic write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def move_to_trash(path: Path) -> Path:
    """Move file to macOS Trash."""
    trash = Path.home() / ".Trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / path.name
    if not target.exists():
        os.replace(path, target)
        return target

    i = 1
    while True:
        candidate = trash / f"{path.stem}-{i}{path.suffix}"
        if not candidate.exists():
            os.replace(path, candidate)
            return candidate
        i += 1


# -------------------------
# MAIN
# -------------------------

def main(argv: list[str]) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    return PDFRenamer(args).run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
