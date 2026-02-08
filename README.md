# Invoice-PDF-Date-Renamer

Rename (French) PDF files based on a date found inside the PDF text. Designed for macOS and works with Homebrew or MacPorts.

This tool is optimized for French utility invoices (GAZ/ELECTRICITÉ) but remains generic. It:
- Extracts text from PDFs (1st page only).
  - Optimized and tested for Oney, Sosh, Orange, OVHcloud, TotalEnergies… invoices.
- Finds candidate dates (numeric and French month formats, including abbreviations).
- Scores candidates with positive/negative context rules.
- Renames to `YYYY-MM-DD[_GAZ|_ELEC].pdf` with collision-safe suffixes.
- Tracks processed files via checksum manifest to avoid reprocessing.

## Install prerequisites

Homebrew:
```bash
brew install poppler
```

MacPorts:
```bash
sudo port install poppler
```

## Usage

Dry-run (preview only):
```bash
python pdf_date_rename.py \
  "/path/to/folder"
```

Apply rename:
```bash
python pdf_date_rename.py \
  "/path/to/folder" \
  --apply
```

Verbose debug:
```bash
python pdf_date_rename.py \
  "/path/to/folder" \
  --verbose
```

Enable duplicate handling (same content, same target name) and ask to move source to Trash:
```bash
python pdf_date_rename.py \
  "/path/to/folder" \
  --apply \
  --trash-duplicates
```

Move duplicates to Trash without confirmation:
```bash
python pdf_date_rename.py \
  "/path/to/folder" \
  --apply \
  --trash-dupes-no-ask
```

Recursive scan:
```bash
python pdf_date_rename.py \
  "/path/to/folder" \
  --recursive
```

If your PDFs use US-style dates (MM/DD/YY), add:
```bash
--month-first
```

Reprocess a folder (ignore existing manifest) by pointing to a new manifest:
```bash
--manifest /tmp/rename_manifest.json
```

## Notes

- Filenames are sanitized by using a safe ISO date: `YYYY-MM-DD.pdf`.
- If a file with the target name already exists, a numeric suffix is added: `YYYY-MM-DD-1.pdf`.
- Default date parsing expects DD/MM/YY (French invoices).
- A manifest file is created in the scan root to avoid reprocessing the same content:
  `.pdf_date_rename_manifest.json` (override with `--manifest`).
- Category suffixes:
  - `_GAZ` if the invoice is for gas.
  - `_ELEC` if the invoice is for electricity.
- The tool scans only the first page of each PDF, which is where invoice titles usually appear.
- If no high-confidence date is found, the file is left unchanged (dry-run shows `no date found`).
