from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PDF_FILE = PROCESSED / "wag_cleaned.pdf"
MAPPING_FILE = PROCESSED / "page_mapping.csv"
DEDUCTIONS_FILE = PROCESSED / "wag_9_3_altalanos_hibak_levonasok.txt"
ELEMENTS_FILE = PROCESSED / "wag_elemtablazat_demo_100_elem.txt"
OUTPUT_FILE = PROCESSED / "rag_chunks.jsonl"

MAX_CHARS = 1200 #kezdetben ekkora chunkokat használok most, de ezt esetleg lehet majd később módosítani / más értékkel tesztelni
OVERLAP_CHARS = 180 #mivel 1200-as most egy chunk ezért 15% átfedés van így most ezzel a 180-al

#szövegtisztító fv
def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

#ez splitteli a szöveget a chunkokra (ha a beadott szöveg < a chunkhossznál, akkor egyben hagyja)

def split_long_text(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS
) -> list[str]:

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            while end < len(text) and not text[end].isspace():
                end += 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)

        if next_start > 0 and not text[next_start - 1].isspace():
            while next_start < end and not text[next_start].isspace():
                next_start += 1

        while next_start < len(text) and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


#a processed torna bírói kézikönyv feldolgozása -> csv is kell hozzá mert ugye ott van az oldalszám mapping
def load_pdf_chunks() -> list[dict]:
    mapping = pd.read_csv(MAPPING_FILE)
    records = []
    with fitz.open(PDF_FILE) as pdf:
        lookup = dict(zip(mapping.cleaned_page.astype(int), mapping.original_page.astype(int)))
        for i, page in enumerate(pdf, start=1):
            original_page = lookup.get(i)
            raw = normalize(page.get_text("text", sort=True))
            raw = re.sub(r"\n{3,}", "\n\n", raw)
            #egyetlen PDF-oldalból származó chunkok; nem keveredik két forrásoldal.
            for j, part in enumerate(split_long_text(raw), start=1):
                records.append({
                    "id": f"pdf_p{original_page:03d}_{j:03d}",
                    "text": part,
                    "metadata": {
                        "source": PDF_FILE.name,
                        "source_type": "pdf_rules",
                        "original_page": original_page,
                        "cleaned_page": i,
                        "section": None,
                    },
                })
    return records

#a txt-k feldolgozása -> elemtáblázat 100 elem demo + levonástáblázat -> itt minden adatsor egy darab chunk lesz, hogy ne keveredjen össze 
def load_txt_records(path: Path, source_type: str, prefix: str) -> list[dict]:
    result = []
    section = ""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = normalize(line)
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            section = line.strip("= ")
            continue
        #a demó TXT fejlécére nemkell index
        if " | " not in line:
            continue
        if source_type == "elements" and not line.startswith("Név:"):
            continue
        if source_type == "deductions" and not (line.startswith("Kategória:") or line.startswith("Hiba:")):
            continue

        original_page = None
        match = re.search(r"PDF\s*(\d+)\.\s*oldal", line, re.IGNORECASE)
        if match:
            original_page = int(match.group(1))
        elif source_type == "deductions":
            match = re.search(r"PDF\s*(\d+)(?:[–-]\d+)?\.\s*oldal", section, re.IGNORECASE)
            if match:
                original_page = int(match.group(1))
            if "PDF 33–34." in section and any(term in line for term in ("Jogosulatlan pódiumon", "Visszatérés a pódiumra", "Beszélgetés aktív bíróval", "Csapattagok nem megfelelő", "Nem egyforma tornadressz", "versenyt, mert nincs", "verseny meg nem engedett")):
                original_page = 34

        fields = {}
        for item in line.split(" | "):
            if ":" in item:
                key, value = item.split(":", 1)
                fields[key.strip().lower()] = value.strip()
        metadata = {
            "source": path.name,
            "source_type": source_type,
            "original_page": original_page,
            "section": section or None,
        }
        if source_type == "elements":
            metadata.update({
                "apparatus": fields.get("szer"),
                "element_type": fields.get("típus"),
                "element_code": fields.get("elem kódszáma"),
                "difficulty": fields.get("d érték"),
            })
        else:
            metadata.update({
                "category": fields.get("kategória"),
                "judge": fields.get("elbíráló"),
            })
        result.append({
            "id": f"{prefix}_{len(result) + 1:03d}",
            "text": line,
            "metadata": metadata,
        })
    return result


def main() -> None:
    chunks = (
        load_pdf_chunks()
        + load_txt_records(DEDUCTIONS_FILE, "deductions", "deduction")
        + load_txt_records(ELEMENTS_FILE, "elements", "element")
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    counts = Counter(c["metadata"]["source_type"] for c in chunks)
    print(f"Összes chunk: {len(chunks)}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"Mentve: {OUTPUT_FILE}")
    if chunks:
        print("\nelső chunk (előnézet):")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
