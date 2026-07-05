#!/usr/bin/env python3
"""
Free page-count survey for the top-10%-by-file-size PDFs — no Reducto calls,
just downloads + local PyMuPDF page counting, to inform an "extreme page
count" cutoff for routing between Reducto (default) and the legacy
extractor (fallback for oversized docs / when out of credits).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import fitz  # PyMuPDF

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
BOT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)"}

DOWNLOAD_DIR = "reducto_pilot/size_survey"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def sanitize_filename(title: str) -> str:
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean).strip('_')
    return clean[:150]


def download(url: str, filepath: str) -> bool:
    for headers in (BOT_HEADERS, BROWSER_HEADERS):
        try:
            resp = requests.get(url, headers=headers, timeout=60, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if os.path.getsize(filepath) > 1000:
                return True
        except Exception:
            continue
    return False


def main():
    docs = json.load(open("reducto_pilot/top10pct_docs.json"))
    results = []

    for i, d in enumerate(docs):
        cid, title, size_mb, url = d["id"], d["title"], d["size_mb"], d["url"]
        if not url:
            continue

        filename = f"{cid}_{sanitize_filename(title)}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        if not os.path.exists(filepath):
            ok = download(url, filepath)
            if not ok:
                print(f"[{i+1}/{len(docs)}] {cid}: DOWNLOAD FAILED")
                results.append({"id": cid, "title": title, "size_mb": size_mb, "pages": None})
                continue

        try:
            doc = fitz.open(filepath)
            pages = doc.page_count
            doc.close()
        except Exception as e:
            print(f"[{i+1}/{len(docs)}] {cid}: OPEN FAILED ({e})")
            results.append({"id": cid, "title": title, "size_mb": size_mb, "pages": None})
            continue

        print(f"[{i+1}/{len(docs)}] {cid}: {pages} pages, {size_mb:.1f} MB — {title[:50]}")
        results.append({"id": cid, "title": title, "size_mb": size_mb, "pages": pages})

    json.dump(results, open("reducto_pilot/size_survey_results.json", "w"), indent=1)
    print("\nDone.")


if __name__ == "__main__":
    main()
