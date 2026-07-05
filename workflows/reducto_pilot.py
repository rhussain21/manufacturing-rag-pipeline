#!/usr/bin/env python3
"""
Reducto Pilot — old (PyMuPDF/pdfplumber) vs. new (Reducto) extraction comparison.

Runs both extractors over a hand-picked set of PDFs (chosen from NB3's
retrieval-eval failure analysis — "anchor" docs — plus near-duplicate
false-positives and control docs, added in later passes), computes
comparison metrics using the pipeline's own existing quality gates, and
writes results into `reducto_pilot_results` in the main DuckDB.

Nothing here touches `content.transcript` or production tables — this is a
fully separate, reversible pilot (drop `reducto_pilot_results` to undo).

Usage:
    python workflows/reducto_pilot.py --bucket anchor
"""

import argparse
import json
import os
import re
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import requests
from dotenv import load_dotenv

load_dotenv(".env.mac", override=True)

from tools.pdf_extractor import PDFExtractor
from tools.reducto_extractor import ReductoExtractor
from etl.data_quality import DataQualityFilter

DB_PATH = "Database/industry_signals.db"
DOWNLOAD_DIR = "reducto_pilot/downloaded"
RAW_DIR = "reducto_pilot/raw"

# ── Anchor docs (from NB3 Section 6/2 failure analysis) ─────────────────────
ANCHOR_DOCS = [
    # content_id, url, title, bucket
    (139, "https://www.eso.org/projects/elt/develop/ifw/ifw-ll/sphinx/plc_config_guide/latexpdf/latex/plc_config_guide.pdf",
     "[PDF] Beckhoff PLC Configuration Guide | ESO", "anchor_miss"),
    (204, "https://assets-global.website-files.com/6861d06c5213bd6666fb9344/68b40b74a41a17d562ab9eb2_mumuxajalijojote.pdf",
     "[PDF] Beckhoff plc programming manual", "anchor_miss"),
    (405, "https://search.abb.com/library/Download.aspx?DocumentID=2TLC172003B02002",
     "[PDF] Safety in control systems according to EN ISO 13849-1 - ABB", "anchor_miss"),
    (922, "https://www.the-digital-insurer.com/wp-content/uploads/2020/01/1554-whitepaper-c11-742528.pdf",
     "[PDF] The Evolution of Industrial Cybersecurity and Cyber Risk White Paper", "anchor_miss"),
    (1039, "https://www.delltechnologies.com/asset/en-us/solutions/business-solutions/briefs-summaries/transforming-manufacturing-with-ai-and-edge-computing-ebook.pdf",
     "[PDF] Transforming Manufacturing with AI and Edge Computing - Dell", "anchor_miss"),
    (1045, "https://indico.cern.ch/event/1009424/contributions/4246135/attachments/2205833/3732073/20210310_Edge%20Computing%20and%20AI%20for%20Industrial%20Applications.pdf",
     "[PDF] Introduction to Siemens Industrial Edge - Managing AI Lifecycle", "anchor_miss"),
    (1029, "http://admin.mantechpublications.com/index.php/IJSCAI/article/download/3098/1700",
     "[PDF] Exploring Intelligent Edge Computing and AI for Industrial Automation", "anchor_freq"),
    (297, "https://assets.new.siemens.com/siemens/assets/api/uuid:1a9305ff-070c-4a21-97cb-33f16bcaada6/whitepaper-industrial-edge-en-0621.pdf",
     "[PDF] Industrial Edge: Real-time intelligence in process plants", "anchor_freq"),
    (932, "https://isagca.org/hubfs/2023%20ISA%20Website%20Redesigns/ISAGCA/PDFs/Industrial%20Cybersecurity%20Knowledge%20FINAL.pdf",
     "[PDF] CURRICULAR GUIDANCE: Industrial Cybersecurity Knowledge", "anchor_freq"),
    (1145, "https://literature.rockwellautomation.com/idc/groups/literature/documents/rm/secure-rm001_-en-p.pdf",
     "[PDF] System Security Design Guidelines - Literature Library", "anchor_freq"),
    (1550, "https://assets.new.siemens.com/siemens/assets/api/uuid:bac0b256-9895-4afe-8802-e7320c75a7f7/EdgeMgmt-PAC-RADAR-Platforms-Europe-2025-reprint-Siemens.pdf",
     "[PDF] Digital Platforms for Industrial Edge Management in Europe 2025", "anchor_freq"),
]

# ── Near-duplicate false-positive candidates (currently flagged by the ──
# SimHash gate) — picked to cover ambiguous cases: same-vendor manuals that
# share heavy template boilerplate but describe different products.
NEAR_DUP_DOCS = [
    (1556, "https://tlauk.net/document/71439/B%26R%2C%20X20SLX410.pdf",
     "[PDF] X20(c)SLXx1x - TLA Distribution", "near_dup"),
    (1557, "https://docs.rs-online.com/871d/A700000013921780.pdf",
     "[PDF] X20(c)SL81xx - RS-online.com", "near_dup"),
    (1747, "https://support.industry.siemens.com/cs/attachments/109767474/MC_ncprogramming_progr_man_0619_en-US.pdf",
     "[PDF] Programming Manual NC Programming - Support", "near_dup"),
    (1748, "https://support.industry.siemens.com/cs/attachments/24763376/PGZ_0108_en_en-US.pdf",
     "[PDF] Programming Manual Cycles PGZ - Support", "near_dup"),
    (1749, "https://cache.industry.siemens.com/dl/files/403/28755403/att_79171/v1/PGT_0407_en.pdf",
     "[PDF] Programming Manual ISO Turning SINUMERIK 802D sl840D", "near_dup"),
    (1300, "https://library.e.abb.com/public/af93d793dee8463a8429a5e6b79af105/EN_ACS880_CSG_A.pdf?x-sign=cloukNOwA5zNvQ%2FJ4Jf6c4odrejKF6hdklDEfzLNGygQpjyWNHtY6MM4dEbRdXvN",
     "ACS880 drives Cyber security guide", "near_dup"),
    (1312, "https://library.e.abb.com/public/b7c27988722d427f9dca5630ac0c2464/SSC600_eng_758920_ENa.pdf?x-sign=qN4iffUUT17eOdMgrXiAVHCbkFZLIHx8rfb9UNdH%2FsU%2BDhk8aF%2F%2BdvIOdEbsErNZ",
     "[PDF] ENGINEERING MANUAL SSC600 - ABB", "near_dup"),
    (1336, "https://library.e.abb.com/public/fc7813f0a7c647599e33e9e6c7b930b2/3BSE041389-600_B_en_System_800xA_6.0_System_Planning.pdf",
     "[PDF] 800xA System, Engineering Planning and Concepts - ABB", "near_dup"),
    (1416, "https://www.mitsubishifa.co.th/files/dl/sh080855engn.pdf",
     "[PDF] Safety Controller User's Manual", "near_dup"),
    (1421, "https://www.mitsubishifa.co.th/files/dl/bfp-a3715c.pdf",
     "[PDF] Hello ASSISTA Quick Set-up Guide", "near_dup"),
]

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def sanitize_filename(title: str) -> str:
    """Match PDFExtractor._download_pdf's exact sanitization so filenames stay
    consistent regardless of which download path (bot UA vs. browser UA) wins."""
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'\s+', '_', clean).strip('_')
    return clean[:180]


def download_with_fallback(url: str, title: str, pdf_path: str) -> Tuple[str, int]:
    """Download via PDFExtractor's bot UA first; retry with a browser UA on failure
    (some sites, e.g. Dell, block the default bot UA with a 403)."""
    extractor = PDFExtractor(pdf_dir=DOWNLOAD_DIR)
    result = extractor.download_and_extract(pdf_url=url, title=title)
    if result:
        return result["text"], result["page_count"]

    # Fallback: browser UA download straight to the expected path, then extract
    resp = requests.get(url, headers=BROWSER_HEADERS, timeout=60)
    resp.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(resp.content)
    return PDFExtractor.extract_text(pdf_path)


def run_bucket(docs, dqf: DataQualityFilter, con):
    for content_id, url, title, bucket in docs:
        print(f"\n=== {content_id}: {title[:60]} ===")

        filename_stem = f"{content_id}_{sanitize_filename(title)}"
        pdf_path = os.path.join(DOWNLOAD_DIR, f"{filename_stem}.pdf")

        # ── Download once, reused for both extractors ──
        if os.path.exists(pdf_path):
            old_text, old_pages = PDFExtractor.extract_text(pdf_path)
        else:
            try:
                old_text, old_pages = download_with_fallback(url, filename_stem, pdf_path)
            except Exception as e:
                print(f"  DOWNLOAD FAILED: {e}")
                continue

        print(f"  OLD: {old_pages} pages, {len(old_text)} chars")

        # ── Reducto extraction ──
        raw_path = os.path.join(RAW_DIR, f"{content_id}.json")
        if os.path.exists(raw_path):
            reducto_result = json.load(open(raw_path))
            print(f"  (using cached Reducto result at {raw_path})")
        else:
            reducto_result = ReductoExtractor.extract_full(pdf_path)
            if reducto_result:
                json.dump(reducto_result, open(raw_path, "w"), indent=1)

        if not reducto_result:
            print("  REDUCTO EXTRACTION FAILED")
            continue

        new_chunks = reducto_result["result"]["chunks"]
        new_text = "\n\n".join(c["content"] for c in new_chunks)
        new_pages = reducto_result["usage"]["num_pages"]
        credits = reducto_result["usage"]["credits"]
        print(f"  NEW: {new_pages} pages, {len(new_text)} chars, {len(new_chunks)} chunks, {credits} credits")

        # ── Comparison metrics (reusing existing DQ gates) ──
        _, old_boil_msg = dqf.gate_boilerplate(old_text)
        _, new_boil_msg = dqf.gate_boilerplate(new_text)
        _, old_div_msg = dqf.gate_token_diversity(old_text)
        _, new_div_msg = dqf.gate_token_diversity(new_text)

        old_screen = dqf.screen(old_text)
        new_screen = dqf.screen(new_text)

        old_simhash = dqf.simhash(old_text)
        new_simhash = dqf.simhash(new_text)

        def extract_ratio(msg):
            m = re.search(r'([\d.]+)%', msg)
            return float(m.group(1)) / 100 if m else None

        def extract_diversity(msg):
            m = re.search(r':\s*([\d.]+)', msg)
            return float(m.group(1)) if m else None

        con.execute("""
            INSERT OR REPLACE INTO reducto_pilot_results (
                content_id, title, bucket,
                old_char_count, new_char_count, old_page_count, new_page_count,
                old_boilerplate_ratio, new_boilerplate_ratio,
                old_token_diversity, new_token_diversity,
                dq_pass_old, dq_fail_reason_old, dq_pass_new, dq_fail_reason_new,
                old_simhash, new_simhash, reducto_chunk_count, reducto_credits, reducto_raw_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            content_id, title, bucket,
            len(old_text), len(new_text), old_pages, new_pages,
            extract_ratio(old_boil_msg), extract_ratio(new_boil_msg),
            extract_diversity(old_div_msg), extract_diversity(new_div_msg),
            old_screen["pass"], old_screen.get("reason"), new_screen["pass"], new_screen.get("reason"),
            old_simhash, new_simhash, len(new_chunks), credits, raw_path,
        ])
        print(f"  DQ pass — old: {old_screen['pass']} ({old_screen.get('reason')})  "
              f"new: {new_screen['pass']} ({new_screen.get('reason')})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", choices=["anchor", "near_dup", "control", "all"], default="anchor")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    con = duckdb.connect(DB_PATH)
    dqf = DataQualityFilter()

    if args.bucket in ("anchor", "all"):
        run_bucket(ANCHOR_DOCS, dqf, con)
    if args.bucket in ("near_dup", "all"):
        run_bucket(NEAR_DUP_DOCS, dqf, con)

    con.close()
    print("\nDone. Query results: SELECT * FROM reducto_pilot_results")


if __name__ == "__main__":
    main()
