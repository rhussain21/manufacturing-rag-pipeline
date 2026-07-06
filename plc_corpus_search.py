"""
Search over the PLC/Structured Text reference corpus for the Coding Agent.

The corpus lives in a separate project (plc_simulation/corpus/ — see that
project's CLAUDE.md for how it was verified and built), not inside this repo.
It's small (~130 files), so a full embedding/vector pipeline would be
overkill — keyword-overlap scoring against filename + content is enough to
find the right file(s) for a given question.
"""

import json
import subprocess
from pathlib import Path

PLC_CORPUS_DIRS = [
    Path("/Users/redwanhussain/Documents/ai-projects/plc_simulation/corpus/beckhoff_el34xx"),
    Path("/Users/redwanhussain/Documents/ai-projects/plc_simulation/corpus/iec_checker_fixtures"),
]


def _load_corpus_files():
    files = []
    for d in PLC_CORPUS_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.st")))
    return files


def _is_function_block(f: Path) -> bool:
    return f.name.startswith("FB_") or f.stem == "EL3443_DPM_CurrentOnly"


# Common English words that show up as noise in every file's comments —
# without filtering these, a query like "what does FB_EL3483 do" can lose
# to longer, unrelated files that just happen to contain "what"/"does" more
# often in their prose, drowning out the one token that actually matters.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "what", "how", "does", "do",
    "this", "which", "when", "where", "who", "why", "can", "could",
    "would", "should", "about", "into", "using", "use", "used", "vs",
    "file", "files", "code", "corpus",
}

# A big score boost when a query token exactly matches a file's stem
# (case-insensitive) — "explain FB_EL3483" naming an exact filename should
# never lose to incidental word overlap in unrelated, longer files.
_EXACT_STEM_MATCH_BOOST = 1000


def search_plc_corpus(query: str, top_k: int = 3) -> list:
    """Keyword-overlap search: score each file by how many query words appear
    in its filename or content, return the top_k matches with full content.
    Stopwords are filtered out and an exact filename match is boosted hard,
    so naming a specific file always surfaces it."""
    query_words = {w.lower() for w in query.split() if len(w) > 2} - _STOPWORDS

    scored = []
    for f in _load_corpus_files():
        content = f.read_text(encoding="utf-8", errors="replace")
        stem_lower = f.stem.lower()
        haystack = (stem_lower + " " + content).lower()

        score = sum(1 for w in query_words if w in haystack)
        if stem_lower in query_words:
            score += _EXACT_STEM_MATCH_BOOST

        if score > 0:
            scored.append((score, f, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"filename": f.name, "content": content, "score": score}
        for score, f, content in scored[:top_k]
    ]


def get_corpus_overview() -> str:
    """Deterministic, accurate inventory of the corpus — real counts and
    filenames computed from disk, not something the model should be asked
    to guess or count itself. Always available as context so "what's in
    here" questions get grounded in fact, not a keyword-search miss."""
    lines = []
    for d in PLC_CORPUS_DIRS:
        if not d.exists():
            continue
        files = sorted(d.glob("*.st"))
        if d.name == "beckhoff_el34xx":
            fbs = [f for f in files if _is_function_block(f)]
            dtypes = [f for f in files if f not in fbs]
            lines.append(
                f"beckhoff_el34xx/ — {len(files)} files total, real vendor code from "
                f"Beckhoff's official EL34xx_Sample repo (3-phase power measurement terminals):"
            )
            lines.append(f"  {len(fbs)} function blocks (the actual logic):")
            lines.extend(f"    - {f.name}" for f in fbs)
            lines.append(f"  {len(dtypes)} data type structs (no logic, just data shapes):")
            lines.extend(f"    - {f.name}" for f in dtypes)
        else:
            lines.append(
                f"{d.name}/ — {len(files)} files, small synthetic ST snippets from "
                f"iec-checker's test suite, each demonstrating one PLCopen coding-guideline rule:"
            )
            lines.extend(f"  - {f.name}" for f in files)
    return "\n".join(lines)


def get_all_function_blocks_content() -> str:
    """Full content of every function block (the 11 files with real logic,
    not the 90 supporting data-type structs) — small and bounded enough to
    always include in full, so the agent can produce a plain-English,
    grouped-by-device overview of what the programs actually do, not just
    list filenames. Data type structs are excluded here since there's no
    "what it does" to summarize for a plain data shape."""
    beckhoff_dir = next((d for d in PLC_CORPUS_DIRS if d.name == "beckhoff_el34xx"), None)
    if not beckhoff_dir or not beckhoff_dir.exists():
        return ""

    fbs = [f for f in sorted(beckhoff_dir.glob("*.st")) if _is_function_block(f)]
    parts = [f"[{f.name}]\n{f.read_text(encoding='utf-8', errors='replace')}" for f in fbs]
    return "\n\n---\n\n".join(parts)


def find_corpus_file(filename_hint: str) -> Path | None:
    """Find a corpus file by exact or near-exact filename match — used to
    resolve which file to run the best-practices checker against."""
    hint = filename_hint.lower().replace(".st", "")
    for f in _load_corpus_files():
        if f.stem.lower() == hint:
            return f
    for f in _load_corpus_files():
        if hint in f.stem.lower():
            return f
    return None


def read_corpus_file(file_path: Path) -> dict:
    """Wrap a resolved corpus file as a match dict, same shape as
    search_plc_corpus's results — used when a file was resolved by name
    (find_corpus_file) rather than by keyword search."""
    return {"filename": file_path.name, "content": file_path.read_text(encoding="utf-8", errors="replace"), "score": 1000}


def run_best_practices_check(file_path: Path) -> dict:
    """Run the real iec-checker static analyzer (PLCopen coding-guideline
    enforcement) against a file via Docker, and return its actual findings —
    not a model guess about whether code "looks right". No macOS prebuilt
    binary exists for this tool, so it runs via the amd64 image under
    Docker's emulation (same pattern as the openplc-runtime setup).

    iec-checker's parser only supports a matiec-compatible ST dialect — some
    genuine TwinCAT vendor syntax (pointer types, symbolic-constant array
    bounds) isn't supported and will fail to parse. That's a real limitation
    of the tool itself, not something to silently paper over by rewriting
    the code's actual type declarations to make the checker happy."""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--platform", "linux/amd64",
                "-v", f"{file_path.parent}:/src", "-w", "/src",
                "jubnzv1/iec-checker:nightly", "-o", "json", file_path.name,
            ],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"ok": False, "error": f"Could not run iec-checker: {e}"}

    try:
        findings = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": result.stderr or result.stdout or "no output from checker"}

    if findings and findings[0].get("id") == "ParserError":
        return {
            "ok": False,
            "error": (
                f"iec-checker couldn't parse this file: {findings[0]['msg']} "
                f"(line {findings[0]['linenr']}). This usually means the file uses "
                f"a TwinCAT vendor construct (pointer types, symbolic-constant array "
                f"bounds) outside iec-checker's supported ST dialect — a real tool "
                f"limitation, not a fixable code issue."
            ),
        }

    return {"ok": True, "findings": findings}
