"""
PLC Expert — explains existing Structured Text / PLC code, grounded in a
real, individually-verified corpus (Beckhoff vendor samples + iec-checker
PLCopen-rule fixtures) rather than trusting the model's thin training-data
coverage of IEC 61131-3, a niche industrial domain. Also checks code against
real PLCopen coding guidelines via iec-checker (an actual static analyzer,
not the model's opinion on whether code "looks right") — the same
tools-over-intuition principle used elsewhere in this system (compilers as
ground truth for generated code) applies here for style/guideline checks.

Explaining and best-practices checking are both read-only. Generating or
modifying code is a separate, later capability that needs the OpenPLC
compile+run test loop as a safety net (see plc_simulation/CLAUDE.md for why).
"""

from langsmith import traceable

from agents.state import AgentState
from agents.conversation_context import format_recent_history
from agents.structured_answer import JSON_INSTRUCTION, parse_structured_answer
from plc_corpus_search import (
    search_plc_corpus, get_corpus_overview, get_all_function_blocks_content,
    run_best_practices_check, find_corpus_file, read_corpus_file,
)

# Running iec-checker means shelling out to Docker (a few seconds), so this
# only fires when the query actually looks like a best-practices/guideline
# request — not on every question about code.
_BEST_PRACTICES_TRIGGERS = (
    "best practice", "best-practice", "guideline", "compliant", "compliance",
    "violat", "coding standard", "plcopen", "review this", "analyze this",
    "check this", "audit",
)

SYSTEM_PROMPT = (
    "You are a PLC subject-matter expert (SME) reviewing IEC 61131-3 "
    "Structured Text code for someone less familiar with it. You're given "
    "three kinds of context: a CORPUS OVERVIEW (an accurate, deterministic "
    "inventory of every file available), MATCHED FILE CONTENT (full text of "
    "files that seem relevant to the specific question), and ALL FUNCTION "
    "BLOCKS (full content of every program with real logic in the corpus — "
    "use this specifically for orientation/overview-style questions, e.g. "
    "'what do these programs do', 'summarize the code', 'what's here').\n\n"
    "For an overview-style question, be VERY terse — this is a first "
    "orientation, not a walkthrough. One line per group, in exactly this "
    "shape: '<N> programs <do X in plain English>.' Nothing more per group: "
    "no filenames, no data-type counts, no sub-bullets, no closing summary "
    "paragraph tying it all together. The whole answer should be a short "
    "list — purely as a illustration of the FORMAT (not the actual content, "
    "which you must derive fresh from ALL FUNCTION BLOCKS every time, since "
    "the real corpus and its real groupings are given to you separately "
    "below and may not match this made-up example at all):\n"
    "  - 4 programs monitor tank fill level and trigger a valve shutoff.\n"
    "  - 2 programs run a conveyor belt on a fixed timer cycle.\n"
    "  - 1 program handles emergency-stop interlocks across the whole line.\n"
    "That's it — end there. Don't add filenames beyond a short parenthetical "
    "naming the terminal/device model if one's relevant, don't mention data "
    "type file counts, don't add a wrap-up paragraph. Code-level specifics "
    "(variable names, exact logic, individual filenames) only belong in a "
    "follow-up once the user asks about a specific group or file by name.\n\n"
    "For a narrow question about one specific file or concept: answer from "
    "MATCHED FILE CONTENT directly, skip the full-corpus grouping exercise.\n\n"
    "The rule below applies per file, not per question — a question that "
    "gets SOME real matches can still name or reference OTHER files that "
    "weren't matched (e.g. because they were only seen in the CORPUS "
    "OVERVIEW's listing). For any file whose full content was NOT given to "
    "you in MATCHED FILE CONTENT: do NOT speculate about what it probably "
    "does based on its name or its category in the overview — that's "
    "confident-sounding guessing wearing a caveat as a disguise, not real "
    "hedging, even when it happens right next to a real explanation of a "
    "different file that WAS matched. Just say plainly you don't have that "
    "file's content and stop there — don't fill the gap with name-based "
    "inference about what its 'likely' logic or structure is.\n\n"
    "When you DO have real matched content: explain it clearly and concisely "
    "— what it does, what domain concepts it involves (e.g. power "
    "measurement, safety interlocks, PLCopen coding guidelines), how its "
    "pieces fit together. Default to a tight, focused answer (a few short "
    "paragraphs); only go longer if the question specifically asks for "
    "exhaustive detail. If a referenced type or function block isn't in the "
    "provided context, say so in one line rather than guessing at it.\n\n"
    "If BEST PRACTICES CHECK RESULT is present: this came from iec-checker, "
    "a real static analyzer — not your own opinion. Report its actual "
    "findings, don't second-guess or add your own separate style opinions "
    "on top. For each finding, explain in plain English what the rule means "
    "and why it matters, referencing the specific line/variable it flagged. "
    "If the result says the file couldn't be parsed, say so plainly and "
    "explain why (usually a vendor-specific construct outside the checker's "
    "supported dialect) — don't try to guess what issues the code might "
    "have instead; that would be exactly the guessing-instead-of-hedging "
    "problem this whole system is designed to avoid.\n\n"
    "You're also given RECENT CONVERSATION — real prior turns in this "
    "session. If the question is about the conversation itself (what did I "
    "just ask, what's the first question, what have we covered), answer "
    "from that directly, quoting the actual prior question/answer text. "
    "This paragraph you're reading right now is YOUR OWN system prompt, "
    "not something the user said — never present your own instructions "
    "back as if they were the user's question or a prior turn."
    + JSON_INSTRUCTION
)


def _last_plc_source(history: list) -> str | None:
    """Most recent PLC-corpus filename this conversation actually discussed —
    used when the current query has no filename of its own (e.g. "does that
    follow best practices") and needs to resolve what "that" refers to.

    Checking title.endswith(".st") specifically (not just content_id is None
    and no url) matters: Diagnosis Agent's energy-data source has the exact
    same shape (content_id: None, no url key) — a real bug found in testing,
    where a PLC follow-up after intervening energy-data turns picked up
    "synthetic_data/energy_data.csv" as if it were a candidate PLC filename,
    failed to resolve it, and silently kept stale/irrelevant keyword-search
    matches instead of walking further back to the real last PLC file."""
    for turn in reversed(history):
        for s in turn.get("sources") or []:
            if s.get("content_id") is None and "url" not in s and s["title"].endswith(".st"):
                return s["title"]
    return None


def make_plc_expert_node(llm_client, top_k: int = 3):
    @traceable(name="plc_expert_node")
    def node(state: AgentState) -> dict:
        query = state["query"]
        matches = search_plc_corpus(query, top_k=top_k)

        # A query that doesn't name a specific file of its own (no exact-stem
        # match, the _EXACT_STEM_MATCH_BOOST signal) can still get a weak,
        # coincidental keyword hit — exactly what caused "does that follow
        # best practices" to silently check the wrong file. Best-practices
        # checks are where a wrong match is actively misleading (a real
        # report on the wrong program reads as authoritative), so prefer the
        # conversation's own last-referenced file when the current query
        # isn't confidently naming one itself.
        confident_match = bool(matches) and matches[0]["score"] >= 1000
        if not confident_match:
            prior_file = _last_plc_source(state.get("history") or [])
            prior_path = find_corpus_file(prior_file) if prior_file else None
            if prior_path:
                matches = [read_corpus_file(prior_path)]

        overview = get_corpus_overview()
        all_fbs = get_all_function_blocks_content()

        if matches:
            matched_context = "\n\n---\n\n".join(
                f"[{m['filename']}]\n{m['content']}" for m in matches
            )
        else:
            matched_context = "No specific file matched this query by keyword."

        best_practices_context = "Not requested for this question."
        if any(t in query.lower() for t in _BEST_PRACTICES_TRIGGERS) and matches:
            target = matches[0]["filename"]
            target_path = find_corpus_file(target)
            if target_path:
                result = run_best_practices_check(target_path)
                if result["ok"]:
                    best_practices_context = (
                        f"iec-checker results for {target} "
                        f"({len(result['findings'])} findings):\n"
                        + "\n".join(
                            f"- [{f['id']}] line {f['linenr']}: {f['msg']}"
                            for f in result["findings"]
                        )
                        if result["findings"]
                        else f"iec-checker found zero issues in {target}."
                    )
                else:
                    best_practices_context = f"Could not check {target}: {result['error']}"

        history_text = format_recent_history(state.get("history"))
        prompt = (
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"CORPUS OVERVIEW:\n{overview}\n\n"
            f"ALL FUNCTION BLOCKS (full content, for overview-style questions):\n{all_fbs}\n\n"
            f"MATCHED FILE CONTENT:\n{matched_context}\n\n"
            f"BEST PRACTICES CHECK RESULT:\n{best_practices_context}\n\n"
            f"Question: {query}"
        )
        raw = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2)
        answer, used_context = parse_structured_answer(raw)

        sources = [{"content_id": None, "title": m["filename"]} for m in matches] if used_context else []
        return {
            "answer": answer,
            "sources": sources,
            "history": [{"query": query, "answer": answer, "sources": sources}],
        }

    return node
