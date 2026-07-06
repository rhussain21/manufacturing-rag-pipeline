"""
Shared formatting for recent conversation history — used by the router (to
classify ambiguous follow-ups) and, now, by every persona's own prompt (to
actually answer meta-questions about the conversation itself, like "what
did I just ask" or "what have we covered").

These are genuinely different uses of the same data: the router needs just
enough to classify intent; a persona answering a meta-question needs real
transcript text to quote from, not a hint to route by.
"""


def format_recent_history(history: list, n: int = 5, answer_chars: int = 300) -> str:
    """Recent turns as readable Q/A text. Used as literal context in a
    prompt — if this is missing, a question like "what did I just ask" has
    nothing genuine to draw from, and a model will guess rather than admit
    it (a real bug this fixed: it pattern-matched onto its own system
    prompt text and presented that back as the user's first question)."""
    recent = (history or [])[-n:]
    if not recent:
        return "No prior turns in this conversation yet."
    return "\n\n".join(
        f"Turn {i+1} — Q: {turn['query']}\nA: {turn['answer'][:answer_chars]}"
        for i, turn in enumerate(recent)
    )
