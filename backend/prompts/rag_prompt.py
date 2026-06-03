"""
RAG Prompt Templates for the Enterprise RAG System.

Intent-aware prompt factory: selects the best response structure and
formatting instructions based on the detected query intent so the
output always feels conversational and purposeful rather than a raw
search dump.
"""

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Intent → formatting instruction mapping
# ---------------------------------------------------------------------------

_INTENT_INSTRUCTIONS = {
    "definition": (
        "Format as a definition:\n"
        "- Lead with the header '📘 Answer' and write a single-sentence definition in bold.\n"
        "- Use bullet points for key details, characteristics, or attributes.\n"
        "- Add a section '💡 Why it matters' explaining its purpose, business value, or impact.\n"
    ),
    "summarize": (
        "Format as a summary:\n"
        "- Lead with the header '📝 Summary: [Subject Name]' and a 2-sentence executive overview.\n"
        "- Group key details under emoji sub-headings (e.g. 🎯 Objectives, 💎 Key Takeaways, 📅 Timeline).\n"
        "- Conclude with a one-sentence closing insight.\n"
    ),
    "compare": (
        "Format as a comparison:\n"
        "- Lead with the header '🔄 Comparison: [Item A] vs [Item B]'.\n"
        "- Use a markdown table with clear column headers to compare attributes side-by-side.\n"
        "- Conclude with a '🔑 Key Takeaway' section explaining differences/similarities.\n"
    ),
    "list": (
        "Format as a structured list:\n"
        "- Lead with the header '📋 Listed Items' and a short opening sentence.\n"
        "- Use a numbered or bulleted list where each item starts with a **Bold Title** followed by a 1-sentence description.\n"
        "- Order by relevance or priority.\n"
    ),
    "count": (
        "Format as a count analysis:\n"
        "- Lead with the header '🔢 Count Analysis'.\n"
        "- State the final count clearly in the first sentence (e.g., 'There are X controls identified').\n"
        "- Add a bulleted list breaking down the items counted.\n"
    ),
    "extract": (
        "Format as an extraction card:\n"
        "- Lead with a relevant emoji header matching the data (e.g. '📊 Tesla Revenue Sources (Q1 2026)' or '👤 Contact Details').\n"
        "- Present the extracted facts or figures as key-value pairs: **Key:** Value.\n"
        "- Add bullet points containing supporting details or context (e.g. amounts, dates, locations).\n"
        "- Conclude with overall totals, averages, or a summary line if numbers are present.\n"
    ),
    "explain": (
        "Format as an explanation:\n"
        "- Lead with the header '💡 Explanation: [Concept]'.\n"
        "- Give a high-level summary of the concept.\n"
        "- Use numbered steps to walk through the mechanics, process, or workflow step-by-step.\n"
        "- Add a section '⚠️ Key Considerations' or warnings if applicable.\n"
    ),
    "general": (
        "Format as general answer:\n"
        "- Lead with the header '💬 Answer'.\n"
        "- Provide a clear, conversational explanation in structured paragraphs.\n"
    ),
}


def get_intent_instruction(intent: str) -> str:
    """Return the formatting instruction for the given intent label."""
    return _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["general"])


# ---------------------------------------------------------------------------
# Dynamic RAG Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_RAG_TEMPLATE = """\
You are an expert Enterprise Knowledge Assistant — professional, friendly, and precise.

RESPONSE FORMATTING RULES:
{intent_instruction}

GROUNDING RULES (non-negotiable):
1. Every claim MUST be grounded in the <documents> below. Cite each fact inline: [Doc X, Page Y].
2. Handle minor spelling variations / typos gracefully — clarify correct spelling and still answer.
3. If the documents do not contain relevant information, say exactly:
   "I wasn't able to find that information in the uploaded documents."
4. Do NOT invent facts, make assumptions, or use external knowledge.

<documents>
{{context}}
</documents>

OUTPUT FORMAT (always use this structure):
---
{response_format}
---
"""

_RESPONSE_FORMAT_WITH_FOLLOWUPS = """\
[ANSWER]
<your formatted answer here, following the formatting rules above>

[SUMMARY]
<a single, concise sentence summarizing the answer>

[SOURCES]
<comma-separated list of cited sources, e.g.: "Resume (Page 1), Resume (Page 2)">

[CONFIDENCE]
<an estimated confidence percentage based on evidence, e.g.: 98% or 95%>

[DOCUMENT_TYPE]
<the type of documents retrieved, e.g.: financial report | standard | resume | policy | manual>

[RESPONSE_TYPE]
<the matched intent classification type, e.g.: definition | summarize | compare | list | count | extract | explain | general>

[FOLLOWUPS]
<3 short, varied follow-up question suggestions the user might want to ask next, as a pipe-separated list>
Example: What are the key projects? | Where did Murali study? | What is the contact information?\
"""

_RESPONSE_FORMAT_NO_FOLLOWUPS = """\
[ANSWER]
<your formatted answer here, following the formatting rules above>

[SUMMARY]
<a single, concise sentence summarizing the answer>

[SOURCES]
<comma-separated list of cited sources, e.g.: "Resume (Page 1), Resume (Page 2)">

[CONFIDENCE]
<an estimated confidence percentage based on evidence, e.g.: 98% or 95%>

[DOCUMENT_TYPE]
<the type of documents retrieved, e.g.: financial report | standard | resume | policy | manual>

[RESPONSE_TYPE]
<the matched intent classification type, e.g.: definition | summarize | compare | list | count | extract | explain | general>\
"""

HUMAN_RAG_PROMPT = "{question}"


def get_rag_prompt(intent: str = "general", include_followups: bool = True) -> ChatPromptTemplate:
    """
    Construct and return an intent-aware ChatPromptTemplate for the RAG chain.

    Args:
        intent:           Detected query intent label (e.g. 'summary', 'skills').
        include_followups: Whether to ask the LLM to generate follow-up suggestions.
    """
    intent_instruction = get_intent_instruction(intent)
    response_format = _RESPONSE_FORMAT_WITH_FOLLOWUPS if include_followups else _RESPONSE_FORMAT_NO_FOLLOWUPS

    system_prompt = _SYSTEM_RAG_TEMPLATE.format(
        intent_instruction=intent_instruction,
        response_format=response_format,
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", HUMAN_RAG_PROMPT),
    ])

