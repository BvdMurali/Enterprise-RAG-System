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
        "Format as a Definition Card:\n"
        "- Lead with the header '📘 Definition' and write a single-sentence definition of the term in bold.\n"
        "- Add '⚙️ Key Components' section to list and explain the core parts or principles using bullets.\n"
        "- Add '💡 Why it matters' section to explain the business value, compliance impact, or importance.\n"
        "- Provide concrete '🔍 Examples' or implementations.\n"
    ),
    "summarize": (
        "Format as a Profile Summary:\n"
        "- Lead with '📝 Summary: [Subject]' and a 2-sentence executive overview.\n"
        "- Group key details under emoji sub-headings (e.g. 👤 Profile, 🎯 Objectives, 💎 Highlights, 🚀 Key Experience).\n"
        "- Conclude with a one-sentence closing insight.\n"
    ),
    "compare": (
        "Format as a Comparison Analysis:\n"
        "- Lead with the header '🔄 Comparison: [Item A] vs [Item B]'.\n"
        "- Use a clean markdown table comparing attributes side-by-side.\n"
        "- Add '⚖️ Key Differences & Similarities' section highlighting 2-3 specific points of contrast in bullet points.\n"
        "- Conclude with a '🔑 Strategic Recommendation' or final evaluation.\n"
    ),
    "list": (
        "Format as a Structured List:\n"
        "- Lead with the header '📋 Listed Items' and a short opening sentence.\n"
        "- Use a numbered or bulleted list where each item starts with a **Bold Title** followed by a concise 1-sentence description.\n"
        "- Add a short summary of patterns or trends observed.\n"
    ),
    "count": (
        "Format as a Count Analysis:\n"
        "- Lead with the header '🔢 Count Analysis'.\n"
        "- State the final count clearly in the first sentence (e.g., 'There are X controls identified').\n"
        "- Add a bulleted list breaking down the items counted, grouped by logical category.\n"
    ),
    "extract": (
        "Format as an Extraction/Financial Card:\n"
        "- Lead with a relevant emoji header matching the data (e.g. '📊 Tesla Revenue Breakdown (Q1 2026)' or '👤 Contact Details').\n"
        "- Present the key metric or overall total clearly in large text (e.g. '**Total Revenue:** $22.4 Billion').\n"
        "- Present the breakdown of categories using custom symbols/emojis and list their values (with percentage calculations of the total if financial data is present).\n"
        "- Conclude with a 'Key Insight' section explaining dependencies, trends, or significance.\n"
    ),
    "explain": (
        "Format as an Explanation Workflow:\n"
        "- Lead with the header '💡 Explanation: [Concept]'.\n"
        "- Give a high-level summary of the concept.\n"
        "- Use numbered steps to walk through the mechanics, stages, or workflow step-by-step.\n"
        "- Add a '⚠️ Key Considerations' or warning section.\n"
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
<your formatted answer here, following the formatting rules above. Do NOT include Key Insights inside the ANSWER block, as it will be generated in its own block below.>

[SUMMARY]
<a single, concise sentence summarizing the answer>

[SOURCES]
<comma-separated list of cited sources, e.g.: "Resume (Page 1), Resume (Page 2)">

[CONFIDENCE_GROUNDING_SCORE]
<an estimated grounding and fact completeness percentage as a float between 0.0 and 1.0 based strictly on context evidence, e.g.: 0.95 or 0.90>

[CONFIDENCE_REASONS]
<bullet list of reasons explaining the grounding score and evidence presence, as a pipe-separated list starting with bullet points, e.g.: • Information found in official table | • Direct textual matches | • No conflicting evidence detected>

[DOCUMENT_TYPE]
<the type of documents retrieved, e.g.: financial report | standard | resume | policy | manual>

[RESPONSE_TYPE]
<the matched intent classification type, e.g.: definition | summarize | compare | list | count | extract | explain | general>

[KEY_INSIGHTS]
<3 short, high-level strategic takeaways or insights summarizing the answer, as a pipe-separated list starting with bullet points, e.g.: • Takeaway 1 | • Takeaway 2 | • Takeaway 3>

[FOLLOWUPS]
<3 short, varied follow-up question suggestions the user might want to ask next, as a pipe-separated list>
Example: What are the key projects? | Where did Murali study? | What is the contact information?\
"""

_RESPONSE_FORMAT_NO_FOLLOWUPS = """\
[ANSWER]
<your formatted answer here, following the formatting rules above. Do NOT include Key Insights inside the ANSWER block, as it will be generated in its own block below.>

[SUMMARY]
<a single, concise sentence summarizing the answer>

[SOURCES]
<comma-separated list of cited sources, e.g.: "Resume (Page 1), Resume (Page 2)">

[CONFIDENCE_GROUNDING_SCORE]
<an estimated grounding and fact completeness percentage as a float between 0.0 and 1.0 based strictly on context evidence, e.g.: 0.95 or 0.90>

[CONFIDENCE_REASONS]
<bullet list of reasons explaining the grounding score and evidence presence, as a pipe-separated list starting with bullet points, e.g.: • Information found in official table | • Direct textual matches | • No conflicting evidence detected>

[DOCUMENT_TYPE]
<the type of documents retrieved, e.g.: financial report | standard | resume | policy | manual>

[RESPONSE_TYPE]
<the matched intent classification type, e.g.: definition | summarize | compare | list | count | extract | explain | general>

[KEY_INSIGHTS]
<3 short, high-level strategic takeaways or insights summarizing the answer, as a pipe-separated list starting with bullet points, e.g.: • Takeaway 1 | • Takeaway 2 | • Takeaway 3>\
"""

HUMAN_RAG_PROMPT = "{question}"


def get_rag_prompt(intent: str = "general", include_followups: bool = True) -> ChatPromptTemplate:
    """
    Construct and return an intent-aware ChatPromptTemplate for the RAG chain.

    Args:
        intent:           Detected query intent label.
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
