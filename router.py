"""
Query routing between the semantic vector index and the structured data
store, plus answer synthesis for the structured path.
"""

from datetime import datetime
from typing import List

from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field

from schema import ITEM_TYPES

ROUTER_PROMPT = PromptTemplate(
    """You are a query router for a documentation Q&A system with two retrieval backends:

1. "semantic" - vector similarity search over document chunks. Good for open-ended,
   explanatory "how/why/what is" questions where any well-matching passage is a fine
   answer.

2. "structured" - a queryable store of extracted items, with exactly these types:
   - decisions    (title, summary, tags)
   - rules        (rule, scope, notes)
   - warnings     (area, message, severity)
   - dependencies (name, purpose, category)
   Every item also carries observed_at (an ISO datetime) and a source file/line
   reference. Prefer "structured" when the question:
   - asks for a full LIST or COUNT of something ("all the ...", "list every ...")
   - asks what is CURRENT / latest / still valid ("what's the current guidance on ...")
   - has a TIME bound ("in the last week", "since Monday", "recently")

When routing to "structured", also fill in:
   - item_types: which of decisions/rules/warnings/dependencies are relevant
     (leave empty to search all of them)
   - keywords: short keywords/phrases (translate to English if needed) to match
     against item text; leave empty if the question is broad
   - date_from / date_to: resolve any relative time expression into absolute ISO
     dates (YYYY-MM-DD) using {today} as "today". Leave both empty if there is no
     time bound.

Today's date is {today}.

Question: {query}
"""
)

STRUCTURED_QA_PROMPT = PromptTemplate(
    """You are a helpful assistant answering a question about a project using a
curated list of facts extracted from its documentation (not raw text search).
Answer using only the items below. Cite each fact's source file and line range
in parentheses. Answer in the same language the question was asked in.
If the items don't actually answer the question, say so honestly instead of
guessing.

Extracted items:
{context_str}

Question: {query_str}

Answer:"""
)


class RouteDecision(BaseModel):
    route: str = Field(description="Either 'semantic' or 'structured'")
    item_types: List[str] = Field(
        default_factory=list,
        description="Subset of decisions/rules/warnings/dependencies to search; empty means all",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords/phrases to filter items by; empty means no keyword filter",
    )
    date_from: str = Field(default="", description="ISO date YYYY-MM-DD, inclusive lower bound")
    date_to: str = Field(default="", description="ISO date YYYY-MM-DD, inclusive upper bound")
    reasoning: str = Field(default="", description="One short sentence explaining the routing choice")


async def decide_route(llm, query: str) -> RouteDecision:
    today = datetime.now().date().isoformat()
    decision = await llm.astructured_predict(RouteDecision, ROUTER_PROMPT, query=query, today=today)

    if decision.route not in ("semantic", "structured"):
        decision.route = "semantic"
    decision.item_types = [t for t in decision.item_types if t in ITEM_TYPES]
    return decision


def _format_item(item: dict) -> str:
    source = item.get("source", {})
    line_range = source.get("line_range", [])
    location = f"{source.get('file', '?')}:{'-'.join(str(x) for x in line_range)}"
    body = {k: v for k, v in item.items() if k not in ("id", "source", "observed_at", "type")}
    return f"- [{item.get('type')}] {body} (observed_at={item.get('observed_at')}, source={location})"


async def synthesize_structured_answer(llm, query: str, items: List[dict]) -> str:
    context_str = "\n".join(_format_item(item) for item in items)
    prompt = STRUCTURED_QA_PROMPT.format(context_str=context_str, query_str=query)
    response = await llm.acomplete(prompt)
    return str(response)
