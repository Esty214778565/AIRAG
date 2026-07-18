import gradio as gr
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from common import get_env_var, get_llm
from router import RouteDecision, decide_route, synthesize_structured_answer
from structured_store import load_store, query_items

PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
COHERE_API_KEY = get_env_var("COHERE_API_KEY")
GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")


TEXT_QA_TEMPLATE = PromptTemplate(
    """You are a helpful assistant.
    Answer the user's question using only the context below.
    If the context does not contain the answer, say that you do not know.

    Context:
    {context_str}

    Question:
    {query_str}

    Answer:"""
    )


def load_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("kiro")

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        # namespace="kiro-steering",
        namespace="Kiro-RAG",
    )

    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

index = load_index()
structured_store = load_store()

llm = get_llm(temperature=0.1)
response_synthesizer = get_response_synthesizer(
    llm=llm,
    text_qa_template=TEXT_QA_TEMPLATE,
    response_mode=ResponseMode.COMPACT
)


# --- Tunables for the validation / routing logic ---------------------------
MIN_QUERY_LENGTH = 2          # anything shorter is rejected before any retrieval/LLM call
LOW_CONFIDENCE_SCORE = 0.3    # best node score below this is not trusted
RETRY_TOP_K = 8               # widened search used on the retry pass
MAX_RETRIEVAL_ATTEMPTS = 2    # first attempt + one retry, then give up


# --- Events ------------------------------------------------------------
# An Event is the handoff contract between two steps: it carries only what
# the *next* step needs in order to do its job - nothing more.

class QueryValidEvent(Event):
    query: str


class SemanticRouteEvent(Event):
    query: str


class StructuredRouteEvent(Event):
    query: str
    decision: RouteDecision


class RetrievalRequestEvent(Event):
    query: str
    top_k: int


class ContextRetrievedEvent(Event):
    query: str
    nodes: list


class NoContextEvent(Event):
    query: str
    reason: str  # "empty_retrieval" | "low_confidence"


class ContextSufficientEvent(Event):
    query: str
    nodes: list


class AnswerDraftedEvent(Event):
    query: str
    answer: str


# --- Workflow ------------------------------------------------------------

class RagChatWorkflow(Workflow):
    """
    validate_input -> route_query
        -> [structured] run_structured_query -> validate_answer
                          | (no matching items) falls back to SemanticRouteEvent
        -> [semantic] retrieve_context -> assess_context_quality
        -> (retry retrieve_context once) | synthesize_answer -> validate_answer
    Any step can short-circuit straight to StopEvent via handle_no_context.
    """

    @step
    async def validate_input(
        self, ctx: Context, ev: StartEvent
    ) -> QueryValidEvent | StopEvent:
        query = (ev.query or "").strip()
        if len(query) < MIN_QUERY_LENGTH:
            return StopEvent(result="Please enter a real question.")

        await ctx.store.set("attempts", 0)
        return QueryValidEvent(query=query)

    @step
    async def route_query(
        self, ev: QueryValidEvent
    ) -> SemanticRouteEvent | StructuredRouteEvent:
        decision = await decide_route(llm, ev.query)
        if decision.route == "structured":
            return StructuredRouteEvent(query=ev.query, decision=decision)
        return SemanticRouteEvent(query=ev.query)

    @step
    async def run_structured_query(
        self, ev: StructuredRouteEvent
    ) -> AnswerDraftedEvent | SemanticRouteEvent:
        items = query_items(
            structured_store,
            ev.decision.item_types,
            ev.decision.keywords,
            ev.decision.date_from,
            ev.decision.date_to,
        )
        if not items:
            # Nothing matched in the structured store - fall back to semantic search
            # rather than telling the user "no results".
            return SemanticRouteEvent(query=ev.query)

        answer = await synthesize_structured_answer(llm, ev.query, items)
        return AnswerDraftedEvent(query=ev.query, answer=answer)

    @step
    async def retrieve_context(
        self, ctx: Context, ev: SemanticRouteEvent | RetrievalRequestEvent
    ) -> ContextRetrievedEvent | NoContextEvent:
        query = ev.query
        top_k = ev.top_k if isinstance(ev, RetrievalRequestEvent) else None

        active_retriever = (
            index.as_retriever(similarity_top_k=top_k) if top_k else index.as_retriever()
        )
        nodes = await active_retriever.aretrieve(query)

        if not nodes:
            return NoContextEvent(query=query, reason="empty_retrieval")
        return ContextRetrievedEvent(query=query, nodes=nodes)

    @step
    async def assess_context_quality(
        self, ctx: Context, ev: ContextRetrievedEvent
    ) -> ContextSufficientEvent | RetrievalRequestEvent | NoContextEvent:
        attempts = await ctx.store.get("attempts", default=0)
        best_score = max((n.score or 0.0) for n in ev.nodes)

        if best_score < LOW_CONFIDENCE_SCORE and attempts + 1 < MAX_RETRIEVAL_ATTEMPTS:
            # low confidence -> try again with a wider net before giving up
            await ctx.store.set("attempts", attempts + 1)
            return RetrievalRequestEvent(query=ev.query, top_k=RETRY_TOP_K)

        if best_score < LOW_CONFIDENCE_SCORE:
            return NoContextEvent(query=ev.query, reason="low_confidence")

        return ContextSufficientEvent(query=ev.query, nodes=ev.nodes)

    @step
    async def synthesize_answer(
        self, ev: ContextSufficientEvent
    ) -> AnswerDraftedEvent:
        response = await response_synthesizer.asynthesize(ev.query, nodes=ev.nodes)
        return AnswerDraftedEvent(query=ev.query, answer=str(response))

    @step
    async def validate_answer(self, ev: AnswerDraftedEvent) -> StopEvent:
        answer = ev.answer.strip()
        if not answer:
            return StopEvent(
                result="I could not generate an answer from the retrieved context."
            )
        return StopEvent(result=answer)

    @step
    async def handle_no_context(self, ev: NoContextEvent) -> StopEvent:
        if ev.reason == "low_confidence":
            return StopEvent(
                result=(
                    "I found some related material, but not enough to answer "
                    "confidently. Could you rephrase or add more detail?"
                )
            )
        return StopEvent(result="No relevant context found.")


rag_workflow = RagChatWorkflow(timeout=60, verbose=False)


async def chat(message, history):
    result = await rag_workflow.run(query=message)
    return str(result)


demo = gr.ChatInterface(
    fn=chat,
    title="RAG Chat",
    description="Ask a question about the indexed documents.",
)


if __name__ == "__main__":
    demo.launch()
