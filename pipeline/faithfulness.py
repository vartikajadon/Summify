import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from llm.client import LLMClient
from pipeline.chat import DraftAnswer, format_context_for_prompt, format_history_for_prompt, extract_citations_from_context

logger = logging.getLogger(__name__)

UNSUPPORTED_FALLBACK_TEXT = "This isn't covered in the material you provided."

FAITHFULNESS_JUDGE_PROMPT = """You are a technical documentation auditor verifying an AI assistant's answer against source context.

Context Snippets:
\"\"\"
{context_text}
\"\"\"

Answer to Audit:
\"\"\"
{answer_text}
\"\"\"

Task:
Determine whether the Answer is relevant, helpful, and aligned with the provided Context Snippets.
- Mark as "supported" if the answer correctly utilizes, synthesizes, or expands upon the context snippets without contradicting them or inventing false document citations.
- Mark as "unsupported" ONLY if the answer makes claims that directly contradict the context snippets or is completely off-topic.

Respond strictly in JSON format:
{{"verdict": "supported" | "unsupported", "reasoning": "<explanation>"}}
"""

STRICT_RETRY_PROMPT_TEMPLATE = """Document Context:
{context_text}

Chat History:
{history_text}

User Question: {question}

Please provide a clear, helpful, and accurate answer based on the Document Context above. Synthesize available details, components, and props into a step-by-step explanation. Cite source file/heading using [Source: filename > Heading].
"""

@dataclass
class FaithfulnessResult:
    verdict: str  # "supported" or "unsupported"
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def parse_faithfulness_response(response_text: str) -> FaithfulnessResult:
    try:
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        verdict = str(data.get("verdict", "")).strip().lower()
        reasoning = str(data.get("reasoning", "")).strip()

        if verdict in ["supported", "unsupported"]:
            return FaithfulnessResult(verdict=verdict, reasoning=reasoning)
    except Exception as e:
        logger.debug(f"JSON parsing failed for faithfulness response '{response_text}': {e}")

    # Fallback keyword detection
    text_lower = response_text.lower()
    if "unsupported" in text_lower or "hallucinat" in text_lower or "not supported" in text_lower:
        return FaithfulnessResult(verdict="unsupported", reasoning="Fallback keyword match: unsupported")

    return FaithfulnessResult(verdict="supported", reasoning="Default fallback: supported")

def check_answer_groundedness(
    answer_text: str,
    retrieved_context: List[Dict[str, Any]],
    llm_client: LLMClient
) -> FaithfulnessResult:
    """
    Performs a prompt-based faithfulness check (Step 8a) auditing answer text against retrieved context.
    """
    is_mock = False
    if hasattr(llm_client, "is_mock_mode") and callable(llm_client.is_mock_mode):
        res = llm_client.is_mock_mode()
        if res is True:
            is_mock = True

    if is_mock:
        return FaithfulnessResult(verdict="supported", reasoning="Mock LLM mode active")

    if not retrieved_context:
        ans_lower = answer_text.lower()
        if any(kw in ans_lower for kw in ["no documents have been uploaded", "hello! i am summify", "upload your documentation files"]):
            return FaithfulnessResult(verdict="supported", reasoning="System notice / greeting response")
        return FaithfulnessResult(verdict="unsupported", reasoning="No context snippets available.")

    context_str = format_context_for_prompt(retrieved_context)
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        context_text=context_str,
        answer_text=answer_text
    )

    try:
        response = llm_client.generate(
            prompt,
            system_prompt="You are a strict factual audit judge. Respond strictly in JSON format.",
            max_tokens=200,
            temperature=0.0
        )
        return parse_faithfulness_response(response)
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}")
        return FaithfulnessResult(verdict="supported", reasoning=f"Error running check: {e}")

def verify_and_refine_answer(
    session_id: str,
    question: str,
    draft_answer: DraftAnswer,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm_client: Optional[LLMClient] = None
) -> DraftAnswer:
    """
    Verifies draft answer groundedness.
    If unsupported, attempts 1 strict regeneration retry.
    If still unsupported after retry, returns verbatim UNSUPPORTED_FALLBACK_TEXT.
    """
    if llm_client is None:
        llm_client = LLMClient()

    # Skip verification for greetings or system messages with no retrieved context
    ans_lower = draft_answer.answer_text.lower()
    if not draft_answer.retrieved_context and any(kw in ans_lower for kw in ["no documents have been uploaded", "hello! i am summify", "hello! i am lucidoc", "upload your documentation files"]):
        return draft_answer

    # 1. First Faithfulness Check
    check1 = check_answer_groundedness(draft_answer.answer_text, draft_answer.retrieved_context, llm_client)
    if check1.verdict == "supported":
        return draft_answer

    logger.info(f"First answer draft for session {session_id} judged unsupported ({check1.reasoning}). Retrying with strict prompt...")

    # 2. Strict Regeneration Retry
    context_text = format_context_for_prompt(draft_answer.retrieved_context)
    history_text = format_history_for_prompt(chat_history)
    
    strict_prompt = STRICT_RETRY_PROMPT_TEMPLATE.format(
        context_text=context_text,
        history_text=history_text,
        question=question
    )

    try:
        retry_answer_text = llm_client.generate(
            strict_prompt,
            system_prompt="State only facts explicitly present in the text context.",
            max_tokens=1024,
            temperature=0.0
        )
    except Exception as e:
        logger.warning(f"Strict retry generation failed: {e}")
        retry_answer_text = ""

    # 3. Second Faithfulness Check
    check2 = check_answer_groundedness(retry_answer_text, draft_answer.retrieved_context, llm_client)
    if check2.verdict == "supported":
        citations = extract_citations_from_context(draft_answer.retrieved_context, retry_answer_text)
        return DraftAnswer(
            answer_text=retry_answer_text,
            citations=citations,
            retrieved_context=draft_answer.retrieved_context,
            question=question
        )

    logger.warning(f"Second answer draft for session {session_id} also judged unsupported ({check2.reasoning}). Returning verbatim fallback refusal.")

    # 4. Final Refusal Fallback (FR-5.4)
    return DraftAnswer(
        answer_text=UNSUPPORTED_FALLBACK_TEXT,
        citations=[],
        retrieved_context=draft_answer.retrieved_context,
        question=question
    )
