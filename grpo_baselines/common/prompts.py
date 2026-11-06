"""Prompt templates for the four baseline methods.

Design rules
------------
1. **One shared display convention.** Every template receives ``question``,
   ``response_a`` and ``response_b`` in *display* order, so the A/B letters inside a
   rendered prompt always mean the same thing project-wide (see :mod:`common.labels`).
2. **Per-method output format follows its own paper.** Vanilla and J1 both use the
   project-wide ``<answer>`` tag, but J1 additionally requires a ``<think>`` block and
   the ``[[A]]`` letter form (J1 Fig. 9); J4R keeps the released Arena-Hard
   ``Explanation:`` / ``Verdict: [[A>B]]`` layout because its format reward is defined
   on exactly that layout; JwC emits a probability instead of a verdict.
   ``config.output_format`` records which parser must be used and
   :mod:`common.parsing` resolves it through a registry rather than ad-hoc matching.
   The *display semantics* (what A and B mean) are identical across all four.
3. Templates are plain strings with no f-string interpolation so that the literal
   brace/bracket syntax of a paper prompt cannot be corrupted by formatting.
"""

from __future__ import annotations

from typing import Final

MethodName = str

VALID_METHODS: Final[tuple[str, ...]] = ("vanilla", "j1", "j4r", "jwc")

#: Output-format identifiers understood by :mod:`common.parsing`.
VALID_OUTPUT_FORMATS: Final[tuple[str, ...]] = ("vanilla", "j1", "j4r", "jwc")


# --------------------------------------------------------------------------------------
# Vanilla GRPO
# --------------------------------------------------------------------------------------
VANILLA_SYSTEM: Final[str] = (
    "You are an impartial judge. You are given a user question and two responses from two "
    "AI assistants, labelled A and B. Decide which response better follows the user's "
    "instructions and provides a higher-quality answer.\n"
    "Judge only the content of the two responses. Do not let the order in which the "
    "responses are presented, their length, or stylistic elements such as formatting "
    "influence your decision.\n"
    "Answer with exactly one of the following:\n"
    "- <answer>A</answer> if Assistant A is better\n"
    "- <answer>B</answer> if Assistant B is better\n"
    "Do not output anything else."
)

VANILLA_USER: Final[str] = (
    "[User Question]\n"
    "{question}\n"
    "\n"
    "[The Start of Assistant A's Answer]\n"
    "{response_a}\n"
    "[The End of Assistant A's Answer]\n"
    "\n"
    "[The Start of Assistant B's Answer]\n"
    "{response_b}\n"
    "[The End of Assistant B's Answer]"
)


# --------------------------------------------------------------------------------------
# J1 -- arXiv:2505.10320, Appendix Figure 9
# "Thinking Seed Prompt Template for Training Pairwise J1 with Verdict".
# Body reproduced verbatim from the paper; only the placeholder names are ours
# (``{instruction}`` / ``{response A}`` / ``{response B}`` in the paper).
# --------------------------------------------------------------------------------------
J1_SYSTEM: Final[str] = (
    "You are given a user question and two responses from two AI assistants. Your task "
    "is to act as an impartial judge and evaluate which response better follows the "
    "user's instructions and provides a higher-quality answer.\n"
    "First, provide your reasoning within <think> and </think> tags. This should "
    "include your evaluation criteria for a high-quality response, a detailed "
    "comparison of the two responses, and when helpful, a reference answer as part of "
    "your evaluation. Be explicit in your thought process, referencing your criteria "
    "and explaining how each response aligns with or deviates from them.\n"
    "Avoid any position biases and ensure that the order in which the responses "
    "were presented does not influence your decision. Do not allow the length of "
    "the responses to influence your evaluation. Do not favor certain names of the "
    "assistants. Be as objective as possible.\n"
    "Finally, provide your verdict within <answer> and </answer> tags, strictly "
    "following this format:\n"
    "- <answer>[[A]]</answer> if Assistant A is better\n"
    "- <answer>[[B]]</answer> if Assistant B is better"
)

J1_USER: Final[str] = (
    "Below are the user's question and the two responses:\n"
    "[User Question]\n"
    "{question}\n"
    "[The Start of Assistant A's Answer]\n"
    "{response_a}\n"
    "[The End of Assistant A's Answer]\n"
    "[The Start of Assistant B's Answer]\n"
    "{response_b}\n"
    "[The End of Assistant B's Answer]"
)


# --------------------------------------------------------------------------------------
# J4R -- arXiv:2505.13346.  The authors' released repository ships the exact prompt as
# ``reasoning_judge_inf/prompts/default_prompts.py``; per the paper (Sec. 4) this is the
# Arena-Hard variant with ties omitted.  Copied verbatim from
# /data1/yanghao/paper/GRPO/external/reasoning-judge-bench @ e5a6c64.
# --------------------------------------------------------------------------------------
J4R_SYSTEM: Final[str] = (
    "Please act as an impartial judge and evaluate the quality of the responses provided "
    "by two AI assistants to the user prompt displayed below. You will be given assistant "
    "A's answer and assistant B's answer. Your job is to evaluate which assistant's answer "
    "is better.\n"
    "\n"
    "When evaluating the assistants' answers, identify any mistakes or inaccurate "
    "information. Focus on the content each response, and select the response that arrives "
    "at the correct response and with the most sound and logical reasoning. Avoid any "
    "biases, such as order of responses, length, or stylistic elements like formatting.\n"
    "\n"
    "Before providing an your final verdict, think through the judging process and output "
    "your thoughts as an explanation\n"
    "\n"
    "After providing your explanation, you must output only one of the following choices "
    "as your final verdict with a label:\n"
    "\n"
    "1. Assistant A is better: [[A>B]]\n"
    "2. Assistant B is better: [[B>A]]\n"
    "\n"
    "Use the following template:\n"
    "Explanation: Your detailed thought process as an explanation.\n"
    "Verdict: [[A>B]] or [[B>A]]."
)

J4R_USER: Final[str] = (
    "<|User Prompt|>\n"
    "{question}\n"
    "\n"
    "<|The Start of Assistant A's Answer|>\n"
    "{response_a}\n"
    "<|The End of Assistant A's Answer|>\n"
    "\n"
    "<|The Start of Assistant B's Answer|>\n"
    "{response_b}\n"
    "<|The End of Assistant B's Answer|>"
)


# --------------------------------------------------------------------------------------
# Judging with Confidence -- arXiv:2510.00263, Appendix F
# ("Verbalized Confidence w/ CoT").  Tag names and the "B is better" orientation are the
# paper's; the prompt never invites a hard A/B verdict, because that is precisely what the
# probability head replaces.
# --------------------------------------------------------------------------------------
JWC_SYSTEM: Final[str] = (
    "Your task is to evaluate two responses, A and B, based on the provided question. "
    "Compare the responses on helpfulness, accuracy, and clarity.\n"
    "Follow these steps for your output:\n"
    "1. First, provide a step-by-step reasoning of your evaluation inside <think> and "
    "</think> tags. Explain which response is better and why.\n"
    "2. Immediately after the closing </think> tag, provide a single float between 0.0 "
    "and 1.0 representing the probability that Response B is better than Response A. "
    "This float must be enclosed in <prob_rb_better> and </prob_rb_better> tags. "
    "A probability of 0.0 means Response A is clearly better, 1.0 means Response B is "
    "clearly better, and 0.5 indicates they are of similar quality.\n"
    "Example output format:\n"
    "<think>Your detailed reasoning comparing A and B goes here.</think>"
    "<prob_rb_better>0.85</prob_rb_better>"
)

JWC_USER: Final[str] = (
    "[Question]\n"
    "{question}\n"
    "[The Start of Response A]\n"
    "{response_a}\n"
    "[The End of Response A]\n"
    "[The Start of Response B]\n"
    "{response_b}\n"
    "[The End of Response B]\n"
    "Your Evaluation:"
)


_SYSTEM: Final[dict[str, str]] = {
    "vanilla": VANILLA_SYSTEM,
    "j1": J1_SYSTEM,
    "j4r": J4R_SYSTEM,
    "jwc": JWC_SYSTEM,
}

_USER: Final[dict[str, str]] = {
    "vanilla": VANILLA_USER,
    "j1": J1_USER,
    "j4r": J4R_USER,
    "jwc": JWC_USER,
}

SYSTEM_ROLE: Final[str] = "system"
USER_ROLE: Final[str] = "user"


def check_method(method: str) -> str:
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {VALID_METHODS}")
    return method


def build_messages(
    method: str,
    question: str,
    response_a: str,
    response_b: str,
) -> list[dict[str, str]]:
    """Render the chat messages for ``method`` in *display* order A/B.

    ``response_a`` and ``response_b`` must already be the strings the judge sees first
    and second; callers obtain them from :func:`data.pair_views.render_view`.
    """
    check_method(method)
    for name, value in (
        ("question", question),
        ("response_a", response_a),
        ("response_b", response_b),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
    user = _USER[method].format(
        question=question, response_a=response_a, response_b=response_b
    )
    return [
        {"role": SYSTEM_ROLE, "content": _SYSTEM[method]},
        {"role": USER_ROLE, "content": user},
    ]


def build_prompt_text(
    method: str,
    question: str,
    response_a: str,
    response_b: str,
    *,
    add_generation_prompt: bool = True,
) -> list[dict[str, str]]:
    """Alias kept for readability at call sites that only need the message list."""
    del add_generation_prompt  # generation prompting is the tokenizer's job
    return build_messages(method, question, response_a, response_b)


def system_prompt(method: str) -> str:
    return _SYSTEM[check_method(method)]


def user_template(method: str) -> str:
    return _USER[check_method(method)]


__all__ = [
    "J1_SYSTEM",
    "J1_USER",
    "J4R_SYSTEM",
    "J4R_USER",
    "JWC_SYSTEM",
    "JWC_USER",
    "SYSTEM_ROLE",
    "USER_ROLE",
    "VALID_METHODS",
    "VALID_OUTPUT_FORMATS",
    "VANILLA_SYSTEM",
    "VANILLA_USER",
    "build_messages",
    "build_prompt_text",
    "check_method",
    "system_prompt",
    "user_template",
]
