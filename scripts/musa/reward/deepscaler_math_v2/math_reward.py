"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
import re
from typing import List, Union

from .globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from .reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from .utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def prompt_to_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        contents = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if content:
                    contents.append(str(content))
            else:
                contents.append(str(message))
        return "\n".join(contents)
    return str(prompt)


def extract_options_advanced(s: str):
    """Extract multiple-choice options from common plain-text and LaTeX formats."""
    marker = re.compile(
        r"\(?\s*\\(?:textbf|mathrm)\s*\{\s*\(?\s*([A-E])\s*\)?\s*\}\s*\)?"
        r"|\(\s*([A-E])\s*\)"
        r"|(?<![A-Za-z0-9])([A-E])\s*[\.:]\s*"
        r"|(?m:^\s*([A-E])\s+)"
    )
    matches = []
    for match in marker.finditer(s):
        letter = next(group for group in match.groups() if group is not None).upper()
        matches.append((letter, match.start(), match.end()))

    options_dict = {}
    for idx, (letter, _start, value_start) in enumerate(matches):
        value_end = matches[idx + 1][1] if idx + 1 < len(matches) else len(s)
        option_value = s[value_start:value_end].strip()
        option_value = re.sub(r"^(?:[-,;]|\\qquad|\\quad|\\;|\\,|\s)+", "", option_value).strip()
        option_value = re.sub(
            r"(?:\\qquad|\\quad|Let\'s think step by step.*|$).*$", "", option_value, flags=re.S
        ).strip()
        option_value = option_value.strip(" $")
        if option_value and letter not in options_dict:
            options_dict[letter] = option_value

    return dict(sorted(options_dict.items()))


def _dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_ground_truth(ground_truth):
    if isinstance(ground_truth, dict):
        for key in ("ground_truth", "answer", "target", "label"):
            if key in ground_truth:
                return normalize_ground_truth(ground_truth[key])
    if isinstance(ground_truth, (list, tuple)):
        return [normalize_ground_truth(value) for value in ground_truth]
    return ground_truth


def _strip_latex_text_wrappers(answer: str) -> str:
    previous = None
    while previous != answer:
        previous = answer
        answer = re.sub(
            r"\\(?:textbf|mathrm|mathbf|text)\s*\{\s*([^{}]*?)\s*\}",
            r"\1",
            answer,
        )
    return answer


def _extract_answer_phrase(solution: str) -> str | None:
    matches = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)",
            solution,
        )
    )
    if not matches:
        return None
    phrase = matches[-1].group(1).strip()
    phrase = re.sub(r"^(?:[:=]|is\b)\s*", "", phrase, flags=re.I).strip()
    phrase = re.split(r"(?:<\|end|<｜end|</s>|$)", phrase, maxsplit=1)[0].strip()
    lines = phrase.splitlines()
    if not lines:
        return None
    phrase = lines[0].strip()
    if not phrase:
        return None
    return phrase.rstrip(".。 \t")


def _extract_hash_answer(solution: str) -> str | None:
    if "####" not in solution:
        return None
    answer = solution.rsplit("####", 1)[-1]
    answer = re.split(r"(?:<\|end|<｜end|</s>|$)", answer, maxsplit=1)[0].strip()
    if not answer:
        return None
    answer = answer.splitlines()[0].strip()
    if not answer:
        return None
    return answer.rstrip(".。 \t")


def extract_answer_candidates(solution: str) -> list[str]:
    """Return plausible final-answer candidates without fragmenting general math."""
    raw_answer = extract_answer(solution)
    if raw_answer is None:
        raw_answer = _extract_hash_answer(solution)
    if raw_answer is None:
        raw_answer = _extract_answer_phrase(solution)
    if raw_answer is None:
        return []

    candidates = [raw_answer]
    cleaned = _strip_latex_text_wrappers(raw_answer)
    cleaned = re.sub(r"\\\s+", " ", cleaned)
    cleaned = cleaned.replace("$", "").strip()
    candidates.append(cleaned)

    choice_match = re.search(
        r"(?is)^\s*(?:\\boxed\s*)?\{?\s*(?:choice|option|answer)?\s*"
        r"(?:\(?\s*([A-E])\s*\)?|([A-E])\s*[\.:])"
        r"\s*(?:[-,:;]|\)|\})?\s*(.*)$",
        cleaned,
    )
    if choice_match:
        letter = next(group for group in choice_match.groups()[:2] if group is not None).upper()
        candidates.append(letter)
        remainder = choice_match.group(3).strip()
        remainder = re.sub(r"^(?:[-,:;]|\\qquad|\\quad|\\;|\\,|\s)+", "", remainder).strip()
        if remainder:
            candidates.append(remainder)

    return _dedupe_preserve_order(candidates)


def grade_candidate_against_truth(model_answer: str, ground_truth: str, options: dict[str, str]) -> bool:
    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)

    if not is_correct and model_answer in ["A", "B", "C", "D", "E"]:
        value = options.get(model_answer, model_answer)
        is_correct = grade_answer_mathd(value, ground_truth) or grade_answer_sympy(value, ground_truth)

    if not is_correct and ground_truth in ["A", "B", "C", "D", "E"]:
        value = options.get(ground_truth, ground_truth)
        is_correct = grade_answer_mathd(model_answer, value) or grade_answer_sympy(model_answer, value)

    return is_correct


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, "Invalid problem type: expected 'MATH', but got '{}'".format(
            input.problem_type
        )

        problem = input.problem
        model_response = input.model_response

        options = extract_options_advanced(problem)
        # print(options)

        # Extract solution.
        # print('THOUGHT_DELIMITER_START',THOUGHT_DELIMITER_START,THOUGHT_DELIMITER_END)
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response
            # return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_answers = extract_answer_candidates(model_solution)

        # print(f"++ {model_answer} \t {input.ground_truth.get('answer', None)}")

        if not model_answers:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        ground_truths = normalize_ground_truth(ground_truths)
        try:
            ground_truths = eval(ground_truths)
        except:
            pass
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # print('processed_ground_truths',processed_ground_truths,model_answers)
        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            # if type(ground_truth) == list:
            #    ground_truth = ground_truth[0]
            for model_answer in model_answers:
                is_correct = grade_candidate_against_truth(model_answer, ground_truth, options)
                if is_correct:
                    return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm=False, problem=None):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    problem = prompt_to_text(problem) if problem is not None else solution_str
    ground_truth = normalize_ground_truth(ground_truth)
    reward_response = reward_fn(
        RewardInput(
            problem=problem,
            problem_type=RewardType.MATH,
            model_response=solution_str,
            ground_truth={"answer": ground_truth},
        )
    )
    return reward_response.is_correct


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(
        problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.",
        problem_type=RewardType.MATH,
        model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.",
        ground_truth={"answer": ["x^{4}-2 x^{3}-13 x^{2}+14 x+24"]},
    )  # True
    input = RewardInput(
        problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.",
        problem_type=RewardType.MATH,
        model_response="<think> I am omniscient. </think> The answer is \\boxed{0}.",
        ground_truth={"answer": ["0"]},
    )  # True
    input = RewardInput(
        problem="11. (4 points) The perimeter of a rectangle is 48 cm, and it is known that the length is twice the width. The length of the rectangle is ( )\nA. 8 cm\nB. 16 cm\nC. 24 cm\n\n Let's think step by step and output the final a",
        problem_type=RewardType.MATH,
        model_response="<think>\nOkay, so the question is asking how many hours is 6 minutes. The options are A through E. Let me think about this step by step.\n\nFirst, I know that time conversions often involve dividing or multiplying by 60. Since there are 60 minutes in an hour, to convert minutes to hours, I should divide the number of minutes by 60. So, if I have 6 minutes, I need to divide 6 by 60 to get the equivalent in hours.\n\nLet me write that down: 6 minutes ÷ 60 = ? hours. Let me do the division. 6 divided by 60. Hmm, 6 divided by 60 is the same as 6/60, which simplifies to 1/10. And 1/10 as a decimal is 0.1. So that would be 0.1 hours. \n\nLooking at the options, option B is 0.1. So that should be the answer. But let me double-check to make sure I didn't make a mistake. \n\nAnother way to think about it: if 60 minutes is 1 hour, then 1 minute is 1/60 hours. Therefore, 6 minutes would be 6*(1/60) = 6/60 = 0.1 hours. Yep, that's the same result. \n\nLet me check the other options to see why they're wrong. Option A is 0.06. That would be if you did 6 divided by 100, but that's not the case here. Option C is 0.6, which is 6/10, but that's not correct because 6 minutes is less than an hour. Option D is 10, which is way too big. Option E is 360, which is 6 minutes multiplied by 60, but that would be converting hours to minutes, not the other way around. So definitely, B is the correct answer.\n</think>\n\nTo determine how many hours 6 minutes corresponds to, we must understand the relationship between minutes and hours. There are **60 minutes in 1 hour**, so to convert minutes to hours, we divide the number of minutes by 60.\n\n---\n\n### Step-by-step Conversion:\n\n$$\n6 \\text{ minutes} \\div 60 = 0.1 \\text{ hours}\n$$\n\nThis is equivalent to:\n\n$$\n\\frac{6}{60} = \\frac{1}{10} = 0.1\n$$\n\n---\n\n### Analysis of Options:\n\n- **A. 0.06** – Incorrect. This would be the result if we divided 6 by 100, not 60.\n- **B. 0.1** – **Correct**. This is the accurate result of the conversion.\n- **C. 0.6** – Incorrect. This would represent 36 minutes (0.6 × 60 = 36).\n- **D. 10** – Incorrect. This would be the number of minutes in 10 hours, not the reverse.\n- **E. 360** – Incorrect. This is 6 minutes multiplied by 60, which would be the number of seconds in 6 minutes, not hours.\n\n---\n\n### Final Answer:\n\n$$\n\\boxed{B}\n$$",
        ground_truth={"answer": '["0.1"]'},
    )  # False
    output = reward(input)
    print(output)
