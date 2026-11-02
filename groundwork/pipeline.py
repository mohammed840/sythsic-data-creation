"""A bounded AutoData-inspired loop; code owns the answer key."""

from .grading import grade
from .errors import ModelError, ModelResponseError


def generate(task, models, max_revisions=2):
    attempts = []
    repair_available = True
    stage = "proposal"
    feedback = "Generate a clear question requiring the supplied policy and the selected case."

    def call_stage(callback, entry):
        nonlocal repair_available
        try:
            return callback(False)
        except ModelResponseError as exc:
            entry.setdefault("format_errors", []).append({"stage": stage, "reason": str(exc)})
            if not repair_available:
                raise
            repair_available = False
            entry["format_repair_attempted"] = True
            return callback(True)

    for attempt in range(max_revisions + 1):
        entry = {"round": attempt, "errors": []}
        attempts.append(entry)
        try:
            stage = "proposal"
            proposal = call_stage(lambda repair: models.propose(task, feedback, repair=repair), entry)
            entry["proposal"] = proposal
            instruction = proposal.get("instruction")
            candidate = proposal.get("answer")
            errors = entry["errors"]
            if not isinstance(instruction, str) or not 10 <= len(instruction) <= 1500:
                errors.append("Instruction must be 10–1500 characters")
            elif task["case_id"] not in instruction:
                errors.append("Instruction must name the selected case ID")
            if not grade(task["expected"], candidate):
                errors.append("Candidate answer disagrees with the executable rule; re-read the policy and case")
            if errors:
                feedback = "; ".join(errors)
                continue
            prompt = instruction + "\n\nRequired task: " + task["instruction"]
            stage = "reviewer"
            review = call_stage(lambda repair: models.review(task, instruction, candidate, repair=repair), entry)
            entry["review"] = review
            if review.get("accepted") is not True:
                feedback = str(review.get("feedback", "The reviewer rejected the question"))[:2000]
                continue
            stage = "target"
            target_answer = call_stage(lambda repair: models.solve(task, prompt, repair=repair), entry)
            target_correct = grade(task["expected"], target_answer)
            entry.update(target_answer=target_answer, target_correct=target_correct)
            if target_correct and attempt < max_revisions:
                feedback = ("The target solved this question. Try a clearer application-focused wording "
                            "requiring the policy and case record, without adding facts, giving away the answer, "
                            "changing the task, or making it ambiguous. " + str(review.get("feedback", "")))
                continue
            return {**task, "instruction": prompt, "accepted": True,
                    "review_status": "model_reviewed", "difficulty": "easy" if target_correct else "target_missed",
                    "attempts": attempts}
        except Exception as exc:
            # Only our own sanitized model errors may contribute diagnostic text.
            return {**task, "accepted": False, "review_status": "error", "error_stage": stage,
                    "error_type": type(exc).__name__,
                    "error_reason": str(exc) if isinstance(exc, ModelError) else "Unexpected stage failure",
                    "attempts": attempts}
    return {**task, "accepted": False, "review_status": "rejected", "attempts": attempts}


class OfflineModels:
    """Explicit fixture mode to check exports without claiming any model generation."""

    usage = {"mode": "offline_template", "calls": 0, "cost_usd": 0}

    def propose(self, task, feedback, *, repair=False):
        return {"instruction": task["instruction"], "answer": task["expected"]}

    def solve(self, task, instruction, *, repair=False):
        return task["expected"]

    def review(self, task, instruction, candidate, *, repair=False):
        return {"accepted": True, "feedback": "Template demonstration; not an LLM review"}
