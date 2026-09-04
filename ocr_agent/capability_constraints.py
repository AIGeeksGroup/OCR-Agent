from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ActionItem:
    text: str
    feasible: bool
    reason: str


class CapabilityConstraint:
    """filter_reflection_actions."""

    def __init__(self, blocked_action_patterns: Iterable[str]) -> None:
        self.blocked_action_patterns = [item.lower() for item in blocked_action_patterns]

    def evaluate(self, action_plan: Iterable[str]) -> List[ActionItem]:
        result: List[ActionItem] = []
        for action in action_plan:
            lower_action = action.strip().lower()
            matched_rule = next((rule for rule in self.blocked_action_patterns if rule in lower_action), None)
            if matched_rule:
                result.append(ActionItem(text=action, feasible=False, reason=f"matched_blocked_rule: {matched_rule}"))
            else:
                result.append(ActionItem(text=action, feasible=True, reason="action is feasible in the current VLM reasoning context."))
        return result
