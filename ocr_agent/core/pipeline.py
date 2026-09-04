from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List

from ..data_models import SingleRequest, StepTrace, PipelineResult, PipelineConfig
from ..capability_constraints import ActionItem, CapabilityConstraint
from ..prompts import SYSTEM_PROMPT, build_initial_prompt, build_reflection_prompt, build_revision_prompt, build_verification_prompt, build_blind_review_prompt
from ..model_service import VisionLanguageBackend
from ..visual_tools import VisualToolExecutor


class OCRAgentPipeline:
    def __init__(self, backend: VisionLanguageBackend, config: PipelineConfig) -> None:
        self.backend = backend
        self.config = config
        self.capability_constraint = CapabilityConstraint(config.blocked_action_patterns)
        self.visual_tool_executor = VisualToolExecutor(config.visual_tool_dir)

    def run(self, request: SingleRequest) -> PipelineResult:
        # CoT is_independent_ablation;other_variants_add_reflection,capability_or_memory.
        enable_chain_of_thought = self.config.variant_name == "cot"
        initial_prompt = build_initial_prompt(
            request.question,
            request.document_context,
            enable_chain_of_thought,
            request.task_type,
            request.dataset_name,
        )
        initial_answer = self.backend.generate(SYSTEM_PROMPT, initial_prompt, request.image_paths)
        initial_reasoning = self._extract_reasoning_text(initial_answer) if enable_chain_of_thought else ""
        initial_conclusion = self._extract_conclusion_text(initial_answer)

        if self.config.variant_name in {"naive", "cot"}:
            return PipelineResult(
                variant_name=self.config.variant_name,
                initial_answer=initial_answer,
                final_answer=initial_answer,
                initial_conclusion=initial_conclusion,
                final_conclusion=initial_conclusion,
                text_changed=False,
                conclusion_changed=False,
                memory=[],
                traces=[],
                initial_reasoning=initial_reasoning,
            )

        current_answer = initial_answer
        current_conclusion = initial_conclusion
        memory: List[str] = []
        traces: List[StepTrace] = []

        for round_index in range(1, request.max_iterations + 1):
            memory_before = list(memory)
            reflection_prompt = build_reflection_prompt(
                question=request.question,
                document_context=request.document_context,
                previous_answer=current_answer,
                memory=memory,
                reflection_language=self.config.reflection_language,
                use_memory=self._use_memory_reflection(),
                task_type=request.task_type,
                dataset_name=request.dataset_name,
            )
            raw_reflection = self.backend.generate(SYSTEM_PROMPT, reflection_prompt, request.image_paths)
            reflection_dict = self._parse_reflection_text(raw_reflection)
            reflection_decision = str(reflection_dict.get("decision", "KEEP"))
            reflection_parse_success = bool(reflection_dict.get("parse_success", False))
            error_found = reflection_decision == "REVISE"
            visual_evidence = reflection_dict.get("evidence", [])
            diagnosis = reflection_dict.get("diagnosis", [])
            action_plan = reflection_dict.get("plan", [])
            avoid_repeated_prompt = reflection_dict.get("avoid_repetition", "")

            if not self.config.enable_capability_reflection:
                feasible_actions = [ActionItem(text=action, feasible=True, reason="Self-Refine keep_all_actions.") for action in action_plan]
            else:
                feasible_actions = self._build_action_list(action_plan)

            structured_memory = self._build_structured_memory_item(
                round_index=round_index,
                diagnosis=diagnosis,
                action_plan=action_plan,
                avoid_repeated_prompt=avoid_repeated_prompt,
                previous_answer=current_answer,
                feasible_actions=feasible_actions,
                error_found=error_found,
                visual_evidence=visual_evidence,
            )
            if self._use_memory_reflection():
                memory.append(self._build_memory_item(structured_memory))
            memory_after = list(memory)

            # The paper's loop sends the feasible revision plan back to the
            # model directly; evidence/verification are optional extensions.
            revision_requested = (
                error_found
                and reflection_parse_success
                and self._has_valid_evidence(visual_evidence)
                and (not self.config.enable_capability_reflection or any(item.feasible for item in feasible_actions))
            )
            visual_tool_image_paths: List[str] = []
            if revision_requested and self.config.enable_visual_tools:
                visual_tool_image_paths = self.visual_tool_executor.generate_high_res_tiles(request.image_paths)
            revision_image_paths = request.image_paths + visual_tool_image_paths
            visual_tool_prompt = (
                "The first image is the original. The remaining images are overlapping high-resolution tiles "
                "from that same image; use them to inspect small text, but preserve the original image context.\n"
                if visual_tool_image_paths else ""
            )
            revision_candidate_answer = current_answer
            revision_evidence: List[str] = []
            revision_parse_success = False
            if revision_requested:
                revision_prompt = build_revision_prompt(
                    question=request.question,
                    document_context=request.document_context,
                    previous_answer=current_answer,
                    reflection_evidence=" ".join(visual_evidence),
                    actions=feasible_actions,
                    memory=memory,
                    use_memory=self._use_memory_reflection(),
                    task_type=request.task_type,
                    dataset_name=request.dataset_name,
                )
                raw_revision = self.backend.generate(SYSTEM_PROMPT, visual_tool_prompt + revision_prompt, revision_image_paths)
                revision_object = self._parse_revision_text(raw_revision)
                revision_candidate_answer = revision_object["answer"] or current_answer
                revision_evidence = revision_object["evidence"]
                revision_parse_success = revision_object["parse_success"]
            if self.config.enable_answer_filter:
                accept_revision = (
                    revision_requested
                    and revision_parse_success
                    and bool(revision_evidence)
                    and self._is_accept_revision_answer(
                        question=request.question, current_answer=current_answer, revision_candidate_answer=revision_candidate_answer
                    )
                )
            else:
                # Faithful OCR-Agent mode follows Algorithm 1:
                # Reflect -> capability filter -> Refine -> memory update.
                accept_revision = (
                    revision_requested
                    and revision_parse_success
                    and bool(revision_candidate_answer)
                    and self._has_valid_evidence(revision_evidence)
                )

            verification_requested = accept_revision and self.config.enable_verification
            verification_parse_success = False
            verification_decision = "REJECT"
            if verification_requested:
                verification_prompt = build_verification_prompt(
                    question=request.question,
                    document_context=request.document_context,
                    previous_answer=current_answer,
                    candidate_answer=revision_candidate_answer,
                    reflection_evidence=" ".join(visual_evidence),
                    revision_evidence=" ".join(revision_evidence),
                    task_type=request.task_type,
                    dataset_name=request.dataset_name,
                )
                raw_verification = self.backend.generate(SYSTEM_PROMPT, visual_tool_prompt + verification_prompt, revision_image_paths)
                verification_object = self._parse_verification_text(raw_verification)
                verification_parse_success = verification_object["parse_success"]
                verification_decision = verification_object["decision"]
                accept_revision = verification_parse_success and verification_decision == "ACCEPT"
            blind_review_requested = (
                accept_revision
                and self.config.enable_blind_review
                and self._needs_blind_review(request.question)
            )
            blind_review_parse_success = False
            blind_review_answer = ""
            if blind_review_requested:
                blind_review_prompt = build_blind_review_prompt(
                    request.question,
                    request.document_context,
                    request.task_type,
                    request.dataset_name,
                )
                raw_blind_review = self.backend.generate(SYSTEM_PROMPT, visual_tool_prompt + blind_review_prompt, revision_image_paths)
                blind_review_object = self._parse_revision_text(raw_blind_review)
                blind_review_parse_success = blind_review_object["parse_success"]
                blind_review_answer = blind_review_object["answer"]
                accept_revision = (
                    blind_review_parse_success
                    and self._normalize_comparison_text(blind_review_answer) == self._normalize_comparison_text(revision_candidate_answer)
                )
            revised_answer = revision_candidate_answer if accept_revision else current_answer
            revised_conclusion = self._extract_conclusion_text(revised_answer)

            feasible_action_texts = [item.text for item in feasible_actions if item.feasible]
            blocked_action_texts = [item.text for item in feasible_actions if not item.feasible]
            text_changed = revised_answer.strip() != current_answer.strip()
            conclusion_changed = revised_conclusion != current_conclusion

            traces.append(
                StepTrace(
                    round_index=round_index,
                    previous_answer=current_answer,
                    revised_answer=revised_answer,
                    previous_conclusion=current_conclusion,
                    revised_conclusion=revised_conclusion,
                    text_changed=text_changed,
                    conclusion_changed=conclusion_changed,
                    answer_changed=conclusion_changed,
                    raw_reflection=raw_reflection,
                    diagnosis=diagnosis,
                    action_plan=action_plan,
                    avoid_repeated_prompt=avoid_repeated_prompt,
                    feasible_actions=[asdict(item) for item in feasible_actions],
                    feasible_action_texts=feasible_action_texts,
                    blocked_action_texts=blocked_action_texts,
                    feasible_action_count=len(feasible_action_texts),
                    blocked_action_count=len(blocked_action_texts),
                    memory_before=memory_before,
                    memory_after=memory_after,
                    structured_memory=structured_memory,
                    reflection_decision=reflection_decision,
                    reflection_parse_success=reflection_parse_success,
                    revision_requested=revision_requested,
                    revision_evidence=revision_evidence,
                    revision_parse_success=revision_parse_success,
                    revision_accepted=accept_revision,
                    verification_requested=verification_requested,
                    verification_parse_success=verification_parse_success,
                    verification_decision=verification_decision,
                    blind_review_requested=blind_review_requested,
                    blind_review_parse_success=blind_review_parse_success,
                    blind_review_answer=blind_review_answer,
                    visual_tool_called=bool(visual_tool_image_paths),
                    visual_tool_image_paths=visual_tool_image_paths,
                )
            )
            current_answer = revised_answer
            current_conclusion = revised_conclusion

        return PipelineResult(
            variant_name=self.config.variant_name,
            initial_answer=initial_answer,
            final_answer=current_answer,
            initial_conclusion=initial_conclusion,
            final_conclusion=current_conclusion,
            text_changed=initial_answer.strip() != current_answer.strip(),
            conclusion_changed=initial_conclusion != current_conclusion,
            memory=memory,
            traces=traces,
            initial_reasoning=initial_reasoning,
        )

    def _use_memory_reflection(self) -> bool:
        return self.config.enable_memory_reflection

    @classmethod
    def _is_accept_revision_answer(cls, question: str, current_answer: str, revision_candidate_answer: str) -> bool:
        """reject_obvious_expansion,format_breaking_revision,avoid_reflection_overwriting_correct."""
        original_answer = cls._clean_answer_shell(current_answer)
        candidate_answer = cls._clean_answer_shell(revision_candidate_answer)
        if not candidate_answer or candidate_answer == original_answer:
            return False

        # space,code_fence_or JSON shell_change_not_revision,avoid_repeated_acceptance.
        original_conclusion = cls._extract_conclusion_text(original_answer)
        candidate_conclusion = cls._extract_conclusion_text(candidate_answer)
        if cls._normalize_comparison_text(original_conclusion) == cls._normalize_comparison_text(candidate_conclusion):
            return False

        lower_question = question.lower()
        # English translation tasks must not accept an unchanged source-language transcription.
        if re.search(r"translate .*\bto english\b|translate_to_english|translate_to_english", lower_question, flags=re.IGNORECASE | re.DOTALL):
            if re.search(r"[\u3400-\u9fff]", candidate_conclusion):
                return False
        options = re.findall(
            r"(?:options?|choices?)\s*:\s*([^\n]+)", question,
            flags=re.IGNORECASE,
        )
        if options:
            option = [
                item.strip(" \\'\".,;:()[]")
                for item in re.split(r",|,", options[-1])
                if item.strip(" \\'\".,;:()[]")
            ]
            if option and not any(re.fullmatch(re.escape(item), candidate_answer, re.IGNORECASE) for item in option):
                return False

        if "json" in lower_question:
            original_object = cls._parse_answer_object(original_answer)
            candidate_object = cls._parse_answer_object(candidate_answer)
            if candidate_object is None:
                return False
            candidate_keys = set(candidate_object)
            # question_quotes_may_contain_examples;textoriginal_answer_field
            # as_allowed_set,reject_unjustified_fields.
            if original_object is not None and not candidate_keys.issubset(set(original_object)):
                return False

        # in_short_answer_tasks,extra_explanation_is_regression.
        if len(original_answer) <= 80 and len(candidate_answer) > max(120, len(original_answer) * 2.5):
            return False
        if re.match(r"^(?:answer|the answer|refined answer|based on)\s*:", candidate_answer, re.IGNORECASE):
            return False

        # number/coordinate_tasks_direct_output;keep_original_answer.
        if re.search(r"how many|exact number|normalized coordinates|bounding box", lower_question):
            if len(re.findall(r"\d+", candidate_answer)) > 8 and len(re.findall(r"\d+", original_answer)) <= 8:
                return False
        return True

    @staticmethod
    def _needs_blind_review(question: str) -> bool:
        return bool(re.search(
            r"options?|choices?|candidate answer|multiple.?choice|option|candidate_answer|how many|exact number|mathematical",
            question,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _normalize_comparison_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    @staticmethod
    def _clean_answer_shell(answer: str) -> str:
        text = str(answer or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    @staticmethod
    def _parse_answer_object(answer: str) -> Dict[str, Any] | None:
        try:
            start = answer.find("{")
            end = answer.rfind("}")
            if start < 0 or end <= start:
                return None
            object = json.loads(answer[start : end + 1])
            return object if isinstance(object, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_conclusion_text(answer_text: str) -> str:
        cleaned_answer = OCRAgentPipeline._clean_answer_shell(answer_text)
        try:
            object = json.loads(cleaned_answer)
            if isinstance(object, dict) and "answer" in object:
                return OCRAgentPipeline._normalize_conclusion_text(str(object["answer"]))
        except (json.JSONDecodeError, TypeError):
            pass
        lines = [row.strip() for row in cleaned_answer.replace("\r\n", "\n").split("\n") if row.strip()]
        if not lines:
            return ""
        candidate_lines = [
            re.sub(r"^\d+[\.\)]\s*", "", row).strip()
            for row in lines
            if not re.match(r"^\d+[\.\)]\s*", row)
        ]
        if candidate_lines:
            return OCRAgentPipeline._normalize_conclusion_text(candidate_lines[-1])
        return OCRAgentPipeline._normalize_conclusion_text(lines[-1])

    @staticmethod
    def _normalize_conclusion_text(text: str) -> str:
        result = re.sub(r"\s+", " ", text).strip()
        if not result:
            return ""
        modes = [
            r"^(?:the|this|it)\b.*?\b(?:is|are|was|were)\b\s+(.+?)\.?$",
            r"^(?:the|this|it)\b.*?\b(?:appears to be|seems to be)\b\s+(.+?)\.?$",
        ]
        for mode in modes:
            textresult = re.match(mode, result, flags=re.IGNORECASE)
            if textresult:
                candidate_text = textresult.group(1).strip()
                if candidate_text:
                    result = candidate_text
                    break
        result = result.strip(" \t\n\r\"'.,;:()[]{}")
        return result

    @staticmethod
    def _extract_reasoning_text(answer_text: str) -> str:
        cleaned_answer = OCRAgentPipeline._clean_answer_shell(answer_text)
        try:
            object = json.loads(cleaned_answer)
            if isinstance(object, dict):
                return str(object.get("reasoning", "")).strip()
        except (json.JSONDecodeError, TypeError):
            pass
        return ""

    def _build_action_list(self, action_plan: List[str]) -> List[ActionItem]:
        if self.config.enable_capability_reflection:
            return self.capability_constraint.evaluate(action_plan)
        return [ActionItem(text=action, feasible=True, reason="capability_constraints_disabled.") for action in action_plan]

    @staticmethod
    def _parse_reflection_text(raw_reflection: str) -> Dict[str, Any]:
        try:
            cleaned_text = OCRAgentPipeline._clean_reflection_text(raw_reflection)
            reflection_dict = json.loads(cleaned_text)
            if not isinstance(reflection_dict, dict):
                raise ValueError("reflection_output_must_be JSON object.")
            decision = str(reflection_dict.get("decision", "")).strip().upper()
            if decision not in {"KEEP", "REVISE"}:
                decision = "REVISE" if OCRAgentPipeline._normalize_bool_field(reflection_dict.get("error_found")) else "KEEP"
            evidence = OCRAgentPipeline._normalize_text_list_field(reflection_dict.get("evidence", []))
            actions = OCRAgentPipeline._normalize_text_list_field(
                reflection_dict.get("action", reflection_dict.get("plan", []))
            )
            if decision == "REVISE" and not OCRAgentPipeline._has_valid_evidence(evidence):
                decision = "KEEP"
            return {
                "parse_success": True,
                "decision": decision,
                "error_found": decision == "REVISE",
                "evidence": evidence,
                "diagnosis": OCRAgentPipeline._normalize_text_list_field(reflection_dict.get("diagnosis", [])),
                "plan": actions,
                "avoid_repetition": OCRAgentPipeline._normalize_text_field(reflection_dict.get("avoid_repetition", "")),
            }
        except Exception:
            return {
                "parse_success": False,
                "decision": "KEEP",
                "error_found": False,
                "evidence": [],
                "diagnosis": ["Failed to parse reflection as JSON; keeping raw reflection as diagnosis."],
                "plan": [],
                "avoid_repetition": "",
            }

    @staticmethod
    def _parse_revision_text(raw_revision: str) -> Dict[str, Any]:
        try:
            cleaned_text = OCRAgentPipeline._clean_reflection_text(raw_revision)
            revision_object = json.loads(cleaned_text)
            if not isinstance(revision_object, dict):
                raise ValueError("revision_output_must_be JSON object.")
            answer = str(revision_object.get("answer", "")).strip()
            evidence = OCRAgentPipeline._normalize_text_list_field(revision_object.get("evidence", []))
            evidence = [evidence for evidence in evidence if OCRAgentPipeline._has_valid_evidence([evidence])]
            return {"parse_success": bool(answer), "answer": answer, "evidence": evidence}
        except Exception:
            return {"parse_success": False, "answer": "", "evidence": []}

    @staticmethod
    def _parse_verification_text(raw_verification: str) -> Dict[str, Any]:
        try:
            cleaned_text = OCRAgentPipeline._clean_reflection_text(raw_verification)
            verification_object = json.loads(cleaned_text)
            if not isinstance(verification_object, dict):
                raise ValueError("verification_output_must_be JSON object.")
            decision = str(verification_object.get("decision", "")).strip().upper()
            if decision not in {"ACCEPT", "REJECT"}:
                raise ValueError("verification_decision_must_be ACCEPT or REJECT.")
            return {"parse_success": True, "decision": decision}
        except Exception:
            return {"parse_success": False, "decision": "REJECT"}

    @staticmethod
    def _clean_reflection_text(raw_reflection: str) -> str:
        text = raw_reflection.strip()
        code_block_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
        if code_block_match:
            text = code_block_match.group(1).strip()
        return OCRAgentPipeline._extract_first_complete_json_object(text)

    @staticmethod
    def _extract_first_complete_json_object(text: str) -> str:
        start_pos = text.find("{")
        if start_pos < 0:
            return text
        depth = 0
        inside_string = False
        in_escape = False
        for position in range(start_pos, len(text)):
            character = text[position]
            if inside_string:
                if in_escape:
                    in_escape = False
                elif character == "\\":
                    in_escape = True
                elif character == '"':
                    inside_string = False
                continue
            if character == '"':
                inside_string = True
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start_pos : position + 1]
        return text

    @staticmethod
    def _normalize_text_list_field(field_value: Any) -> List[str]:
        if isinstance(field_value, list):
            return [str(item).strip() for item in field_value if str(item).strip()]
        if isinstance(field_value, str):
            text = field_value.strip()
            return [text] if text else []
        if field_value is None:
            return []
        text = str(field_value).strip()
        return [text] if text else []

    @staticmethod
    def _normalize_text_field(field_value: Any) -> str:
        if field_value is None:
            return ""
        if isinstance(field_value, list):
            return " ".join(str(item).strip() for item in field_value if str(item).strip())
        return str(field_value).strip()

    @staticmethod
    def _has_valid_evidence(evidence: List[str]) -> bool:
        placeholder_patterns = [
            r"^\s*$",
            r"visible evidence or empty string",
            r"visible evidence for this answer",
            r"short visible evidence",
            r"real visible text",
            r"visible text,? value,? or location",
            r"empty string",
            r"^none$",
            r"^n/a$",
            r"^unknown$",
            r"uncertain",
            r"guess",
        ]
        for evidence in evidence:
            text = str(evidence or "").strip()
            if not text:
                continue
            lower_text = text.lower()
            if any(re.search(mode, lower_text, flags=re.IGNORECASE) for mode in placeholder_patterns):
                continue
            return True
        return False

    @staticmethod
    def _normalize_bool_field(field_value: Any) -> bool:
        if isinstance(field_value, bool):
            return field_value
        return str(field_value).strip().lower() in {"1", "true", "yes", "revise"}

    @staticmethod
    def _build_structured_memory_item(
        round_index: int,
        diagnosis: List[str],
        action_plan: List[str],
        avoid_repeated_prompt: str,
        previous_answer: str,
        feasible_actions: List[ActionItem],
        error_found: bool,
        visual_evidence: List[str],
    ) -> Dict[str, Any]:
        return {
            "iteration": round_index,
            "diagnosis": diagnosis,
            "plan": action_plan,
            "avoid_repetition": avoid_repeated_prompt,
            "previous_answer": previous_answer,
            "feasible_actions": [item.text for item in feasible_actions if item.feasible],
            "blocked_actions": [item.text for item in feasible_actions if not item.feasible],
            "error_found": error_found,
            "evidence": visual_evidence,
        }

    @staticmethod
    def _build_memory_item(structured_memory: Dict[str, Any]) -> str:
        # pass_full_structure_across_rounds,preserve_action_plan.
        return json.dumps(structured_memory, ensure_ascii=False, separators=(",", ":"))
