from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Dict, List

import numpy as np
import openai
import torch


class LLMTextLabelClassifier:
    """Classifies object labels to nearest class name using an LLM."""

    def __init__(
        self,
        device: str = "cuda:0",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.device = device
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self._cache: Dict[tuple[str, tuple[str, ...]], int] = {}

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set. It is required for classify_label_with=llm.")
        openai.api_key = key

    def _extract_query_text(self, obj: dict) -> str:
        if "label" in obj and obj["label"] not in (None, ""):
            return str(obj["label"])
        if "class_name" in obj and obj["class_name"]:
            if isinstance(obj["class_name"], list):
                return str(obj["class_name"][0])
            return str(obj["class_name"])
        return "unknown"

    def _fallback_by_string_similarity(self, query_text: str, candidate_names: List[str]) -> int:
        query = query_text.lower().strip()
        best_idx = 0
        best_score = -1.0
        for i, cand in enumerate(candidate_names):
            score = SequenceMatcher(None, query, cand.lower()).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _parse_index(self, text: str, n_candidates: int) -> int | None:
        match = re.search(r"-?\d+", text)
        if match is None:
            return None
        idx = int(match.group(0))
        if idx < 0 or idx >= n_candidates:
            return None
        return idx

    def _pick_class_with_llm(self, query_text: str, candidate_names: List[str]) -> tuple[int, bool, str]:
        cache_key = (query_text.lower().strip(), tuple(candidate_names))
        if cache_key in self._cache:
            return self._cache[cache_key], False, "cache"

        numbered = "\n".join([f"{i}: {name}" for i, name in enumerate(candidate_names)])
        n_candidates = len(candidate_names)
        last_response = ""
        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                user_prompt = (
                    "You are a strict semantic label matcher.\n"
                    "Task: map the query label to the single closest class candidate.\n"
                    f"Return ONLY one integer index in [0, {n_candidates - 1}]. No extra text.\n\n"
                    f"Query label: {query_text}\n"
                    "Candidate classes:\n"
                    f"{numbered}\n"
                )
            else:
                user_prompt = (
                    f"Your previous response was invalid: {last_response}\n"
                    f"Return ONLY one integer in [0, {n_candidates - 1}]. No words.\n"
                    f"Query label: {query_text}\n"
                    "Candidate classes:\n"
                    f"{numbered}\n"
                )

            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Output only a valid class index integer."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=8,
            )
            last_response = completion["choices"][0]["message"]["content"].strip()
            idx = self._parse_index(last_response, n_candidates)
            if idx is not None:
                self._cache[cache_key] = idx
                return idx, False, last_response

        fallback_idx = self._fallback_by_string_similarity(query_text, candidate_names)
        self._cache[cache_key] = fallback_idx
        return fallback_idx, True, last_response

    @torch.no_grad()
    def classify_object_list(
        self,
        objects,
        class_names: List[str],
        ignore_index: np.ndarray,
        object_feat_key: str = "clip_ft",
        print_assignments: bool = True,
    ) -> torch.Tensor:
        del object_feat_key  # interface compatibility with embedding-based classifiers

        all_ids = np.arange(len(class_names), dtype=np.int64)
        keep_ids = np.setdiff1d(all_ids, np.asarray(ignore_index, dtype=np.int64))
        candidate_names = [class_names[int(i)] for i in keep_ids]
        if len(candidate_names) == 0:
            raise ValueError("No class candidates left after ignore_index filtering.")

        missing_label_indices = [i for i in range(len(objects)) if "label" not in objects[i]]
        if missing_label_indices:
            raise ValueError(
                "LLM classification requires a 'label' field for every object. "
                f"Missing 'label' in {len(missing_label_indices)} objects, e.g. indices "
                f"{missing_label_indices[:10]}."
            )

        if print_assignments:
            print("Assigned object classes:")

        n_fallback = 0
        object_class_ids = []
        for i in range(len(objects)):
            query_text = self._extract_query_text(objects[i])
            local_idx, used_fallback, raw_response = self._pick_class_with_llm(query_text, candidate_names)
            global_idx = int(keep_ids[local_idx])
            object_class_ids.append(global_idx)
            if used_fallback:
                n_fallback += 1
                print(
                    f"Warning: invalid LLM output for object {i} (query: {query_text!r}, response: {raw_response!r}); "
                    f"using fallback class {global_idx} ({class_names[global_idx]})."
                )

            if print_assignments:
                print(
                    f"Object {i}: (orig: {query_text}) class {global_idx} - {class_names[global_idx]}"
                )

        if n_fallback > 0:
            print(f"LLM fallback used for {n_fallback}/{len(objects)} objects.")

        return torch.tensor(object_class_ids, dtype=torch.long)
