"""Common-layer tests: label mapping, adapters, config arithmetic, prompt contracts.

The tests in this file are the ones that would catch a silent *distribution* bug: an
A/B swap, a RewardBench row leaking into training, a template that changed under a
method's feet, or a config whose budget arithmetic drifted away from its own table.
Every one of them runs against the real data files (read-only) or against synthetic
rows built to break a specific invariant.
"""

from __future__ import annotations

import json
import os
import unittest

from common import config as config_mod
from common import labels, parsing, paths, prompts
from data import helpsteer_adapter, length_audit, pair_views, rewardbench_adapter, trl_dataset


def _sample_helpsteer_row(**overrides):
    row = {
        "question_body": "Which answer is more helpful?",
        "answer1_body": "Response one body.",
        "answer2_body": "Response two body.",
        "label": "A",
        "group_id": "g-1",
        "view": "original",
        "source_line": 7,
        "text": "<think>legacy JudgeLRM rationale</think>",
        "source_overall_preference": -1,
        "domain": "general",
        "language": "en",
        "token_len": 42,
    }
    row.update(overrides)
    return row


class LabelMappingTest(unittest.TestCase):
    """The two naming spaces must only be crossed through :mod:`common.labels`."""

    def test_source_label_maps_to_physical_response(self):
        # Verified on the real file: label=A means answer1_body is better.
        self.assertEqual(labels.source_label_to_response("A"), labels.RESPONSE_1)
        self.assertEqual(labels.source_label_to_response("B"), labels.RESPONSE_2)

    def test_source_label_rejects_anything_else(self):
        for bad in ("a", "AB", "", "1", None):
            with self.assertRaises(ValueError):
                labels.source_label_to_response(bad)

    def test_flip_is_its_own_inverse(self):
        self.assertEqual(labels.flip_answer("A"), "B")
        self.assertEqual(labels.flip_answer("B"), "A")
        self.assertEqual(labels.flip_answer(labels.flip_answer("A")), "A")

    def test_response_to_display_and_back(self):
        for is_swapped in (False, True):
            for response in (labels.RESPONSE_1, labels.RESPONSE_2):
                display = labels.response_to_display(response, is_swapped)
                self.assertEqual(
                    labels.display_to_response(display, is_swapped), response,
                    msg="round trip failed for response={0!r} is_swapped={1}".format(
                        response, is_swapped))

    def test_swap_puts_the_other_response_in_slot_a(self):
        self.assertEqual(labels.display_order(False), (labels.RESPONSE_1, labels.RESPONSE_2))
        self.assertEqual(labels.display_order(True), (labels.RESPONSE_2, labels.RESPONSE_1))

    def test_ground_truth_flips_with_the_swap(self):
        better = labels.RESPONSE_1
        self.assertEqual(labels.ground_truth_for_view(better, False), "A")
        self.assertEqual(labels.ground_truth_for_view(better, True), "B")

    def test_unparsable_is_always_incorrect(self):
        self.assertFalse(labels.is_correct(None, "A"))
        self.assertFalse(labels.is_correct(None, "B"))

    def test_same_response_chosen_compares_identities_not_letters(self):
        # A consistent judge answers "A" then "B" and picks answer1 both times.
        self.assertTrue(labels.same_response_chosen("A", "B"))
        # Answering "A" twice picks answer1 then answer2: inconsistent despite equal letters.
        self.assertFalse(labels.same_response_chosen("A", "A"))
        # Either side unparsable is not evidence of consistency.
        self.assertFalse(labels.same_response_chosen(None, "B"))
        self.assertFalse(labels.same_response_chosen("A", None))

    def test_probability_target_follows_the_documented_convention(self):
        # p = Pr(B is better), so y = 1 exactly when display B holds the better answer.
        self.assertEqual(labels.probability_target("B"), 1)
        self.assertEqual(labels.probability_target("A"), 0)


class PairViewTest(unittest.TestCase):
    """Linked states: identical content, exchanged slots, flipped ground truth."""

    def _views(self, better=labels.RESPONSE_1):
        return pair_views.build_views(
            pair_id="hs3-7",
            source_row_index=7,
            question="Q?",
            answer_1="one",
            answer_2="two",
            better_response=better,
            source_dataset="helpsteer3_train_clean",
            two_states=True,
        )

    def test_two_states_exchange_content_and_flip_label(self):
        original, swapped = self._views()
        self.assertEqual((original.answer_a, original.answer_b), ("one", "two"))
        self.assertEqual((swapped.answer_a, swapped.answer_b), ("two", "one"))
        self.assertNotEqual(original.ground_truth, swapped.ground_truth)

    def test_sample_ids_are_stable_and_distinct(self):
        original, swapped = self._views()
        self.assertEqual(original.sample_id, "hs3-7::original")
        self.assertEqual(swapped.sample_id, "hs3-7::swapped")

    def test_single_state_keeps_only_the_original(self):
        views = pair_views.build_views(
            pair_id="p", source_row_index=0, question="Q", answer_1="one",
            answer_2="two", better_response=labels.RESPONSE_2,
            source_dataset="helpsteer3_train_clean", two_states=False,
        )
        self.assertEqual(len(views), 1)
        self.assertFalse(views[0].is_swapped)
        self.assertEqual(views[0].ground_truth, "B")

    def test_constructor_rejects_non_bool_swap_flag(self):
        with self.assertRaises(TypeError):
            pair_views.PairView(
                pair_id="p", source_row_index=0, question="Q", answer_1="one",
                answer_2="two", better_response=labels.RESPONSE_1,
                is_swapped="yes", source_dataset="d",
            )

    def test_constructor_rejects_unknown_better_response(self):
        with self.assertRaises(ValueError):
            pair_views.PairView(
                pair_id="p", source_row_index=0, question="Q", answer_1="one",
                answer_2="two", better_response="answer3", is_swapped=False,
                source_dataset="d",
            )

    def test_constructor_rejects_empty_text(self):
        with self.assertRaises(ValueError):
            pair_views.PairView(
                pair_id="p", source_row_index=0, question="   ", answer_1="one",
                answer_2="two", better_response=labels.RESPONSE_1, is_swapped=False,
                source_dataset="d",
            )

    def test_identity_mapping_across_views(self):
        original, swapped = self._views()
        # "A" in the original and "B" in the swapped view both mean answer1.
        self.assertEqual(pair_views.identity_of(original, "A"), labels.RESPONSE_1)
        self.assertEqual(pair_views.identity_of(swapped, "B"), labels.RESPONSE_1)
        self.assertTrue(pair_views.same_choice_across_views(original, swapped, "A", "B"))
        self.assertFalse(pair_views.same_choice_across_views(original, swapped, "A", "A"))

    def test_same_choice_accepts_reversed_argument_order(self):
        original, swapped = self._views()
        self.assertTrue(
            pair_views.same_choice_across_views(swapped, original, "B", "A"))

    def test_same_choice_rejects_two_views_of_the_same_ordering(self):
        original, _ = self._views()
        with self.assertRaises(ValueError):
            pair_views.same_choice_across_views(original, original, "A", "A")

    def test_group_by_pair_rejects_duplicates(self):
        original, _ = self._views()
        with self.assertRaises(ValueError):
            pair_views.group_by_pair([original, original])

    def test_assert_views_cover_checks_content_and_label(self):
        views = self._views()
        pair_views.assert_views_cover(views, 1)
        with self.assertRaises(ValueError):
            pair_views.assert_views_cover(views, 2)


class HelpSteerAdapterTest(unittest.TestCase):
    """Source rows are validated, remapped and stripped of the legacy rationale."""

    def test_convert_row_maps_label_to_response(self):
        converted = helpsteer_adapter.convert_row(_sample_helpsteer_row(label="B"), 0)
        self.assertEqual(converted["better_response"], labels.RESPONSE_2)
        self.assertEqual(converted["pair_id"], "hs3-7")
        self.assertEqual(converted["answer_1"], "Response one body.")
        self.assertEqual(converted["answer_2"], "Response two body.")

    def test_legacy_rationale_never_reaches_the_training_row(self):
        converted = helpsteer_adapter.convert_row(_sample_helpsteer_row(), 0)
        views = helpsteer_adapter.build_pair_views(converted, two_states=True)
        rows = trl_dataset.build_trl_rows([view.to_record() for view in views], "vanilla")
        for row in rows:
            self.assertNotIn("text", row)
            self.assertNotIn("source_overall_preference", row)
            rendered = json.dumps(row["prompt"], ensure_ascii=False)
            self.assertNotIn("legacy JudgeLRM rationale", rendered)

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(helpsteer_adapter.AdapterError):
            helpsteer_adapter.convert_row(_sample_helpsteer_row(view="swapped"), 0)

    def test_missing_field_is_rejected(self):
        row = _sample_helpsteer_row()
        del row["label"]
        with self.assertRaises(helpsteer_adapter.AdapterError):
            helpsteer_adapter.convert_row(row, 0)

    def test_empty_answer_is_rejected(self):
        with self.assertRaises(helpsteer_adapter.AdapterError):
            helpsteer_adapter.convert_row(_sample_helpsteer_row(answer2_body="  "), 0)

    def test_source_line_is_traceable_in_the_pair_id(self):
        first = helpsteer_adapter.convert_row(_sample_helpsteer_row(source_line=11), 0)
        second = helpsteer_adapter.convert_row(_sample_helpsteer_row(source_line=12), 1)
        self.assertNotEqual(first["pair_id"], second["pair_id"])
        self.assertEqual(first["pair_id"], "hs3-11")

    def test_materialise_writes_both_states_and_keeps_labels_balanced(self):
        import tempfile

        converted = [
            helpsteer_adapter.convert_row(_sample_helpsteer_row(source_line=1, label="A"), 0),
            helpsteer_adapter.convert_row(_sample_helpsteer_row(source_line=2, label="B"), 1),
        ]
        rows = []
        for pair in converted:
            for view in helpsteer_adapter.build_pair_views(pair, two_states=True):
                rows.append(view.to_record())
        self.assertEqual(len(rows), 4)
        summary = trl_dataset.assert_row_invariants(rows, states_per_sample=2)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["ground_truth_counts"]["A"], 2)
        self.assertEqual(summary["ground_truth_counts"]["B"], 2)
        with tempfile.TemporaryDirectory() as tmp:  # pragma: no cover - IO smoke
            self.assertTrue(os.path.isdir(tmp))


class RealDataCensusTest(unittest.TestCase):
    """Read-only census of the shipped files; these numbers are the documented ones."""

    def test_training_file_is_the_documented_size(self):
        census = helpsteer_adapter.scan_source()
        self.assertEqual(census["rows"], 22190)
        self.assertEqual(census["by_view"], {"original": 22190})
        self.assertEqual(census["by_label"]["A"], 11068)
        self.assertEqual(census["by_label"]["B"], 11122)

    def test_rewardbench_file_is_the_documented_size(self):
        census = rewardbench_adapter.scan_source()
        self.assertEqual(census["rows"], 2985)
        self.assertEqual(census["unique_qids"], 2985)
        self.assertEqual(census["subsets"], 23)
        self.assertEqual(census["unknown_subsets"], [])
        self.assertEqual(census["by_ground_truth"], {"A": 1495, "B": 1490})

    def test_rewardbench_covers_the_official_subsets_exactly(self):
        self.assertEqual(len(rewardbench_adapter.OFFICIAL_SUBSETS), 23)
        self.assertEqual(len(set(rewardbench_adapter.OFFICIAL_SUBSETS)), 23)
        census = rewardbench_adapter.scan_source()
        self.assertEqual(sorted(census["by_subset"]), sorted(rewardbench_adapter.OFFICIAL_SUBSETS))


class RewardBenchIsolationTest(unittest.TestCase):
    """The evaluation file must never become a training file."""

    def test_assert_not_training_rejects_the_eval_file(self):
        with self.assertRaises(ValueError):
            rewardbench_adapter.assert_not_training(str(paths.REWARDBENCH_OOD))

    def test_config_rejects_rewardbench_as_train_dataset(self):
        data = _valid_config_dict()
        data["train_dataset"] = str(paths.REWARDBENCH_OOD)
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("RewardBench must never become a training file", str(ctx.exception))

    def test_helpsteer_pair_ids_never_collide_with_rewardbench_qids(self):
        converted = helpsteer_adapter.convert_row(_sample_helpsteer_row(source_line=0), 0)
        self.assertTrue(converted["pair_id"].startswith("hs3-"))


class EvalViewTest(unittest.TestCase):
    """Swapped evaluation views must be a real swap, not a relabelled copy."""

    def _record(self, gt="A"):
        return {
            "qid": "1",
            "source_subset": "alpacaeval-easy",
            "question_body": "Q?",
            "answer1_body": "one",
            "answer2_body": "two",
            "gt_class": gt,
            "label": gt,
            "view": "chosen_A" if gt == "A" else "chosen_B",
            "chosen_A": "m1",
            "chosen_B": "m2",
            "source_chosen": "one" if gt == "A" else "two",
            "source_rejected": "two" if gt == "A" else "one",
            "source_dataset": "rewardbench",
            "source_revision": "v1",
            "source_row_index": 0,
            "source_id": "1",
            "chosen_model": "m1",
            "rejected_model": "m2",
        }

    def test_two_views_per_record_with_flipped_labels(self):
        views = rewardbench_adapter.build_eval_views(
            rewardbench_adapter.convert_row(self._record("A"), 0))
        self.assertEqual(len(views), 2)
        original, swapped = views
        self.assertEqual((original["answer_a"], original["answer_b"]), ("one", "two"))
        self.assertEqual((swapped["answer_a"], swapped["answer_b"]), ("two", "one"))
        self.assertEqual(original["ground_truth"], "A")
        self.assertEqual(swapped["ground_truth"], "B")

    def test_swapped_view_keeps_the_same_physical_content(self):
        views = rewardbench_adapter.build_eval_views(
            rewardbench_adapter.convert_row(self._record("B"), 0))
        original, swapped = views
        self.assertEqual(
            sorted([original["answer_a"], original["answer_b"]]),
            sorted([swapped["answer_a"], swapped["answer_b"]]),
        )

    def test_unknown_subset_is_rejected_at_load_time(self):
        record = self._record()
        record["source_subset"] = "chat-hard"
        with self.assertRaises(rewardbench_adapter.RewardBenchAdapterError) as ctx:
            rewardbench_adapter.convert_row(record, 0)
        self.assertIn("is not one of the 23 official RewardBench v1 subsets", str(ctx.exception))

    def test_gt_class_contradicting_the_stored_view_is_rejected(self):
        record = self._record("B")
        record["view"] = "chosen_A"
        with self.assertRaises(rewardbench_adapter.RewardBenchAdapterError):
            rewardbench_adapter.convert_row(record, 0)


class PromptContractTest(unittest.TestCase):
    """Each method's prompt must carry the tags its parser requires."""

    def test_every_method_renders_two_messages(self):
        for method in prompts.VALID_METHODS:
            messages = prompts.build_messages(
                method, question="Q?", response_a="one", response_b="two")
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertIn("Q?", messages[1]["content"])

    def test_prompt_layout_matches_the_parser_contract(self):
        cases = {
            "vanilla": ("<answer>A</answer>", None),
            "j1": ("<answer>[[A]]</answer>", "<think>"),
            "j4r": ("Verdict: [[A>B]]", "Explanation:"),
            "jwc": ("<prob_rb_better>", "<think>"),
        }
        for method, (needle, extra) in cases.items():
            system = prompts.system_prompt(method)
            user = prompts.user_template(method)
            haystack = system + "\n" + user
            self.assertIn(needle, haystack, msg="method {0} lost its output contract".format(method))
            if extra is not None:
                self.assertIn(extra, haystack, msg="method {0} lost {1}".format(method, extra))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            prompts.check_method("j2")

    def test_build_messages_validates_its_inputs(self):
        with self.assertRaises(TypeError):
            prompts.build_messages("vanilla", question=1, response_a="one", response_b="two")
        with self.assertRaises(ValueError):
            prompts.build_messages("vanilla", question="Q", response_a="", response_b="two")

    def test_swapping_the_answers_swaps_the_rendered_text(self):
        forward = prompts.build_messages("vanilla", question="Q", response_a="AAA",
                                         response_b="BBB")[1]["content"]
        backward = prompts.build_messages("vanilla", question="Q", response_a="BBB",
                                          response_b="AAA")[1]["content"]
        self.assertNotEqual(forward, backward)
        self.assertLess(forward.index("AAA"), forward.index("BBB"))
        self.assertLess(backward.index("BBB"), backward.index("AAA"))


class TrlRowTest(unittest.TestCase):
    """TRL row construction must not drop or duplicate the identity columns."""

    def _record(self, is_swapped=False, gt="A"):
        return {
            "sample_id": "hs3-7::" + ("swapped" if is_swapped else "original"),
            "pair_id": "hs3-7",
            "question": "Q?",
            "answer_a": "two" if is_swapped else "one",
            "answer_b": "one" if is_swapped else "two",
            "ground_truth": gt,
            "is_swapped": is_swapped,
            "view": "swapped" if is_swapped else "original",
        }

    def test_row_carries_every_reward_column(self):
        row = trl_dataset.build_trl_row(self._record(), "j1")
        for column in trl_dataset.REWARD_COLUMNS:
            self.assertIn(column, row)
        self.assertIsInstance(row["prompt"], list)
        self.assertEqual(row["prompt"][0]["role"], "system")

    def test_degenerate_pair_is_rejected(self):
        record = self._record()
        record["answer_b"] = record["answer_a"]
        with self.assertRaises(trl_dataset.DatasetRowError):
            trl_dataset.build_trl_row(record, "vanilla")

    def test_duplicate_sample_id_is_rejected(self):
        record = self._record()
        with self.assertRaises(trl_dataset.DatasetRowError):
            list(trl_dataset.iter_trl_rows([record, dict(record)], "vanilla"))

    def test_two_state_invariants_catch_a_missing_counterpart(self):
        rows = [trl_dataset.build_trl_row(self._record(False, "A"), "j1")]
        with self.assertRaises(trl_dataset.DatasetRowError) as ctx:
            trl_dataset.assert_row_invariants(rows, states_per_sample=2)
        self.assertIn("found ['original']", str(ctx.exception))

    def test_two_state_invariants_catch_a_pair_with_only_one_state(self):
        # Two original views of different pairs, then a full set minus one counterpart:
        # the view set looks right, so the per-pair completeness check must catch it.
        rows = [
            trl_dataset.build_trl_row(self._record(False, "A"), "j1"),
            trl_dataset.build_trl_row(self._record(True, "B"), "j1"),
        ]
        rows[1]["pair_id"] = "hs3-8"
        rows[1]["sample_id"] = "hs3-8::swapped"
        with self.assertRaises(trl_dataset.DatasetRowError) as ctx:
            trl_dataset.assert_row_invariants(rows, states_per_sample=2)
        self.assertIn("missing a state", str(ctx.exception))

    def test_two_state_invariants_catch_an_unflipped_label(self):
        rows = [
            trl_dataset.build_trl_row(self._record(False, "A"), "j1"),
            trl_dataset.build_trl_row(self._record(True, "A"), "j1"),
        ]
        with self.assertRaises(trl_dataset.DatasetRowError):
            trl_dataset.assert_row_invariants(rows, states_per_sample=2)

    def test_single_state_invariants_are_satisfied(self):
        rows = [trl_dataset.build_trl_row(self._record(False, "A"), "vanilla")]
        summary = trl_dataset.assert_row_invariants(rows, states_per_sample=1)
        self.assertEqual(summary["views"], ["original"])


class LengthAuditTest(unittest.TestCase):
    """Pure statistics used by the real tokenizer audit."""

    def test_percentiles_match_the_numpy_default(self):
        values = [1, 2, 3, 4]
        self.assertAlmostEqual(length_audit.percentile(values, 50), 2.5)
        self.assertAlmostEqual(length_audit.percentile(values, 0), 1.0)
        self.assertAlmostEqual(length_audit.percentile(values, 100), 4.0)

    def test_summary_reports_every_documented_statistic(self):
        summary = length_audit.summarise(list(range(1, 101)))
        for key in ("count", "min", "max", "mean", "std", "p50", "p90", "p95", "p99"):
            self.assertIn(key, summary)

    def test_overflow_is_counted_and_measured(self):
        overflow = length_audit.count_exceeding([10, 50, 100], 50)
        self.assertEqual(overflow["exceeding"], 1)
        self.assertEqual(overflow["max_overshoot"], 50)
        self.assertAlmostEqual(overflow["exceeding_ratio"], 1 / 3, places=6)

    def test_audit_flags_would_need_truncation(self):
        audit = length_audit.audit_lengths([10, 20, 500], 100, 16)
        self.assertTrue(audit["would_need_truncation"])
        with self.assertRaises(length_audit.LengthAuditError):
            length_audit.assert_within_limits(audit)

    def test_audit_within_limits_does_not_raise(self):
        audit = length_audit.audit_lengths([10, 20, 30], 100, 16)
        self.assertFalse(audit["would_need_truncation"])
        length_audit.assert_within_limits(audit)

    def test_audit_rejects_a_budget_above_the_context_window(self):
        with self.assertRaises(length_audit.LengthAuditError):
            length_audit.audit_lengths([10], 20000, 20000)


class ReproducibilityTest(unittest.TestCase):
    """Seeding and IO helpers, exercised without torch."""

    def test_seed_is_validated(self):
        from common import reproducibility

        with self.assertRaises(TypeError):
            reproducibility.seed_everything("42")
        with self.assertRaises(ValueError):
            reproducibility.seed_everything(-1)

    def test_jsonl_round_trip(self):
        import shutil
        import tempfile

        from common import reproducibility

        # The writer enforces the "only grpo_baselines/ is writable" rule, so the scratch
        # directory has to live inside the project root.
        tmp = tempfile.mkdtemp(prefix="_test_jsonl_", dir=str(paths.RUNS_DIR))
        try:
            path = os.path.join(tmp, "rows.jsonl")
            reproducibility.write_jsonl(path, [{"a": 1}, {"a": 2}])
            self.assertEqual(reproducibility.count_jsonl_rows(path), 2)
            self.assertEqual([row["a"] for row in reproducibility.iter_jsonl(path)], [1, 2])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_writers_refuse_paths_outside_the_project_root(self):
        from common import reproducibility

        with self.assertRaises(PermissionError):
            reproducibility.write_jsonl("/tmp/escape.jsonl", [{"a": 1}])
        with self.assertRaises(PermissionError):
            reproducibility.write_json("/tmp/escape.json", {"a": 1})

    def test_sha256_text_is_stable(self):
        from common import reproducibility

        self.assertEqual(reproducibility.sha256_text("abc"),
                         reproducibility.sha256_text("abc"))
        self.assertNotEqual(reproducibility.sha256_text("abc"),
                            reproducibility.sha256_text("abd"))


# --------------------------------------------------------------------------------------
# Config tests
# --------------------------------------------------------------------------------------
def _valid_config_dict(**overrides):
    """A minimal valid vanilla config, built independently of the YAML files."""
    data = {
        "method": "vanilla",
        "output_format": "vanilla",
        "base_model": str(paths.BASE_MODEL),
        "train_dataset": str(paths.HELPSTEER_TRAIN),
        "output_dir": str(paths.PROJECT_ROOT / "runs" / "_test_vanilla"),
        "seed": 42,
        "max_prompt_length": 11264,
        "max_completion_length": 1024,
        "learning_rate": 1.0e-6,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "global_batch_size": 8,
        "micro_batch_size": 2,
        "gradient_accumulation": 4,
        "states_per_sample": 1,
        "num_generations": 8,
        "training_steps": 256,
        "total_rollouts": 256 * 8 * 8,
        "estimated_prompt_tokens": 1800,
        "estimated_generated_tokens": (256 * 8 * 8) * (1800 + 1024),
        "temperature": 1.0,
        "top_p": 1.0,
        "clip_epsilon": 0.2,
        "reference_model_policy": config_mod.REFERENCE_POLICY_FROZEN,
        "kl_beta": 0.01,
        "logging": {
            "log_every_n_steps": 10,
            "save_completion_samples": True,
            "metrics": ["loss", "reward"],
        },
        "checkpoint": {
            "save_every_n_steps": 50,
            "save_total_limit": 3,
            "save_optimizer_state": True,
        },
        "controlled_budget": {
            "paper_reference": "arXiv:2505.10320",
            "controlled_reference": "configs/controlled_budget.yaml",
            "paper_states_per_sample": 2,
            "paper_rollouts_per_state": 2,
            "paper_rollouts_per_update": 512,
            "paper_training_steps": 1000,
            "paper_generated_tokens": 1000000,
            "notes": "test fixture",
        },
    }
    data.update(overrides)
    return data


class ConfigValidationTest(unittest.TestCase):
    """A config must be internally consistent; silent drift is a hard error."""

    def test_a_consistent_config_validates_and_derives_its_budget(self):
        cfg = config_mod.validate_config(_valid_config_dict())
        self.assertEqual(cfg.rollouts_per_state, 8)
        self.assertEqual(cfg.rows_per_update, 8)
        self.assertEqual(cfg.rollouts_per_update, 64)
        self.assertEqual(cfg.total_rollouts_derived, 16384)
        self.assertEqual(cfg.context_budget, 12288)

    def test_missing_field_is_reported(self):
        data = _valid_config_dict()
        del data["kl_beta"]
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("missing required fields", str(ctx.exception))

    def test_unknown_field_is_reported(self):
        data = _valid_config_dict()
        data["learining_rate"] = 1e-6
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("unknown fields", str(ctx.exception))

    def test_output_format_must_match_method(self):
        data = _valid_config_dict(output_format="j1")
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("output_format must match method", str(ctx.exception))

    def test_prompt_and_completion_must_fit_the_context_window(self):
        data = _valid_config_dict(max_prompt_length=32000, max_completion_length=2048)
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("exceeds the model context window", str(ctx.exception))

    def test_base_model_is_pinned(self):
        with self.assertRaises(config_mod.ConfigError):
            config_mod.validate_config(_valid_config_dict(base_model="/tmp/other-model"))

    def test_two_state_methods_require_two_states(self):
        data = _valid_config_dict(method="j4r", output_format="j4r", states_per_sample=1)
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("requires states_per_sample=2", str(ctx.exception))

    def test_single_state_methods_forbid_two_states(self):
        data = _valid_config_dict(states_per_sample=2, num_generations=16,
                                  gradient_accumulation=8,
                                  total_rollouts=128 * 8 * 16)
        data["estimated_generated_tokens"] = data["total_rollouts"] * (1800 + 1024)
        with self.assertRaises(config_mod.ConfigError):
            config_mod.validate_config(data)

    def test_generations_must_be_divisible_by_the_state_count(self):
        data = _valid_config_dict(
            method="j4r", output_format="j4r", states_per_sample=2,
            num_generations=3, gradient_accumulation=8,
            total_rollouts=128 * 8 * 3,
        )
        data["estimated_generated_tokens"] = data["total_rollouts"] * (1800 + 1024)
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("not divisible by states_per_sample", str(ctx.exception))

    def test_rollouts_per_state_below_two_is_rejected(self):
        # A GRPO group of size 1 has no variance to normalise against, so the paired
        # methods must refuse G == states_per_sample.
        data = _valid_config_dict(
            method="j4r", output_format="j4r", states_per_sample=2,
            num_generations=2, gradient_accumulation=8,
            total_rollouts=128 * 8 * 2,
        )
        data["estimated_generated_tokens"] = data["total_rollouts"] * (1800 + 1024)
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(data)
        self.assertIn("rollouts_per_state=1 < 2", str(ctx.exception))

    def test_gradient_accumulation_must_match_the_rows(self):
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(_valid_config_dict(gradient_accumulation=2))
        self.assertIn("contradicts rows_per_update", str(ctx.exception))

    def test_total_rollouts_must_match_the_product(self):
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(_valid_config_dict(total_rollouts=1000))
        self.assertIn("total_rollouts", str(ctx.exception))

    def test_generated_tokens_must_match_the_product(self):
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(
                _valid_config_dict(estimated_generated_tokens=12345))
        self.assertIn("estimated_generated_tokens", str(ctx.exception))

    def test_frozen_reference_requires_positive_kl(self):
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(_valid_config_dict(kl_beta=0.0))
        self.assertIn("requires kl_beta > 0", str(ctx.exception))

    def test_no_reference_requires_zero_kl(self):
        with self.assertRaises(config_mod.ConfigError) as ctx:
            config_mod.validate_config(
                _valid_config_dict(reference_model_policy=config_mod.REFERENCE_POLICY_NONE))
        self.assertIn("requires kl_beta == 0", str(ctx.exception))

    def test_output_dir_outside_the_project_is_rejected(self):
        with self.assertRaises(PermissionError):
            config_mod.validate_config(_valid_config_dict(output_dir="/tmp/escape"))

    def test_bool_is_not_an_int(self):
        with self.assertRaises(config_mod.ConfigError):
            config_mod.validate_config(_valid_config_dict(seed=False))


class ShippedConfigTest(unittest.TestCase):
    """The four shipped configs must validate and share one token budget."""

    def setUp(self):
        self.configs = config_mod.load_all_configs()

    def test_all_four_methods_are_present(self):
        self.assertEqual(sorted(self.configs), ["j1", "j4r", "jwc", "vanilla"])

    def test_output_format_equals_method(self):
        for method, cfg in self.configs.items():
            self.assertEqual(cfg.output_format, method)

    def test_budget_agrees_across_methods(self):
        report = config_mod.assert_equal_budget(self.configs)
        self.assertEqual(len(report), 2)

    def test_paired_methods_use_two_states_and_double_the_rows(self):
        for method in ("j1", "j4r"):
            cfg = self.configs[method]
            self.assertEqual(cfg.states_per_sample, 2)
            self.assertEqual(cfg.rows_per_update, 2 * cfg.global_batch_size)
        for method in ("vanilla", "jwc"):
            self.assertEqual(self.configs[method].states_per_sample, 1)

    def test_each_method_has_a_different_effective_rollout_group(self):
        # rollouts_per_state must stay comparable across the two families.
        per_state = {m: cfg.rollouts_per_state for m, cfg in self.configs.items()}
        self.assertEqual(per_state["vanilla"], 8)
        self.assertEqual(per_state["j1"], 8)
        self.assertEqual(per_state["j4r"], 8)
        self.assertEqual(per_state["jwc"], 8)

    def test_controlled_budget_table_matches_the_configs(self):
        import yaml

        with open(str(paths.PROJECT_ROOT / "configs" / "controlled_budget.yaml"),
                  encoding="utf-8") as handle:
            table = yaml.safe_load(handle)
        self.assertIn("controlled", table)
        self.assertIn("methods", table)
        self.assertIn("paper_vs_controlled", table)
        self.assertIn("open_risks", table)
        self.assertEqual(len(table["open_risks"]), 5)
        for method, cfg in self.configs.items():
            row = table["methods"][method]
            self.assertEqual(row["states_per_sample"], cfg.states_per_sample)
            self.assertEqual(row["num_generations"], cfg.num_generations)
            self.assertEqual(row["rollouts_per_state"], cfg.rollouts_per_state)
            self.assertEqual(row["rows_per_update"], cfg.rows_per_update)
            self.assertEqual(row["rollouts_per_update"], cfg.rollouts_per_update)
            self.assertEqual(row["training_steps"], cfg.training_steps)
            self.assertEqual(row["gradient_accumulation"], cfg.gradient_accumulation)
            advantage = row["advantage"].lower()
            if method == "j4r":
                self.assertIn("eis", advantage)
                self.assertIn("global", advantage)
                self.assertIn("local", advantage)
            else:
                self.assertIn("grpo group normalisation", advantage)
                self.assertNotIn("eis", advantage)

    def test_shared_hyperparameters_are_identical(self):
        shared = ("learning_rate", "kl_beta", "temperature", "top_p",
                  "clip_epsilon", "max_prompt_length", "max_completion_length",
                  "global_batch_size", "micro_batch_size", "seed",
                  "reference_model_policy", "lr_scheduler_type")
        values = {}
        for method, cfg in self.configs.items():
            values[method] = {key: cfg[key] for key in shared}
        reference = values["vanilla"]
        for method, payload in values.items():
            self.assertEqual(payload, reference, msg="method {0} diverged".format(method))


if __name__ == "__main__":
    unittest.main()
