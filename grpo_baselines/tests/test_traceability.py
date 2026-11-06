"""Traceability tests: source row -> pair view -> TRL row -> rollout log.

The brief's Sec. 3.3 asks for ``source IDs`` to stay recoverable. That chain has four
links and it was broken in the middle twice: ``PairView.to_record`` dropped ``group_id``
and ``build_trl_row`` rebuilt its row from a whitelist that dropped it again. Each link
tested alone looked fine, which is why these tests walk the whole chain on real rows
rather than checking ``to_record()`` in isolation.
"""

from __future__ import annotations

import json
import unittest

from common import paths
from data import helpsteer_adapter
from data import pair_views
from data import rewardbench_adapter
from data import trl_dataset
from rewards import rollout_log


class PairViewCarriesSourceIdentityTest(unittest.TestCase):
    def test_to_record_carries_group_id_and_source_line(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=1))
        record = rows[0]
        self.assertIn("group_id", record)
        self.assertIn("source_line", record)
        self.assertTrue(record["group_id"])
        self.assertIsNotNone(record["source_line"])

    def test_group_id_is_the_source_group_not_the_pair_id(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=3))
        # pair_id is "hs3-<source_line>"; group_id is HelpSteer's own grouping. They are
        # different namespaces and collapsing them would make a group ambiguous the
        # moment two rows share a group.
        pair = rows[0]
        self.assertNotEqual(pair["group_id"], pair["pair_id"])
        self.assertTrue(pair["pair_id"].startswith("hs3-"))

    def test_both_states_agree_on_group_id(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=True, limit=2))
        by_pair = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], set()).add(row["group_id"])
        for pair_id, groups in by_pair.items():
            self.assertEqual(len(groups), 1, "pair {0!r} has groups {1}".format(pair_id, groups))

    def test_metadata_only_construction_leaves_the_fields_empty(self):
        views = pair_views.build_views(
            pair_id="p1", source_row_index=0, question="q",
            answer_1="a", answer_2="b", better_response="answer1",
            source_dataset="synthetic", two_states=True, metadata=None)
        for view in views:
            self.assertIsNone(view.group_id)
            self.assertIsNone(view.source_line)
            record = view.to_record()
            self.assertIsNone(record["group_id"])
            self.assertIsNone(record["source_line"])


class TrlRowKeepsTheTraceTest(unittest.TestCase):
    def test_row_carries_group_id_and_source_line(self):
        record = next(helpsteer_adapter.iter_training_rows(two_states=False, limit=1))
        row = trl_dataset.build_trl_row(record, "vanilla")
        self.assertEqual(row["group_id"], record["group_id"])
        self.assertEqual(row["source_line"], record["source_line"])
        for field in ("group_id", "source_line"):
            self.assertIn(field, trl_dataset.REWARD_COLUMNS,
                          "{0} must be forwarded to the reward function".format(field))

    def test_row_names_the_corpus_it_came_from(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=1))
        row = trl_dataset.build_trl_row(rows[0], "vanilla")
        # Sec. 3.3 lists source_dataset among the fields a derived example must carry.
        # Without it a reward function can only guess the corpus from an id prefix.
        self.assertEqual(row["source_dataset"], helpsteer_adapter.SOURCE_DATASET_NAME)
        self.assertIn("source_dataset", trl_dataset.REWARD_COLUMNS)

    def test_the_eight_required_fields_survive_into_the_training_row(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=1))
        row = trl_dataset.build_trl_row(rows[0], "vanilla")
        # `question` is rendered into `prompt` rather than duplicated as a column; the
        # other seven are carried by name.
        for field in ("sample_id", "pair_id", "answer_a", "answer_b",
                      "ground_truth", "is_swapped", "source_dataset"):
            with self.subTest(field=field):
                self.assertIn(field, row)
        self.assertIn(rows[0]["question"], row["prompt"][1]["content"])

    def test_row_does_not_start_looking_multimodal(self):
        record = next(helpsteer_adapter.iter_training_rows(two_states=False, limit=1))
        row = trl_dataset.build_trl_row(record, "vanilla")
        for forbidden in trl_dataset.FORBIDDEN_COLUMNS:
            self.assertNotIn(forbidden, row)

    def test_rewardbench_views_do_not_claim_a_training_corpus(self):
        views = list(rewardbench_adapter.iter_eval_views(limit=1))
        # An OOD row has no HelpSteer group and must not pretend to. Its own corpus name
        # lives in `source_dataset` on the eval view, which is a different value.
        self.assertNotIn("group_id", views[0])
        self.assertEqual(views[0]["source_dataset"], rewardbench_adapter.SOURCE_DATASET_NAME)

    def test_rewardbench_rows_have_no_group_id(self):
        # RewardBench is OOD evaluation data: it must not carry a training group label
        # that could later be mistaken for provenance.
        view = next(rewardbench_adapter.iter_eval_views(limit=1))
        self.assertIsNone(view.get("group_id"))


class RolloutLogRecordsTheGroupTest(unittest.TestCase):
    def test_rollout_record_keeps_group_id(self):
        records = rollout_log.records_from_batch(
            ["c1", "c2"], [1.0, 0.0], ["hs3-1::original"] * 2,
            group_ids=["train_000001", "train_000001"])
        self.assertEqual([r.group_id for r in records], ["train_000001"] * 2)
        self.assertEqual(records[0].as_dict()["group_id"], "train_000001")

    def test_group_ids_length_mismatch_is_rejected(self):
        with self.assertRaises(rollout_log.RolloutLogError) as caught:
            rollout_log.records_from_batch(
                ["c1"], [1.0], ["hs3-1::original"], group_ids=["a", "b"])
        self.assertIn("group_ids has 2 entries but there are 1 rollouts", str(caught.exception))

    def test_absent_group_ids_become_none_not_empty_string(self):
        records = rollout_log.records_from_batch(["c1"], [1.0], ["hs3-1::original"])
        self.assertIsNone(records[0].group_id)

    def test_wrapper_forwards_the_group_column(self):
        def reward_fn(prompts=None, completions=None, completion_ids=None,
                      sample_id=None, ground_truth=None, group_id=None, **kwargs):
            return [1.0 for _ in completions]

        wrapped = rollout_log.logged_reward_fn(reward_fn)
        wrapped(completions=["c1", "c2"], sample_id=["p::original", "p::swapped"],
                group_id=["g1", "g1"])
        self.assertEqual([r.group_id for r in wrapped.last_rollouts], ["g1", "g1"])

    def test_diagnostics_can_name_the_flat_groups(self):
        # A flat group is one where every reward is identical -- that is the case TRL
        # silently turns into all-zero advantages, so it is the case worth naming.
        records = rollout_log.records_from_batch(
            ["c1", "c2", "c3", "c4"], [1.0, 1.0, 1.0, 1.0],
            ["p::original"] * 4, group_ids=["gA", "gA", "gA", "gA"])
        result = rollout_log.group_diagnostics(records)
        self.assertEqual(result["flat_group_count"], 1)
        self.assertEqual(result["groups"]["p::original"]["rollouts"], 4)
        # The point of carrying group_id: the diagnostic can be traced to a source group.
        self.assertEqual(records[0].group_id, "gA")

    def test_a_mixed_group_is_not_reported_as_flat(self):
        # Guard against a check that only looks at group size.
        records = rollout_log.records_from_batch(
            ["c1", "c2", "c3", "c4"], [0.0, 0.0, 1.0, 0.0],
            ["p::original"] * 4, group_ids=["gA"] * 4)
        result = rollout_log.group_diagnostics(records)
        self.assertEqual(result["flat_group_count"], 0)


class RealDataTraceIsReversibleTest(unittest.TestCase):
    def test_every_sampled_row_can_be_traced_back_to_its_source_line(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=25))
        self.assertTrue(rows)
        for row in rows:
            self.assertIsNotNone(row["source_line"])
            # pair_id is derived from source_line, so recovering one recovers the other.
            self.assertEqual(row["pair_id"], "hs3-{0}".format(row["source_line"]))

    def test_group_id_matches_the_source_file(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=10))
        # `source_line` is an int in the JSONL, so normalise both sides to str before
        # matching; comparing an int set against str keys silently matches nothing.
        wanted = {str(row["source_line"]) for row in rows}
        source_groups = {}
        with open(paths.HELPSTEER_TRAIN, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                key = str(payload.get("source_line"))
                if key in wanted:
                    source_groups[key] = payload.get("group_id")
        self.assertEqual(len(source_groups), len(wanted))
        for row in rows:
            expected = source_groups.get(str(row["source_line"]))
            self.assertEqual(row["group_id"], expected,
                             "row {0!r} claims group {1!r}, source says {2!r}".format(
                                 row["sample_id"], row["group_id"], expected))

    def test_group_ids_are_unique_per_pair_in_the_sample(self):
        rows = list(helpsteer_adapter.iter_training_rows(two_states=False, limit=50))
        pairs = {row["pair_id"] for row in rows}
        groups = {row["group_id"] for row in rows}
        # HelpSteer groups are source conversations; in the clean file each pair carries
        # its own group, so the two counts agree over a sample. If they ever diverge the
        # field is still meaningful, but the assumption should be restated, not assumed.
        self.assertEqual(len(pairs), len(groups))


if __name__ == "__main__":
    unittest.main()
