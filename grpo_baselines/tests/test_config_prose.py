"""A config's prose must not disagree with the number it configures.

`j4r.yaml` shipped for a while with `kl_beta: 0.01` and a `controlled_reference` sentence
saying it used the paper's `1e-4`. Both sentences were in the same file; the wrong one was
the one a reader would quote. Numbers in a config are checked by `validate_config`, but the
prose around them is checked by nothing, and prose is what gets copied into a report.

So: wherever a config *describes its own* KL value, the number in the sentence must equal
`kl_beta`. Statements about the paper's value are fine and expected -- they live in
`paper_reference` or are explicitly attributed ("the paper's 1e-4"), and the check only
applies to sentences claiming to describe this configuration.
"""
from __future__ import annotations

import re
import unittest

import yaml

from common import config as config_mod
from common import paths

# method -> config file stem (the brief fixes the JwC filename)
CONFIG_FILES = {"vanilla": "vanilla", "j1": "j1", "j4r": "j4r", "jwc": "jwc_rl_brier"}

# "KL beta is 1e-2", "kl_beta is 0.01", "KL coefficient is 1e-2"
_OWN_KL = re.compile(
    r"\b(?:kl[_ ]beta|kl[ _]coefficient)\b[^.\n]{0,24}?\bis\b\s*"
    r"([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)",
    re.IGNORECASE)

# "the paper's 1e-4" is a citation, not a claim about this configuration. Strip those
# clauses and keep the rest of the sentence, rather than skipping any sentence that
# mentions the paper: the corrected j4r text reads "KL beta is 1e-2 ... not the paper's
# 1e-4", and skipping it whole would also hide a regression in the same sentence.
_PAPER_CITATION = re.compile(
    r"\b(?:the\s+)?paper(?:'s|\u2019s)\s+[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?",
    re.IGNORECASE)


def payload_for(method):
    stem = CONFIG_FILES[method]
    return yaml.safe_load((paths.CONFIGS_DIR / "{0}.yaml".format(stem)).read_text(
        encoding="utf-8"))


class ConfigProseAgreesWithItsNumbersTest(unittest.TestCase):
    def setUp(self):
        self.configs = config_mod.load_all_configs()

    def test_any_stated_kl_value_matches_the_field(self):
        for method, cfg in self.configs.items():
            payload = payload_for(method)
            prose = payload.get("controlled_budget", {})
            for field in ("controlled_reference", "notes"):
                text = str(prose.get(field, ""))
                for sentence in re.split(r"(?<=[.;])\s+", text):
                    own_claim = _PAPER_CITATION.sub(" ", sentence)
                    for match in _OWN_KL.finditer(own_claim):
                        quoted = float(match.group(1))
                        self.assertAlmostEqual(
                            quoted, cfg.kl_beta, places=10,
                            msg="{0}.{1} says KL is {2} but kl_beta is {3}: {4!r}".format(
                                method, field, quoted, cfg.kl_beta, sentence.strip()))

    def test_the_paper_reference_states_the_papers_value(self):
        """The one place a number differing from kl_beta is correct."""
        paper = payload_for("j4r")["controlled_budget"]["paper_reference"]
        self.assertIn("1e-4", paper,
                      "j4r's paper_reference should record the paper's own KL")

    def test_every_config_documents_both_references(self):
        for method, cfg in self.configs.items():
            budget = payload_for(method)["controlled_budget"]
            self.assertIn("paper_reference", budget, method)
            self.assertIn("controlled_reference", budget, method)
            self.assertIn("notes", budget, method)
            payload = payload_for(method)
            self.assertEqual(payload["reference_model_policy"],
                             cfg["reference_model_policy"], method)
            self.assertEqual(float(payload["kl_beta"]), cfg.kl_beta, method)

    def test_no_config_claims_the_papers_kl_as_its_own(self):
        """The exact failure that was fixed: 'KL beta is 1e-4' beside kl_beta: 0.01."""
        for method, cfg in self.configs.items():
            if abs(cfg.kl_beta - 1e-4) < 1e-12:
                continue
            prose = payload_for(method)["controlled_budget"].get("controlled_reference", "")
            self.assertNotIn(
                "1e-4 as in the paper rather than", prose,
                "{0} claims to use the paper's 1e-4 while kl_beta is {1}".format(
                    method, cfg.kl_beta))


if __name__ == "__main__":
    unittest.main()
