"""The dialog layer must survive the customer being reworded.

``docs/competition_specification.md`` allows the organizer to paraphrase the
simulated customer on the private set. Measured with ``paraphrase_eval.py``,
frame-only parsing collapses from 0.9738 to 0.0259 when that happens, because
``observe()`` used to fall through and set nothing at all - not even the
category. These tests pin the free-form fallback that closes that hole, and
pin the shipped frames alongside it so the fallback cannot quietly replace them.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.dialog import SessionState
from starter.retrieval import CatalogIndex

CATALOG_ROWS = [
    {
        "parent_asin": "A",
        "title": "Wide leather belt",
        "features": ["Buckle closure"],
        "details": {"Closure": "Buckle"},
        "description": ["a belt"],
        "categories": ["Clothing", "Accessories, Belts"],
        "store": "Example",
        "average_rating": 4.5,
        "rating_number": 100,
        "price": 25.0,
    },
    {
        "parent_asin": "B",
        "title": "Padded winter coat",
        "features": ["Down"],
        "details": {"Fill": "Down"},
        "description": ["a coat"],
        "categories": ["Clothing", "Coats, Parkas"],
        "store": "Example",
        "average_rating": 4.1,
        "rating_number": 80,
        "price": 120.0,
    },
]


class DialogRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        path = Path(cls._directory.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS), encoding="utf-8"
        )
        cls.index = CatalogIndex(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def observe(self, text: str, turn: int = 1) -> SessionState:
        state = SessionState("s", {}, self.index)
        state.observe(text, turn)
        return state

    # -- the shipped frames still win -------------------------------------
    def test_original_frame_still_parsed_exactly(self) -> None:
        state = self.observe(
            "I'm looking for Accessories Belts. A key requirement is: Buckle closure."
        )
        self.assertEqual(state.category, "Accessories Belts")
        self.assertEqual(state.constraints, ["Buckle closure"])
        self.assertEqual(state.scenario, "buying")

    # -- the fallback picks up what the frames no longer match -------------
    def test_reworded_opening_still_yields_category_and_constraint(self) -> None:
        state = self.observe(
            "hey there! so i'm hunting for Accessories Belts and the big "
            "thing for me is Buckle closure. what have you got?"
        )
        self.assertEqual(state.category, "Accessories Belts")
        self.assertIn("Buckle closure", state.constraints)

    def test_reworded_browsing_opening_still_yields_category(self) -> None:
        state = self.observe("just poking around at Accessories Belts, nothing firm yet")
        self.assertEqual(state.category, "Accessories Belts")

    def test_case_and_punctuation_damage_still_resolves(self) -> None:
        # A rewriter that lower-cases and de-punctuates the quoted value must
        # still land on the catalog's own spelling, or exact-match scoring dies.
        state = self.observe("Good question. buckle closure. Does that narrow it?")
        self.assertIn("Buckle closure", state.constraints)

    def test_override_cue_recognised_without_the_frame(self) -> None:
        state = self.observe("actually hold on - forget that. Buckle closure is what i need", 3)
        self.assertEqual(state.scenario, "intent_override")
        self.assertIn("Buckle closure", state.constraints)

    def test_exhaustion_cue_recognised_without_the_frame(self) -> None:
        state = self.observe("nope, nothing more on other sorry", 4)
        self.assertTrue(state.exhausted)

    # -- and it does not invent constraints -------------------------------
    def test_category_name_is_not_read_back_as_a_constraint(self) -> None:
        state = self.observe("Window shopping for Accessories Belts really.")
        self.assertEqual(state.constraints, [])

    def test_loosely_named_category_still_resolves(self) -> None:
        # A rewritten customer says "belts", not the catalog's own
        # "Accessories Belts". The category is the most valuable field on
        # turn 1, so partial token overlap has to recover it.
        state = self.observe("hey, just looking for belts really")
        self.assertEqual(state.category, "Accessories Belts")

    def test_fuzzy_category_does_not_fire_when_exact_name_present(self) -> None:
        state = self.observe("Window shopping for Coats Parkas really.")
        self.assertEqual(state.category, "Coats Parkas")

    def test_bare_filler_word_is_not_a_disclosure(self) -> None:
        # "Down" is a real constraint on product B, but as a lone filler word in
        # a carrier sentence it is noise, not a disclosure.
        state = self.observe("Not seeing it. Narrow it Down for me?")
        self.assertEqual(state.constraints, [])


if __name__ == "__main__":
    unittest.main()
