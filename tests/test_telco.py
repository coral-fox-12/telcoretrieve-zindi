"""Tests for the TelcoRetrieve scaffold. Run: python3 -m unittest discover -s tests -v"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco.prompt import (
    MAX_PROMPT_TOKENS,
    MULTIPLE_CHOICE_INSTRUCTION,
    SYSTEM_MESSAGE,
    TRUE_FALSE_INSTRUCTION,
    Passage,
    PromptTooLong,
    build_prompt,
    build_prompt_budgeted,
)
from telco.questions import ChallengeQuestion, load_test_csv, load_validation_json
from telco.retrieval import BM25Index
from telco.scoring import (
    MIN_ACCURACY,
    score_submission,
    token_efficiency,
)
from telco.submission import (
    SubmissionError,
    SubmissionRow,
    check_submission,
    parse_answer,
    write_submission,
)
from telco.tokens import HAVE_TIKTOKEN, count_tokens

ROOT = Path(__file__).resolve().parents[1]


def mc_question(qid="q1", options=None):
    return ChallengeQuestion(
        id=qid,
        question="What does SRVCC stand for?",
        options=options
        or {
            "A": "Single Radio Voice Call Continuity",
            "B": "Single Radio Video Call Continuity",
            "C": "Seamless Radio Voice Call Control",
            "D": "Single RAN Voice Call Continuity",
        },
        answer="A",
    )


def tf_question(qid="q2", answer="TRUE"):
    return ChallengeQuestion(
        id=qid,
        question="There shall be one pair of TEID-C per UE on the Sv interface.",
        options={},
        answer=answer,
    )


class TestTokens(unittest.TestCase):
    def test_cl100k_known_value(self):
        # tiktoken is installed; exact cl100k_base count.
        self.assertTrue(HAVE_TIKTOKEN)
        self.assertEqual(count_tokens("Hello world, this is a test."), 8)
        self.assertEqual(count_tokens(""), 0)

    def test_longer_text_counts_more(self):
        self.assertGreater(count_tokens("a " * 100), count_tokens("a"))


class TestPrompt(unittest.TestCase):
    def test_system_and_instructions(self):
        p = build_prompt(mc_question(), [Passage(text="SRVCC means Single Radio Voice Call Continuity.", spec="TS 29.280", release="19")])
        self.assertTrue(p.input_prompt.startswith(SYSTEM_MESSAGE + "\n\n"))
        self.assertIn("Passages:\n[1] Spec: TS 29.280 | Release: 19\n", p.input_prompt)
        self.assertIn("Question: What does SRVCC stand for?", p.input_prompt)
        self.assertIn("Options:\nA. Single Radio Voice Call Continuity", p.input_prompt)
        self.assertIn(MULTIPLE_CHOICE_INSTRUCTION, p.input_prompt)
        self.assertEqual(p.n_passages, 1)
        self.assertEqual(p.n_tokens, count_tokens(p.input_prompt))

    def test_tf_prompt_has_no_options(self):
        p = build_prompt(tf_question(), [])
        self.assertIn(TRUE_FALSE_INSTRUCTION, p.input_prompt)
        self.assertNotIn("Options:", p.input_prompt)
        self.assertIn("Passages:\nNone", p.input_prompt)
        self.assertEqual(p.n_passages, 0)

    def test_prompt_drops_passages_to_fit_cap(self):
        big = [Passage(text="word " * 1500) for _ in range(4)]  # ~1500 tok each
        p = build_prompt(mc_question(), big, max_prompt_tokens=4096)
        self.assertLessEqual(p.n_tokens, 4096)
        self.assertLess(p.n_passages, 4)

    def test_prompt_too_long_without_passages(self):
        q = ChallengeQuestion(id="q", question="x " * 5000, options={"A": "a"})
        with self.assertRaises(PromptTooLong):
            build_prompt(q, [], max_prompt_tokens=100)

    def test_budgeted_fill_keeps_passages_within_budget(self):
        # 3 passages of ~100 tokens each; budget 250 keeps the top 2.
        passages = [Passage(text="word " * 100) for _ in range(3)]
        p = build_prompt_budgeted(mc_question(), passages, 250)
        self.assertEqual(p.n_passages, 2)

    def test_budgeted_fill_always_keeps_top_passage(self):
        # A single passage over budget is still kept (top-ranked).
        passages = [Passage(text="word " * 500)]
        p = build_prompt_budgeted(mc_question(), passages, 100)
        self.assertEqual(p.n_passages, 1)

    def test_budgeted_fill_zero_budget_keeps_top_passage(self):
        passages = [Passage(text="word " * 10) for _ in range(3)]
        p = build_prompt_budgeted(mc_question(), passages, 0)
        self.assertEqual(p.n_passages, 1)


class TestQuestions(unittest.TestCase):
    def test_load_test_csv(self):
        qs = load_test_csv(ROOT / "data/Test.csv")
        self.assertEqual(len(qs), 100)
        n_mc = sum(1 for q in qs if not q.is_true_false)
        n_tf = sum(1 for q in qs if q.is_true_false)
        self.assertEqual(n_mc, 50)
        self.assertEqual(n_tf, 50)
        self.assertTrue(all(q.id for q in qs))
        self.assertEqual(len({q.id for q in qs}), 100)

    def test_load_validation_json(self):
        qs = load_validation_json(ROOT / "data/validation_questions.json")
        self.assertEqual(len(qs), 12)
        self.assertTrue(all(q.answer for q in qs))
        self.assertTrue(all(q.evidence_chunk for q in qs))

    def test_valid_answers(self):
        self.assertEqual(mc_question().valid_answers, ["A", "B", "C", "D"])
        self.assertEqual(tf_question().valid_answers, ["TRUE", "FALSE"])


class TestSubmission(unittest.TestCase):
    def test_parse_answer_mc(self):
        q = mc_question()
        self.assertEqual(parse_answer("B", q), "B")
        self.assertEqual(parse_answer("  b.  ", q), "B")
        self.assertEqual(parse_answer("The answer is C.", q), "C")
        self.assertEqual(parse_answer("I think it is A", q), "A")
        self.assertEqual(parse_answer("", q), "")
        self.assertEqual(parse_answer("Zebra", q), "")
        # Reasoning trace: early mentions are not the answer; the last
        # valid token is the conclusion.
        self.assertEqual(
            parse_answer("Option A mentions voice. Option B is video. So B.", q), "B"
        )

    def test_parse_answer_tf(self):
        q = tf_question()
        self.assertEqual(parse_answer("TRUE", q), "TRUE")
        self.assertEqual(parse_answer("false.", q), "FALSE")
        self.assertEqual(parse_answer("It is true.", q), "TRUE")
        self.assertEqual(parse_answer("maybe", q), "")

    def test_write_and_check_roundtrip(self):
        qs = [mc_question("q1"), tf_question("q2")]
        rows = [
            SubmissionRow(id="q1", input_prompt="sys\n\nuser", answer="A"),
            SubmissionRow(id="q2", input_prompt="sys\n\nuser", answer="TRUE"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "submission.csv"
            write_submission(rows, path)
            self.assertEqual(check_submission(path, qs), rows)
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))

    def test_check_rejects_bad_answer(self):
        qs = [mc_question("q1")]
        rows = [SubmissionRow(id="q1", input_prompt="x", answer="Z")]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s.csv"
            write_submission(rows, path)
            with self.assertRaises(SubmissionError):
                check_submission(path, qs)

    def test_check_rejects_missing_id(self):
        qs = [mc_question("q1"), tf_question("q2")]
        rows = [SubmissionRow(id="q1", input_prompt="x", answer="A")]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s.csv"
            write_submission(rows, path)
            with self.assertRaises(SubmissionError):
                check_submission(path, qs)

    def test_check_rejects_overlong_prompt(self):
        qs = [mc_question("q1")]
        rows = [SubmissionRow(id="q1", input_prompt="word " * 5000, answer="A")]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s.csv"
            write_submission(rows, path)
            with self.assertRaises(SubmissionError):
                check_submission(path, qs)


class TestGenerate(unittest.TestCase):
    def test_default_standin_model(self):
        # The 12/12 baseline ran on nemotron-3-nano-omni; gemma-4 is
        # 429-limited upstream as of 2026-11-17, so the default is the
        # nemotron-omni id (see generate.py).
        from telco.generate import DEFAULT_FREE_MODEL

        self.assertEqual(
            DEFAULT_FREE_MODEL,
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        )


class TestScoring(unittest.TestCase):
    def test_perfect_score(self):
        s = score_submission({"a": "A", "b": "B"}, {"a": "A", "b": "B"}, 0)
        self.assertEqual(s.accuracy, 1.0)
        self.assertEqual(s.token_efficiency, 100.0)
        self.assertEqual(s.overall, 100.0)
        self.assertTrue(s.passes_accuracy_gate)

    def test_baseline_token_efficiency(self):
        # T == B (850 tokens/question) -> efficiency exactly 50.
        self.assertAlmostEqual(token_efficiency(850 * 4, 4), 50.0)

    def test_overall_formula(self):
        # accuracy 0.9, T = 850/q -> eff 50 -> overall 0.7*90 + 0.3*50 = 78.
        s = score_submission(
            {f"q{i}": "A" for i in range(9)} | {"q9": "B"},
            {f"q{i}": "A" for i in range(10)},
            8500,
        )
        self.assertAlmostEqual(s.accuracy, 0.9)
        self.assertAlmostEqual(s.overall, 78.0)

    def test_accuracy_gate(self):
        s = score_submission({"a": "A"}, {"a": "A", "b": "B"}, 0)
        self.assertEqual(s.accuracy, 0.5)
        self.assertFalse(s.passes_accuracy_gate)
        self.assertGreaterEqual(MIN_ACCURACY, 0.8)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index.from_jsonl(ROOT / "data/mini_index.jsonl")

    def test_index_loads(self):
        self.assertEqual(len(self.index.chunks), 688)
        self.assertTrue(all(c["text"] for c in self.index.chunks))

    def test_gtp_query_finds_sv_chunks(self):
        hits = self.index.search("Sv interface GTPv2-C messages MME MSC SRVCC", top_k=5)
        self.assertEqual(len(hits), 5)
        specs = {h.spec for h in hits}
        self.assertIn("TS 29.280", specs)

    def test_evidence_chunk_retrievable(self):
        # The val-003 evidence chunk (message-type table) must be found
        # for its own question wording. BM25-only ranks the table 15th:
        # table cells repeat "SRVCC PS to CS" across many chunks, so the
        # signal dilutes — a known lexical-retrieval weakness the dense
        # + reranker path must fix. Top-20 documents it; the eval
        # report's evidence_in_context (top-5) tracks the real number.
        hits = self.index.search(
            "Message Type value SRVCC PS to CS Request Sv interface", top_k=20
        )
        ids = [h.id for h in hits]
        self.assertIn("TS29280-c0009", ids)


if __name__ == "__main__":
    unittest.main()
