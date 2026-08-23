"""Unit tests for dewie.enrichment.content_validator."""

from __future__ import annotations

from unittest.mock import MagicMock

from dewie.enrichment.content_validator import (
    ContentValidator,
    EnrichmentQualityChecker,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc(
    body: str = "",
    title: str = "A Normal Article Title",
    url: str = "https://example.com/article",
    summary: str = "",
    embed_summary: str = "",
    answers_questions: list[str] | None = None,
):
    d = MagicMock()
    d.body = body
    d.body_text = body
    d.title = title
    d.url = url
    d.summary = summary
    d.embed_summary = embed_summary
    d.answers_questions = answers_questions or []
    d.id = "00000000-0000-0000-0000-000000000001"
    return d


GOOD_BODY = (
    "Artificial intelligence has made remarkable progress in recent years. "
    "Large language models are now capable of solving complex reasoning tasks. "
    "Researchers at several institutions have published findings showing dramatic improvements. "
    "This article explores the implications for software development, healthcare, and education. "
    "The techniques described here represent a significant leap forward in machine learning."
)  # ~300 chars, 50+ words, high alpha ratio


# ── ContentValidator — passing cases ─────────────────────────────────────────


class TestContentValidatorPass:
    def test_good_body_passes(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY))
        assert result.ok

    def test_custom_min_chars_passes(self):
        body = "a " * 500
        result = ContentValidator.validate(_doc(body=body), min_body_chars=50)
        assert result.ok

    def test_long_body_passes(self):
        body = (GOOD_BODY + " ") * 50
        result = ContentValidator.validate(_doc(body=body))
        assert result.ok


# ── ContentValidator — failing cases ─────────────────────────────────────────


class TestContentValidatorFail:
    def test_empty_body_fails(self):
        result = ContentValidator.validate(_doc(body=""))
        assert not result.ok
        assert "min_body_chars" in result.checks_failed

    def test_short_body_fails(self):
        result = ContentValidator.validate(_doc(body="hello world"))
        assert not result.ok
        assert "min_body_chars" in result.checks_failed

    def test_too_few_words_fails(self):
        # Long enough chars but few words
        result = ContentValidator.validate(_doc(body="x" * 300), min_body_words=40)
        assert not result.ok

    def test_body_over_max_fails(self):
        result = ContentValidator.validate(_doc(body="x " * 10_000), max_body_chars=100)
        assert not result.ok
        assert "max_body_chars" in result.checks_failed

    def test_boilerplate_javascript_required_fails(self):
        body = (
            "JavaScript required to view this content. Please enable JavaScript in your browser settings. "
            + "x " * 100
        )
        result = ContentValidator.validate(_doc(body=body))
        assert not result.ok
        assert "boilerplate" in result.checks_failed

    def test_boilerplate_subscribe_fails(self):
        body = (
            "Subscribe to read this article. Sign up now to continue reading the full content. "
            + "x " * 100
        )
        result = ContentValidator.validate(_doc(body=body))
        assert not result.ok
        assert "boilerplate" in result.checks_failed

    def test_boilerplate_access_denied_fails(self):
        body = (
            "Access Denied. You do not have permission to access this page on this server. "
            + "x " * 100
        )
        result = ContentValidator.validate(_doc(body=body))
        assert not result.ok
        assert "boilerplate" in result.checks_failed

    def test_high_noise_ratio_fails(self):
        # Lots of HTML/symbols mixed in
        noisy = "<div>{{{[[[|||]]]}}}</div>" * 30 + " " + "a" * 5
        result = ContentValidator.validate(_doc(body=noisy))
        assert not result.ok

    def test_low_alpha_ratio_fails(self):
        # Base64-like garbage
        garbage = "dGhpcyBpcyBub3QgcmVhbGx5IHRleHQ=" * 20 + "!!!@@##$$$" * 30
        result = ContentValidator.validate(_doc(body=garbage), min_alpha_ratio=0.5)
        assert not result.ok

    def test_high_repetition_fails(self):
        # Nav menu repeated 50 times
        line = "Home About Services Contact Blog"
        repeated = (line + "\n") * 50
        result = ContentValidator.validate(_doc(body=repeated + GOOD_BODY))
        assert not result.ok
        assert "repetition" in result.checks_failed

    def test_short_title_fails(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY, title="ab"))
        assert not result.ok
        assert "title_length" in result.checks_failed

    def test_empty_title_fails(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY, title=""))
        assert not result.ok
        assert "title_length" in result.checks_failed

    def test_generic_title_untitled_fails(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY, title="Untitled"))
        assert not result.ok
        assert "generic_title" in result.checks_failed

    def test_generic_title_home_fails(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY, title="Home | Example"))
        assert not result.ok
        assert "generic_title" in result.checks_failed


# ── ContentValidator — check toggles ─────────────────────────────────────────


class TestContentValidatorToggles:
    def test_boilerplate_check_disabled(self):
        body = "JavaScript required " + GOOD_BODY
        result = ContentValidator.validate(_doc(body=body), check_boilerplate=False)
        assert result.ok

    def test_title_check_disabled(self):
        result = ContentValidator.validate(_doc(body=GOOD_BODY, title=""), check_title=False)
        assert result.ok

    def test_repetition_check_disabled(self):
        repeated = ("Nav link\n") * 50 + GOOD_BODY
        result = ContentValidator.validate(_doc(body=repeated), check_repetition=False)
        # May still fail other checks, but not repetition
        if not result.ok:
            assert "repetition" not in result.checks_failed


# ── ContentValidator — validate_many ─────────────────────────────────────────


class TestValidateMany:
    def test_all_pass(self):
        docs = [_doc(body=GOOD_BODY) for _ in range(5)]
        passed, rejected = ContentValidator.validate_many(docs)
        assert len(passed) == 5
        assert len(rejected) == 0

    def test_some_rejected(self):
        docs = [_doc(body=GOOD_BODY), _doc(body="too short"), _doc(body=GOOD_BODY)]
        passed, rejected = ContentValidator.validate_many(docs)
        assert len(passed) == 2
        assert len(rejected) == 1
        assert rejected[0][1].ok is False

    def test_all_rejected(self):
        docs = [_doc(body="x") for _ in range(3)]
        passed, rejected = ContentValidator.validate_many(docs)
        assert len(passed) == 0
        assert len(rejected) == 3


# ── EnrichmentQualityChecker ──────────────────────────────────────────────────


GOOD_AQ = [
    "What are the main benefits of transformer architectures?",
    "How do attention mechanisms improve language model performance?",
    "What datasets were used to train the model described in this paper?",
    "What are the computational requirements for deploying large language models?",
]

GOOD_SUMMARY = (
    "This paper presents a novel approach to training large language models using "
    "a combination of supervised fine-tuning and reinforcement learning from human feedback. "
    "The authors demonstrate state-of-the-art performance on multiple benchmarks."
)


class TestEnrichmentQualityCheckerPass:
    def test_clean_doc_no_flags(self):
        doc = _doc(
            body=GOOD_BODY,
            summary=GOOD_SUMMARY,
            embed_summary=GOOD_SUMMARY,
            answers_questions=GOOD_AQ,
        )
        flags = EnrichmentQualityChecker.check(doc)
        assert flags == []

    def test_custom_thresholds_pass(self):
        doc = _doc(
            body=GOOD_BODY,
            summary=GOOD_SUMMARY,
            answers_questions=GOOD_AQ,
        )
        flags = EnrichmentQualityChecker.check(doc, min_summary_chars=10, min_aq_count=1)
        assert flags == []


class TestEnrichmentQualityCheckerFail:
    def test_short_summary_flagged(self):
        doc = _doc(summary="Too short.", answers_questions=GOOD_AQ)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "summary_too_short" in checks

    def test_long_summary_flagged(self):
        doc = _doc(summary="x" * 6000, answers_questions=GOOD_AQ)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "summary_too_long" in checks

    def test_summary_body_bleed_flagged(self):
        bleed_text = (
            "this is the raw body text starting here and going on for a while about many topics"
        )
        # Summary IS the beginning of the body — classic bleed
        doc = _doc(body=bleed_text + " " + GOOD_BODY, summary=bleed_text, answers_questions=GOOD_AQ)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "summary_body_bleed" in checks

    def test_too_few_aq_flagged(self):
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=["Only one question?"])
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "aq_count_low" in checks

    def test_too_many_aq_flagged(self):
        many_aq = [f"Question number {i}?" for i in range(25)]
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=many_aq)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "aq_count_high" in checks

    def test_generic_aq_flagged(self):
        generic = ["What is this article about?", "Can you summarize this?"]
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=generic + GOOD_AQ[:2])
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "generic_aq" in checks

    def test_duplicate_aq_flagged(self):
        duped = GOOD_AQ[:2] + [GOOD_AQ[0], GOOD_AQ[1]]  # exact duplicates
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=duped)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "duplicate_aq" in checks

    def test_short_aq_strings_flagged(self):
        short_aq = ["Why?", "How?", "What?"] + GOOD_AQ
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=short_aq)
        flags = EnrichmentQualityChecker.check(doc)
        checks = [f.check for f in flags]
        assert "aq_too_short" in checks

    def test_severity_error_on_generic_aq(self):
        generic = ["What is this article about?"] * 3 + GOOD_AQ[:1]
        doc = _doc(summary=GOOD_SUMMARY, answers_questions=generic)
        flags = EnrichmentQualityChecker.check(doc)
        error_flags = [f for f in flags if f.severity == "error"]
        assert any(f.check == "generic_aq" for f in error_flags)


class TestCheckMany:
    def test_clean_docs_no_results(self):
        docs = [_doc(summary=GOOD_SUMMARY, answers_questions=GOOD_AQ) for _ in range(3)]
        results = EnrichmentQualityChecker.check_many(docs)
        assert results == {}

    def test_flagged_docs_in_results(self):
        docs = [
            _doc(summary=GOOD_SUMMARY, answers_questions=GOOD_AQ),
            _doc(summary="too short", answers_questions=["What is this about?"]),
        ]
        results = EnrichmentQualityChecker.check_many(docs)
        assert len(results) == 1
        doc_id = list(results.keys())[0]
        assert len(results[doc_id]) >= 1
