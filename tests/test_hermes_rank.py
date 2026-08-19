"""Tests for the vendored hermes_rank.py ranking script.

Offline-safe (repo invariant): the module is importlib-loaded, the
Artificial Analysis client is exercised through a monkeypatched fake
`_opener` (no network), and `main()` is run against tmp data with no
API key set.
"""

import email.message
import importlib.util
import io
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# hermes_rank.py does `from common import _opener`, so the repo root must be
# importable even when pytest does not add it to sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def hermes():
    return load_module("hermes_rank", "hermes_rank.py")


# ── WEIGHTS ───────────────────────────────────────────────────────────────────


def test_weights_sum_to_one(hermes):
    assert sum(hermes.WEIGHTS.values()) == pytest.approx(1.0)


# ── normalize ─────────────────────────────────────────────────────────────────


def test_normalize_strips_provider_prefixes(hermes):
    assert hermes.normalize("@cf/qwen3") == "qwen3"
    assert hermes.normalize("ollama:llama-4") == "llama-4"
    assert hermes.normalize("kilo:hy3") == "hy3"
    assert hermes.normalize("mistral:ministral-8b") == "ministral-8b"
    assert hermes.normalize("nvidia:nemotron-3-super") == "nemotron-3-super"
    assert hermes.normalize("zen:qwen3.6") == "qwen3.6"
    assert hermes.normalize("cloudflare:foo") == "foo"
    assert hermes.normalize("groq:foo") == "foo"


def test_normalize_lowercases_and_replaces_separators(hermes):
    assert hermes.normalize("OpenAI GPT-OSS 120B") == "openai gpt-oss 120b"
    assert hermes.normalize("minimax/minimax-m3") == "minimax minimax-m3"
    assert hermes.normalize("MiniMax/MiniMax-M3") == "minimax minimax-m3"
    assert hermes.normalize("  GEMINI 2.5 ") == "gemini 2.5"
    assert hermes.normalize("foo@bar:baz") == "foo bar baz"  # :/@ become spaces


def test_normalize_none_and_garbage(hermes):
    assert hermes.normalize(None) == ""
    # Non-alphanumerics stripped, hyphens kept.
    assert hermes.normalize("gpt-oss!!!120b") == "gpt-oss120b"


# ── compact ───────────────────────────────────────────────────────────────────


def test_compact_cascade(hermes):
    assert hermes.compact("gpt-oss-120b") == "gptoss120b"
    assert hermes.compact("@cf/qwen3") == "qwen3"
    assert hermes.compact("OpenAI GPT-OSS 120B") == "openaigptoss120b"
    assert hermes.compact(None) == ""


# ── percentile_normalize ──────────────────────────────────────────────────────


def test_percentile_minmax(hermes):
    assert hermes.percentile_normalize({"a": 0.0, "b": 50.0, "c": 100.0}) == {
        "a": 0.0,
        "b": 50.0,
        "c": 100.0,
    }
    out = hermes.percentile_normalize({"a": 10.0, "b": 30.0})
    assert out["a"] == pytest.approx(0.0)
    assert out["b"] == pytest.approx(100.0)


def test_percentile_missing_values_stay_none(hermes):
    out = hermes.percentile_normalize({"a": 1.0, "b": None, "c": 2.0})
    assert out["b"] is None
    assert out["a"] == pytest.approx(0.0)
    assert out["c"] == pytest.approx(100.0)
    assert hermes.percentile_normalize({"a": None}) == {"a": None}
    assert hermes.percentile_normalize({}) == {}


def test_percentile_hi_equals_lo(hermes):
    assert hermes.percentile_normalize({"a": 5.0, "b": 5.0}) == {
        "a": 50.0,
        "b": 50.0,
    }
    assert hermes.percentile_normalize({"a": 5.0, "b": None}) == {
        "a": 50.0,
        "b": None,
    }


# ── load_csv ──────────────────────────────────────────────────────────────────


def test_load_csv_none_path(hermes):
    assert hermes.load_csv(None) == {}


def test_load_csv_malformed_rows_skipped(hermes, tmp_path):
    csv_path = tmp_path / "arena.csv"
    csv_path.write_text(
        "model,elo\n"
        "minimax/minimax-m3,1273\n"
        "broken,\n"
        ",88.0\n"
        "garbage,not-a-number\n"
        "openai/gpt-oss-120b,62.4\n"
    )
    out = hermes.load_csv(str(csv_path))
    assert out == {"minimax minimax-m3": 1273.0, "openai gpt-oss-120b": 62.4}


def test_load_csv_accepts_score_column(hermes, tmp_path):
    csv_path = tmp_path / "swebench.csv"
    csv_path.write_text("model,score\nqwen/qwen3.6-27b,59.0\n")
    assert hermes.load_csv(str(csv_path)) == {"qwen qwen3.6-27b": 59.0}


def test_load_csv_header_errors(hermes, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="no header"):
        hermes.load_csv(str(empty))
    bad = tmp_path / "bad.csv"
    bad.write_text("model,other\nx,1\n")
    with pytest.raises(ValueError, match="expected columns"):
        hermes.load_csv(str(bad))


def test_load_csv_missing_file_raises(hermes, tmp_path):
    # NOTE: the vendored code does NOT swallow a missing file — it raises
    # FileNotFoundError. Only a None path short-circuits to {}.
    with pytest.raises(FileNotFoundError):
        hermes.load_csv(str(tmp_path / "nope.csv"))


# ── extract_models ────────────────────────────────────────────────────────────


def test_extract_models_schema(hermes):
    data = {
        "updated": "2026-08-19",
        "providers": {
            "minimax": {
                "name": "MiniMax",
                "url": "https://example.com",
                "models": [
                    {
                        "id": "minimax/minimax-m3",
                        "name": "MiniMax M3",
                        "context": 128000,
                        "params_b": 3.0,
                        "limits": {"max_tokens": 65536},
                    },
                    {"id": "no-name-model"},  # name falls back to id
                    {"name": "missing-id"},  # skipped: no id
                    "not-a-dict",  # skipped: not a dict
                ],
            },
            "not-a-provider": "skip me",  # skipped: not a dict
        },
    }
    models = hermes.extract_models(data)
    assert len(models) == 2
    m = models[0]
    assert m["provider_id"] == "minimax"
    assert m["provider"] == "MiniMax"
    assert m["id"] == "minimax/minimax-m3"
    assert m["name"] == "MiniMax M3"
    assert m["context"] == 128000
    assert m["params_b"] == 3.0
    assert m["limits"] == {"max_tokens": 65536}
    assert models[1]["name"] == "no-name-model"  # falls back to id


def test_extract_models_invalid_inputs(hermes):
    with pytest.raises(ValueError, match="JSON object"):
        hermes.extract_models(["not", "a", "dict"])
    with pytest.raises(ValueError, match="'providers'"):
        hermes.extract_models({"providers": "nope"})


# ── matching: aliases and substring fallback ──────────────────────────────────


def test_match_alias_resolution(hermes):
    aa = [
        {"slug": "minimax-m3", "name": "MiniMax M3"},
        {"slug": "hy3", "name": "Hunyuan 3"},
        {"slug": "inkling", "name": "Inkling"},
    ]
    normal, compact = hermes.build_aa_lookup(aa)

    hit, kind = hermes.match_model("minimax/minimax-m3", "Completely Different", normal, compact)
    assert kind == "exact"
    assert hit["slug"] == "minimax-m3"

    hit, kind = hermes.match_model("tencent/hy3", "Completely Different", normal, compact)
    assert kind == "exact" and hit["slug"] == "hy3"

    hit, kind = hermes.match_model("thinkingmachines/inkling", "Completely Different", normal, compact)
    assert kind == "exact" and hit["slug"] == "inkling"


def test_match_substring_unique_candidate(hermes):
    aa = [{"slug": "gpt-oss-120b-extra", "name": "Extra"}]
    normal, compact = hermes.build_aa_lookup(aa)
    hit, kind = hermes.match_model("gpt-oss-120b", "Not Matching", normal, compact)
    assert kind == "substring"
    assert hit["slug"] == "gpt-oss-120b-extra"


def test_match_substring_ambiguous_is_missing(hermes):
    aa = [
        {"slug": "gpt-oss-120b-extra", "name": "Extra"},
        {"slug": "gpt-oss-120b-v2", "name": "V2"},
    ]
    normal, compact = hermes.build_aa_lookup(aa)
    hit, kind = hermes.match_model("gpt-oss-120b", "Not Matching", normal, compact)
    assert kind == "missing" and hit is None


def test_match_substring_skips_short_keys(hermes):
    # Lookup data whose compact keys do not collide with the source's compact
    # key: "ABC" would compact to "abc" and hit the compact pass first, so use
    # a name that keeps "abc" out of the compact lookup entirely.
    aa = [{"slug": "xyz-abc-model", "name": "Something Else"}]
    normal, compact = hermes.build_aa_lookup(aa)
    # "a-b-c" compacts to "abc" (length 3 < 6), so the substring fallback is skipped.
    hit, kind = hermes.match_model("a-b-c", "Not Matching", normal, compact)
    assert kind == "missing" and hit is None


# ── calculate_score: reweighting, clamping, coverage ─────────────────────────


FULL_RANGES = {
    "agentic": (0.0, 100.0),
    "coding": (0.0, 100.0),
    "intelligence": (0.0, 100.0),
    "arena": (0.0, 100.0),
    "swebench": (0.0, 100.0),
}


def test_calculate_score_missing_metrics_reweight(hermes):
    score, coverage, normalized = hermes.calculate_score(
        {"agentic": 100.0, "coding": None, "intelligence": None},
        None,
        None,
        FULL_RANGES,
    )
    # Only agentic (weight 0.40) present: score renormalizes to 100, not 40.
    assert score == pytest.approx(100.0)
    assert coverage == pytest.approx(0.40)
    assert normalized["agentic"] == pytest.approx(100.0)
    assert normalized["coding"] is None


def test_calculate_score_partial_data(hermes):
    ranges = {**FULL_RANGES, "arena": (0.0, 200.0)}
    score, coverage, normalized = hermes.calculate_score(
        {"agentic": 100.0, "coding": None, "intelligence": None},
        80.0,  # arena → (80 - 0) / (200 - 0) * 100 = 40
        None,
        ranges,
    )
    # (100 * 0.40 + 40 * 0.10) / 0.50 → 88.0
    assert score == pytest.approx(88.0)
    assert coverage == pytest.approx(0.50)
    assert normalized["arena"] == pytest.approx(40.0)
    assert normalized["swebench"] is None


def test_calculate_score_clamps_out_of_range(hermes):
    score, coverage, normalized = hermes.calculate_score(
        {"agentic": 150.0, "coding": None, "intelligence": None},
        None,
        None,
        FULL_RANGES,
    )
    assert normalized["agentic"] == 100.0  # clamped
    assert score == pytest.approx(100.0)
    assert coverage == pytest.approx(0.40)


def test_calculate_score_flat_range_midpoint(hermes):
    ranges = {**FULL_RANGES, "arena": (50.0, 50.0)}
    score, coverage, normalized = hermes.calculate_score(
        {"agentic": None, "coding": None, "intelligence": None},
        50.0,
        None,
        ranges,
    )
    assert normalized["arena"] == 50.0  # hi == lo → midpoint
    assert score == pytest.approx(50.0)
    assert coverage == pytest.approx(0.10)


def test_calculate_score_no_data(hermes):
    score, coverage, normalized = hermes.calculate_score(
        {"agentic": None, "coding": None, "intelligence": None},
        None,
        None,
        FULL_RANGES,
    )
    assert score == 0.0 and coverage == 0.0
    assert all(v is None for v in normalized.values())


# ── main(): --min-coverage filtering ─────────────────────────────────────────


def test_main_min_coverage_filters(hermes, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
    models_json = {
        "providers": {
            "minimax": {
                "name": "MiniMax",
                "models": [
                    {"id": "minimax/minimax-m3", "name": "MiniMax M3"},
                    {"id": "some/other-model", "name": "Other Model"},
                ],
            }
        }
    }
    (tmp_path / "models.json").write_text(json.dumps(models_json))
    (tmp_path / "arena.csv").write_text("model,elo\nminimax/minimax-m3,1273\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes_rank",
            "--models",
            "models.json",
            "--arena",
            "arena.csv",
            "--min-coverage",
            "0.05",
        ],
    )

    assert hermes.main() == 0
    out = json.loads((tmp_path / "hermes_ranking.json").read_text())
    assert out["metadata"]["ranked_count"] == 1
    assert out["ranking"][0]["model_id"] == "minimax/minimax-m3"
    # arena-only model has coverage 0.10 → kept; no-data model 0.0 → dropped.
    assert out["ranking"][0]["data_coverage"] == 10.0
    assert (tmp_path / "hermes_ranking.csv").exists()


# ── Artificial Analysis client (fake _opener, no network) ────────────────────


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Replays a list of page payloads; records (url, headers) per request."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []  # [(url, headers), ...]

    def open(self, request, timeout=None):
        # urllib stores header keys capitalized (Request.add_header), so
        # normalize to lowercase before recording for case-insensitive asserts.
        self.calls.append((request.full_url, {k.lower(): v for k, v in request.headers.items()}))
        return FakeResponse(self.pages.pop(0))


class AlwaysMoreOpener:
    """Every page says has_more=True — drives the page > 100 abort."""

    def __init__(self):
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        payload = {
            "data": [{"slug": f"m{self.calls}", "name": f"Model {self.calls}"}],
            "pagination": {"has_more": True},
        }
        return FakeResponse(payload)


class ErrorOpener:
    """Raises a urllib HTTPError with the configured status."""

    def __init__(self, code, headers=None):
        self.code = code
        self.headers = headers or {}
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        msg = email.message.Message()
        for k, v in self.headers.items():
            msg[k] = v
        raise urllib.error.HTTPError(
            request.full_url,
            self.code,
            "boom",
            msg,
            io.BytesIO(),
        )


def test_aa_client_pagination(hermes, monkeypatch):
    pages = [
        {
            "intelligence_index_version": "3.14",
            "data": [{"slug": "a", "name": "A"}, {"slug": "b", "name": "B"}],
            "pagination": {"has_more": True},
        },
        {
            "data": [{"slug": "c", "name": "C"}],
            "pagination": {"has_more": False},
        },
    ]
    fake = FakeOpener(pages)
    sleeps = []
    monkeypatch.setattr(hermes, "_opener", fake)
    monkeypatch.setattr(hermes, "time", types.SimpleNamespace(sleep=sleeps.append))

    models, version = hermes.fetch_aa_models("test-key-123")

    assert len(models) == 3
    assert version == pytest.approx(3.14)
    assert len(fake.calls) == 2
    assert fake.calls[0][0].endswith("?page=1")
    assert fake.calls[1][0].endswith("?page=2")
    # x-api-key is sent on every request.
    assert all(headers.get("x-api-key") == "test-key-123" for _, headers in fake.calls)
    # One 0.15s sleep between the two pages.
    assert sleeps == [0.15]


def test_aa_client_env_key_fallback(hermes, monkeypatch):
    fake = FakeOpener([{"data": [], "pagination": {"has_more": False}}])
    monkeypatch.setattr(hermes, "_opener", fake)
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "env-key")

    models, version = hermes.fetch_aa_models()

    assert fake.calls[0][1]["x-api-key"] == "env-key"


def test_aa_client_no_key_returns_empty(hermes, monkeypatch):
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
    assert hermes.fetch_aa_models() == ([], None)


def test_aa_client_aborts_after_100_pages(hermes, monkeypatch):
    fake = AlwaysMoreOpener()
    sleeps = []
    monkeypatch.setattr(hermes, "_opener", fake)
    monkeypatch.setattr(hermes, "time", types.SimpleNamespace(sleep=sleeps.append))

    with pytest.raises(RuntimeError, match="unexpectedly many API pages"):
        hermes.fetch_aa_models("test-key-123")

    assert fake.calls == 100  # pages 1..100 fetched before the abort
    assert len(sleeps) == 99


@pytest.mark.parametrize(
    "code,headers,match",
    [
        (429, {"Retry-After": "12"}, "Retry-After=12"),
        (401, {}, "key is invalid"),
        (403, {}, "tier does not permit"),
    ],
)
def test_aa_client_http_errors_raise_runtime_error(hermes, monkeypatch, code, headers, match):
    fake = ErrorOpener(code, headers)
    monkeypatch.setattr(hermes, "_opener", fake)

    with pytest.raises(RuntimeError, match=match):
        hermes.fetch_aa_models("test-key-123")

    assert fake.calls == 1
