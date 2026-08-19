"""Unit tests for generate_site.load_quality_rankings (offline-safe).

The loader reads docs/availability/rankings.json (produced by hermes_rank.py)
and returns {normalize_model_id(model_id): hermes_score}. It must tolerate any
file shape and return {} on any failure. generate_site.py is loaded via
importlib exactly like tests/test_dashboard.py does — no network at import.
"""

import importlib.util
import json
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


@pytest.fixture(scope="module")
def generate_site():
    return load_module("generate_site", "generate_site.py")


def _write(tmp_path, data):
    """Write a rankings.json payload into tmp_path and return its path."""
    p = tmp_path / "rankings.json"
    p.write_text(json.dumps(data))
    return p


def test_valid_file_returns_normalized_scores(generate_site, tmp_path):
    p = _write(
        tmp_path,
        {
            "metadata": {"source_count": 2},
            "ranking": [
                {"model_id": "openai/gpt-4o:free", "hermes_score": 88.5},
                {"model_id": "@cf/meta/llama-3.3-70b-instruct", "hermes_score": 91},
            ],
        },
    )
    assert generate_site.load_quality_rankings(p) == {
        "gpt4o": 88.5,
        "llama3370binstruct": 91.0,
    }


def test_missing_file_returns_empty(generate_site, tmp_path):
    assert generate_site.load_quality_rankings(tmp_path / "nope.json") == {}


def test_malformed_json_returns_empty(generate_site, tmp_path):
    p = tmp_path / "rankings.json"
    p.write_text("{not valid json")
    assert generate_site.load_quality_rankings(p) == {}


def test_ranking_key_missing_or_not_list_returns_empty(generate_site, tmp_path):
    # "ranking" key missing entirely
    assert generate_site.load_quality_rankings(_write(tmp_path, {"metadata": {}})) == {}
    # "ranking" present but not a list
    assert (
        generate_site.load_quality_rankings(_write(tmp_path, {"ranking": {"model_id": "x"}}))
        == {}
    )
    # top-level value not a dict at all
    assert generate_site.load_quality_rankings(_write(tmp_path, ["not", "a", "dict"])) == {}


def test_nan_hermes_score_skipped(generate_site, tmp_path):
    p = _write(
        tmp_path,
        {
            "ranking": [
                {"model_id": "openai/gpt-4o", "hermes_score": float("nan")},
                {"model_id": "openai/gpt-4o:free", "hermes_score": 80},
            ]
        },
    )
    # json.dumps emits NaN, json.loads parses it back to float nan → row skipped.
    assert generate_site.load_quality_rankings(p) == {"gpt4o": 80.0}


def test_non_dict_row_skipped(generate_site, tmp_path):
    p = _write(
        tmp_path,
        {
            "ranking": [
                "garbage-row",
                {"model_id": "deepseek/deepseek-v4-flash-free", "hermes_score": 70},
                None,
                42,
            ]
        },
    )
    assert generate_site.load_quality_rankings(p) == {"deepseekv4flash": 70.0}


def test_suffix_and_prefix_normalization(generate_site, tmp_path):
    """@cf/ provider prefix and :free suffix must be stripped by normalization."""
    p = _write(
        tmp_path,
        {
            "ranking": [
                {"model_id": "@cf/meta/llama-3.3-70b-instruct", "hermes_score": 90},
                {"model_id": "meta-llama/llama-3.1-8b-instruct:free", "hermes_score": 60},
            ]
        },
    )
    assert generate_site.load_quality_rankings(p) == {
        "llama3370binstruct": 90.0,
        "metallamallama318binstruct": 60.0,
    }


def test_missing_model_id_or_score_skipped(generate_site, tmp_path):
    p = _write(
        tmp_path,
        {
            "ranking": [
                {"model_id": "", "hermes_score": 99},  # empty id
                {"model_id": None, "hermes_score": 99},  # None id
                {"model_id": "x/y", "hermes_score": "n/a"},  # non-numeric score
                {"model_id": "x/y", "hermes_score": None},  # None score
                {"model_id": "x/y"},  # missing score
            ]
        },
    )
    assert generate_site.load_quality_rankings(p) == {}
