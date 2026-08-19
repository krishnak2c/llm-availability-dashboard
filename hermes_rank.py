#!/usr/bin/env python3

"""
Hermes Agentic Model Ranker

Inputs:
    models.json                         Required
    arena.csv                           Optional
    swebench.csv                        Optional
    ARTIFICIAL_ANALYSIS_API_KEY        Required for AA data

Outputs:
    hermes_ranking.csv
    hermes_ranking.json

Example:
    export ARTIFICIAL_ANALYSIS_API_KEY="your_key"
    python hermes_rank.py \
        --models models.json \
        --arena arena.csv \
        --swebench swebench.csv \
        --out-prefix hermes_ranking

Arena CSV format:
    model,elo

Example:
    minimax/minimax-m3,1273
    tencent/hy3,1227

SWE-bench CSV format:
    model,score

Example:
    minimax/minimax-m3,59.0
    openai/gpt-oss-120b,62.4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from common import _opener

AA_BASE_URL = "https://artificialanalysis.ai/api/v2"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WEIGHTS = {
    "agentic": 0.40,
    "coding": 0.25,
    "intelligence": 0.15,
    "arena": 0.10,
    "swebench": 0.10,
}

# Substring fallback ignores compact keys shorter than this to avoid
# noise matches from very short ids.
SUBSTRING_MIN_COMPACT_LEN = 6

# Explicit aliases for common provider/model naming differences.
ALIASES = {
    # MiniMax
    "minimax/minimax-m3": [
        "minimax-m3",
        "minimax m3",
        "minimaxai/minimax-m3",
    ],

    # Tencent
    "tencent/hy3": [
        "hy3",
        "tencent hy3",
    ],

    # Thinking Machines
    "thinkingmachines/inkling": [
        "inkling",
        "thinking machines inkling",
    ],

    # Mistral
    "mistralai/mistral-large-2512": [
        "mistral-large-2512",
        "mistral large 3",
        "mistral large 2512",
        "mistral-large-3",
    ],
    "mistralai/mistral-medium-3": [
        "mistral-medium-3",
        "mistral medium 3",
    ],
    "mistralai/mistral-medium-3-5": [
        "mistral-medium-3-5",
        "mistral medium 3.5",
    ],
    "mistralai/codestral-2508": [
        "codestral-2508",
        "codestral",
    ],
    "mistralai/ministral-14b-2512": [
        "ministral-14b-2512",
        "ministral 14b",
    ],
    "mistralai/ministral-8b-2512": [
        "ministral-8b-2512",
        "ministral 8b",
    ],
    "mistralai/ministral-3b-2512": [
        "ministral-3b-2512",
        "ministral 3b",
    ],

    # OpenAI
    "openai/gpt-oss-120b": [
        "gpt-oss-120b",
        "gpt-oss 120b",
    ],
    "openai/gpt-oss-20b": [
        "gpt-oss-20b",
        "gpt-oss 20b",
    ],

    # Meta
    "meta-llama/llama-4-scout": [
        "llama-4-scout",
        "llama 4 scout",
    ],

    # NVIDIA
    "nvidia/nemotron-3-super-120b-a12b": [
        "nemotron-3-super-120b-a12b",
        "nemotron 3 super",
        "nemotron-3-super",
    ],
    "nvidia/nemotron-3-nano-30b-a3b": [
        "nemotron-3-nano-30b-a3b",
        "nemotron 3 nano",
    ],
    "nvidia/nemotron-3.5-lightning": [
        "nemotron-3.5-lightning",
        "nemotron 3.5 lightning",
    ],
    "nvidia/nemotron-3-ultra-550b-a55b": [
        "nemotron-3-ultra-550b-a55b",
        "nemotron 3 ultra",
    ],

    # Qwen
    "qwen/qwen3.6-27b": [
        "qwen3.6-27b",
        "qwen 3.6 27b",
    ],

    # Poolside
    "poolside/laguna-s-2.1:free": [
        "laguna-s-2.1",
        "laguna s 2.1",
        "laguna-s-2.1:free",
    ],

    # Step
    "stepfun/step-3.7-flash:free": [
        "step-3.7-flash",
        "step 3.7 flash",
    ],
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize(value: Any) -> str:
    """Normalize a model identifier for comparison."""
    if value is None:
        return ""

    s = str(value).lower().strip()

    # Remove provider URL decorations.
    s = s.replace("@cf/", "")
    s = s.replace("ollama:", "")
    s = s.replace("kilo:", "")
    s = s.replace("mistral:", "")
    s = s.replace("nvidia:", "")
    s = s.replace("groq:", "")
    s = s.replace("cloudflare:", "")
    s = s.replace("zen:", "")

    # Common separators.
    s = s.replace("_", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[:/@]", " ", s)

    # Keep alphanumeric + spaces/hyphens.
    s = re.sub(r"[^a-z0-9.\- ]+", "", s)

    return s.strip()


def compact(value: Any) -> str:
    """Very aggressive normalization for fallback matching."""
    return re.sub(r"[^a-z0-9]+", "", normalize(value))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (ValueError, TypeError):
        return None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def percentile_normalize(
    values: dict[str, float | None],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float | None]:
    """
    Convert arbitrary benchmark values into 0-100 using min-max.

    Without `ranges`, each value is normalized against the min/max of the
    values in this dict; with `ranges`, against each metric's (lo, hi)
    range instead. Missing values remain None. Flat ranges (lo == hi)
    yield the midpoint 50.0. Results are clamped to [0, 100].
    """
    result: dict[str, float | None] = {}

    for k, v in values.items():
        if v is None:
            result[k] = None
            continue

        if ranges is not None and k in ranges:
            lo, hi = ranges[k]
        else:
            valid = [x for x in values.values() if x is not None]

            if not valid:
                result[k] = None
                continue

            lo, hi = min(valid), max(valid)

        if hi == lo:
            result[k] = 50.0
        else:
            result[k] = clamp(
                ((v - lo) / (hi - lo)) * 100.0
            )

    return result


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str | None) -> dict[str, float]:
    if not path:
        return {}

    result: dict[str, float] = {}

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header")

        fields = {x.lower().strip(): x for x in reader.fieldnames}

        model_col = fields.get("model")
        score_col = (
            fields.get("elo")
            or fields.get("score")
            or fields.get("value")
        )

        if not model_col or not score_col:
            raise ValueError(
                f"{path}: expected columns 'model' and "
                f"'elo' or 'score'"
            )

        for row in reader:
            model = row.get(model_col)
            score = safe_float(row.get(score_col))

            if model and score is not None:
                result[normalize(model)] = score

    return result


# ---------------------------------------------------------
# Models.json parsing
# ---------------------------------------------------------

def extract_models(models_json: Any) -> list[dict[str, Any]]:
    """
    Supports the user's llm-availability-dashboard schema:

    {
      "providers": {
        "provider": {
          "name": "...",
          "models": [...]
        }
      }
    }
    """

    if not isinstance(models_json, dict):
        raise ValueError("models.json must contain a JSON object")

    providers = models_json.get("providers", {})

    if not isinstance(providers, dict):
        raise ValueError("models.json has no valid 'providers' object")

    models: list[dict[str, Any]] = []

    for provider_id, provider in providers.items():
        if not isinstance(provider, dict):
            continue

        provider_name = provider.get("name", provider_id)

        for model in provider.get("models", []):
            if not isinstance(model, dict):
                continue

            model_id = model.get("id")
            model_name = model.get("name", model_id)

            if not model_id:
                continue

            models.append(
                {
                    "provider_id": provider_id,
                    "provider": provider_name,
                    "id": model_id,
                    "name": model_name,
                    "context": model.get("context"),
                    "params_b": model.get("params_b"),
                    "limits": model.get("limits"),
                }
            )

    return models


# ---------------------------------------------------------
# Artificial Analysis API
# ---------------------------------------------------------

class ArtificialAnalysisClient:

    def __init__(self, api_key: str):
        self.api_key = api_key

        self.headers = {
            "x-api-key": api_key,
            "Accept": "application/json",
            "User-Agent": "Hermes-Agent-Ranker/1.0",
        }

    def get_all_models(self) -> tuple[list[dict[str, Any]], float | None]:
        """
        Fetch all pages from:
            /api/v2/language/models/free

        The free endpoint exposes the headline:
            artificial_analysis_intelligence_index
            artificial_analysis_coding_index
            artificial_analysis_agentic_index
        """

        all_models: list[dict[str, Any]] = []
        page = 1
        index_version: float | None = None

        while True:
            url = f"{AA_BASE_URL}/language/models/free?page={page}"

            request = urllib.request.Request(
                url,
                headers=self.headers,
            )

            try:
                with _opener.open(request, timeout=30) as response:
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    raise RuntimeError(
                        "Artificial Analysis API rate limit reached. "
                        f"Retry-After={retry_after}"
                    ) from exc

                if exc.code == 401:
                    raise RuntimeError(
                        "Artificial Analysis API key is invalid."
                    ) from exc

                if exc.code == 403:
                    raise RuntimeError(
                        "Your Artificial Analysis API tier does not "
                        "permit this endpoint."
                    ) from exc

                raise

            if index_version is None:
                index_version = safe_float(
                    payload.get("intelligence_index_version")
                )

            page_data = payload.get("data", [])
            if isinstance(page_data, list):
                all_models.extend(page_data)

            pagination = payload.get("pagination", {})

            has_more = bool(
                pagination.get("has_more", False)
            )

            if not has_more:
                break

            page += 1

            # Avoid accidental runaway loops.
            if page > 100:
                raise RuntimeError(
                    "Aborting: unexpectedly many API pages."
                )

            time.sleep(0.15)

        return all_models, index_version


def fetch_aa_models(
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], float | None]:
    """
    Fetch all pages of free models from the Artificial Analysis API.

    Falls back to the ARTIFICIAL_ANALYSIS_API_KEY environment variable
    when no key is passed. Returns empty results (no crash) when no
    key is configured.
    """
    if not api_key:
        api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")

    if not api_key:
        return [], None

    return ArtificialAnalysisClient(api_key).get_all_models()


# ---------------------------------------------------------
# Model matching
# ---------------------------------------------------------

def build_aa_lookup(
    aa_models: Iterable[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:

    normal_lookup: dict[str, dict[str, Any]] = {}
    compact_lookup: dict[str, dict[str, Any]] = {}

    for m in aa_models:
        slug = m.get("slug")
        name = m.get("name")

        for candidate in [slug, name]:
            if not candidate:
                continue

            # Exact (normalized) keys stay authoritative: match_model
            # consults normal_lookup before compact_lookup, so a compact
            # collision below never shadows an exact match.
            normal_lookup[normalize(candidate)] = m

            key = compact(candidate)
            prev = compact_lookup.get(key)

            if prev is not None and prev is not m:
                print(
                    f"WARNING: AA compact-key collision on {key!r}: "
                    f"{candidate!r} collides with "
                    f"{prev.get('slug')!r}; keeping the first.",
                    file=sys.stderr,
                )
                continue

            compact_lookup[key] = m

    return normal_lookup, compact_lookup


def match_model(
    source_id: str,
    source_name: str,
    normal_lookup: dict[str, dict[str, Any]],
    compact_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:

    candidates = [
        source_id,
        source_name,
    ]

    # Explicit aliases.
    alias_list = ALIASES.get(source_id, [])
    candidates.extend(alias_list)

    for candidate in candidates:
        key = normalize(candidate)

        if key in normal_lookup:
            return normal_lookup[key], "exact"

    # Compact match.
    for candidate in candidates:
        key = compact(candidate)

        if key in compact_lookup:
            return compact_lookup[key], "compact"

    # Substring fallback.
    candidate_keys = [
        compact(x)
        for x in candidates
        if x
    ]

    for source_key in candidate_keys:
        if len(source_key) < SUBSTRING_MIN_COMPACT_LEN:
            continue

        possible = [
            (key, model)
            for key, model in compact_lookup.items()
            if source_key in key or key in source_key
        ]

        if len(possible) == 1:
            return possible[0][1], "substring"

    return None, "missing"


# ---------------------------------------------------------
# Extract AA benchmark values
# ---------------------------------------------------------

def get_aa_evaluations(
    aa_model: dict[str, Any] | None
) -> dict[str, float | None]:

    if not aa_model:
        return {
            "agentic": None,
            "coding": None,
            "intelligence": None,
        }

    evaluations = aa_model.get("evaluations", {})

    if not isinstance(evaluations, dict):
        evaluations = {}

    return {
        "agentic": safe_float(
            evaluations.get(
                "artificial_analysis_agentic_index"
            )
        ),
        "coding": safe_float(
            evaluations.get(
                "artificial_analysis_coding_index"
            )
        ),
        "intelligence": safe_float(
            evaluations.get(
                "artificial_analysis_intelligence_index"
            )
        ),
    }


# ---------------------------------------------------------
# Scoring
# ---------------------------------------------------------

def calculate_score(
    aa_values: dict[str, float | None],
    arena_score: float | None,
    swebench_score: float | None,
    normalization_ranges: dict[str, tuple[float, float]],
) -> tuple[float, float, dict[str, float | None]]:

    raw_values = {
        "agentic": aa_values.get("agentic"),
        "coding": aa_values.get("coding"),
        "intelligence": aa_values.get("intelligence"),
        "arena": arena_score,
        "swebench": swebench_score,
    }

    normalized = percentile_normalize(
        raw_values,
        normalization_ranges,
    )

    total_weight = 0.0
    weighted_score = 0.0

    for metric, weight in WEIGHTS.items():
        value = normalized.get(metric)

        # Missing benchmark is NOT treated as 0.
        if value is None:
            continue

        weighted_score += value * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, 0.0, normalized

    # Renormalize when benchmarks are missing.
    score = weighted_score / total_weight

    coverage = total_weight / sum(WEIGHTS.values())

    return score, coverage, normalized


def get_ranges(
    records: list[dict[str, Any]]
) -> dict[str, tuple[float, float]]:

    ranges: dict[str, tuple[float, float]] = {}

    for metric in WEIGHTS:

        vals = [
            r[metric]
            for r in records
            if r.get(metric) is not None
        ]

        if not vals:
            ranges[metric] = (0.0, 1.0)
            continue

        ranges[metric] = (
            min(vals),
            max(vals),
        )

    return ranges


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def _coverage_fraction(value: str) -> float:
    """argparse type= for --min-coverage: a float in [0, 1]."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"invalid min-coverage value: {value!r} "
            "(expected a number between 0 and 1)"
        ) from None

    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(
            f"min-coverage must be between 0 and 1, got {x}"
        )

    return x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank models for Hermes agentic work."
    )

    parser.add_argument(
        "--models",
        required=True,
        help="Path to models.json",
    )

    parser.add_argument(
        "--arena",
        default=None,
        help="Optional Arena CSV: model,elo",
    )

    parser.add_argument(
        "--swebench",
        default=None,
        help="Optional SWE-bench CSV: model,score",
    )

    parser.add_argument(
        "--out-prefix",
        default="hermes_ranking",
        help="Output prefix",
    )

    parser.add_argument(
        "--min-coverage",
        type=_coverage_fraction,
        default=0.0,
        help=(
            "Hide models whose score uses less than this "
            "fraction of available weights. Example: 0.50"
        ),
    )

    return parser


def load_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    Path,
    dict[str, float],
    dict[str, float],
]:
    """Load models.json and the optional arena/swebench CSVs."""
    models_path = Path(args.models)

    if not models_path.exists():
        raise FileNotFoundError(
            f"models file not found: {models_path}"
        )

    models_json = load_json(str(models_path))
    models = extract_models(models_json)

    print(f"Loaded {len(models)} models from {models_path}")

    arena = load_csv(args.arena)
    swebench = load_csv(args.swebench)

    if arena:
        print(f"Loaded {len(arena)} Arena scores")

    if swebench:
        print(f"Loaded {len(swebench)} SWE-bench scores")

    return models, models_path, arena, swebench


def rank_models(
    models: list[dict[str, Any]],
    arena: dict[str, float],
    swebench: dict[str, float],
    aa_models: list[dict[str, Any]],
    min_coverage: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Match dashboard models against AA data and benchmark scores,
    then compute normalized scores and ranks.
    """
    aa_normal, aa_compact = build_aa_lookup(aa_models)

    # First pass: collect raw values.
    raw_records: list[dict[str, Any]] = []

    for model in models:

        source_id = model["id"]
        source_name = model["name"]

        aa_match, match_type = match_model(
            source_id,
            source_name,
            aa_normal,
            aa_compact,
        )

        aa_values = get_aa_evaluations(aa_match)

        arena_score = arena.get(
            normalize(source_id)
        )

        if arena_score is None:
            arena_score = arena.get(
                normalize(source_name)
            )

        swebench_score = swebench.get(
            normalize(source_id)
        )

        if swebench_score is None:
            swebench_score = swebench.get(
                normalize(source_name)
            )

        raw_records.append(
            {
                "provider": model["provider"],
                "provider_id": model["provider_id"],
                "model_id": source_id,
                "model_name": source_name,

                "aa_slug": (
                    aa_match.get("slug")
                    if aa_match
                    else None
                ),

                "aa_name": (
                    aa_match.get("name")
                    if aa_match
                    else None
                ),

                "aa_match": match_type,

                "agentic": aa_values["agentic"],
                "coding": aa_values["coding"],
                "intelligence": aa_values["intelligence"],

                "arena": arena_score,
                "swebench": swebench_score,
            }
        )

    # Normalize all metrics relative to the current dataset.
    ranges = get_ranges(raw_records)

    # Calculate final score.
    results: list[dict[str, Any]] = []

    for r in raw_records:

        score, coverage, normalized = calculate_score(
            {
                "agentic": r["agentic"],
                "coding": r["coding"],
                "intelligence": r["intelligence"],
            },
            r["arena"],
            r["swebench"],
            ranges,
        )

        if coverage < min_coverage:
            continue

        results.append(
            {
                **r,

                "agentic_norm": (
                    round(normalized["agentic"], 2)
                    if normalized["agentic"] is not None
                    else None
                ),

                "coding_norm": (
                    round(normalized["coding"], 2)
                    if normalized["coding"] is not None
                    else None
                ),

                "intelligence_norm": (
                    round(normalized["intelligence"], 2)
                    if normalized["intelligence"] is not None
                    else None
                ),

                "arena_norm": (
                    round(normalized["arena"], 2)
                    if normalized["arena"] is not None
                    else None
                ),

                "swebench_norm": (
                    round(normalized["swebench"], 2)
                    if normalized["swebench"] is not None
                    else None
                ),

                "hermes_score": round(score, 2),
                "data_coverage": round(coverage * 100, 1),
            }
        )

    # Rank.
    results.sort(
        key=lambda x: x["hermes_score"],
        reverse=True,
    )

    for rank, row in enumerate(results, 1):
        row["rank"] = rank

    return results


def write_outputs(
    results: list[dict[str, Any]],
    prefix: str,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    """Write ranking.csv and ranking.json; return their paths."""
    csv_path = f"{prefix}.csv"

    if results:
        fieldnames = list(results[0].keys())

        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(results)

    json_path = f"{prefix}.json"

    output = {
        "metadata": metadata,
        "ranking": results,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return csv_path, json_path


def print_summary(
    results: list[dict[str, Any]],
    paths: tuple[str, str],
    version: float | None,
) -> None:
    """Print the console ranking summary."""
    csv_path, json_path = paths

    print()
    print("=" * 80)
    print("HERMES AGENTIC MODEL RANKING")
    print("=" * 80)

    print(
        f"{'RANK':<6}"
        f"{'MODEL':<42}"
        f"{'SCORE':>8}"
        f"{'COVERAGE':>11}"
    )

    print("-" * 80)

    for row in results[:30]:

        name = row["model_id"]

        if len(name) > 40:
            name = name[:37] + "..."

        print(
            f"{row['rank']:<6}"
            f"{name:<42}"
            f"{row['hermes_score']:>8.2f}"
            f"{row['data_coverage']:>10.1f}%"
        )

    print("=" * 80)
    print()
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")

    if version is not None:
        print(
            f"Artificial Analysis Intelligence Index: v{version}"
        )

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        models, models_path, arena, swebench = load_inputs(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # -----------------------------------------------------
    # Artificial Analysis
    # -----------------------------------------------------

    api_key = os.getenv(
        "ARTIFICIAL_ANALYSIS_API_KEY"
    )

    aa_models: list[dict[str, Any]] = []
    aa_version: float | None = None

    if api_key:
        print(
            "Downloading Artificial Analysis model data..."
        )

        aa_models, aa_version = fetch_aa_models(api_key)

        print(
            f"Loaded {len(aa_models)} Artificial Analysis models "
            f"(index v{aa_version})"
        )
    else:
        print(
            "WARNING: ARTIFICIAL_ANALYSIS_API_KEY not set. "
            "AA metrics will be unavailable."
        )

    results = rank_models(
        models,
        arena,
        swebench,
        aa_models,
        min_coverage=args.min_coverage,
    )

    metadata = {
        "weights": WEIGHTS,
        "artificial_analysis_index_version": aa_version,
        "models_input": str(models_path),
        "source_count": len(models),
        "ranked_count": len(results),
        "normalization": "min-max within current dataset",
    }

    paths = write_outputs(results, args.out_prefix, metadata)

    print_summary(results, paths, aa_version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
