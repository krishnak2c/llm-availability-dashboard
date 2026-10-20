#!/usr/bin/env python3
# =============================================================================
# Free-LLM availability dashboard — external cron for probe.yml
# =============================================================================
# GitHub Actions' `schedule` trigger is best-effort: runs are dropped under
# load and top-of-hour crons are the worst case. This script registers the
# probe workflow on cron-job.org, which POSTs `workflow_dispatch` to the
# GitHub API every hour (default :17 UTC). probe.yml keeps a sparse 6-hour
# `schedule` backstop.
#
# Environment (read from the process env — never hardcode, never printed):
#   CRON_JOB_API_KEY           cron-job.org API key (Console > Settings)
#   GITHUB_DISPATCH_TOKEN      fine-grained PAT, repo Actions:Write only
#   GITHUB_REPO                default krishnak2c/llm-availability-dashboard
#   GITHUB_WORKFLOW            default probe.yml
#   CRON_SCHEDULE_MINUTE       default 17 (minute of the hour, 0-59)
#   CRON_TIMEZONE              default UTC
#
# Usage:
#   CRON_JOB_API_KEY=... GITHUB_DISPATCH_TOKEN=... \
#     python3 scripts/setup-cron-dispatch.py --create
#
#   --create         create the job, or PATCH it if the title already exists
#   --update         PATCH an existing job only (fails if not found)
#   --list           show the job's schedule + last run summary
#   --history        last 10 executions (planned vs actual, jitter, duration)
#   --test-dispatch  call GitHub's workflow_dispatch endpoint directly
#   --dry-run        print the cron-job.org payload without calling its API
#
# Notes:
#   * The cron-job.org free tier allows 100 API requests/day per account;
#     one job firing hourly is well inside that budget.
#   * cron-job.org cannot create GitHub tokens, and GitHub has no REST API
#     for fine-grained PATs — mint GITHUB_DISPATCH_TOKEN in the web UI with
#     `Actions: Write` on this repository and nothing else.
# =============================================================================

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CRON_API = "https://api.cron-job.org"
GITHUB_API = "https://api.github.com"

JOB_TITLE = "llm-dashboard-probe"
DEFAULT_REPO = "krishnak2c/llm-availability-dashboard"
DEFAULT_WORKFLOW = "probe.yml"
DEFAULT_MINUTE = "17"
DEFAULT_TIMEZONE = "UTC"

REQUEST_TIMEOUT = 30


class CronError(RuntimeError):
    """A cron-job.org or GitHub API call failed."""


def env(name, default=None):
    """Read an env var, treating blank as unset."""
    value = os.environ.get(name, "").strip()
    return value or default


def require_env(name):
    value = env(name)
    if not value:
        raise CronError(f"{name} is not set (see the module docstring)")
    return value


def redact(text, secret):
    """Never print a token, even inside an API error body."""
    if secret and secret in text:
        return text.replace(secret, "<redacted>")
    return text


def request_json(method, url, headers, payload=None, secret=None):
    """Minimal urllib JSON client; raises CronError with the response body."""
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise CronError(
            redact(f"{method} {url} -> HTTP {exc.code}: {body.strip()}", secret)
        ) from None
    except urllib.error.URLError as exc:
        raise CronError(f"{method} {url} -> {exc.reason}") from None
    if not body.strip():
        return status, {}
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, {"raw": body}


def cron_headers(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def build_job(repo, workflow, minute, timezone, token):
    """The exact cron-job.org job config (see their REST docs).

    `extendedData.body` must be a JSON *string*. -1 in a schedule field means
    "every"; hours=[-1] plus a single minute gives one run per hour.
    """
    return {
        "job": {
            "title": JOB_TITLE,
            "url": f"{GITHUB_API}/repos/{repo}/actions/workflows/{workflow}/dispatches",
            "requestMethod": 1,
            "extendedData": {
                "headers": {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"ref": "main"}, separators=(",", ":")),
            },
            "schedule": {
                "timezone": timezone,
                "minutes": [minute],
                "hours": [-1],
                "mdays": [-1],
                "months": [-1],
                "wdays": [-1],
            },
            "enabled": True,
            "saveResponses": False,
            "requestTimeout": 30,
        }
    }


def list_jobs(api_key):
    _, payload = request_json("GET", f"{CRON_API}/jobs", cron_headers(api_key))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise CronError(f"unexpected /jobs response: {payload!r}")
    return jobs


def find_job(api_key):
    for job in list_jobs(api_key):
        if job.get("title") == JOB_TITLE:
            return job
    return None


def resolve_job_id(api_key, explicit):
    if explicit:
        return explicit
    job = find_job(api_key)
    if not job:
        raise CronError(f"no cron-job.org job titled {JOB_TITLE!r} (run --create)")
    return job.get("jobId")


def cmd_create(api_key, body):
    existing = find_job(api_key)
    if existing:
        job_id = existing["jobId"]
        request_json("PATCH", f"{CRON_API}/jobs/{job_id}", cron_headers(api_key), body)
        action = "updated"
    else:
        _, payload = request_json("PUT", f"{CRON_API}/jobs", cron_headers(api_key), body)
        job_id = payload.get("jobId")
        if job_id is None:
            raise CronError(f"create returned no jobId: {payload!r}")
        action = "created"
    print(f"{action} cron-job.org job {job_id} ({JOB_TITLE})")
    print(f"jobId: {job_id}")
    print(f"url: https://console.cron-job.org/jobs/{job_id}")


def cmd_update(api_key, body):
    existing = find_job(api_key)
    if not existing:
        raise CronError(f"no job titled {JOB_TITLE!r} to update — run --create first")
    job_id = existing["jobId"]
    request_json("PATCH", f"{CRON_API}/jobs/{job_id}", cron_headers(api_key), body)
    print(f"updated cron-job.org job {job_id} ({JOB_TITLE})")
    print(f"url: https://console.cron-job.org/jobs/{job_id}")


def _schedule_line(schedule):
    if not isinstance(schedule, dict):
        return str(schedule)
    minutes = schedule.get("minutes", [])
    hours = schedule.get("hours", [])
    return (
        f"minutes={minutes} hours={hours} mdays={schedule.get('mdays')} "
        f"months={schedule.get('months')} wdays={schedule.get('wdays')} "
        f"tz={schedule.get('timezone')}"
    )


def cmd_list(api_key):
    job = find_job(api_key)
    if not job:
        raise CronError(f"no job titled {JOB_TITLE!r} (run --create)")
    print(f"jobId:        {job.get('jobId')}")
    print(f"title:        {job.get('title')}")
    print(f"enabled:      {job.get('enabled')}")
    print(f"schedule:     {_schedule_line(job.get('schedule'))}")
    print(f"lastStatus:   {job.get('lastStatus')}")
    print(f"nextExecution: {job.get('nextExecution')}")
    print(f"lastDuration: {job.get('lastDuration')}")


def cmd_history(api_key, job_id):
    _, payload = request_json(
        "GET", f"{CRON_API}/jobs/{job_id}/history", cron_headers(api_key)
    )
    history = payload.get("history")
    if not isinstance(history, list):
        raise CronError(f"unexpected history response: {payload!r}")
    if not history:
        print(f"job {job_id}: no executions recorded yet")
        return
    print(f"job {job_id}: last {min(len(history), 10)} executions")
    for entry in history[-10:]:
        print(
            "  date={date} planned={planned} jitter={jitter}ms "
            "duration={duration}ms status={status} httpStatus={http}".format(
                date=entry.get("date"),
                planned=entry.get("datePlanned"),
                jitter=entry.get("jitter"),
                duration=entry.get("duration"),
                status=entry.get("statusText", entry.get("status")),
                http=entry.get("httpStatus"),
            )
        )


def cmd_test_dispatch(repo, workflow, token):
    url = f"{GITHUB_API}/repos/{repo}/actions/workflows/{workflow}/dispatches"
    status, payload = request_json("POST", url, github_headers(token), {"ref": "main"}, token)
    body = "" if not payload else json.dumps(payload)
    print(f"POST {url}")
    print(f"HTTP {status}")
    print(f"body: {body or '(empty)'}")
    if status not in (200, 204):
        raise CronError(f"workflow_dispatch failed with HTTP {status}")
    print("dispatch accepted")


def cmd_dry_run(body, token):
    rendered = json.dumps(body, indent=2)
    if token:
        rendered = rendered.replace(token, "<redacted>")
    print("# cron-job.org payload (GITHUB_DISPATCH_TOKEN redacted)")
    print(rendered)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="setup-cron-dispatch.py",
        description=(
            "Register the probe workflow on cron-job.org so it is dispatched "
            "hourly via workflow_dispatch (GitHub Actions keeps a sparse backstop)."
        ),
        epilog=(
            "cron-job.org free tier: 100 API requests/day per account. "
            "Secrets are read from CRON_JOB_API_KEY and GITHUB_DISPATCH_TOKEN."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true", help="create the job (or PATCH if it exists)")
    group.add_argument("--update", action="store_true", help="PATCH an existing job")
    group.add_argument("--list", action="store_true", help="show the matching job")
    group.add_argument(
        "--test-dispatch",
        action="store_true",
        help="POST workflow_dispatch directly to GitHub",
    )
    group.add_argument(
        "--history", action="store_true", help="show the last 10 executions"
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cron-job.org payload without calling its API",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="job id for --history (defaults to the llm-dashboard-probe job)",
    )
    args = parser.parse_args(argv)

    repo = env("GITHUB_REPO", DEFAULT_REPO)
    workflow = env("GITHUB_WORKFLOW", DEFAULT_WORKFLOW)
    timezone = env("CRON_TIMEZONE", DEFAULT_TIMEZONE)

    minute_raw = env("CRON_SCHEDULE_MINUTE", DEFAULT_MINUTE)
    try:
        minute = int(minute_raw)
    except ValueError:
        raise CronError(f"CRON_SCHEDULE_MINUTE must be an integer, got {minute_raw!r}") from None
    if not 0 <= minute <= 59:
        raise CronError(f"CRON_SCHEDULE_MINUTE must be 0-59, got {minute}")

    token = require_env("GITHUB_DISPATCH_TOKEN")

    if args.test_dispatch:
        cmd_test_dispatch(repo, workflow, token)
        return 0

    body = build_job(repo, workflow, minute, timezone, token)

    if args.dry_run:
        cmd_dry_run(body, token)
        return 0

    api_key = require_env("CRON_JOB_API_KEY")

    if args.create:
        cmd_create(api_key, body)
    elif args.update:
        cmd_update(api_key, body)
    elif args.list:
        cmd_list(api_key)
    elif args.history:
        cmd_history(api_key, resolve_job_id(api_key, args.job_id))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CronError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
