"""End-to-end tests for the rule-combination fault daily ledger.

The ledger is a strict read-only view over *stored* confirmed faults: it must
merge selected rules' intervals, split at UTC midnight and never trigger a
re-evaluation.
"""

import datetime
import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import DAY_SECONDS, to_epoch


def day(n: int) -> float:
    return n * DAY_SECONDS


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _insert_fault(rule_id, start, end):
    """Write a confirmed fault interval directly (the ledger's source data)."""
    db.execute(
        "INSERT INTO fault_intervals"
        " (rule_id, start_ts, end_ts, duration_required, confirmed_at, evidence_json)"
        " VALUES (?,?,?,?,?,?)",
        (rule_id, start, end, 0.0, start, "[]"),
    )


def _insert_rule(rule_id, name):
    db.execute(
        "INSERT INTO rules (id, name, logic, duration, created_at) VALUES (?,?,?,?,?)",
        (rule_id, name, "AND", 0, 0.0),
    )
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (?,?,?,?,?,?)",
        (rule_id, "d", "m", ">", 1, 0),
    )


def test_service_merges_selected_rules_and_splits_by_utc_day(scratch_db):
    _insert_rule(1, "R1")
    _insert_rule(2, "R2")
    # R1 covers day0 23:00 -> day1 01:00; R2 covers day1 00:30 -> 02:00.
    _insert_fault(1, day(0) + 23 * 3600, day(1) + 3600)
    _insert_fault(2, day(1) + 1800, day(1) + 2 * 3600)

    res = services.fault_daily_ledger([1, 2], day(0), day(2))
    days = {d["date"]: d for d in res["days"]}
    d0 = datetime.datetime.utcfromtimestamp(day(0)).strftime("%Y-%m-%d")
    d1 = datetime.datetime.utcfromtimestamp(day(1)).strftime("%Y-%m-%d")
    assert days[d0]["fault_seconds"] == 3600
    assert (days[d0]["new_segments"], days[d0]["continued_segments"]) == (1, 0)
    # R1's tail and R2 overlap on day1; merged first, so day1 carries a single
    # continued segment covering 00:00 -> 02:00 (deduped seconds).
    assert days[d1]["fault_seconds"] == 2 * 3600
    assert (days[d1]["new_segments"], days[d1]["continued_segments"]) == (0, 1)
    # selected rules echoed back for display
    assert [r["id"] for r in res["rules"]] == [1, 2]
    assert res["totals"]["fault_seconds"] == 3 * 3600


def test_only_selected_rules_are_combined(scratch_db):
    _insert_rule(1, "R1")
    _insert_rule(2, "R2")
    _insert_fault(1, day(0) + 3600, day(0) + 2 * 3600)
    _insert_fault(2, day(0) + 5 * 3600, day(0) + 6 * 3600)
    res = services.fault_daily_ledger([1], day(0), day(1))
    assert res["days"][0]["fault_seconds"] == 3600
    assert res["days"][0]["new_segments"] == 1


def test_duplicate_rule_ids_deduped(scratch_db):
    _insert_rule(1, "R1")
    _insert_fault(1, day(0) + 3600, day(0) + 2 * 3600)
    res = services.fault_daily_ledger([1, 1, 1], day(0), day(1))
    assert [r["id"] for r in res["rules"]] == [1]
    assert res["days"][0]["fault_seconds"] == 3600


def test_missing_rule_raises_keyerror(scratch_db):
    _insert_rule(1, "R1")
    with pytest.raises(KeyError):
        services.fault_daily_ledger([1, 99], day(0), day(1))


def test_ledger_uses_stored_faults_without_reevaluation(scratch_db, monkeypatch):
    _insert_rule(1, "R1")
    _insert_fault(1, day(0) + 3600, day(0) + 2 * 3600)

    def boom(*a, **k):
        raise AssertionError("the ledger must not re-evaluate rules")

    monkeypatch.setattr(services, "evaluate_rule", boom)
    res = services.fault_daily_ledger([1], day(0), day(1))
    assert res["days"][0]["fault_seconds"] == 3600
    # read-only: stored rows untouched
    assert len(services.query_faults(None, None)) == 1


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


def test_api_ledger_happy_path(client):
    _insert_rule(1, "R1")
    _insert_fault(1, day(0) + 23 * 3600, day(1) + 3600)
    r = client.post(
        "/api/fault-daily-ledger",
        json={"rule_ids": [1], "start": day(0), "end": day(2)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["start"] == day(0) and body["end"] == day(2)
    assert body["rules"][0]["name"] == "R1"
    assert len(body["days"]) == 2
    assert body["totals"]["new_segments"] == 1


def test_api_ledger_accepts_iso8601_utc(client):
    _insert_rule(1, "R1")
    _insert_fault(1, to_epoch("1970-01-01T00:00:00Z"), to_epoch("1970-01-01T01:00:00Z"))
    r = client.post(
        "/api/fault-daily-ledger",
        json={
            "rule_ids": [1],
            "start": "1970-01-01T00:00:00Z",
            "end": "1970-01-02T00:00:00Z",
        },
    )
    assert r.status_code == 200
    assert r.json()["days"][0]["fault_seconds"] == 3600


def test_api_ledger_missing_rule_404(client):
    r = client.post(
        "/api/fault-daily-ledger",
        json={"rule_ids": [42], "start": day(0), "end": day(1)},
    )
    assert r.status_code == 404


def test_api_ledger_invalid_range_400(client):
    _insert_rule(1, "R1")
    r = client.post(
        "/api/fault-daily-ledger",
        json={"rule_ids": [1], "start": day(1), "end": day(0)},
    )
    assert r.status_code == 400


def test_api_ledger_empty_rules_422(client):
    r = client.post(
        "/api/fault-daily-ledger",
        json={"rule_ids": [], "start": day(0), "end": day(1)},
    )
    assert r.status_code == 422


def test_api_ledger_too_many_days_400(client):
    _insert_rule(1, "R1")
    r = client.post(
        "/api/fault-daily-ledger",
        json={"rule_ids": [1], "start": day(0), "end": day(0) + 400 * DAY_SECONDS},
    )
    assert r.status_code == 400
