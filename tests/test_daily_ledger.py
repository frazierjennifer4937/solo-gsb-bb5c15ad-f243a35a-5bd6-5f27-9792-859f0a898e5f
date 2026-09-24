"""Tests for the rule-combination fault daily ledger core.

The ledger reads already-confirmed half-open fault intervals, merges them as
one covering (overlap/touch dedup), clips to a UTC query window and splits at
UTC midnight.  These tests cover merging, day splitting, new-vs-continued
counting, midnight-touch semantics, partial first/last day denominators and
the coverage rate.
"""

import datetime

import pytest

from app.engine import DAY_SECONDS, split_daily_ledger


def day(n: int) -> float:
    # Epoch day n is a multiple of 86400 and hence exactly a UTC midnight.
    return n * DAY_SECONDS


def by_date(result) -> dict:
    return {d["date"]: d for d in result["days"]}


def test_single_day_full_window_segment_is_new():
    res = split_daily_ledger([(day(2), day(2) + 3600)], day(2), day(3))
    assert len(res["days"]) == 1
    d = res["days"][0]
    assert d["queried_seconds"] == 86400
    assert d["fault_seconds"] == 3600
    assert d["new_segments"] == 1
    assert d["continued_segments"] == 0
    assert d["coverage_rate"] == pytest.approx(3600 / 86400)
    assert d["segments"][0]["kind"] == "new"
    assert res["totals"]["fault_seconds"] == 3600
    assert res["totals"]["new_segments"] == 1
    assert res["totals"]["coverage_rate"] == pytest.approx(3600 / 86400)


def test_segment_crossing_midnight_new_then_continued():
    # [23:00, next day 02:00): new on day 0 (3600s), continued on day 1 (7200s)
    res = split_daily_ledger(
        [(day(0) + 23 * 3600, day(1) + 2 * 3600)], day(0), day(2)
    )
    d0, d1 = res["days"]
    assert d0["fault_seconds"] == 3600
    assert (d0["new_segments"], d0["continued_segments"]) == (1, 0)
    assert d1["fault_seconds"] == 7200
    # the crossing segment is only "new" on the day it starts; day 1 sees it
    # purely as a continuation
    assert (d1["new_segments"], d1["continued_segments"]) == (0, 1)
    assert d0["segments"][0]["kind"] == "new"
    assert d1["segments"][0]["kind"] == "continued"
    # the new segment is counted exactly once across the whole query
    assert res["totals"]["new_segments"] == 1
    assert res["totals"]["continued_segments"] == 1
    assert res["totals"]["fault_seconds"] == 3 * 3600


def test_overlapping_segments_from_rules_are_deduped():
    # two rules overlap on the same day: merged first, so the covering is one
    # segment — seconds and counts are both deduped
    res = split_daily_ledger(
        [(day(0) + 3600, day(0) + 5 * 3600), (day(0) + 2 * 3600, day(0) + 7 * 3600)],
        day(0),
        day(1),
    )
    d = res["days"][0]
    assert d["fault_seconds"] == 6 * 3600
    assert len(d["segments"]) == 1
    assert d["new_segments"] == 1
    assert d["continued_segments"] == 0


def test_overlap_with_crossing_segment_merges_counts():
    # Rule A's segment crosses midnight into day1 (continued there); rule B's
    # segment starts on day1 but overlaps A's tail.  The merge runs first, so
    # day1 sees a single continued merged segment — B's start is absorbed.
    a = (day(0) + 23 * 3600, day(1) + 3600)      # crosses into day1
    b = (day(1) + 1800, day(1) + 2 * 3600)        # starts on day1, overlaps a
    res = split_daily_ledger([a, b], day(0), day(2))
    d0, d1 = res["days"]
    assert (d0["new_segments"], d0["continued_segments"]) == (1, 0)
    assert d0["fault_seconds"] == 3600
    # merged covering on day1 is 00:00 -> 02:00, all from one continued segment
    assert (d1["new_segments"], d1["continued_segments"]) == (0, 1)
    assert d1["fault_seconds"] == 2 * 3600
    assert d1["segments"] == [
        {
            "start": day(0) + 23 * 3600,
            "end": day(1) + 2 * 3600,
            "day_start": day(1),
            "day_end": day(1) + 2 * 3600,
            "kind": "continued",
        }
    ]


def test_touching_segments_merge_into_one():
    # [a,b) and [b,c) touch at b: merged into one continuous segment — one
    # "new" count and deduped seconds
    res = split_daily_ledger(
        [(day(0) + 3600, day(0) + 2 * 3600), (day(0) + 2 * 3600, day(0) + 4 * 3600)],
        day(0),
        day(1),
    )
    d = res["days"][0]
    assert d["fault_seconds"] == 3 * 3600
    assert d["new_segments"] == 1
    assert d["continued_segments"] == 0
    assert res["segments"] == [{"start": day(0) + 3600, "end": day(0) + 4 * 3600}]


def test_segment_ending_exactly_at_midnight_does_not_enter_next_day():
    res = split_daily_ledger(
        [(day(0) + 22 * 3600, day(1))], day(0), day(2)
    )
    d0, d1 = res["days"]
    assert d0["fault_seconds"] == 2 * 3600
    assert (d0["new_segments"], d0["continued_segments"]) == (1, 0)
    assert d1["fault_seconds"] == 0
    assert (d1["new_segments"], d1["continued_segments"]) == (0, 0)
    assert d1["segments"] == []


def test_query_ends_exactly_at_midnight_excludes_new_day():
    # window [day0 23:00, day1 00:00) only covers day 0
    res = split_daily_ledger(
        [(day(0) + 23 * 3600, day(1))], day(0) + 23 * 3600, day(1)
    )
    assert [d["date"] for d in res["days"]] == [
        datetime.datetime.utcfromtimestamp(day(0)).strftime("%Y-%m-%d")
    ]
    assert res["days"][0]["fault_seconds"] == 3600


def test_partial_first_and_last_day_use_actual_queried_seconds():
    # window from day0 18:00 through day1 06:00: both boundary days carry a
    # 6h/21600s partial denominator, coverage uses those, not 86400.
    start = day(0) + 18 * 3600
    end = day(1) + 6 * 3600
    res = split_daily_ledger([(start, end)], start, end)
    d0, d1 = res["days"]
    assert d0["queried_seconds"] == 6 * 3600
    assert d1["queried_seconds"] == 6 * 3600
    assert d0["fault_seconds"] == 6 * 3600
    assert d1["fault_seconds"] == 6 * 3600
    assert d0["coverage_rate"] == pytest.approx(1.0)
    assert d1["coverage_rate"] == pytest.approx(1.0)
    assert (d0["new_segments"], d1["continued_segments"]) == (1, 1)
    assert res["totals"]["queried_seconds"] == 12 * 3600


def test_segment_starting_before_query_is_continued_on_first_day():
    # segment crosses the query start: the first queried day sees a
    # continuation, never a new segment
    start = day(0) + 12 * 3600
    res = split_daily_ledger(
        [(day(0) - 3600, day(0) + 13 * 3600)], start, day(1)
    )
    d = res["days"][0]
    assert d["fault_seconds"] == 3600
    assert (d["new_segments"], d["continued_segments"]) == (0, 1)
    assert d["segments"][0]["kind"] == "continued"
    assert res["totals"]["new_segments"] == 0


def test_query_start_exactly_at_midnight_segment_then_new():
    # segment starts at the query start, which is itself a midnight: new
    res = split_daily_ledger([(day(0), day(0) + 3600)], day(0), day(1))
    d = res["days"][0]
    assert (d["new_segments"], d["continued_segments"]) == (1, 0)


def test_empty_days_still_listed_with_zero_coverage():
    res = split_daily_ledger(
        [(day(0) + 3600, day(0) + 2 * 3600)], day(0), day(3)
    )
    assert len(res["days"]) == 3
    assert res["days"][1]["fault_seconds"] == 0
    assert res["days"][1]["new_segments"] == 0
    assert res["days"][1]["coverage_rate"] == 0.0
    assert res["days"][1]["queried_seconds"] == 86400


def test_two_segments_on_separate_days_are_each_new():
    res = split_daily_ledger(
        [(day(0) + 3600, day(0) + 2 * 3600), (day(1) + 5 * 3600, day(1) + 6 * 3600)],
        day(0),
        day(2),
    )
    d0, d1 = res["days"]
    assert (d0["new_segments"], d0["continued_segments"]) == (1, 0)
    assert (d1["new_segments"], d1["continued_segments"]) == (1, 0)
    assert res["totals"]["new_segments"] == 2
    assert res["totals"]["continued_segments"] == 0


def test_segment_fully_outside_window_is_clipped_away():
    res = split_daily_ledger(
        [(day(5), day(5) + 3600)], day(0), day(1)
    )
    assert res["segments"] == []
    assert res["days"][0]["fault_seconds"] == 0


def test_invalid_range_rejected():
    with pytest.raises(ValueError):
        split_daily_ledger([], day(0), day(0))
    with pytest.raises(ValueError):
        split_daily_ledger([], day(1), day(0))


def test_multi_day_segment_crossing_two_midnights():
    # day0 06:00 -> day2 06:00: new day0, continued day1 and day2
    res = split_daily_ledger(
        [(day(0) + 6 * 3600, day(2) + 6 * 3600)], day(0), day(3)
    )
    d0, d1, d2 = res["days"]
    assert (d0["new_segments"], d0["continued_segments"]) == (1, 0)
    assert (d1["new_segments"], d1["continued_segments"]) == (0, 1)
    assert (d2["new_segments"], d2["continued_segments"]) == (0, 1)
    assert d0["fault_seconds"] == 18 * 3600
    assert d1["fault_seconds"] == 86400
    assert d2["fault_seconds"] == 6 * 3600


def test_date_labels_are_utc_calendar_dates():
    res = split_daily_ledger(
        [(day(0), day(0) + 60)], day(0), day(1)
    )
    # epoch day 0 starts 1970-01-01T00:00:00Z
    assert res["days"][0]["date"] == "1970-01-01"
    assert by_date(res)["1970-01-01"]["day_start"] == 0.0
