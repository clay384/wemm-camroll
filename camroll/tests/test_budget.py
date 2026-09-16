from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import sqlite3

import pytest

from camroll_agent.budget import BudgetExceeded, BudgetLedger


def _reserve_in_process(path):
    ledger = BudgetLedger(path, limit_cny=3)
    try:
        return ledger.reserve(1)
    except BudgetExceeded:
        return None


def test_atomic_reservation_never_overdraws(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")

    def reserve(_):
        try:
            return ledger.reserve(10, {"role": "caption"})
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(100)))
    assert len([item for item in results if item]) == 30
    assert ledger.summary()["remaining_cny"] == 0
    assert ledger.summary()["reserved_cny"] == 300


def test_independent_processes_share_one_allowance(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    ledger = BudgetLedger(path, limit_cny=3)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_reserve_in_process, [path] * 12))
    assert sum(result is not None for result in results) == 3
    assert ledger.summary()["remaining_cny"] == 0


def test_uncertain_retained_explicit_rejection_released_and_settlement_idempotent(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite", limit_cny=3)
    unknown = ledger.reserve(1)
    ledger.finish(unknown, None)
    released = ledger.reserve(1)
    ledger.finish(released, 0, rejected=True)
    settled = ledger.reserve(1)
    ledger.finish(settled, 0.25, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    ledger.finish(settled, 0.5)
    summary = ledger.summary()
    assert summary["estimated_spent_cny"] == 0.25
    assert summary["reserved_cny"] == 1
    assert summary["uncertain_requests"] == 1
    assert summary["remaining_cny"] == 1.75
    assert summary["usage"]["total_tokens"] == 150
    ledger.finish(unknown, 0.1)
    assert ledger.summary()["reserved_cny"] == 0


def test_ledger_reopen_does_not_reset_or_increase_budget(tmp_path):
    path = tmp_path / "budget.sqlite"
    first = BudgetLedger(path, limit_cny=2)
    first.reserve(1.9)
    second = BudgetLedger(path)
    assert second.summary()["limit_cny"] == 2
    with pytest.raises(BudgetExceeded):
        second.reserve(0.2)
    with pytest.raises(ValueError):
        BudgetLedger(path, limit_cny=301)


def test_metadata_drops_payload_and_credentials(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = BudgetLedger(path)
    call = ledger.reserve(1, {"prompt": "private text", "api_key": "secret-key", "role": "caption"})
    ledger.finish(call, 0.1, {"image": "private-image", "messages": ["private"], "total_tokens": 100})
    with sqlite3.connect(path) as conn:
        metadata = json.loads(conn.execute("SELECT metadata FROM budget_calls").fetchone()[0])
    assert metadata == {"role": "caption", "total_tokens": 100}


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_reject_invalid_amounts(tmp_path, amount):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    with pytest.raises(ValueError):
        ledger.reserve(amount)
