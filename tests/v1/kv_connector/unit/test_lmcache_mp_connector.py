# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LMCacheMPConnector.on_new_request.

The full connector requires a running LMCache backend. These tests instead
bind the methods under test to a MagicMock so we exercise the real logic
against a stubbed scheduler_adapter without spinning up LMCache.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("lmcache")

from tests.v1.core.utils import create_requests  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

pytestmark = pytest.mark.cpu_test


def _bind(connector: MagicMock, *method_names: str) -> None:
    for name in method_names:
        method = getattr(LMCacheMPConnector, name)
        setattr(connector, name, method.__get__(connector, LMCacheMPConnector))


def _make_connector(*, eager_prefetch: bool = True) -> MagicMock:
    connector = MagicMock(spec=LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = MagicMock()
    connector._eager_prefetch = eager_prefetch
    _bind(
        connector,
        "on_new_request",
        "_get_or_create_request_tracker",
    )
    return connector


def test_on_new_request_submits_lookup():
    """on_new_request must call maybe_submit_lookup_request immediately."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]

    connector.on_new_request(request)

    submit = connector.scheduler_adapter.maybe_submit_lookup_request
    submit.assert_called_once()
    call = submit.call_args
    assert call.args[0] == request.request_id
    assert call.kwargs["token_ids"] == list(request.all_token_ids)
    assert call.kwargs["cache_salt"] == (request.cache_salt or "")


def test_on_new_request_creates_tracker():
    """on_new_request must create a request tracker as a side-effect."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]

    assert request.request_id not in connector.request_trackers
    connector.on_new_request(request)
    assert request.request_id in connector.request_trackers


def test_on_new_request_skips_resumable():
    """Resumable (streaming-input) requests must not trigger an early lookup."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]
    request.resumable = True

    connector.on_new_request(request)

    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()


def test_on_new_request_disabled_by_default():
    """With eager_prefetch=False (default) on_new_request must be a no-op."""
    connector = _make_connector(eager_prefetch=False)
    request = create_requests(num_requests=1, num_tokens=10)[0]

    connector.on_new_request(request)

    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
    assert request.request_id not in connector.request_trackers


def test_on_new_request_delegates_dedup_to_adapter():
    """on_new_request forwards every call to the adapter; deduplication
    is handled by maybe_submit_lookup_request's _pending_lookups guard."""
    connector = _make_connector()
    request = create_requests(num_requests=1, num_tokens=10)[0]

    connector.on_new_request(request)
    connector.on_new_request(request)

    assert connector.scheduler_adapter.maybe_submit_lookup_request.call_count == 2
