import json

import pytest

from fxforecast.data.fetch import FetchError, fetch_series, load_series

GOOD_CSV = "observation_date,TEST\n2020-01-01,1.5\n2020-02-01,.\n2020-03-01,2.5\n"


class FakeHttp:
    """Returns queued responses; an Exception in the queue is raised instead."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_writes_csv_and_metadata(tmp_path):
    http = FakeHttp(GOOD_CSV)
    path = fetch_series("TEST", tmp_path, http_get=http)
    assert path.read_text(encoding="utf-8") == GOOD_CSV
    meta = json.loads((tmp_path / "TEST.meta.json").read_text(encoding="utf-8"))
    assert meta["n_obs"] == 2
    assert meta["first"] == "2020-01-01" and meta["last"] == "2020-03-01"
    assert len(meta["sha256"]) == 64


def test_cached_file_is_not_downloaded_again(tmp_path):
    http = FakeHttp(GOOD_CSV)
    fetch_series("TEST", tmp_path, http_get=http)
    fetch_series("TEST", tmp_path, http_get=http)
    assert http.calls == 1


def test_refresh_downloads_again(tmp_path):
    http = FakeHttp(GOOD_CSV, GOOD_CSV)
    fetch_series("TEST", tmp_path, http_get=http)
    fetch_series("TEST", tmp_path, http_get=http, refresh=True)
    assert http.calls == 2


def test_retries_then_succeeds(tmp_path):
    http = FakeHttp(TimeoutError("slow"), GOOD_CSV)
    fetch_series("TEST", tmp_path, http_get=http, backoff_seconds=0)
    assert http.calls == 2


def test_gives_up_after_retries_and_caches_nothing(tmp_path):
    http = FakeHttp(*[TimeoutError("down")] * 3)
    with pytest.raises(FetchError):
        fetch_series("TEST", tmp_path, http_get=http, retries=3, backoff_seconds=0)
    assert not (tmp_path / "TEST.csv").exists()


def test_html_error_page_is_rejected(tmp_path):
    http = FakeHttp("<html><body>Error</body></html>")
    with pytest.raises(FetchError):
        fetch_series("TEST", tmp_path, http_get=http, retries=1, backoff_seconds=0)
    assert not (tmp_path / "TEST.csv").exists()


def test_load_series_drops_missing_values(tmp_path):
    fetch_series("TEST", tmp_path, http_get=FakeHttp(GOOD_CSV))
    s = load_series("TEST", tmp_path)
    assert list(s.values) == [1.5, 2.5]
    assert s.name == "TEST"


def test_load_series_without_fetch_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_series("MISSING", tmp_path)
