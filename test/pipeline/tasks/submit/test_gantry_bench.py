"""Tests for the Gantry Bench submission task, against a real local HTTP server."""

import http.server
import json
import pathlib
import re
import tempfile
import threading
import urllib.error
import zipfile
from typing import ClassVar

import pytest

from src.pipeline.tasks.submit.gantry_bench import SubmitGantryBench


class _Bench(http.server.BaseHTTPRequestHandler):
    """Just enough of the bench API: create mints a cookie, upload requires it."""

    requests: ClassVar[list[dict]] = []
    fail_create: ClassVar[bool] = False

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        record = {
            "path": self.path,
            "cookie": self.headers.get("Cookie", ""),
            "content_type": self.headers.get("Content-Type", ""),
            "body": body,
        }
        _Bench.requests.append(record)

        if self.path == "/api/submissions":
            if _Bench.fail_create:
                self.send_response(500)
                self.end_headers()
                return
            payload = json.dumps({"id": "sub_test", "gates": []}).encode()
            self.send_response(200)
            self.send_header("Set-Cookie", "bench_visitor=jar; Path=/")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if re.fullmatch(r"/api/submissions/sub_test/dataset", self.path):
            payload = json.dumps(
                {"id": "sub_test", "gates": [{"key": "g0", "status": "queued"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, *args) -> None:  # noqa: ANN002 -- silence request logging
        pass


@pytest.fixture()
def bench() -> str:
    _Bench.requests = []
    _Bench.fail_create = False
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Bench)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture()
def export() -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp(prefix="lerobot-export-")) / "export"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text("{}")
    (root / "data" / "episode_000.parquet").write_bytes(b"not-really-parquet")
    return root


def _bare_task(source: str, api_url: str, **kwargs) -> SubmitGantryBench:  # noqa: ANN003
    task = SubmitGantryBench(source=source, api_url=api_url, **kwargs)
    # Pipeline.build sets these; unit tests set them directly.
    task._name = "submit"
    task._pipeline = "grasp_windows"
    task._site = "lab"
    task._asset = "arm_1"
    task._path = source
    task._log_id = "log"
    task._upload = False
    return task


def test_should_zip_a_directory_and_upload_it(bench: str, export: pathlib.Path) -> None:
    # GIVEN
    task = _bare_task(str(export), bench)

    # WHEN
    artifacts = task.execute(asof_seconds=1.0, lookback=None)

    # THEN the upload is multipart and carries a zip holding the export's files
    upload = next(r for r in _Bench.requests if r["path"].endswith("/dataset"))
    assert upload["content_type"].startswith("multipart/form-data")
    boundary = upload["content_type"].split("boundary=")[1]
    payload = upload["body"].split(f"--{boundary}".encode())[1].split(b"\r\n\r\n", 1)[1]
    with tempfile.NamedTemporaryFile(suffix=".zip") as f:
        f.write(payload.rsplit(b"\r\n", 1)[0])
        f.flush()
        names = set(zipfile.ZipFile(f.name).namelist())
    assert names == {"meta/info.json", "data/episode_000.parquet"}
    assert any(str(p).endswith(".receipt.json") for p in artifacts)


def test_should_present_the_minted_cookie_on_the_upload(bench: str, export: pathlib.Path) -> None:
    # GIVEN a bench that identifies visitors by the cookie create() mints
    task = _bare_task(str(export), bench)

    # WHEN
    task.execute(asof_seconds=1.0, lookback=None)

    # THEN the upload presented it, so it lands on the creator's submission
    upload = next(r for r in _Bench.requests if r["path"].endswith("/dataset"))
    assert "bench_visitor=jar" in upload["cookie"]


def test_should_substitute_the_name_template(bench: str, export: pathlib.Path) -> None:
    # GIVEN
    task = _bare_task(str(export), bench, name="{pipeline} on {asset}")

    # WHEN
    task.execute(asof_seconds=1.0, lookback=None)

    # THEN
    create = next(r for r in _Bench.requests if r["path"] == "/api/submissions")
    assert json.loads(create["body"])["name"] == "grasp_windows on arm_1"


def test_should_write_an_honest_receipt(bench: str, export: pathlib.Path) -> None:
    # GIVEN
    task = _bare_task(str(export), bench)

    # WHEN
    artifacts = task.execute(asof_seconds=1.0, lookback=None)

    # THEN the receipt names the submission and the gates' states at upload time
    receipt = json.loads(next(p for p in artifacts if str(p).endswith(".receipt.json")).read_text())
    assert receipt["submission_id"] == "sub_test"
    assert receipt["report_url"].endswith("/submissions/sub_test")
    assert receipt["gates"] == [{"key": "g0", "status": "queued"}]


def test_should_accept_an_existing_zip_without_rezipping(bench: str, export: pathlib.Path) -> None:
    # GIVEN an already-built archive
    archive = export.parent / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(export / "meta" / "info.json", "meta/info.json")
    task = _bare_task(str(archive), bench)

    # WHEN
    task.execute(asof_seconds=1.0, lookback=None)

    # THEN the uploaded bytes are the archive's own
    upload = next(r for r in _Bench.requests if r["path"].endswith("/dataset"))
    assert archive.read_bytes() in upload["body"]


def test_should_fail_the_task_when_create_fails(bench: str, export: pathlib.Path) -> None:
    # GIVEN a bench that refuses the submission
    _Bench.fail_create = True
    task = _bare_task(str(export), bench)

    # WHEN / THEN nothing is uploaded and the failure is loud
    with pytest.raises(urllib.error.HTTPError):
        task.execute(asof_seconds=1.0, lookback=None)
    assert not any(r["path"].endswith("/dataset") for r in _Bench.requests)


def test_should_refuse_a_file_that_is_not_a_zip(bench: str, export: pathlib.Path) -> None:
    # GIVEN
    not_zip = export / "meta" / "info.json"

    # WHEN / THEN
    task = _bare_task(str(not_zip), bench)
    with pytest.raises(ValueError, match="not a zip"):
        task.execute(asof_seconds=1.0, lookback=None)


def test_should_refuse_a_non_http_url(export: pathlib.Path) -> None:
    # WHEN / THEN
    with pytest.raises(ValueError, match="http"):
        SubmitGantryBench(source=str(export), api_url="ftp://bench")
