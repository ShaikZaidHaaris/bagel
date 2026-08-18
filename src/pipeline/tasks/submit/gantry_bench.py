"""Submit a LeRobot export to a Gantry Bench instance for evaluation.

Bagel's pipelines can already cut detected events into a LeRobot dataset (the
``lerobot`` output). Gantry Bench (https://github.com/ShaikZaidHaaris/gantry)
answers the question that comes next: is that dataset actually worth training
on? Each submission runs a gauntlet of gates -- intake, a data report, a
learnability probe against a shuffled control, a robot test -- and the verdict
exports back out as an evidence bundle Bagel reads natively (the
``gantry.evidence`` data source).

This task is the delivery leg: zip the export if it is a directory, create a
submission, upload the archive, and write a receipt artifact recording the
submission id and report URL. The bench queues its free intake and data-report
gates on upload by itself; the deeper gates are a cost decision the bench
deliberately leaves to a person, and this task does not press that button.

Identity note: the bench scopes submissions to the visiting address (no
accounts). The receipt's report URL therefore works from the network that
submitted, or wherever the submission was published from.
"""

import json
import logging
import pathlib
import urllib.request
import uuid
import zipfile

from src.di import module
from src.pipeline import base


def _multipart(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    """One-file multipart/form-data body, stdlib-only. Returns (body, content_type)."""
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: application/zip\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


class SubmitGantryBench(base.ArtifactMixin, base.Task):
    """Submit a LeRobot export to a Gantry Bench instance for evaluation.

    The ``source`` may be a directory (a LeRobot export, zipped by the task) or
    an already-built ``.zip``. The artifact is a small JSON receipt: submission
    id, report URL, and the state of every gate at submission time -- enough
    for the next tool (or person) to find the verdict without this task having
    waited around for it. Evaluation takes minutes to hours; a pipeline task
    should not.
    """

    def __init__(
        self,
        source: str,
        api_url: str,
        name: str = "{pipeline} {asset}",
        benchmark: str = "pick_dual_bottles",
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize the task.

        Args:
            source (str): A LeRobot export directory (zipped recursively) or an
                existing ``.zip`` archive.
            api_url (str): Base URL of the bench, e.g. ``https://gantry.gurasees.com``.
            name (str, optional): Submission name template; ``{pipeline}``,
                ``{site}``, ``{asset}`` are substituted at execution time.
            benchmark (str, optional): Benchmark key on the bench instance.
            timeout_seconds (int, optional): HTTP timeout for the upload call.

        """
        if not api_url.startswith(("https://", "http://")):
            raise ValueError("api_url must be http(s).")
        self._source = source
        self._api_url = api_url.rstrip("/")
        # Not stored as ``_name``: that attribute belongs to the Operator
        # contract and is set by Pipeline.build to the task's own name.
        self._name_template = name
        self._benchmark = benchmark
        self._timeout_seconds = timeout_seconds
        # One opener with a cookie jar for the whole conversation: the bench
        # identifies a visitor by a minted cookie (falling back to address), so
        # the create and the upload must present the same one or the upload
        # lands on a submission its uploader cannot see.
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(),
        )

    def setup(self, path: str, **kwargs) -> None:  # noqa: ANN003
        """Nothing to set up in this task."""

    def _call(self, route: str, data: bytes | None = None, content_type: str = "") -> dict:
        request = urllib.request.Request(self._api_url + route, data=data)  # noqa: S310 -- scheme enforced in __init__
        if content_type:
            request.add_header("Content-Type", content_type)
        with self._opener.open(request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode())

    def _archive(self, asof_seconds: float) -> pathlib.Path:
        source = pathlib.Path(self._source)
        if source.is_file():
            if not zipfile.is_zipfile(source):
                raise ValueError(f"{source} is not a zip archive")
            return source
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = self.artifact_path(asof_seconds, ".zip")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(p for p in source.rglob("*") if p.is_file()):
                zf.write(file, file.relative_to(source))
        return target

    def execute(self, asof_seconds: float, lookback: base.Lookback | None) -> list[pathlib.Path]:
        """Zip if needed, create the submission, upload, and write the receipt."""
        archive = self._archive(asof_seconds)

        name = self._name_template.format_map(
            {"pipeline": self.pipeline, "site": self.site, "asset": self.asset}
        )
        created = self._call(
            "/api/submissions",
            data=json.dumps({"name": name, "benchmark": self._benchmark}).encode(),
            content_type="application/json",
        )
        submission_id = created["id"]

        body, content_type = _multipart("file", archive.name, archive.read_bytes())
        uploaded = self._call(
            f"/api/submissions/{submission_id}/dataset", data=body, content_type=content_type
        )

        receipt_path = self.artifact_path(asof_seconds, ".receipt.json")
        receipt = {
            "submission_id": submission_id,
            "report_url": f"{self._api_url}/submissions/{submission_id}",
            "benchmark": self._benchmark,
            "uploaded_bytes": archive.stat().st_size,
            "gates": [
                {"key": g.get("key"), "status": g.get("status")} for g in uploaded.get("gates", [])
            ],
        }
        receipt_path.write_text(json.dumps(receipt, indent=1))
        logging.info("Submitted %s to %s as %s", archive.name, self._api_url, submission_id)
        return [receipt_path]


def register() -> None:
    """Register module for dependency injection."""
    module.global_registry[__name__] = SubmitGantryBench
