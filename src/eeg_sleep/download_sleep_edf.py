from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_SLEEP_CASSETTE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
DEFAULT_DOWNLOAD_BASE_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/"


@dataclass(frozen=True)
class RemoteFile:
    """Remote EDF file metadata."""

    record_id: str
    file_name: str
    url: str
    fallback_url: str | None = None


class _HrefParser(HTMLParser):
    """Extract href values from the directory listing page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.hrefs.append(href)


def _create_progress(total: int | None, description: str):
    """Prefer tqdm when available, otherwise return None."""

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(total=total, unit="B", unit_scale=True, desc=description, leave=False, dynamic_ncols=True)


def _fetch_directory_listing(base_url: str) -> list[str]:
    """Read the PhysioNet listing page and extract EDF filenames."""

    request = Request(base_url, headers={"User-Agent": "eeg-sleep-repro/0.1"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")

    parser = _HrefParser()
    parser.feed(html)
    return sorted(
        href
        for href in parser.hrefs
        if href.lower().endswith(".edf") and not href.startswith("?")
    )


def _build_download_plan(
    file_names: list[str],
    base_url: str,
    download_base_url: str | None = None,
    record_prefix: str = "SC",
    max_records: int = 0,
) -> list[RemoteFile]:
    """Build paired PSG/Hypnogram downloads from the directory listing."""

    grouped: dict[str, dict[str, str]] = {}

    for file_name in file_names:
        if not file_name.startswith(record_prefix):
            continue
        if len(file_name) < 6:
            continue

        record_id = file_name[:6]
        bucket = grouped.setdefault(record_id, {})
        if file_name.endswith("-PSG.edf"):
            bucket["psg"] = file_name
        elif file_name.endswith("-Hypnogram.edf"):
            bucket["hypnogram"] = file_name

    selected_record_ids = sorted(
        record_id
        for record_id, bucket in grouped.items()
        if "psg" in bucket and "hypnogram" in bucket
    )
    if max_records > 0:
        selected_record_ids = selected_record_ids[:max_records]

    primary_base_url = download_base_url or base_url
    download_plan: list[RemoteFile] = []
    for record_id in selected_record_ids:
        bucket = grouped[record_id]
        for key in ("psg", "hypnogram"):
            file_name = bucket[key]
            download_plan.append(
                RemoteFile(
                    record_id=record_id,
                    file_name=file_name,
                    url=urljoin(primary_base_url, file_name),
                    fallback_url=urljoin(base_url, file_name) if primary_base_url != base_url else None,
                )
            )

    return download_plan


def _download_one_file(remote_file: RemoteFile, output_dir: Path, overwrite: bool = False) -> Path:
    """Download one EDF file with fallback to the original PhysioNet URL."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / remote_file.file_name
    if destination.exists() and not overwrite:
        return destination

    candidate_urls = [remote_file.url]
    if remote_file.fallback_url is not None:
        candidate_urls.append(remote_file.fallback_url)

    last_error: Exception | None = None
    for candidate_url in candidate_urls:
        progress = None
        try:
            request = Request(candidate_url, headers={"User-Agent": "eeg-sleep-repro/0.1"})
            with urlopen(request, timeout=30) as response, destination.open("wb") as file:
                total_size = response.headers.get("Content-Length")
                total_bytes = int(total_size) if total_size is not None else None
                progress = _create_progress(total_bytes, f"下载 {remote_file.file_name}")

                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))

            if progress is not None:
                progress.close()
            return destination
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if progress is not None:
                progress.close()
            if destination.exists():
                destination.unlink()

    raise RuntimeError(f"下载失败: {remote_file.file_name}: {last_error}") from last_error


def download_sleep_cassette(
    output_dir: str | Path,
    base_url: str = DEFAULT_SLEEP_CASSETTE_URL,
    download_base_url: str = DEFAULT_DOWNLOAD_BASE_URL,
    record_prefix: str = "SC",
    max_records: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[RemoteFile]:
    """Download the Sleep-EDF Expanded sleep-cassette subset."""

    output_path = Path(output_dir)
    file_names = _fetch_directory_listing(base_url)
    plan = _build_download_plan(
        file_names=file_names,
        base_url=base_url,
        download_base_url=download_base_url,
        record_prefix=record_prefix,
        max_records=max_records,
    )

    if dry_run:
        return plan

    for remote_file in plan:
        _download_one_file(remote_file, output_path, overwrite=overwrite)

    return plan
