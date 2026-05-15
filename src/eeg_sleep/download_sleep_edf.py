from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


DEFAULT_SLEEP_CASSETTE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
DEFAULT_DOWNLOAD_BASE_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/"
DEFAULT_DOMESTIC_ZIP_URL = (
    "https://cdn-xlab-data.openxlab.org.cn/objects/"
    "a6d978fd0f9fd4693c05222f667f4d532169f8dac65ea201168b6e11ae3a80a0"
    "?Expires=1776144277"
    "&OSSAccessKeyId=LTAI5tSqABbitQcgeNNd8dAE"
    "&Signature=bofxoV%2B2O9%2BrzxKEXqWhEuFn%2FRA%3D"
    "&response-content-disposition=attachment%3B%20filename%3D%22sleep-edf-database-expanded-1.0.0.zip%22"
    "&response-content-type=application%2Foctet-stream"
)


@dataclass(frozen=True)
class RemoteFile:
    """一条待下载的远程文件信息。"""

    record_id: str
    file_name: str
    url: str
    fallback_url: str | None = None


class _HrefParser(HTMLParser):
    """从目录页面里提取 href。"""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def _create_progress(total: int | None, description: str):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )


def _fetch_directory_listing(base_url: str) -> list[str]:
    """读取 PhysioNet 目录页面，拿到全部 EDF 文件名。"""

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
    """把目录文件整理成 PSG/Hypnogram 成对下载计划。"""

    grouped: dict[str, dict[str, str]] = {}

    for file_name in file_names:
        if not file_name.startswith(record_prefix) or len(file_name) < 6:
            continue

        record_id = file_name[:6]
        bucket = grouped.setdefault(record_id, {})
        if file_name.endswith("-PSG.edf"):
            bucket["psg"] = file_name
        elif file_name.endswith("-Hypnogram.edf"):
            bucket["hypnogram"] = file_name

    record_ids = sorted(
        record_id
        for record_id, bucket in grouped.items()
        if "psg" in bucket and "hypnogram" in bucket
    )
    if max_records > 0:
        record_ids = record_ids[:max_records]

    primary_base_url = download_base_url or base_url
    plan: list[RemoteFile] = []
    for record_id in record_ids:
        for key in ("psg", "hypnogram"):
            file_name = grouped[record_id][key]
            plan.append(
                RemoteFile(
                    record_id=record_id,
                    file_name=file_name,
                    url=urljoin(primary_base_url, file_name),
                    fallback_url=urljoin(base_url, file_name) if primary_base_url != base_url else None,
                )
            )

    return plan


def _download_to_path(url: str, destination: Path, description: str) -> Path:
    """下载单个文件到本地。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    progress = None

    try:
        request = Request(url, headers={"User-Agent": "eeg-sleep-repro/0.1"})
        with urlopen(request, timeout=60) as response, destination.open("wb") as file:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size is not None else None
            progress = _create_progress(total_bytes, description)

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))

        if progress is not None:
            progress.close()
        return destination
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        if progress is not None:
            progress.close()
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"download failed: {url}: {error}") from error


def _download_one_file(remote_file: RemoteFile, output_dir: Path, overwrite: bool = False) -> Path:
    """优先走主下载源，失败再走 fallback。"""

    destination = output_dir / remote_file.file_name
    if destination.exists() and not overwrite:
        return destination

    candidate_urls = [remote_file.url]
    if remote_file.fallback_url is not None:
        candidate_urls.append(remote_file.fallback_url)

    last_error: Exception | None = None
    for candidate_url in candidate_urls:
        try:
            return _download_to_path(candidate_url, destination, f"download {remote_file.file_name}")
        except RuntimeError as error:
            last_error = error
            if destination.exists():
                destination.unlink()

    raise RuntimeError(f"download failed: {remote_file.file_name}: {last_error}") from last_error


def _extract_domestic_archive(
    archive_path: Path,
    output_dir: Path,
    record_prefix: str,
    max_records: int,
    overwrite: bool,
) -> list[RemoteFile]:
    """从国内镜像 zip 中直接解出需要的 EDF。"""

    with ZipFile(archive_path) as archive:
        archive_members = {
            Path(member).name: member
            for member in archive.namelist()
            if member.lower().endswith(".edf")
        }
        plan = _build_download_plan(
            file_names=sorted(archive_members),
            base_url=DEFAULT_SLEEP_CASSETTE_URL,
            download_base_url=DEFAULT_DOWNLOAD_BASE_URL,
            record_prefix=record_prefix,
            max_records=max_records,
        )

        missing_files = [item.file_name for item in plan if item.file_name not in archive_members]
        if missing_files:
            raise RuntimeError(f"domestic archive is missing expected files: {missing_files[:4]}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for remote_file in plan:
            destination = output_dir / remote_file.file_name
            if destination.exists() and not overwrite:
                continue

            with archive.open(archive_members[remote_file.file_name]) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)

    return plan


def _try_domestic_mirror(
    output_dir: Path,
    record_prefix: str,
    max_records: int,
    overwrite: bool,
    domestic_mirror_url: str,
) -> list[RemoteFile]:
    """先下载国内镜像 zip，再直接解压。"""

    archive_path = output_dir.parent / "sleep-edf-database-expanded-1.0.0.zip"
    if overwrite or not archive_path.exists():
        _download_to_path(domestic_mirror_url, archive_path, "download domestic Sleep-EDF mirror")

    try:
        return _extract_domestic_archive(
            archive_path=archive_path,
            output_dir=output_dir,
            record_prefix=record_prefix,
            max_records=max_records,
            overwrite=overwrite,
        )
    except BadZipFile as error:
        # 国内镜像偶尔会返回错误页而不是 zip，这里删掉坏文件，让外层自动回退。
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(f"domestic archive is invalid: {archive_path}") from error


def download_sleep_cassette(
    output_dir: str | Path,
    base_url: str = DEFAULT_SLEEP_CASSETTE_URL,
    download_base_url: str = DEFAULT_DOWNLOAD_BASE_URL,
    record_prefix: str = "SC",
    max_records: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
    domestic_mirror_url: str = DEFAULT_DOMESTIC_ZIP_URL,
) -> list[RemoteFile]:
    """下载 Sleep-EDF Expanded 的 sleep-cassette 子集。"""

    output_path = Path(output_dir)
    plan = _build_download_plan(
        file_names=_fetch_directory_listing(base_url),
        base_url=base_url,
        download_base_url=download_base_url,
        record_prefix=record_prefix,
        max_records=max_records,
    )

    if dry_run:
        return plan

    # 优先走国内镜像，失败再回退到逐文件下载。
    if domestic_mirror_url:
        try:
            return _try_domestic_mirror(
                output_dir=output_path,
                record_prefix=record_prefix,
                max_records=max_records,
                overwrite=overwrite,
                domestic_mirror_url=domestic_mirror_url,
            )
        except RuntimeError as error:
            print(f"[download-sleep-edf] domestic mirror unavailable, fallback to per-file download: {error}")
            pass

    output_path.mkdir(parents=True, exist_ok=True)
    for remote_file in plan:
        _download_one_file(remote_file, output_path, overwrite=overwrite)

    return plan
