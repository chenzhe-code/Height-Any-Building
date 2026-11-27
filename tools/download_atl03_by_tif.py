#!/usr/bin/env python3
"""Query and download ATL03 granules intersecting one or more GeoTIFF extents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import rasterio
import rasterio.warp
import requests
from requests import Response
from requests.auth import AuthBase, HTTPBasicAuth
from tqdm import tqdm


CMR_COLLECTIONS_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
HARMONY_COVERAGES_URL_TEMPLATE = (
    "https://harmony.earthdata.nasa.gov/{collection_id}/ogc-api-coverages/1.0.0/"
    "collections/all/coverage/rangeset"
)
HARMONY_JOB_URL_TEMPLATE = "https://harmony.earthdata.nasa.gov/jobs/{job_id}"
DEFAULT_CLIENT_ID = "HeightAnyBuilding-ATL03-Downloader"
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


class AuthenticationError(RuntimeError):
    """Raised when Earthdata authentication is missing or rejected."""


class EarthdataLoginAuth(AuthBase):
    """Apply Basic Auth only when requests are redirected to Earthdata Login."""

    def __init__(self, username: str, password: str):
        self._auth = HTTPBasicAuth(username, password)

    def __call__(self, request):
        url = urlparse(request.url)
        if url.scheme == "https" and url.hostname == "urs.earthdata.nasa.gov":
            return self._auth(request)
        return request


def parse_utc_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse YYYY-MM-DD or ISO datetime and return a UTC-aware datetime."""
    value = value.strip()
    if not value:
        raise ValueError("Date value cannot be empty.")

    if len(value) == 10:
        parsed_date = date.fromisoformat(value)
        parsed_time = datetime_time.max if end_of_day else datetime_time.min
        return datetime.combine(parsed_date, parsed_time, tzinfo=timezone.utc)

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_cmr_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_temporal(start_date: str, end_date: str) -> str:
    start = parse_utc_datetime(start_date)
    end = parse_utc_datetime(end_date, end_of_day=len(end_date.strip()) == 10)
    if start > end:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    return f"{format_cmr_datetime(start)},{format_cmr_datetime(end)}"


def build_datetime_interval(start_date: str, end_date: str) -> tuple[str, str]:
    start = parse_utc_datetime(start_date)
    end = parse_utc_datetime(end_date, end_of_day=len(end_date.strip()) == 10)
    if start > end:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    return format_cmr_datetime(start), format_cmr_datetime(end)


def format_harmony_datetime(value: str, *, end_of_day: bool = False) -> str:
    parsed = parse_utc_datetime(value, end_of_day=end_of_day)
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def geotiff_wgs84_bbox(tif_path: Path) -> tuple[float, float, float, float]:
    """Return GeoTIFF bounds as (west, south, east, north) in EPSG:4326."""
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS; cannot build CMR bounding_box.")

        bounds = src.bounds
        if src.crs.to_epsg() == 4326:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        else:
            west, south, east, north = rasterio.warp.transform_bounds(
                src.crs,
                "EPSG:4326",
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )

    west, east = min(west, east), max(west, east)
    south, north = min(south, north), max(south, north)
    return west, south, east, north


def geotiffs_wgs84_bbox(tif_paths: list[Path]) -> tuple[float, float, float, float]:
    """Return the WGS84 bounding box covering every reference GeoTIFF."""
    if not tif_paths:
        raise ValueError("At least one reference GeoTIFF is required.")
    bounds = [geotiff_wgs84_bbox(path) for path in tif_paths]
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def natural_path_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_tif_paths(args: argparse.Namespace) -> list[Path]:
    if args.tif_path:
        paths = [Path(value) for value in args.tif_path]
    else:
        tif_dir = Path(args.tif_dir)
        if not tif_dir.is_dir():
            raise NotADirectoryError(f"GeoTIFF directory does not exist: {tif_dir}")
        paths = sorted(
            (
                path
                for path in tif_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            ),
            key=natural_path_key,
        )

    if not paths:
        raise FileNotFoundError("No reference GeoTIFF files were found.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GeoTIFF does not exist: {missing[0]}")
    return paths


def format_bbox(bbox: Iterable[float]) -> str:
    return ",".join(f"{value:.10f}".rstrip("0").rstrip(".") for value in bbox)


def make_session(user_agent: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent or DEFAULT_CLIENT_ID,
            "Client-Id": user_agent or DEFAULT_CLIENT_ID,
        }
    )

    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if username or password:
        if not username or not password:
            raise ValueError(
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set together."
            )
        session.auth = EarthdataLoginAuth(username, password)

    return session


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict,
    timeout: float,
) -> dict:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def find_collection(
    session: requests.Session,
    short_name: str,
    version: str,
    timeout: float,
) -> dict:
    params = {
        "short_name": short_name,
        "version": version,
        "page_size": 10,
    }
    data = get_json(session, CMR_COLLECTIONS_URL, params=params, timeout=timeout)
    entries = data.get("feed", {}).get("entry", [])
    if not entries:
        raise RuntimeError(f"No CMR collection found for {short_name} version {version}.")

    for entry in entries:
        if entry.get("short_name") == short_name and str(entry.get("version_id")) == str(version):
            return entry
    return entries[0]


def search_granules(
    session: requests.Session,
    collection_concept_id: str,
    bbox: tuple[float, float, float, float],
    temporal: str,
    page_size: int,
    timeout: float,
) -> list[dict]:
    granules: list[dict] = []
    page_num = 1

    while True:
        params = {
            "collection_concept_id": collection_concept_id,
            "bounding_box": format_bbox(bbox),
            "temporal": temporal,
            "page_size": page_size,
            "page_num": page_num,
            "sort_key": "start_date",
        }
        data = get_json(session, CMR_GRANULES_URL, params=params, timeout=timeout)
        entries = data.get("feed", {}).get("entry", [])
        granules.extend(entries)
        if len(entries) < page_size:
            break
        page_num += 1

    return granules


def is_hdf5_https_link(link: dict) -> bool:
    href = str(link.get("href") or "")
    if not href.lower().startswith("https://"):
        return False

    parsed = urlparse(href)
    if not parsed.path.lower().endswith(".h5"):
        return False

    rel = str(link.get("rel") or "").lower()
    title = str(link.get("title") or "").lower()
    href_lower = href.lower()
    excluded_terms = (
        "browse",
        "metadata",
        "opendap",
        "documentation",
        "s3credentials",
        ".xml",
        ".jpg",
        ".jpeg",
        ".png",
    )
    haystack = " ".join((rel, title, href_lower))
    return not any(term in haystack for term in excluded_terms)


def extract_hdf5_urls(granules: list[dict]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for granule in granules:
        for link in granule.get("links", []):
            if is_hdf5_https_link(link):
                href = str(link["href"])
                if href not in seen:
                    seen.add(href)
                    urls.append(href)
    return urls


def write_outputs(output_dir: Path, urls: list[str], granules: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atl03_urls.txt").write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    payload = {
        "summary": summary,
        "granules": granules,
    }
    (output_dir / "atl03_granules.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def harmony_request_signature(
    *,
    tif_paths: list[Path],
    bbox: tuple[float, float, float, float],
    temporal: str,
    collection_concept_id: str,
    granules: list[dict],
) -> dict:
    return {
        "tif_paths": [str(path.resolve()) for path in tif_paths],
        "bbox_wgs84": [round(value, 10) for value in bbox],
        "temporal": temporal,
        "collection_concept_id": collection_concept_id,
        "granule_ids": [str(item.get("id") or item.get("title")) for item in granules],
    }


def reusable_harmony_downloads(output_dir: Path, signature: dict):
    marker = output_dir / "atl03_download_complete.json"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("request") != signature or payload.get("status") != "complete":
        return None

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return None
    resolved = []
    for item in files:
        if not isinstance(item, dict) or not item.get("name"):
            return None
        path = output_dir / Path(str(item["name"])).name
        size = path.stat().st_size if path.is_file() else 0
        if size <= 0 or size != item.get("size_bytes"):
            return None
        resolved.append(path)
    return resolved


def write_harmony_result(
    output_dir: Path,
    signature: dict,
    downloaded: list[Path],
    job: dict,
    *,
    partial: bool,
) -> None:
    status = "partial" if partial else "complete"
    marker = output_dir / f"atl03_download_{status}.json"
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    payload = {
        "request": signature,
        "status": status,
        "harmony_status": harmony_status_value(job),
        "errors": job.get("errors") or [],
        "files": [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in downloaded
        ],
    }
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    previous_status = "complete" if partial else "partial"
    (output_dir / f"atl03_download_{previous_status}.json").unlink(missing_ok=True)


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Cannot infer filename from URL: {url}")
    return name


def earthdata_auth_for_url(session: requests.Session, url: str):
    hostname = urlparse(url).hostname or ""
    if hostname.endswith("urs.earthdata.nasa.gov"):
        return getattr(session, "auth", None)
    return None


def is_earthdata_login_response(response: Response) -> bool:
    final_host = urlparse(response.url).hostname or ""
    content_type = response.headers.get("content-type", "").lower()
    if final_host.endswith("urs.earthdata.nasa.gov"):
        return True
    if "text/html" not in content_type:
        return False
    return "earthdata login" in response.text[:2000].lower()


def earthdata_auth_message(context: str) -> str:
    return (
        f"{context} reached the Earthdata Login page instead of data. "
        "Configure ~/.netrc or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD. "
        "For ~/.netrc, use: machine urs.earthdata.nasa.gov login <username> password <password> "
        "and run chmod 600 ~/.netrc."
    )


def earthdata_unauthorized_message(context: str) -> str:
    return (
        f"{context} was not authorized by Earthdata Login. "
        "If you entered credentials in the terminal prompt, check that the username/password "
        "are correct and that the account has accepted the required Earthdata/NSIDC access terms."
    )


def open_download_response(
    session: requests.Session,
    url: str,
    timeout: float,
    headers: dict | None = None,
) -> Response:
    return open_earthdata_response(session, "GET", url, timeout, headers=headers, stream=True)


def open_earthdata_response(
    session: requests.Session,
    method: str,
    url: str,
    timeout: float,
    *,
    headers: dict | None = None,
    params: dict | list[tuple[str, str]] | None = None,
    stream: bool = False,
    max_redirects: int = 10,
    allow_redirects: bool = False,
) -> Response:
    if allow_redirects:
        return session.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            stream=stream,
            allow_redirects=True,
        )

    current_url = url
    current_method = method
    for _ in range(max_redirects + 1):
        response = session.request(
            current_method,
            current_url,
            headers=headers,
            params=params,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
            auth=earthdata_auth_for_url(session, current_url),
        )
        if response.status_code not in REDIRECT_STATUS_CODES:
            return response

        location = response.headers.get("Location")
        session.cookies.update(response.cookies)
        response.close()
        if not location:
            raise RuntimeError(f"Redirect response from {current_url} did not include Location.")
        current_url = urljoin(current_url, location)
        current_method = "GET" if response.status_code == 303 else current_method
        params = None

    raise RuntimeError(f"Too many redirects while requesting {url}.")


def parse_json_response(response: Response, *, context: str) -> dict:
    content_type = response.headers.get("content-type", "").lower()
    if response.status_code in (401, 403):
        raise AuthenticationError(earthdata_unauthorized_message(context))
    if is_earthdata_login_response(response):
        raise AuthenticationError(earthdata_auth_message(context))
    if response.status_code >= 400:
        text = response.text[:1000]
        raise RuntimeError(
            f"{context} failed with HTTP {response.status_code}. "
            f"Content-Type={content_type}. Body={text}"
        )
    if "json" not in content_type:
        text = response.text[:500]
        raise RuntimeError(f"{context} did not return JSON. Content-Type={content_type}. Body={text}")
    return response.json()


def validate_download_response(response: Response) -> None:
    if response.status_code in (401, 403):
        raise AuthenticationError(earthdata_unauthorized_message("Earthdata download"))
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if is_earthdata_login_response(response) or "text/html" in content_type:
        raise AuthenticationError(earthdata_auth_message("Earthdata download"))


def download_one(
    session: requests.Session,
    url: str,
    output_dir: Path,
    *,
    overwrite: bool,
    timeout: float,
    max_retries: int,
) -> Path:
    output_path = output_dir / filename_from_url(url)
    if output_path.exists() and not overwrite:
        print(f"[SKIP] {output_path} already exists")
        return output_path

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists() and overwrite:
        part_path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DOWNLOAD] {url}")
            response = open_download_response(session, url, timeout)
            with response:
                validate_download_response(response)
                total = int(response.headers.get("content-length") or 0)
                with open(part_path, "wb") as file_obj, tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_obj.write(chunk)
                            progress.update(len(chunk))
            part_path.replace(output_path)
            return output_path
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()
            if attempt < max_retries:
                sleep_seconds = min(2 ** attempt, 30)
                print(f"[RETRY] attempt {attempt}/{max_retries} failed: {exc}")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_urls(
    urls: list[str],
    output_dir: Path,
    *,
    overwrite: bool,
    timeout: float,
    max_retries: int,
    user_agent: str | None,
) -> list[Path]:
    session = make_session(user_agent)
    downloaded: list[Path] = []
    for url in urls:
        downloaded.append(
            download_one(
                session,
                url,
                output_dir,
                overwrite=overwrite,
                timeout=timeout,
                max_retries=max_retries,
            )
        )
    return downloaded


def build_harmony_subset_params(
    bbox: tuple[float, float, float, float],
    granules: list[dict],
    start_datetime: str,
    end_datetime: str,
    ignore_errors: bool,
) -> list[tuple[str, str]]:
    west, south, east, north = bbox
    params: list[tuple[str, str]] = [
        ("subset", f"lat({south:.8f}:{north:.8f})"),
        ("subset", f"lon({west:.8f}:{east:.8f})"),
        ("subset", f'time("{start_datetime}":"{end_datetime}")'),
        ("maxResults", str(max(len(granules), 1))),
        ("skipPreview", "true"),
        ("ignoreErrors", "true" if ignore_errors else "false"),
        ("forceAsync", "true"),
        ("label", "heightanybuilding-atl03-subset"),
    ]
    return params


def submit_harmony_subset_job(
    session: requests.Session,
    collection_concept_id: str,
    bbox: tuple[float, float, float, float],
    granules: list[dict],
    start_datetime: str,
    end_datetime: str,
    timeout: float,
    ignore_errors: bool,
) -> dict:
    url = HARMONY_COVERAGES_URL_TEMPLATE.format(collection_id=collection_concept_id)
    params = build_harmony_subset_params(
        bbox, granules, start_datetime, end_datetime, ignore_errors
    )
    response = open_earthdata_response(session, "GET", url, timeout, params=params)
    with response:
        return parse_json_response(response, context="Harmony subset request")


def get_harmony_job(session: requests.Session, job_id: str, timeout: float) -> dict:
    url = HARMONY_JOB_URL_TEMPLATE.format(job_id=job_id)
    response = open_earthdata_response(session, "GET", url, timeout)
    with response:
        return parse_json_response(response, context=f"Harmony job {job_id}")


def harmony_status_value(job: dict) -> str:
    return str(job.get("status") or "").strip().replace(" ", "_").lower()


HARMONY_COMPLETE = {"successful", "success", "complete", "completed"}
HARMONY_PARTIAL = {"successful_with_errors", "complete_with_errors", "completed_with_errors"}


def wait_for_harmony_job(
    session: requests.Session,
    job_id: str,
    *,
    timeout: float,
    poll_interval: float,
    request_timeout: float,
    max_retries: int,
) -> dict:
    started = time.time()
    while True:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                job = get_harmony_job(session, job_id, request_timeout)
                break
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                retry_delay = min(2 ** attempt, 30)
                print(
                    f"[RETRY] Harmony status request {attempt}/{max_retries} failed: "
                    f"{type(exc).__name__}; retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)
                if time.time() - started > timeout:
                    raise TimeoutError(
                        f"Harmony job {job_id} did not finish within {timeout} seconds."
                    ) from last_error
        status = harmony_status_value(job)
        progress = job.get("progress")
        message = job.get("message") or job.get("statusMessage") or ""
        print(f"[HARMONY] job={job_id} status={status} progress={progress} {message}")

        if status in HARMONY_COMPLETE | HARMONY_PARTIAL:
            return job
        if status in {"failed", "canceled", "cancelled"}:
            errors = job.get("errors") or []
            raise RuntimeError(
                f"Harmony job {job_id} failed at progress={progress}; "
                f"errors={json.dumps(errors, ensure_ascii=False)}"
            )
        if time.time() - started > timeout:
            raise TimeoutError(f"Harmony job {job_id} did not finish within {timeout} seconds.")

        time.sleep(poll_interval)


def extract_harmony_result_urls(job: dict) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add_url(value: str) -> None:
        if value.lower().startswith("https://") and value not in seen:
            seen.add(value)
            urls.append(value)

    for key in ("links", "results"):
        for item in job.get(key, []) or []:
            if isinstance(item, str):
                add_url(item)
            elif isinstance(item, dict):
                href = item.get("href") or item.get("url")
                if href:
                    add_url(str(href))

    return [url for url in urls if urlparse(url).path.lower().endswith(".h5")]


def write_harmony_outputs(output_dir: Path, urls: list[str], job: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atl03_subset_urls.txt").write_text(
        "\n".join(urls) + ("\n" if urls else ""),
        encoding="utf-8",
    )
    (output_dir / "atl03_harmony_job.json").write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def download_harmony_subset(
    *,
    collection_concept_id: str,
    bbox: tuple[float, float, float, float],
    granules: list[dict],
    start_datetime: str,
    end_datetime: str,
    output_dir: Path,
    overwrite: bool,
    timeout: float,
    max_retries: int,
    harmony_timeout: float,
    harmony_poll_interval: float,
    user_agent: str | None,
    harmony_job_id: str | None = None,
    harmony_ignore_errors: bool = False,
) -> tuple[list[Path], dict, bool]:
    session = make_session(user_agent)
    if harmony_job_id:
        job_id = harmony_job_id
        print(f"[RESUME] Harmony job={job_id}")
    else:
        job = submit_harmony_subset_job(
            session,
            collection_concept_id,
            bbox,
            granules,
            start_datetime,
            end_datetime,
            timeout,
            harmony_ignore_errors,
        )
        job_id = job.get("jobID") or job.get("jobId") or job.get("id")
        if not job_id:
            raise RuntimeError(
                f"Harmony response did not include a job id: {json.dumps(job, ensure_ascii=False)}"
            )

    final_job = wait_for_harmony_job(
        session,
        str(job_id),
        timeout=harmony_timeout,
        poll_interval=harmony_poll_interval,
        request_timeout=timeout,
        max_retries=max_retries,
    )
    urls = extract_harmony_result_urls(final_job)
    write_harmony_outputs(output_dir, urls, final_job)
    errors = final_job.get("errors") or []
    partial = harmony_status_value(final_job) in HARMONY_PARTIAL or bool(errors)
    if partial and (not harmony_ignore_errors or not urls):
        return [], final_job, partial
    if not urls:
        raise RuntimeError("Harmony job succeeded but returned no .h5 result URLs.")
    if partial:
        print(
            f"[WARN] Partial Harmony result: downloading {len(urls)} available subset(s).",
            file=sys.stderr,
        )
    downloaded = download_urls(
        urls,
        output_dir,
        overwrite=overwrite,
        timeout=timeout,
        max_retries=max_retries,
        user_agent=user_agent,
    )
    return downloaded, final_job, partial


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query NASA CMR and download ATL03 granules intersecting one or more GeoTIFF extents."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tif-path",
        action="append",
        help="Reference GeoTIFF path; repeat to use their combined extent.",
    )
    source.add_argument(
        "--tif-dir",
        help="Directory whose top-level .tif/.tiff files define one combined extent.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for manifests and downloads.")
    parser.add_argument("--start-date", required=True, help="Start date: YYYY-MM-DD or ISO UTC.")
    parser.add_argument("--end-date", required=True, help="End date: YYYY-MM-DD or ISO UTC.")
    parser.add_argument("--short-name", default="ATL03", help="CMR collection short name.")
    parser.add_argument("--version", default="007", help="CMR collection version.")
    parser.add_argument("--page-size", type=int, default=2000, help="CMR granule page size.")
    parser.add_argument("--download", action="store_true", help="Download files; otherwise query only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--timeout", type=float, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Download retry count.")
    parser.add_argument(
        "--harmony-subset",
        action="store_true",
        help="Submit a Harmony spatial subset job and download subsetted HDF5 files.",
    )
    parser.add_argument(
        "--download-full-granule",
        action="store_true",
        help="Download full ATL03 granule HDF5 files. This is also the default when --download is used without --harmony-subset.",
    )
    parser.add_argument(
        "--harmony-job-id",
        help="Resume an already submitted Harmony job instead of submitting a new job.",
    )
    parser.add_argument(
        "--harmony-ignore-errors",
        action="store_true",
        help="Allow partial ATL03 outputs if individual Harmony inputs fail.",
    )
    parser.add_argument(
        "--harmony-timeout",
        type=float,
        default=7200,
        help="Maximum seconds to wait for a Harmony job.",
    )
    parser.add_argument(
        "--harmony-poll-interval",
        type=float,
        default=20,
        help="Seconds between Harmony job status checks.",
    )
    parser.add_argument("--user-agent", default=DEFAULT_CLIENT_ID, help="HTTP User-Agent/Client-Id.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    tif_paths = resolve_tif_paths(args)
    output_dir = Path(args.output_dir)
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")
    if args.harmony_subset and args.download_full_granule:
        raise ValueError("--harmony-subset and --download-full-granule are mutually exclusive.")

    bbox = geotiffs_wgs84_bbox(tif_paths)
    temporal = build_temporal(args.start_date, args.end_date)
    start_datetime, end_datetime = build_datetime_interval(args.start_date, args.end_date)
    harmony_start_datetime = format_harmony_datetime(args.start_date)
    harmony_end_datetime = format_harmony_datetime(
        args.end_date,
        end_of_day=len(args.end_date.strip()) == 10,
    )

    session = make_session(args.user_agent)
    collection = find_collection(session, args.short_name, args.version, args.timeout)
    collection_concept_id = collection["id"]
    granules = search_granules(
        session,
        collection_concept_id,
        bbox,
        temporal,
        args.page_size,
        args.timeout,
    )
    urls = extract_hdf5_urls(granules)
    request_signature = harmony_request_signature(
        tif_paths=tif_paths,
        bbox=bbox,
        temporal=temporal,
        collection_concept_id=collection_concept_id,
        granules=granules,
    )

    summary = {
        "tif_path": str(tif_paths[0]) if len(tif_paths) == 1 else None,
        "tif_paths": [str(path) for path in tif_paths],
        "reference_tif_count": len(tif_paths),
        "bbox_wgs84": {
            "west": bbox[0],
            "south": bbox[1],
            "east": bbox[2],
            "north": bbox[3],
        },
        "temporal": temporal,
        "short_name": args.short_name,
        "version": args.version,
        "collection_concept_id": collection_concept_id,
        "granule_count": len(granules),
        "hdf5_url_count": len(urls),
        "download_mode": "harmony_subset" if args.harmony_subset else "full_granule",
    }
    write_outputs(output_dir, urls, granules, summary)

    print(f"GeoTIFF bbox (WGS84): {format_bbox(bbox)}")
    print(f"Reference GeoTIFFs: {len(tif_paths)}")
    print(f"Temporal: {temporal}")
    print(f"Collection concept id: {collection_concept_id}")
    print(f"Granules matched: {len(granules)}")
    print(f"HDF5 URLs found: {len(urls)}")
    print(f"Wrote: {output_dir / 'atl03_urls.txt'}")
    print(f"Wrote: {output_dir / 'atl03_granules.json'}")

    if not granules:
        print("No matching granules. Query completed without downloads.")
        return 0

    if not args.download:
        print("Query only. Pass --download to download data.")
        return 0

    if args.harmony_subset:
        if not args.overwrite and not args.harmony_job_id:
            cached = reusable_harmony_downloads(output_dir, request_signature)
            if cached is not None:
                print(f"[REUSE] Complete Harmony subset already exists: {len(cached)} files")
                return 0
        (output_dir / "atl03_download_complete.json").unlink(missing_ok=True)
        (output_dir / "atl03_download_partial.json").unlink(missing_ok=True)
        downloaded, final_job, partial = download_harmony_subset(
            collection_concept_id=collection_concept_id,
            bbox=bbox,
            granules=granules,
            start_datetime=harmony_start_datetime,
            end_datetime=harmony_end_datetime,
            output_dir=output_dir,
            overwrite=args.overwrite,
            timeout=args.timeout,
            max_retries=args.max_retries,
            harmony_timeout=args.harmony_timeout,
            harmony_poll_interval=args.harmony_poll_interval,
            user_agent=args.user_agent,
            harmony_job_id=args.harmony_job_id,
            harmony_ignore_errors=args.harmony_ignore_errors,
        )
        write_harmony_result(output_dir, request_signature, downloaded, final_job, partial=partial)
        if partial and not downloaded:
            hint = ("No usable subsets were returned." if args.harmony_ignore_errors
                    else "Use --harmony-ignore-errors to explicitly allow available subsets.")
            raise RuntimeError(
                "Harmony did not complete all inputs; see "
                f"{output_dir / 'atl03_download_partial.json'} and atl03_harmony_job.json. {hint}"
            )
    else:
        if not urls:
            print("No downloadable .h5 HTTPS URLs were found in CMR links.")
            return 0
        downloaded = download_urls(
            urls,
            output_dir,
            overwrite=args.overwrite,
            timeout=args.timeout,
            max_retries=args.max_retries,
            user_agent=args.user_agent,
        )

    print(f"Downloaded/skipped files: {len(downloaded)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthenticationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
