"""Logic related to downloading the data from Google Drive.

As of April 21, 2025, the authors of https://github.com/MIRACLE-Center/CTSpine1K
recommend accessing the dataset via Google Drive. While manual downloading through
the web UI is possible, it becomes cumbersome when working on remote machines or
when automating dataset preparation workflows.

Several approaches were evaluated for robustness, maintainability, and ease of use:

1. Google Drive API: Requires OAuth setup and credentials management, adding
   configuration overhead that complicates deployment.

2. Third-party tools (e.g., rclone): Also require initial configuration and
   may have dependencies that affect portability.

3. Web scraping with link extraction: For directories with >50 files, Google Drive's
   pagination and dynamic loading makes this approach unreliable.

4. Manual shareable links collection (current approach): Most reliable solution that
   works "out of the box" without configuration, though requires manual maintenance.

The implementation uses a collection of manually saved shareable links stored in text
files within a 'shareable' directory. These links are processed to download files
concurrently using multiple threads, with retry logic for resilience. In case the
pagination approach becomes more robust, it is also possible to switch, but only the
case of expected active changes to the files in Google Drive.

This module is designed to be the first step in a data preparation pipeline, ensuring
that researchers can quickly obtain the CTSpine1K dataset programmatically without
manual intervention, regardless of their computing environment.

"""

import concurrent.futures
import re
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

URL = "https://drive.google.com/drive/folders/1Acyuu7ZmbjnS4mkJRdiUfkXx5SBta4EM"
FOLDER_ID = URL.split("/")[-1]  # shortcut to get ID


def download_from_google_drive(
    output_dir: Path,
    downloaded_files: list,
    max_workers: int = 4,
    max_attempts: int = 5,
) -> None:
    """Download the data CTSpine1K dataset from Google Drive.

    Args:
        output_dir: where to write the data. Mind to use the exact directory to
            when specifying the `cache_dir` later on.
        downloaded_files: files downloaded from hugging face. These should also
            contain data related to the shareable files with the links to the
            actual files.
        max_workers: how many threads will load the data concurrently.
        max_attempts: how often we try to download in the case of a failure.

    """
    data_key = "shareable"
    shareables: list[str] = [file for file in downloaded_files if data_key in file]
    data_dir = Path(shareables[0].split(data_key)[0]) / data_key

    workload = _generate_workload(data_dir, output_dir)
    failed_links = []

    with (
        tqdm(total=len(workload), desc="Downloading files") as pbar,
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        # Map the download function across all links
        future_to_link = {}

        for link, path in workload.items():
            future = executor.submit(
                _download_file_from_google_drive_link,
                link,
                path,
            )
            future_to_link[future] = link

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_link):
            link = future_to_link[future]
            result = future.result()

            # Update progress bar for each completed file
            pbar.update(1)

            if not result:
                failed_links.append(link)

    if failed_links:
        print(f"Found {len(failed_links)} failed links. Retrying...")
        for failed_link in failed_links:
            path = workload[failed_link]
            attempt = 0
            while attempt < max_attempts:
                if _download_file_from_google_drive_link(failed_link, path) is not None:
                    break  # Download succeeded, break the retry loop
                attempt += 1


def _download_file_from_google_drive_link(  # noqa: C901, PLR0912, PLR0914, PLR0915
    shareable_link: str,
    destination_folder: Path,
) -> Path | None:
    """Download a publicly shared Google Drive file from its shareable link.

    Handling various edge cases (virus scans, large files, failed previews).

    Args:
        shareable_link (str): The public shareable link to the Google Drive file.
        destination_folder (str): The local folder path to save the downloaded file.

    Returns:
        The path to the downloaded file if successful, None otherwise.

    """
    # all links exhibit the same pattern
    file_id = re.search(r"/file/d/([^/]+)", shareable_link).group(1)

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # noqa: E501
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",  # noqa: E501
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://drive.google.com/",
    }

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url, stream=True)

    success_code = 200
    if response.status_code != success_code:
        print(f"Failed to access the file (HTTP {response.status_code})")
        return None

    # Check if we got the file directly or a confirmation page
    if "Content-Disposition" in response.headers:
        filename = _extract_filename(response.headers.get("Content-Disposition", ""))
        if not filename:
            filename = f"drive_file_{file_id}"

        return _save_response_to_file(response, destination_folder / filename)

    # Handle the confirmation page (virus scan warning)
    confirmation_token = None
    # Parse HTML with BeautifulSoup to find the token
    soup = BeautifulSoup(response.content, "html.parser")

    # Check for form with confirm button
    form = soup.find("form")
    if form:
        for input_tag in form.find_all("input"):
            if input_tag.get("name") == "confirm":
                confirmation_token = input_tag.get("value")
                break

    # Check for download links with confirmation token
    if not confirmation_token:
        for link in soup.find_all("a"):
            href = link.get("href")
            match = re.search(r"confirm=([0-9A-Za-z_-]+)", href)
            if href and match:
                confirmation_token = match.group(1)
                break

    if confirmation_token:
        # Make a request with the confirmation token
        download_url = f"https://drive.google.com/uc?export=download&confirm={confirmation_token}&id={file_id}"
        response = session.get(download_url, stream=True)

        if (
            response.status_code == success_code
            and "Content-Disposition" in response.headers
        ):
            # Success - got the file after confirmation
            filename = _extract_filename(
                response.headers.get("Content-Disposition", ""),
            )
            if not filename:
                filename = f"drive_file_{file_id}"

            return _save_response_to_file(response, destination_folder / filename)

    # Try an alternative approach for large files

    # First, get cookies with an initial request
    session.get(f"https://drive.google.com/file/d/{file_id}/view", headers=headers)

    # Try to download with cookies set
    params = {
        "id": file_id,
        "export": "download",
        "confirm": "t",  # Add confirmation parameter
    }

    response = session.get(
        "https://drive.google.com/uc",
        params=params,
        headers=headers,
        stream=True,
    )

    if response.status_code == success_code:
        content_type = response.headers.get("Content-Type", "")

        # Check if we're getting HTML instead of a file
        if "text/html" not in content_type or "Content-Disposition" in response.headers:
            # Try to get filename from Content-Disposition, or use default
            if "Content-Disposition" in response.headers:
                filename = _extract_filename(
                    response.headers.get("Content-Disposition", ""),
                )
            else:
                filename = f"large_file_{file_id}"

            return _save_response_to_file(response, destination_folder / filename)

    # Handle "virus scan warning" page with UUID-based download
    # Try direct download to get the warning page
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url)

    # We need to parse this page to get the form and its parameters
    filename = None
    form_url = None
    form_params = {}

    soup = BeautifulSoup(response.content, "html.parser")

    # Try to extract the original filename and size from the warning page
    name_size_elem = soup.select_one(".uc-name-size a")
    if name_size_elem:
        filename = name_size_elem.text.strip()

    # Look for the download form (the "Download anyway" button)
    form = soup.find("form", id="download-form")
    if form:
        form_url = form.get("action")

        # Get all input values from the form
        for input_tag in form.find_all("input"):
            name = input_tag.get("name")
            value = input_tag.get("value")
            if name and value:
                form_params[name] = value

    # If we found the form details, use them to make a direct download request
    if form_url and form_params:
        response = session.get(form_url, params=form_params, stream=True)

        if response.status_code == success_code:
            content_type = response.headers.get("Content-Type", "")

            # Verify we're not getting HTML again
            if "text/html" not in content_type:
                if not filename:
                    filename = _extract_filename(
                        response.headers.get("Content-Disposition", ""),
                    )
                if not filename:
                    filename = f"file_{file_id}"

                return _save_response_to_file(response, destination_folder / filename)

            print("Warning: Received HTML instead of file content.")

    print(f"All download methods failed for file ID {file_id}")

    with open(f"debug_html_{file_id}.html", "wb") as debug_file:
        debug_file.write(response.content)

    return None


def _extract_filename(content_disposition: str) -> str:
    """Help function to extract filename from Content-Disposition header.

    Returns:
        The name of file as in Google Drive.

    """
    fname_match = re.search(r'filename="?([^"]+)"?', content_disposition)
    return unquote(fname_match.group(1))


def _save_response_to_file(
    response: requests.Response,
    destination_path: Path,
) -> Path | None:
    """Save the content from the drive to destination.

    Returns:
        In the case of failure it is None, otherwise the Path to where the data was written.

    """
    with destination_path.open("wb") as f:
        block_size = 8192

        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)

    # Verify the downloaded file is not tiny (which would suggest an error)
    file_size = destination_path.stat().st_size
    one_kilobyte = 1024
    if file_size < one_kilobyte:
        with destination_path.open("rb") as f:
            content = f.read()

        # Check if the tiny file is actually HTML
        if content.startswith((b"<!DOCTYPE html>", b"<html")):
            print(
                f"Warning: Downloaded file appears to be HTML ({file_size} bytes)",
            )
            destination_path.unlink()  # Delete the HTML error page
            return None

    return destination_path


def _generate_workload(
    shareable_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    data_glob = sorted((shareable_dir / "data").iterdir())
    label_glob = sorted((shareable_dir / "label").iterdir())
    links_dict: dict[str, list[Path]] = {"data": data_glob, "label": label_glob}

    workload = {}
    for key, links_files in links_dict.items():
        for links_file in links_files:
            path = output_dir / key / links_file.stem
            path.mkdir(exist_ok=True, parents=True)

            for link in links_file.read_text().split(", "):
                workload[link] = path

    return workload
