"""Logic related to downloading the data from Google Drive.

As of the 21.04.2025, authors of https://github.com/MIRACLE-Center/CTSpine1K
mentioned that the go to way is taking the data from Google Drive. Using the UI and
placing the data somewhere on the machine is a deasible way but is not streamlined and
problematic when working on a remote machine.

To make life easier this code programmatically fetches data from the drive and places
it in the specified directory from which it can easily be used to for the actual
training pipeline. There were many experiments done to see which approach is the best
maintainable and easiest out of the box. Unfortuantely for directories with more than
50 files there is no easy fix and unfortunately the CTSpine1K dataset in Google Drive
have directories with hundreds of files. Alternatives such as the official Google API
or rclone all need configuration overhead in the beginning which is unwanted. It should
work out of the box from anywhere. Hence the decision was to just manually save all
shareable links. This has the downside of manually adapting the files in case of changes
and if it gets easier over the time definitely something to change

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
LOCAL_DIR = Path(__file__).parent / "shareable"


def download_from_google_drive(
    output_dir: Path,
    max_workers: int = 8,
    max_attempts: int = 5,
) -> None:
    """Download the data CTSpine1K dataset from Google Drive.

    Args:
        output_dir: where to write the data. Mind to use the exact directory to
            when specifying the `cache_dir` later on.
        max_workers: how many threads will load the data concurrently.
        max_attempts: how often we try to download in the case of a failure.

    Raises:
        RuntimeError: violation of assumption about location of .txt files
            with shareable links.

    """
    if not LOCAL_DIR.is_dir():
        msg = (
            f"Expected directory {LOCAL_DIR!s} to exist on the same level as script. "
            "Ensure directory `shareable` contains all .txt files with links to GDrive."
        )
        raise RuntimeError(msg)

    workload = _generate_workload(LOCAL_DIR, output_dir)
    failed_links = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks to the executor
        future_to_link = {
            executor.submit(_download_file_from_google_drive_link, link, path): link
            for link, path in list(workload.items())[:10]
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_link):
            link = future_to_link[future]
            downloaded_file_path = future.result()
            if not downloaded_file_path:
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


def _download_file_from_google_drive_link(
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
        filename = extract_filename(response.headers.get("Content-Disposition", ""))
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
            filename = extract_filename(response.headers.get("Content-Disposition", ""))
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
                filename = extract_filename(
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
                    filename = extract_filename(
                        response.headers.get("Content-Disposition", ""),
                    )
                if not filename:
                    filename = f"file_{file_id}"

                return _save_response_to_file(response, destination_folder / filename)

            print("Warning: Received HTML instead of file content.")

    # Last resort for very stubborn files - try multiple approaches
    print("All standard methods failed. Trying last resort methods...")
    # Try cookies and direct download
    headers["Cookie"] = "download_warning_13058876669=yes; "

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url, headers=headers, stream=True)

    if response.status_code == success_code:
        content_type = response.headers.get("Content-Type", "")

        # Only save if we're reasonably sure it's the actual file (not HTML)
        if "text/html" not in content_type:
            if not filename:
                filename = f"file_{file_id}"

            return _save_response_to_file(response, destination_folder / filename)

    print(f"All download methods failed for file ID {file_id}")
    return None


def extract_filename(content_disposition):
    """Helper function to extract filename from Content-Disposition header."""
    filename = None

    # Try to extract filename*= format first (handles UTF-8 encoding)
    fname_match = re.search(r"filename\*=([^;]+)", content_disposition)
    if fname_match:
        try:
            parts = fname_match.group(1).split("''", 1)
            if len(parts) == 2:
                filename = unquote(parts[1])
        except Exception:
            pass

    # If that didn't work, try the simpler filename= format
    if not filename:
        fname_match_fallback = re.search(r'filename="?([^"]+)"?', content_disposition)
        if fname_match_fallback:
            filename = unquote(fname_match_fallback.group(1))

    return filename


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
        total_size = int(response.headers.get("content-length", 0))

        if total_size > 0:
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {destination_path!s}",
            )

            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()
        else:
            # If we don't know the total size, download without progress bar
            print(
                f"Downloading {destination_path!s}...",
            )
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


def _generate_workload(shareable_dir: Path, output_dir: Path) -> dict[str, Path]:
    data_glob = sorted((shareable_dir / "data").iterdir())[-2:]
    label_glob = sorted((shareable_dir / "label").iterdir())[-2:]
    links_dict: dict[str, list[Path]] = {"data": data_glob, "label": label_glob}

    workload = {}
    for key, links_files in links_dict.items():
        for links_file in links_files:
            path = output_dir / links_file.stem / key
            path.mkdir(exist_ok=True, parents=True)

            for link in links_file.read_text().split(", "):
                workload[link] = path

    return workload


if __name__ == "__main__":
    download_from_google_drive(Path.cwd() / "test")
