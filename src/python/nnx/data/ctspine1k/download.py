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
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

URL = "https://drive.google.com/drive/folders/1Acyuu7ZmbjnS4mkJRdiUfkXx5SBta4EM"
FOLDER_ID = URL.split("/")[-1]  # shortcut to get ID
LOCAL_DIR = Path(__file__).parent / "shareable"


def download_from_google_drive(output_dir: Path, max_workers: int = 8) -> None:
    """Download the data CTSpine1K dataset from Google Drive.

    Args:
        output_dir: where to write the data. Mind to use the exact directory to
            when specifying the `cache_dir` later on.
        max_workers: how many threads will load the data concurrently.

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks to the executor
        future_to_link = {
            executor.submit(_download_file_from_google_drive_link, link, path): link
            for path, link in workload
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_link):
            link = future_to_link[future]
            downloaded_file_path = future.result()
            if downloaded_file_path:
                pass
            else:
                print(f"Failed to download {link}")


def _download_file_from_google_drive_link(shareable_link, destination_folder):
    """Download a publicly shared Google Drive file from its shareable link.

    Handling various edge cases (virus scans, large files, failed previews).

    Args:
        shareable_link (str): The public shareable link to the Google Drive file.
        destination_folder (str): The local folder path to save the downloaded file.

    Returns:
        str: The path to the downloaded file if successful, None otherwise.

    """
    import os
    import re

    try:
        from bs4 import BeautifulSoup

        has_bs4 = True
    except ImportError:
        print("BeautifulSoup not installed. Some methods may not work as effectively.")
        print("Consider installing it with: pip install beautifulsoup4")
        has_bs4 = False

    # Extract file ID from the shareable link
    file_id = None
    parsed_url = urlparse(shareable_link)

    if "drive.google.com" in parsed_url.netloc:
        if "/file/d/" in parsed_url.path:
            # Format: https://drive.google.com/file/d/{fileId}/view...
            match = re.search(r"/file/d/([^/]+)", parsed_url.path)
            if match:
                file_id = match.group(1)
        elif "/open" in parsed_url.path:
            # Format: https://drive.google.com/open?id={fileId}
            query_params = parse_qs(parsed_url.query)
            if "id" in query_params:
                file_id = query_params["id"][0]

    if not file_id:
        print(f"Could not extract file ID from the link: {shareable_link}")
        return None

    print(f"Extracted file ID: {file_id}")

    # Create a session to maintain cookies between requests
    session = requests.Session()

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Method 1: Direct download attempt
    print("Trying direct download method...")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url, stream=True)

    if response.status_code != 200:
        print(f"Failed to access the file (HTTP {response.status_code})")
        return None

    # Check if we got the file directly or a confirmation page
    if "Content-Disposition" in response.headers:
        # Success - Direct download worked
        filename = extract_filename(response.headers.get("Content-Disposition", ""))
        if not filename:
            filename = f"drive_file_{file_id}"

        destination_path = os.path.join(destination_folder, filename)

        return save_response_to_file(response, destination_path)

    # Method 2: Handle the confirmation page (virus scan warning)
    print("Direct download did not work. Handling confirmation page...")

    # Extract the confirmation token
    confirmation_token = None

    if has_bs4:
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
                if href and "confirm=" in href:
                    match = re.search(r"confirm=([0-9A-Za-z_-]+)", href)
                    if match:
                        confirmation_token = match.group(1)
                        break
    else:
        # Fallback to regex if BeautifulSoup is not available
        match = re.search(r"confirm=([0-9A-Za-z_-]+)", response.text)
        if match:
            confirmation_token = match.group(1)

    if confirmation_token:
        print(f"Found confirmation token: {confirmation_token}")

        # Make a request with the confirmation token
        download_url = f"https://drive.google.com/uc?export=download&confirm={confirmation_token}&id={file_id}"
        response = session.get(download_url, stream=True)

        if response.status_code == 200 and "Content-Disposition" in response.headers:
            # Success - got the file after confirmation
            filename = extract_filename(response.headers.get("Content-Disposition", ""))
            if not filename:
                filename = f"drive_file_{file_id}"

            destination_path = os.path.join(destination_folder, filename)

            return save_response_to_file(response, destination_path)

    # Method 3: Try an alternative approach for large files
    print("Confirmation token method did not work. Trying large file method...")

    # Clear session and try with new headers
    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://drive.google.com/",
    }

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

    if response.status_code == 200:
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

            destination_path = os.path.join(destination_folder, filename)

            return save_response_to_file(response, destination_path)

    # Method 4: Handle "virus scan warning" page with UUID-based download
    print("Trying to extract download parameters from the virus scan warning page...")

    # Try direct download to get the warning page
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url)

    # We need to parse this page to get the form and its parameters
    filename = None
    form_url = None
    form_params = {}

    if has_bs4:
        soup = BeautifulSoup(response.content, "html.parser")

        # Try to extract the original filename and size from the warning page
        name_size_elem = soup.select_one(".uc-name-size a")
        if name_size_elem:
            filename = name_size_elem.text.strip()
            print(f"Found original filename from warning page: {filename}")

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

            if form_url and form_params:
                print(f"Found download form: {form_url}")
                print(f"With parameters: {form_params}")
    else:
        # Fallback regex approach if BeautifulSoup is not available
        filename_match = re.search(
            r'<a href="/open\?id=[^"]+">([^<]+)</a>',
            response.text,
        )
        if filename_match:
            filename = filename_match.group(1)

        # Extract form action URL
        form_url_match = re.search(r'<form[^>]+action="([^"]+)"', response.text)
        if form_url_match:
            form_url = form_url_match.group(1)

        # Extract all hidden inputs
        for name, value in re.findall(
            r'<input type="hidden" name="([^"]+)" value="([^"]+)">',
            response.text,
        ):
            form_params[name] = value

    # If we found the form details, use them to make a direct download request
    if form_url and form_params:
        print("Attempting download using extracted form parameters...")

        response = session.get(form_url, params=form_params, stream=True)

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")

            # Verify we're not getting HTML again
            if "text/html" not in content_type:
                if not filename:
                    filename = extract_filename(
                        response.headers.get("Content-Disposition", ""),
                    )
                if not filename:
                    filename = f"file_{file_id}"

                destination_path = os.path.join(destination_folder, filename)
                return save_response_to_file(response, destination_path)
            print("Warning: Received HTML instead of file content.")

            # Check if we got a small HTML content - could be an error page
            if len(response.content) < 10000:
                print("HTML content preview:")
                print(response.text[:500])

    # Method 5: Last resort for very stubborn files - try multiple approaches
    print("All standard methods failed. Trying last resort methods...")

    # Approach 1: Try cookies and direct download
    cookies = {"download_warning_13058876669": "yes"}
    headers["Cookie"] = "; ".join([f"{k}={v}" for k, v in cookies.items()])

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url, headers=headers, stream=True)

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length", "0")

        # Only save if we're reasonably sure it's the actual file (not HTML)
        if "text/html" not in content_type and int(content_length) > 10000:
            if not filename:
                filename = f"file_{file_id}"

            destination_path = os.path.join(destination_folder, filename)
            return save_response_to_file(response, destination_path)

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


def save_response_to_file(response, destination_path):
    """Helper function to save a response to a file with progress bar."""
    from tqdm import tqdm

    try:
        # Check the first few bytes of the response to detect HTML content
        # Peek at the content without consuming it
        peek_content = next(response.iter_content(256, False), b"")

        # If it looks like HTML, we're probably getting an error page, not the actual file
        if peek_content.startswith(b"<!DOCTYPE html>") or peek_content.startswith(
            b"<html",
        ):
            print("Warning: Response appears to be HTML, not a file.")
            print("HTML preview:")
            print(peek_content.decode("utf-8", errors="replace")[:100] + "...")
            print(
                "Aborting download - this is likely an error page, not the actual file.",
            )
            return None

        # Reset the response stream if possible, or re-request if needed
        if hasattr(response, "seek") and callable(response.seek):
            try:
                response.seek(0)
            except:
                # Can't seek, continue with what we have
                pass

        with open(destination_path, "wb") as f:
            # Write the peeked content first if we couldn't reset the stream
            if not hasattr(response, "seek"):
                f.write(peek_content)

            block_size = 8192
            total_size = int(response.headers.get("content-length", 0))

            if total_size > 0:
                # Adjust total size if we've already written some bytes
                if not hasattr(response, "seek"):
                    total_size = max(0, total_size - len(peek_content))

                progress_bar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(destination_path)}",
                )

                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

                progress_bar.close()
            else:
                # If we don't know the total size, download without progress bar
                print(
                    f"Downloading {os.path.basename(destination_path)} (unknown size)...",
                )
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)

        # Verify the downloaded file is not tiny (which would suggest an error)
        file_size = os.path.getsize(destination_path)
        if file_size < 1000:  # Less than 1KB
            with open(destination_path, "rb") as f:
                content = f.read()

            # Check if the tiny file is actually HTML
            if content.startswith(b"<!DOCTYPE html>") or content.startswith(b"<html"):
                print(
                    f"Warning: Downloaded file appears to be HTML ({file_size} bytes)",
                )
                os.remove(destination_path)  # Delete the HTML error page
                return None

        print(f"Successfully downloaded to {destination_path} ({file_size} bytes)")
        return destination_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def _generate_workload(shareable_dir: Path, output_dir: Path) -> list[tuple]:
    data_glob = sorted((shareable_dir / "data").iterdir())
    label_glob = sorted((shareable_dir / "label").iterdir())
    links_dict: dict[str, list[Path]] = {"data": data_glob, "label": label_glob}

    workload = []
    for key, links in links_dict.items():
        for link in links:
            path = output_dir / link.stem / key
            path.mkdir(exist_ok=True, parents=True)

            for link in link.read_text().split(", "):
                workload.append((path, link))

    return workload


if __name__ == "__main__":
    download_from_google_drive(Path.cwd() / "test")
