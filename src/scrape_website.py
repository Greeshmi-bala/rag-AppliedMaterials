import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# Official Applied Materials India website pages
URLS = [
    "https://www.appliedmaterials.com/in/en.html",
    "https://www.appliedmaterials.com/in/en/about/india-overview.html",
    "https://www.appliedmaterials.com/in/en/about.html",
    "https://www.appliedmaterials.com/in/en/careers.html",
    "https://www.appliedmaterials.com/in/en/frequently-asked-questions.html",
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


# --------------------------------------------------
# 2. EXTRACT TEXT FROM A WEBPAGE
# --------------------------------------------------

def scrape_page(url: str):

    print("\n" + "=" * 60)
    print(f"Collecting webpage: {url}")

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        )

        if "text/html" not in content_type.lower():

            print("Skipped: The response is not an HTML page.")

            return None

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Extract the page title
        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True,
            )

        else:

            title = urlparse(url).path

        # Remove unnecessary page elements
        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
            ]
        ):

            element.decompose()

        # Prefer the main article or main content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = main_content.get_text(
            separator="\n",
            strip=True,
        )

        # Clean excessive whitespace
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = text.strip()

        if len(text) < 100:

            print(
                "Warning: Very little text was extracted. "
                "The webpage may require JavaScript."
            )

        # Prepare a structured record
        page_data = {
            "title": title,
            "source_url": url,
            "text": text,
            "collected_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }

        # Generate a safe filename
        filename = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            urlparse(url).path.strip("/") or "homepage",
        )

        output_file = RAW_DIR / f"{filename}.json"

        # Save the webpage information
        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                page_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Page title: {title}")
        print(f"Characters collected: {len(text)}")
        print(f"Saved to: {output_file}")

        return page_data

    except requests.RequestException as error:

        print(f"Failed to collect webpage: {error}")

        return None


# --------------------------------------------------
# 3. RUN THE SCRAPER
# --------------------------------------------------

def main():

    successful_pages = 0

    total_pages = len(URLS)

    print("\n")
    print("=" * 60)
    print("APPLIED MATERIALS WEBSITE DATA COLLECTION")
    print("=" * 60)

    for url in URLS:

        result = scrape_page(url)

        if result is not None:

            successful_pages += 1

        # Wait briefly between requests
        time.sleep(1)

    print("\n")
    print("=" * 60)
    print("DATA COLLECTION FINISHED")
    print("=" * 60)

    print(f"Total webpages: {total_pages}")

    print(f"Successfully collected: {successful_pages}")

    print(f"Failed or skipped: {total_pages - successful_pages}")

    print(f"\nOutput folder: {RAW_DIR}")


if __name__ == "__main__":

    main()