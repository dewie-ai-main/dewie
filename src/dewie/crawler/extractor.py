# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pure synchronous link extractor. No I/O — accepts raw HTML and a base URL,
returns a deduplicated list of absolute URLs.
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def extract_links(html: str, base_url: str, *, same_domain: bool = True) -> list[str]:
    """
    Parse *html* and return absolute HTTP(S) URLs found in <a href> tags.

    Parameters
    ----------
    html:        Raw HTML string.
    base_url:    Page URL used to resolve relative hrefs.
    same_domain: When True, only keep links whose netloc matches base_url's netloc.
    """
    base_netloc = urlparse(base_url).netloc.lower()
    soup = BeautifulSoup(html, "lxml")

    seen: set[str] = set()
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        # Normalise: lowercase scheme+netloc, drop fragment
        normalised = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            fragment="",
        ).geturl()

        if parsed.scheme not in ("http", "https"):
            continue
        if same_domain and parsed.netloc.lower() != base_netloc:
            continue
        if normalised in seen:
            continue

        seen.add(normalised)
        links.append(normalised)

    return links
