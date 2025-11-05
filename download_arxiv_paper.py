#!/usr/bin/env python3
"""
Download arXiv papers - Script to fetch CALM paper (2510.27688)
"""
import urllib.request
import sys
import os

def download_arxiv_paper(arxiv_id, output_path=None):
    """
    Download a paper from arXiv

    Args:
        arxiv_id: The arXiv paper ID (e.g., '2510.27688')
        output_path: Where to save the PDF (default: current directory)
    """
    if output_path is None:
        output_path = f"arxiv_{arxiv_id.replace('.', '_')}.pdf"

    # Try multiple URL formats
    urls = [
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }

    for url in urls:
        print(f"Trying {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

                # Check if we actually got a PDF (should start with %PDF)
                if data[:4] == b'%PDF':
                    with open(output_path, 'wb') as f:
                        f.write(data)
                    print(f"✓ Successfully downloaded {len(data):,} bytes to {output_path}")
                    return True
                else:
                    print(f"✗ Received non-PDF data from {url}")
        except Exception as e:
            print(f"✗ Failed: {e}")

    print("\nAll download attempts failed.")
    print(f"\nManual download options:")
    print(f"1. Visit https://arxiv.org/abs/{arxiv_id}")
    print(f"2. Click 'Download PDF' or 'View HTML'")
    print(f"3. Or use: wget https://arxiv.org/pdf/{arxiv_id}.pdf")
    return False

if __name__ == "__main__":
    arxiv_id = "2510.27688"  # CALM paper

    if len(sys.argv) > 1:
        arxiv_id = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = f"CALM_paper_{arxiv_id.replace('.', '_')}.pdf"

    print(f"Downloading arXiv paper {arxiv_id}...")
    download_arxiv_paper(arxiv_id, output_path)
