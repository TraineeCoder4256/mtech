#!/usr/bin/env python3
"""Render the research-paper PDF from its HTML fragments.

Assembles scratch fragments + figures into a two-column A4 paper and prints
it through headless Chromium.  Requires: pip install playwright pymupdf
(the browser itself ships with the environment).

Usage: python scripts/build_paper.py [--src docs/gmc_paper_src.html]
                                     [--out docs/gmc_paper.pdf]
"""
import argparse, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/gmc_paper_src.html")
    ap.add_argument("--out", default="docs/gmc_paper.pdf")
    ap.add_argument("--chrome", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    args = ap.parse_args()

    src = ROOT / args.src
    if not src.exists():
        sys.exit(f"missing {src} (the assembled single-file paper HTML)")

    from playwright.sync_api import sync_playwright
    out = ROOT / args.out
    with sync_playwright() as pw:
        kw = {"args": ["--no-sandbox"]}
        if pathlib.Path(args.chrome).exists():
            kw["executable_path"] = args.chrome
        b = pw.chromium.launch(**kw)
        pg = b.new_page()
        pg.goto(src.as_uri(), wait_until="load")
        pg.wait_for_timeout(2500)
        pg.pdf(path=str(out), format="A4", print_background=True,
               margin={"top": "19mm", "bottom": "20mm",
                       "left": "16mm", "right": "16mm"},
               display_header_footer=True, header_template="<div></div>",
               footer_template=(
                   '<div style="width:100%;font-family:Liberation Serif,serif;'
                   'font-size:8pt;text-align:center;color:#000;">'
                   '<span class="pageNumber"></span></div>'))
        b.close()
    print(f"wrote {out}  ({out.stat().st_size / 2**20:.2f} MB)")


if __name__ == "__main__":
    main()
