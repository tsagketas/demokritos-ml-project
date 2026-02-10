#!/usr/bin/env python3
"""
Compose slide_*.html files into a single project_presentation.html (HTML-only)

This script reads files named `slide_<n>.html` in the slides directory,
orders them by <n>, inserts a blank slide placeholder for missing indices,
and writes a combined `project_presentation.html` containing simple
previous/next navigation. It does not modify source slide files.
"""
from pathlib import Path
import re
import argparse
import sys
import html as html_lib


def extract_head_body(text: str):
    import re as _re
    head_m = _re.search(r"<head[^>]*>(.*?)</head>", text, _re.S | _re.I)
    body_m = _re.search(r"<body[^>]*>(.*?)</body>", text, _re.S | _re.I)
    head = head_m.group(1).strip() if head_m else ""
    body = body_m.group(1).strip() if body_m else text.strip()
    return head, body


def read_text_fallback(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1253", errors="replace")


def extract_head_parts(head: str):
    # Pull out meta + title from the head (best effort)
    meta_title = []
    meta_re = re.compile(r"<meta[^>]+?>", re.I | re.S)
    title_re = re.compile(r"<title[^>]*>.*?</title>", re.I | re.S)
    meta_title.extend(meta_re.findall(head))
    meta_title.extend(title_re.findall(head))
    return meta_title


def make_blank_slide():
    return (
        '<div class="slide-container p-10 bg-white text-gray-800" '
        'style="display:flex;align-items:center;justify-content:center;">\n'
        '  <div style="width:100%;height:100%;background:white;box-shadow:none;display:flex;align-items:center;justify-content:center;">\n'
        '    <h2 style="color:#9CA3AF;font-family:Montserrat,\'sans-serif\';">Blank slide</h2>\n'
        '  </div>\n'
        '</div>'
    )


def compose(slides_dir: Path, out_name: str = "project_presentation.html"):
    files = list(slides_dir.glob("slide_*.html"))
    slides = {}
    number_re = re.compile(r"slide_(\d+)")

    for f in files:
        m = number_re.search(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        text = read_text_fallback(f)
        head, body = extract_head_body(text)
        slides[idx] = {"path": f, "head": head, "body": body}

    if not slides:
        print("No slide_*.html files found in", slides_dir)
        return None

    max_idx = max(slides.keys())
    first_head = slides[min(slides.keys())]["head"] if slides else ""

    slide_blocks = []
    for i in range(1, max_idx + 1):
        if i in slides:
            slide_blocks.append(read_text_fallback(slides[i]["path"]))
        else:
            slide_blocks.append(
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>"
                "<title>Blank slide</title></head><body>"
                f"{make_blank_slide()}</body></html>"
            )

    nav_css = (
        '<style>\n'
        '  body { margin:0; font-family: "Open Sans", sans-serif; background:#111; }\n'
        '  #slideshow { position:relative; width:1280px; height:720px; margin:20px auto; overflow:hidden; background:#fff; }\n'
        '  .slides { width:100%; height:100%; }\n'
        '  .slide { display:none; width:100%; height:100%; }\n'
        '  .slide.active { display:block; }\n'
        '  .slide iframe { width:100%; height:100%; border:0; display:block; }\n'
        '  #prev, #next { position:absolute; top:50%; transform:translateY(-50%); background:rgba(0,0,0,0.5); color:#fff; border:0; padding:12px 16px; cursor:pointer; font-size:20px; border-radius:6px; }\n'
        '  #prev { left:10px }\n'
        '  #next { right:10px }\n'
        '  #indicator { position:absolute; right:16px; bottom:12px; background:rgba(255,255,255,0.9); padding:6px 10px; border-radius:6px; font-weight:600; }\n'
        '</style>\n'
    )

    nav_js = (
        '<script>\n'
        '(function(){\n'
        '  var slides = Array.from(document.querySelectorAll(\'.slide\'));\n'
        '  var current = 0;\n'
        '  function show(n){\n'
        '    if(!slides.length) return;\n'
        '    slides.forEach(function(s){ s.classList.remove(\'active\'); });\n'
        '    slides[n].classList.add(\'active\');\n'
        '    var ind = document.getElementById(\'indicator\');\n'
        '    if(ind) ind.textContent = (n+1) + \" / \" + slides.length;\n'
        '  }\n'
        '  function next(){ current = (current+1) % slides.length; show(current); }\n'
        '  function prev(){ current = (current-1 + slides.length) % slides.length; show(current); }\n'
        '  document.addEventListener(\'DOMContentLoaded\', function(){\n'
        '    document.getElementById(\'next\').addEventListener(\'click\', next);\n'
        '    document.getElementById(\'prev\').addEventListener(\'click\', prev);\n'
        '    document.addEventListener(\'keydown\', function(e){ if(e.key===\'ArrowRight\') next(); if(e.key===\'ArrowLeft\') prev(); });\n'
        '    show(0);\n'
        '  });\n'
        '  window.showSlide = function(n){ if(n>=0 && n<slides.length){ current = n; show(current); } }\n'
        '})();\n'
        '</script>\n'
    )

    meta_title = extract_head_parts(first_head)

    html = ["<!DOCTYPE html>", "<html>", "<head>"]
    html.extend(meta_title or ['<meta charset="utf-8"/>', '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>', "<title>Project Presentation</title>"])
    html.append(nav_css)
    html.append("</head>")
    html.append("<body>")
    html.append("\n<!-- Slides in order (1..%d) -->\n" % max_idx)
    html.append('<div id="slideshow">')
    html.append('<div class="slides">')

    for idx, block in enumerate(slide_blocks, start=1):
        srcdoc = html_lib.escape(block, quote=True)
        html.append(f'<section class="slide" data-index="{idx}">')
        html.append(f'<iframe srcdoc="{srcdoc}"></iframe>')
        html.append('</section>')

    html.append('</div>')
    html.append('<button id="prev" aria-label="Previous">◀</button>')
    html.append('<button id="next" aria-label="Next">▶</button>')
    html.append('<div id="indicator"></div>')
    html.append('</div>')
    html.append(nav_js)
    html.append("</body>")
    html.append("</html>")

    out_path = slides_dir / out_name
    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote combined presentation to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compose slide_*.html into one file")
    parser.add_argument("--slides-dir", "-d", default=".", help="directory with slide_*.html files")
    parser.add_argument("--out", "-o", default="project_presentation.html", help="output filename")
    args = parser.parse_args()

    slides_dir = Path(args.slides_dir)
    if not slides_dir.exists():
        print("slides dir does not exist:", slides_dir)
        sys.exit(1)

    compose(slides_dir, args.out)


if __name__ == "__main__":
    main()
