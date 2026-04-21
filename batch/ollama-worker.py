#!/usr/bin/env python3
"""Ollama batch worker for career-ops — replaces claude -p workers.

Pre-loads all context (JD, cv.md, article-digest.md, etc.) and sends to
Ollama in a single prompt. Parses structured output and writes report,
PDF HTML, and tracker TSV without requiring any tool-calling.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:27b")


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._chunks.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._chunks)).strip()


def _strip_html(html: str) -> str:
    p = _Stripper()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.get_text()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str, timeout: int = 30) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; career-ops-ollama/1.0)"})
        with urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
        return _strip_html(raw) if "<html" in raw.lower() else raw
    except Exception as e:
        print(f"WARN fetch {url}: {e}", file=sys.stderr)
        return None


def web_search(query: str, n: int = 3) -> str:
    """Optional DuckDuckGo search. Returns empty string if library absent."""
    try:
        from duckduckgo_search import DDGS  # type: ignore
        results = list(DDGS().text(query, max_results=n))
        return "\n".join(f"- {r['title']}: {r['body'][:200]}" for r in results)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def call_ollama(system: str, user: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }).encode()

    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        return data["message"]["content"]
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL} — is it running? ({e})") from e


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def next_tracker_num(applications_md: Path) -> int:
    try:
        for line in reversed(applications_md.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line.startswith("|") and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts and parts[0].isdigit():
                    return int(parts[0]) + 1
    except Exception:
        pass
    return 1


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    for pattern in [
        r"```json\s*(\{.*?\})\s*```",
        r'(\{"status"\s*:.*?\})\s*$',
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                continue
    return {}


def extract_html(text: str) -> str | None:
    m = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(<!DOCTYPE html>.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def extract_tracker_tsv(text: str) -> str | None:
    m = re.search(r"^TRACKER_TSV:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_system(
    base_prompt: str,
    jd: str, cv: str, digest: str, llms: str,
    scan_hist: str, cv_tmpl: str, comp_data: str,
    url: str, report_num: str, date: str, offer_id: str, tracker_num: int,
) -> str:
    preamble = f"""\
# IMPORTANT — Ollama Non-Agentic Mode

You are running as a standalone LLM worker (not Claude Code). All file contents
are pre-loaded in this prompt. DO NOT attempt to read files, call tools, use
WebFetch, or use WebSearch — everything needed is already provided here.

Resolved placeholders:
- {{{{URL}}}} = {url}
- {{{{JD_FILE}}}} = (content pre-loaded below — skip Paso 1)
- {{{{REPORT_NUM}}}} = {report_num}
- {{{{DATE}}}} = {date}
- {{{{ID}}}} = {offer_id}

## Required output format (in this exact order)

1. Full evaluation report in markdown (Blocks A–G + Score Global)
2. Complete tailored CV HTML enclosed in triple-backtick html fences
3. JSON summary enclosed in triple-backtick json fences
4. Tracker TSV line on a line starting with: TRACKER_TSV:

---

# Pre-loaded Context

## Job Description (from {url})

{jd[:10000]}

---

## cv.md

{cv[:8000]}

---

## article-digest.md

{digest[:3000] if digest else "(not present)"}

---

## llms.txt

{llms[:2000] if llms else "(not present)"}

---

## data/scan-history.tsv (most recent entries)

{scan_hist[-2000:] if scan_hist else "(empty)"}

---

## templates/cv-template.html

{cv_tmpl[:10000] if cv_tmpl else "(not present)"}

---

## Compensation Research

{comp_data if comp_data else "(web search unavailable — note this in Block D)"}

---

## Next tracker number available: {tracker_num}

---

# Original Batch Worker Instructions

"""
    return preamble + base_prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ollama batch worker for career-ops")
    ap.add_argument("--id",          required=True, help="Offer ID from batch-input.tsv")
    ap.add_argument("--url",         required=True, help="Job offer URL")
    ap.add_argument("--jd-file",     required=True, help="Path to pre-fetched JD text file")
    ap.add_argument("--report-num",  required=True, help="Zero-padded report number (e.g. 042)")
    ap.add_argument("--date",        required=True, help="Processing date YYYY-MM-DD")
    ap.add_argument("--prompt-file", required=True, help="Path to batch-prompt.md")
    ap.add_argument("--project-dir", required=True, help="Absolute path to career-ops root")
    ap.add_argument("--model",       default=None,  help="Override OLLAMA_MODEL env var")
    args = ap.parse_args()

    proj  = Path(args.project_dir)
    model = args.model or OLLAMA_MODEL

    def fail(msg: str):
        out = {
            "status": "failed", "id": args.id, "report_num": args.report_num,
            "company": "unknown", "role": "unknown", "score": None,
            "pdf": None, "report": None, "error": msg,
        }
        print(json.dumps(out))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Fetch JD
    # ------------------------------------------------------------------
    jd = ""
    jd_path = Path(args.jd_file)
    if jd_path.exists() and jd_path.stat().st_size > 0:
        jd = jd_path.read_text(encoding="utf-8")
    if not jd:
        print(f"Fetching JD from {args.url}", file=sys.stderr)
        jd = fetch_url(args.url) or ""
    if not jd:
        fail("Could not fetch JD — URL unreachable and JD file empty")

    # ------------------------------------------------------------------
    # 2. Context files
    # ------------------------------------------------------------------
    cv      = read(proj / "cv.md")
    digest  = read(proj / "article-digest.md")
    llms    = read(proj / "llms.txt")
    scan    = read(proj / "data" / "scan-history.tsv")
    tmpl    = read(proj / "templates" / "cv-template.html")
    base    = read(args.prompt_file)

    # ------------------------------------------------------------------
    # 3. Optional comp search
    # ------------------------------------------------------------------
    comp = web_search(f"software engineer salary glassdoor levels.fyi {args.url[:80]}")

    # ------------------------------------------------------------------
    # 4. Tracker number
    # ------------------------------------------------------------------
    tnum = next_tracker_num(proj / "data" / "applications.md")

    # ------------------------------------------------------------------
    # 5. Build prompt and call Ollama
    # ------------------------------------------------------------------
    system = build_system(
        base, jd, cv, digest, llms, scan, tmpl, comp,
        args.url, args.report_num, args.date, args.id, tnum,
    )
    user = f"Evaluate job offer #{args.id} and produce the full pipeline output."

    print(f"Calling Ollama model={model} ...", file=sys.stderr)
    try:
        response = call_ollama(system, user, model)
    except Exception as e:
        fail(str(e))

    # ------------------------------------------------------------------
    # 6. Parse response
    # ------------------------------------------------------------------
    parsed       = extract_json(response)
    company      = parsed.get("company", "unknown")
    company_slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-") or "unknown"
    role         = parsed.get("role", "unknown")
    score        = parsed.get("score")
    legitimacy   = parsed.get("legitimacy", "unknown")

    # ------------------------------------------------------------------
    # 7. Save report .md (everything before the ```html block)
    # ------------------------------------------------------------------
    html_match = re.search(r"```html", response)
    report_body = response[: html_match.start()].strip() if html_match else response

    reports_dir = proj / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_file = f"{args.report_num}-{company_slug}-{args.date}.md"
    report_path = reports_dir / report_file
    report_path.write_text(report_body, encoding="utf-8")
    print(f"Report: {report_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 8. Generate PDF (if HTML was produced)
    # ------------------------------------------------------------------
    pdf_path  = None
    pdf_emoji = "❌"
    html      = extract_html(response)
    if html:
        tmp_html = os.path.join(tempfile.gettempdir(), f"cv-candidate-{company_slug}.html")
        Path(tmp_html).write_text(html, encoding="utf-8")
        out_dir  = proj / "output"
        out_dir.mkdir(exist_ok=True)
        pdf_out  = out_dir / f"cv-candidate-{company_slug}-{args.date}.pdf"
        try:
            res = subprocess.run(
                ["node", "generate-pdf.mjs", tmp_html, str(pdf_out), "--format=letter"],
                cwd=str(proj), capture_output=True, text=True, timeout=90,
            )
            if res.returncode == 0:
                pdf_path  = str(pdf_out)
                pdf_emoji = "✅"
                print(f"PDF: {pdf_path}", file=sys.stderr)
            else:
                print(f"WARN PDF failed: {res.stderr[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"WARN PDF error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 9. Write tracker TSV
    # ------------------------------------------------------------------
    tsv_dir = proj / "batch" / "tracker-additions"
    tsv_dir.mkdir(exist_ok=True)
    score_str   = f"{score:.2f}/5" if score is not None else "N/A"
    report_link = f"[{args.report_num}](reports/{report_file})"
    note        = f"Score {score_str} — {legitimacy}"

    tsv_line = extract_tracker_tsv(response) or (
        f"{tnum}\t{args.date}\t{company}\t{role}\tEvaluada\t{score_str}\t{pdf_emoji}\t{report_link}\t{note}"
    )
    (tsv_dir / f"{args.id}.tsv").write_text(tsv_line + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 10. JSON summary to stdout (parsed by batch-runner.sh)
    # ------------------------------------------------------------------
    summary = {
        "status": "completed",
        "id": args.id,
        "report_num": args.report_num,
        "company": company_slug,
        "role": role,
        "score": score,
        "legitimacy": legitimacy,
        "pdf": pdf_path,
        "report": str(report_path),
        "error": None,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
