#!/usr/bin/env python3
"""PyTorch course — self-contained builder.

Inlines all CSS so there's no separate styles file to lose.
Each module file imports `emit` and calls it with (slug, title, body).
Outputs go to out/html/ and out/md/.
"""
import re
import sys
import subprocess
from pathlib import Path

try:
    import html2text
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "html2text",
         "--break-system-packages", "--quiet"]
    )
    import html2text


ROOT = Path(__file__).parent
OUT_HTML = ROOT / "out" / "html"
OUT_MD = ROOT / "out" / "md"
OUT_HTML.mkdir(parents=True, exist_ok=True)
OUT_MD.mkdir(parents=True, exist_ok=True)


STYLES = r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800;9..144,900&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Caveat:wght@400;500;600;700&family=Patrick+Hand&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
  :root {
    --paper: #f6efe1; --paper-deep: #ede2cc;
    --ink: #1a1612; --ink-soft: #3a322a; --ink-muted: #6b5d4f;
    --terracotta: #c1502e; --terracotta-deep: #8c3a20;
    --teal: #1f5f5b; --teal-deep: #133e3b;
    --mustard: #d4a017; --rose: #b85a6c;
    --pytorch: #ee4c2c;
    --code-bg: #2a2520; --code-bg-light: #f0e8d4;
    --rule: #c8b89a;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; padding: 0; background: var(--paper);
    background-image:
      radial-gradient(circle at 20% 30%, rgba(193,80,46,0.04) 0%, transparent 40%),
      radial-gradient(circle at 80% 70%, rgba(31,95,91,0.04) 0%, transparent 40%);
    color: var(--ink);
    font-family: 'Lora', Georgia, serif;
    font-size: 17px; line-height: 1.7;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 60px 40px 120px; }
  .module-header { border-bottom: 3px double var(--ink); padding-bottom: 32px; margin-bottom: 48px; position: relative; }
  .module-tag {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--terracotta-deep); background: var(--paper-deep);
    padding: 6px 14px; border: 1.5px solid var(--terracotta-deep);
    border-radius: 2px; margin-bottom: 20px;
  }
  .module-title {
    font-family: 'Fraunces', serif; font-weight: 900; font-size: 56px;
    line-height: 1.05; margin: 0 0 14px; color: var(--ink); letter-spacing: -0.02em;
  }
  .module-title em { font-style: italic; font-weight: 400; color: var(--terracotta); }
  .module-sub { font-family: 'Caveat', cursive; font-size: 26px; color: var(--ink-soft); margin: 0; }
  h2 { font-family: 'Fraunces', serif; font-weight: 800; font-size: 32px; margin: 56px 0 18px; color: var(--ink); line-height: 1.15; letter-spacing: -0.01em; }
  h2::before { content: "§ "; color: var(--terracotta); font-weight: 600; }
  h3 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 22px; margin: 36px 0 12px; color: var(--teal-deep); letter-spacing: -0.005em; }
  h4 { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 0.1em; margin: 28px 0 10px; color: var(--ink-soft); }
  p { margin: 0 0 18px; }
  p strong { color: var(--terracotta-deep); font-weight: 600; }
  p em { color: var(--teal-deep); }
  blockquote { border-left: 4px solid var(--terracotta); padding: 4px 0 4px 22px; margin: 24px 0; font-style: italic; color: var(--ink-soft); font-size: 18px; }
  a { color: var(--teal-deep); text-decoration: underline wavy var(--mustard); text-underline-offset: 4px; }
  code { font-family: 'IBM Plex Mono', monospace; font-size: 0.88em; background: var(--code-bg-light); padding: 1px 6px; border-radius: 3px; color: var(--terracotta-deep); border: 1px solid rgba(0,0,0,0.06); }
  pre { background: var(--code-bg); color: #e8d9b8; font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; line-height: 1.55; padding: 22px 26px; border-radius: 4px; overflow-x: auto; margin: 24px 0; border-left: 4px solid var(--pytorch); box-shadow: 4px 4px 0 var(--ink); position: relative; }
  pre::before { content: ">_"; position: absolute; top: 8px; right: 14px; font-size: 11px; color: var(--pytorch); opacity: 0.7; letter-spacing: 0.1em; }
  pre code { background: transparent; color: inherit; padding: 0; border: none; font-size: inherit; }
  pre .kw { color: #ff8a65; }
  pre .str { color: #a5d6a7; }
  pre .num { color: #ffcc80; }
  pre .com { color: #8d7d65; font-style: italic; }
  pre .fn { color: #82b1ff; }
  pre .ty { color: #ce93d8; }
  .sticky { background: #fff8a8; background-image: linear-gradient(180deg, #fff8a8 0%, #ffeb7a 100%); padding: 22px 26px; margin: 28px 0; box-shadow: 3px 4px 12px rgba(0,0,0,0.18), inset 0 -8px 14px rgba(0,0,0,0.04); transform: rotate(-0.4deg); font-family: 'Patrick Hand', cursive; font-size: 19px; line-height: 1.5; color: var(--ink); position: relative; }
  .sticky::before { content: ""; position: absolute; top: -10px; left: 50%; transform: translateX(-50%) rotate(-3deg); width: 80px; height: 22px; background: rgba(193,80,46,0.5); border-radius: 1px; box-shadow: 0 2px 4px rgba(0,0,0,0.15); }
  .sticky.pink { background-image: linear-gradient(180deg, #ffd5dc 0%, #ffb8c4 100%); transform: rotate(0.5deg); }
  .sticky.blue { background-image: linear-gradient(180deg, #d3e9f5 0%, #aed4ec 100%); transform: rotate(-0.6deg); }
  .sticky.green { background-image: linear-gradient(180deg, #d4ecc8 0%, #b3d99e 100%); transform: rotate(0.3deg); }
  .sticky strong { color: var(--terracotta-deep); }
  .brainpower { border: 2.5px solid var(--ink); background: var(--paper-deep); padding: 24px 28px 26px; margin: 32px 0; border-radius: 2px; position: relative; box-shadow: 5px 5px 0 var(--terracotta); }
  .brainpower::before { content: "BRAIN POWER"; position: absolute; top: -13px; left: 22px; background: var(--terracotta); color: #fff; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; padding: 4px 12px; letter-spacing: 0.18em; }
  .brainpower p { margin: 8px 0; font-size: 16px; }
  .brainpower p:first-child { margin-top: 6px; }
  .ndq { background: linear-gradient(180deg, #fdf6e3 0%, #f5ead0 100%); border: 1.5px dashed var(--ink-soft); border-radius: 8px; padding: 28px 30px; margin: 36px 0; }
  .ndq h4 { font-family: 'Fraunces', serif; font-size: 22px; text-transform: none; letter-spacing: 0; color: var(--terracotta-deep); margin: 0 0 18px; border-bottom: 1.5px solid var(--rule); padding-bottom: 8px; }
  .ndq h4::before { content: "Q&A — "; color: var(--ink-muted); font-style: italic; }
  .ndq .q { font-family: 'Caveat', cursive; font-size: 22px; color: var(--teal-deep); margin: 14px 0 4px; line-height: 1.3; }
  .ndq .q::before { content: "Q: "; font-weight: 700; color: var(--terracotta); }
  .ndq .a { margin: 0 0 16px 14px; padding-left: 14px; border-left: 3px solid var(--mustard); color: var(--ink-soft); font-size: 16px; }
  .ndq .a::before { content: "A: "; font-weight: 700; color: var(--ink); font-family: 'Fraunces', serif; }
  .table-wrap { margin: 28px 0; overflow-x: auto; border: 2px solid var(--ink); box-shadow: 4px 4px 0 var(--teal); background: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 14.5px; font-family: 'Lora', serif; }
  table caption { font-family: 'Caveat', cursive; font-size: 22px; color: var(--terracotta-deep); padding: 12px 0; text-align: left; caption-side: top; }
  th { background: var(--ink); color: #fff; text-align: left; padding: 12px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
  td { padding: 12px 14px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  tr:nth-child(even) td { background: var(--paper-deep); }
  tr:last-child td { border-bottom: none; }
  td strong { color: var(--terracotta-deep); }
  td code { font-size: 12.5px; }
  .decision { background: #fff; border: 2px solid var(--ink); border-radius: 2px; padding: 24px 28px; margin: 32px 0; box-shadow: 5px 5px 0 var(--mustard); font-family: 'IBM Plex Mono', monospace; font-size: 14px; line-height: 1.8; white-space: pre; overflow-x: auto; color: var(--ink); }
  .warn { background: #fcecec; border-left: 6px solid var(--terracotta); padding: 18px 22px; margin: 28px 0; border-radius: 0 4px 4px 0; position: relative; }
  .warn::before { content: "⚠"; position: absolute; top: 14px; right: 18px; font-size: 24px; color: var(--terracotta); }
  .warn strong { color: var(--terracotta-deep); }
  .keyidea { background: var(--teal); color: #f6efe1; padding: 26px 30px; margin: 32px 0; border-radius: 2px; font-family: 'Fraunces', serif; font-size: 19px; font-weight: 500; line-height: 1.5; position: relative; box-shadow: 4px 4px 0 var(--ink); }
  .keyidea::before { content: "★ KEY IDEA"; position: absolute; top: -11px; left: 22px; background: var(--mustard); color: var(--ink); font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 700; padding: 4px 10px; letter-spacing: 0.18em; }
  .keyidea strong { color: var(--mustard); }
  .exercise { border: 2px dashed var(--teal-deep); padding: 24px 28px; margin: 36px 0; background: rgba(255,255,255,0.5); border-radius: 2px; }
  .exercise::before { content: "✎ SHARPEN YOUR PENCIL"; display: block; font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; letter-spacing: 0.18em; color: var(--teal-deep); margin-bottom: 14px; padding-bottom: 10px; border-bottom: 2px dotted var(--teal); }
  .exercise .answer { margin-top: 14px; padding: 14px 18px; background: var(--paper-deep); border-radius: 4px; font-size: 15px; }
  .exercise .answer summary { cursor: pointer; font-family: 'Caveat', cursive; font-size: 20px; color: var(--terracotta); }
  ul, ol { margin: 16px 0; padding-left: 28px; }
  li { margin: 8px 0; }
  ul li::marker { color: var(--terracotta); }
  ol li::marker { color: var(--teal-deep); font-weight: 600; }
  .dialogue { margin: 28px 0; padding: 20px 24px; background: var(--paper-deep); border-radius: 4px; border-left: 4px solid var(--rose); }
  .dialogue .speaker { font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--rose); font-weight: 700; display: block; margin-top: 10px; }
  .dialogue .speaker:first-child { margin-top: 0; }
  .dialogue .speaker.b { color: var(--teal-deep); }
  .twocol { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 28px 0; }
  .twocol > div { padding: 20px 22px; border-radius: 2px; background: #fff; border: 1.5px solid var(--rule); }
  .twocol > div h4 { margin-top: 0; }
  .twocol .good { border-color: var(--teal); border-left: 5px solid var(--teal); }
  .twocol .bad { border-color: var(--terracotta); border-left: 5px solid var(--terracotta); }
  .ascii { background: var(--paper-deep); padding: 22px 26px; margin: 28px 0; border: 1.5px solid var(--rule); border-radius: 2px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; line-height: 1.5; white-space: pre; overflow-x: auto; color: var(--ink); }
  .module-footer { margin-top: 80px; padding-top: 28px; border-top: 3px double var(--ink); display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink-muted); }
  .module-footer .num { font-family: 'Fraunces', serif; font-size: 32px; font-weight: 800; color: var(--terracotta); line-height: 1; }
  .toc { background: #fff; border: 2px solid var(--ink); padding: 30px 40px; margin: 40px 0; box-shadow: 6px 6px 0 var(--teal); }
  .toc h3 { font-family: 'Fraunces', serif; font-size: 26px; margin: 0 0 20px; color: var(--ink); }
  .toc ol { columns: 2; column-gap: 40px; padding-left: 24px; }
  .toc li { font-size: 15px; break-inside: avoid; margin: 4px 0; }
  .toc li strong { color: var(--terracotta-deep); }
  .toc .part { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.15em; color: var(--teal-deep); margin: 16px 0 6px; column-span: all; border-bottom: 1px dashed var(--rule); padding-bottom: 4px; }

  /* CODE MAGNETS — fridge-magnet style code snippets to be arranged */
  .magnets { margin: 36px 0; padding: 26px 28px 30px; background: #ede2cc; border: 2px solid var(--ink); border-radius: 4px; box-shadow: 4px 4px 0 var(--teal-deep); position: relative; }
  .magnets::before { content: "✦ CODE MAGNETS"; display: block; font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; letter-spacing: 0.18em; color: var(--teal-deep); margin-bottom: 12px; padding-bottom: 10px; border-bottom: 2px dotted var(--ink-soft); }
  .magnets p { margin: 0 0 18px; font-size: 16px; }
  .magnet-pool { display: flex; flex-wrap: wrap; gap: 10px; padding: 16px; background: #c8c0a8; background-image: linear-gradient(135deg, #c8c0a8 25%, transparent 25%), linear-gradient(225deg, #c8c0a8 25%, transparent 25%), linear-gradient(45deg, #c8c0a8 25%, transparent 25%), linear-gradient(315deg, #c8c0a8 25%, #b5ad95 25%); background-size: 14px 14px; background-position: 7px 0, 7px 0, 0 0, 0 0; border-radius: 6px; box-shadow: inset 0 2px 6px rgba(0,0,0,0.2); margin: 12px 0; min-height: 70px; }
  .magnet { display: inline-block; background: #fff; padding: 10px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink); border-radius: 3px; box-shadow: 2px 2px 4px rgba(0,0,0,0.25), inset 0 -2px 0 rgba(0,0,0,0.05); border: 1px solid #d8d0b4; transform: rotate(-1deg); transition: transform 0.15s; cursor: default; user-select: all; }
  .magnet:nth-child(2n) { transform: rotate(1.5deg); background: #fff8e8; }
  .magnet:nth-child(3n) { transform: rotate(-2deg); background: #f0f5ee; }
  .magnet:nth-child(5n) { transform: rotate(0.8deg); background: #f5e8e0; }
  .magnet:hover { transform: rotate(0) scale(1.05); z-index: 2; }
  .magnets .answer { margin-top: 16px; padding: 14px 18px; background: #fff8d8; border-radius: 4px; font-size: 15px; border-left: 4px solid var(--mustard); }
  .magnets .answer summary { cursor: pointer; font-family: 'Caveat', cursive; font-size: 20px; color: var(--terracotta); }
  .magnets .answer pre { margin: 12px 0 0; }

  /* CHARACTER — anthropomorphized objects with avatar and speech */
  .character { margin: 30px 0; padding: 20px 22px 22px; background: #fff; border: 2px solid var(--ink); border-radius: 4px; box-shadow: 4px 4px 0 var(--rose); position: relative; display: grid; grid-template-columns: 80px 1fr; gap: 18px; align-items: start; }
  .character .avatar { width: 72px; height: 72px; border-radius: 50%; background: var(--mustard); display: flex; align-items: center; justify-content: center; font-family: 'Fraunces', serif; font-weight: 900; font-size: 30px; color: var(--ink); border: 2.5px solid var(--ink); box-shadow: 2px 2px 0 var(--ink); }
  .character.tensor .avatar { background: var(--pytorch); color: #fff; }
  .character.stride .avatar { background: var(--teal); color: #fff; }
  .character.storage .avatar { background: var(--mustard); }
  .character.dtype .avatar { background: var(--rose); color: #fff; }
  .character.autograd .avatar { background: var(--terracotta); color: #fff; }
  .character .who { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.15em; color: var(--ink-muted); font-weight: 700; margin: 0 0 4px; }
  .character .name { font-family: 'Fraunces', serif; font-weight: 800; font-size: 22px; color: var(--ink); margin: 0 0 8px; line-height: 1; }
  .character .says { font-family: 'Lora', serif; font-size: 16px; line-height: 1.55; color: var(--ink); margin: 0; font-style: italic; }
  .character .says::before { content: '"'; font-family: 'Fraunces', serif; font-size: 28px; color: var(--terracotta); line-height: 0; vertical-align: -8px; margin-right: 4px; }
  .character .says::after { content: '"'; font-family: 'Fraunces', serif; font-size: 28px; color: var(--terracotta); line-height: 0; vertical-align: -8px; margin-left: 2px; }
  .character + .character { margin-top: -10px; }

  /* BULLET POINTS — recap section with checklist vibe */
  .bullet-points { margin: 48px 0 36px; padding: 32px 36px 36px; background: var(--paper-deep); border: 3px solid var(--ink); border-radius: 4px; position: relative; box-shadow: 6px 6px 0 var(--mustard); }
  .bullet-points::before { content: "BULLET POINTS"; position: absolute; top: -16px; left: 28px; background: var(--ink); color: var(--mustard); font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; padding: 6px 16px; letter-spacing: 0.2em; }
  .bullet-points h3 { font-family: 'Caveat', cursive; font-size: 28px; color: var(--terracotta); margin: 0 0 16px; }
  .bullet-points ul { list-style: none; padding-left: 0; margin: 0; }
  .bullet-points ul li { position: relative; padding: 6px 0 6px 38px; font-size: 16px; line-height: 1.5; border-bottom: 1px dashed var(--rule); }
  .bullet-points ul li:last-child { border-bottom: none; }
  .bullet-points ul li::before { content: ""; position: absolute; left: 6px; top: 11px; width: 18px; height: 18px; border: 2px solid var(--terracotta); border-radius: 3px; background: #fff; }
  .bullet-points ul li::after { content: "✓"; position: absolute; left: 9px; top: 6px; font-family: 'Caveat', cursive; font-size: 22px; font-weight: 700; color: var(--terracotta); }
  .bullet-points ul li strong { color: var(--terracotta-deep); }
  .bullet-points ul li code { font-size: 13px; }

  /* MATCHING — "Who Does What?" two-column matching exercise */
  .matching { margin: 36px 0; padding: 26px 28px 28px; background: #fff; border: 2px solid var(--ink); border-radius: 4px; box-shadow: 4px 4px 0 var(--teal); position: relative; }
  .matching::before { content: "✦ WHO DOES WHAT?"; display: block; font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 700; letter-spacing: 0.18em; color: var(--teal-deep); margin-bottom: 6px; padding-bottom: 10px; border-bottom: 2px dotted var(--ink-soft); }
  .matching p.intro { margin: 8px 0 18px; font-size: 15px; font-style: italic; color: var(--ink-soft); }
  .match-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; border: 1.5px solid var(--ink); }
  .match-grid > div { padding: 12px 14px; border-bottom: 1px solid var(--rule); font-size: 14.5px; vertical-align: middle; display: flex; align-items: center; }
  .match-grid > div:nth-child(4n+1), .match-grid > div:nth-child(4n+2) { background: var(--paper-deep); }
  .match-grid > div:nth-child(2n+1) { font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: var(--terracotta-deep); border-right: 1.5px dashed var(--ink); }
  .match-grid > div:nth-child(2n) { color: var(--ink); }
  .match-grid > div:nth-last-child(-n+2) { border-bottom: none; }
  .match-grid .header { background: var(--ink) !important; color: #fff !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; padding: 10px 14px; border-bottom: none; }
  .matching .answer { margin-top: 16px; padding: 14px 18px; background: var(--paper-deep); border-radius: 4px; font-size: 15px; }
  .matching .answer summary { cursor: pointer; font-family: 'Caveat', cursive; font-size: 20px; color: var(--terracotta); }

  @media (max-width: 720px) {
    .wrap { padding: 40px 22px 80px; }
    .module-title { font-size: 40px; }
    .twocol { grid-template-columns: 1fr; }
    .toc ol { columns: 1; }
    .character { grid-template-columns: 56px 1fr; gap: 12px; }
    .character .avatar { width: 56px; height: 56px; font-size: 22px; }
    .match-grid { grid-template-columns: 1fr 1fr; font-size: 13px; }
  }
</style>
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{styles}
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""


def html_to_md(body: str, title: str) -> str:
    src = body
    def strip_spans(m):
        inner = m.group(0)
        inner = re.sub(r'<span class="[^"]*">', '', inner)
        return inner.replace('</span>', '')
    src = re.sub(r'<pre[^>]*>.*?</pre>', strip_spans, src, flags=re.S)

    src = re.sub(
        r'<div class="module-header">\s*<div class="module-tag">([^<]+)</div>\s*'
        r'<h1 class="module-title">(.*?)</h1>\s*<p class="module-sub">(.*?)</p>\s*</div>',
        lambda m: f'<h1>{m.group(2)}</h1>\n<p><em>{m.group(1)}</em></p>\n<p>{m.group(3)}</p>\n\n---\n',
        src, flags=re.S,
    )

    callouts = [
        (r'<div class="keyidea"[^>]*>(.*?)</div>', '★ KEY IDEA'),
        (r'<div class="sticky[^"]*"[^>]*>(.*?)</div>', '📝 NOTE'),
        (r'<div class="warn"[^>]*>(.*?)</div>', '⚠ WARNING'),
        (r'<div class="brainpower"[^>]*>(.*?)</div>', '🧠 BRAIN POWER'),
    ]
    for pat, label in callouts:
        src = re.sub(
            pat,
            lambda m, l=label: f'\n<blockquote><strong>{l}</strong><br>{m.group(1)}</blockquote>\n',
            src, flags=re.S,
        )

    def ndq_block(m):
        inner = m.group(1)
        inner = re.sub(r'<h4>([^<]*)</h4>', r'\n#### Q&A — \1\n', inner)
        inner = re.sub(r'<p class="q">(.*?)</p>', r'\n**Q:** \1\n', inner, flags=re.S)
        inner = re.sub(r'<p class="a">(.*?)</p>', r'\n**A:** \1\n', inner, flags=re.S)
        return f'\n{inner}\n'
    src = re.sub(r'<div class="ndq"[^>]*>(.*?)</div>', ndq_block, src, flags=re.S)

    def ex_block(m):
        inner = m.group(1)
        inner = re.sub(
            r'<details[^>]*class="answer"[^>]*>\s*<summary>([^<]+)</summary>(.*?)</details>',
            r'\n\n<details><summary>\1</summary>\2</details>\n',
            inner, flags=re.S,
        )
        return f'\n> **✎ SHARPEN YOUR PENCIL**\n>\n> {inner}\n'
    src = re.sub(r'<div class="exercise"[^>]*>(.*?)</div>', ex_block, src, flags=re.S)

    # Replace ascii/decision divs with <pre> so html2text formats them as code blocks
    # (preserves newlines, unlike a stripped div).
    src = re.sub(
        r'<div class="decision"[^>]*>(.*?)</div>',
        lambda m: '<pre>' + re.sub(r'<[^>]+>', '', m.group(1)).strip('\n') + '</pre>',
        src, flags=re.S,
    )
    src = re.sub(
        r'<div class="ascii"[^>]*>(.*?)</div>',
        lambda m: '<pre>' + re.sub(r'<[^>]+>', '', m.group(1)).strip('\n') + '</pre>',
        src, flags=re.S,
    )

    def twocol(m):
        inner = m.group(1)
        return re.sub(
            r'<div class="(good|bad)">(.*?)</div>',
            lambda mm: f'\n**{"DO" if mm.group(1)=="good" else "DON\'T"}**\n\n{mm.group(2)}\n',
            inner, flags=re.S,
        )
    src = re.sub(r'<div class="twocol"[^>]*>(.*?)</div>', twocol, src, flags=re.S)

    src = re.sub(
        r'<div class="dialogue"[^>]*>(.*?)</div>',
        lambda m: re.sub(r'<span class="speaker[^"]*">([^<]+)</span>', r'\n**\1:** ', m.group(1)),
        src, flags=re.S,
    )

    src = re.sub(r'<div class="module-footer">.*?</div>', '', src, flags=re.S)

    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.bypass_tables = False
    h.use_automatic_links = True
    h.unicode_snob = True
    md = h.handle(src)

    md = f"# {title}\n\n{md.strip()}\n"
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md


def emit(slug: str, title: str, body: str):
    html = TEMPLATE.format(title=title, styles=STYLES, body=body)
    (OUT_HTML / f"{slug}.html").write_text(html)
    md_text = html_to_md(body, title)
    (OUT_MD / f"{slug}.md").write_text(md_text)
    print(f"  emitted {slug}: {len(html):,} html / {len(md_text):,} md")
