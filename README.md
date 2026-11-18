# CoPlay — project page

Built page for **CoPlay: Learning When to Speak and What to Say as Game Companions**.

**https://openraiser.github.io/CoPlayer/**

This branch (`gh-pages`) is what GitHub Pages serves. It holds only the page and its
assets; no source code and no paper material live here.

## Structure

The page follows the pattern of the other project pages in this family: a wordmark header
with section anchors, a full-bleed themed hero with a centred title, pill links and a stat
strip, then hairline-separated section blocks — Problem, Benchmark, Method, Results, Cases,
Audit, Citation. Every table sits in a `.table-module` whose large insight line states what
that table proves, with the paper's own caption below it.

## What is in here

| path | what it is |
|---|---|
| `index.html` | the whole page: markup, CSS and JavaScript inline, no build step, no runtime dependency |
| `assets/figures/` | the paper's figures and recorded model-input frames as PNG/JPEG + WebP, with `figures.manifest.json` recording source, size and alt text |
| `assets/icons/` | the provider marks the paper puts above its leaderboard bars |
| `assets/fonts/` | subset WOFF2 of TeX Gyre Termes (the serif in the paper's own figures), Work Sans and JetBrains Mono, with their licences |
| `assets/brand/` | the CoPlayer mark, used as the favicon and in the header |
| `assets/social-card.png` | 1200x630 Open Graph card |

## How it was produced

All 11 of the paper's tables are generated from its LaTeX source rather than retyped: each
cell is checked against the extracted table ledger and every numeric token against the
original `tabular`, and the generator refuses to emit output on any mismatch. The two
recorded decision timelines are recovered from the authors' own figure source and redrawn
natively, so the marks are that video's recorded decision steps. The leaderboard, the
per-game pace chart, the recall scatter and the rank matrix are drawn from the measured
values rather than pasted as images.

## Notes

- No network at runtime: no CDN, webfont service or analytics. The page renders from the
  filesystem and reads completely with JavaScript disabled — all 11 tables, all four case
  studies, every figure as a link; the interactive parts are enhancements.
- No horizontal scrolling at any width from 360 to 1920 px. Wide tables take a row-card
  form (each cell carries its column name) below 1040 px, so no value is ever hidden.
- Every text/background pair clears WCAG AA, measured with translucent layers composited.
- The manuscript is anonymous and states no code, data or model release, so the page names
  no authors or institutions and carries a draft BibTeX entry.
