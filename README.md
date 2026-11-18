# CoPlay — project page

Built page for **CoPlay: Learning When to Speak and What to Say as Game Companions**.

This branch (`gh-pages`) is what GitHub Pages serves. It holds only the page and its
assets; no source code and no paper material live here.

## Two hero variants

Everything below the hero is identical in both; only the first screen differs.

| | file | first screen |
|---|---|---|
| **A** | [`index.html`](index.html) | a draggable recorded decision moment: the real model input, the required speaking window, and what each model said at that step |
| **B** | [`hero-b.html`](hero-b.html) | a two-mode diptych: the player asks on the left, nobody asks on the right, each with a real case |

A switcher in the bottom-right corner moves between them. It is there for review and
should be removed once one variant is chosen.

## What is in here

| path | what it is |
|---|---|
| `index.html`, `hero-b.html` | the whole page: markup, CSS and JavaScript inline, no build step, no runtime dependency |
| `assets/figures/` | the paper's figures and recorded model-input frames as PNG/JPEG + WebP, with `figures.manifest.json` recording source, size and alt text |
| `assets/icons/` | the provider marks the paper puts above its leaderboard bars |
| `assets/fonts/` | subset WOFF2 of TeX Gyre Termes (the serif in the paper's own figures), Work Sans and JetBrains Mono, with their licences |
| `assets/brand/` | the CoPlayer mark, used as the favicon and in the header |
| `assets/social-card.png` | 1200x630 Open Graph card |

## How it was produced

All 11 of the paper's tables are generated from its LaTeX source rather than retyped:
each cell is checked against the extracted table ledger and every numeric token against
the original `tabular`, and the generator refuses to emit output on any mismatch. The two
recorded decision timelines are recovered from the authors' own figure source and redrawn
natively, so the marks are that video's recorded decision steps. The leaderboard, the
per-game pace chart, the recall scatter and the rank matrix are drawn from the measured
values, not pasted as images.

## Notes

- No network at runtime: no CDN, webfont service or analytics. The page renders from the
  filesystem and reads completely with JavaScript disabled, including all four case
  studies and every table; the interactive parts are enhancements.
- No horizontal scrolling anywhere from 360 to 1920 px. The two widest tables and the rank
  matrix split into complete per-group sub-tables on narrow screens, so no value is hidden.
- Every text/background pair clears WCAG AA, measured with translucent layers composited.
- The manuscript is anonymous and states no code, data or model release, so the page names
  no authors or institutions and carries a draft BibTeX entry.
