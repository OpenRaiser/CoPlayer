# CoPlay — project page

Source of the project webpage for **CoPlay: Learning When to Speak and What to Say as
Game Companions** (CVPR 2027 submission, under review).

This branch (`gh-pages`) holds only the built, self-contained page and its assets. It is
the branch GitHub Pages serves; no source code or paper material lives here.

## What is in here

| path | what it is |
|---|---|
| `index.html` | the whole page: markup, CSS and JS inline, no build step, no runtime dependency |
| `assets/figures/` | the paper's figures, converted to PNG + WebP, with `figures.manifest.json` recording source, size and alt text |
| `assets/fonts/` | subset WOFF2 of TeX Gyre Termes (the serif used in the paper's own figures), Work Sans and JetBrains Mono, with their licences |
| `assets/brand/` | the CoPlayer mark, used as the favicon and in the header |
| `assets/social-card.png` | 1200x630 Open Graph card |
| `.nojekyll` | tells Pages to serve the files as-is |

## How it was produced

Every table on the page is generated from the paper's LaTeX source rather than retyped:
each cell is checked against the extracted table ledger and every numeric token against
the original `tabular`, and the generator refuses to emit output on any mismatch. The two
recorded decision timelines are redrawn natively from the authors' own figure source.

## Notes

- The page requires no network at runtime: there is no CDN, webfont service or analytics,
  and it renders correctly from the filesystem. It also reads completely with JavaScript
  disabled.
- The paper is an anonymous submission under review, so the page names no authors or
  institutions, states no code or data release, and carries a draft BibTeX entry.
- The review-copy PDF is deliberately **not** published here: every page of it is stamped
  "CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE."
