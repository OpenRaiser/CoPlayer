# CoPlayer

Project page for **CoPlay: Learning When to Speak and What to Say as Game Companions**.

A game companion watches the same screen as the player and has to decide both what to say
and whether to speak at all. Question-driven video benchmarks pose the question, so they
score the answer and never the decision to intervene. CoPlayBench scores that decision:
818 first-person videos from nine games carry 5,153 event-level annotations, each with a
speaking window, success criteria and a reference reply. CoPlayer, trained with shared
supervised fine-tuning followed by track-specific reinforcement learning, pairs sound
advice with fitting support on 34.3% of advice questions, against 14.5% for the best of
eight frontier models tested, and still fails about two thirds of advice questions.

## The page

**https://openraiser.github.io/CoPlayer/**

The built page lives on the [`gh-pages`](../../tree/gh-pages) branch, which is what GitHub
Pages serves. This `main` branch holds nothing but this README, so the page and its assets
stay isolated on their own branch.

## About the page

- One self-contained `index.html`: markup, CSS and JavaScript inline, no build step and no
  runtime dependency. No CDN, webfont service or analytics, so it renders correctly from
  the filesystem and reads completely with JavaScript disabled.
- All 11 of the paper's tables are generated from its LaTeX source rather than retyped.
  Each cell is checked against the extracted table ledger and every numeric token against
  the original `tabular`; the generator refuses to emit output on any mismatch.
- The two recorded proactive decision timelines are redrawn as native HTML from the
  authors' own figure source, so the marks are that video's recorded decision steps.
- No horizontal scrolling at any width from 360 to 1440 px: the two widest tables switch to
  complete per-group sub-tables on narrow screens, so no value is ever hidden.

## Status

The paper is an anonymous submission under review, so the page names no authors or
institutions, states no code, data or model release, and carries a draft BibTeX entry with
its pending fields flagged. The confidential review-copy PDF is not published here.
