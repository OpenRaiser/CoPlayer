# CoPlay — project page

Built page for **CoPlay: Learning When to Speak and What to Say as Game Companions**.

**https://openraiser.github.io/CoPlayer/**

This branch (`gh-pages`) is what GitHub Pages serves. It holds only the static page and its web-optimized media assets.

## Structure

The page is organized into five focused acts:

1. **Hero & Live Proactive Stage** — project thesis, headline KPIs, and an interactive *Plants vs. Zombies 2* proactive intervention clip synced to recorded decision steps.
2. **Overview & CoPlayBench** — reactive vs. proactive formulation, consensus annotation funnel, the 92.6% silence principle, a 9-game visual rhythm grid, and the two headline metrics (`StratEmo` and `TimingLift`).
3. **CoPlayer Method** — shared supervised backbone, track-specific GRPO policies, and a visual decision-first reward priority scale.
4. **Headline Results & Diagnostics** — interactive 5-metric leaderboard, `Recall` vs. `SpeakRate` scatter, collapsible full benchmark tables (`Table 2`, `Table 3`), and honest RL trade-offs / 2-second live deadline analysis.
5. **Multi-Game Live Cases & Citation** — five playable benchmark clips (*Honor of Kings*, *Dou Dizhu*, *Stardew Valley*, *Hearthstone*, *Carrot Fantasy*) comparing CoPlayer, GPT-5.6-Sol, and DeepSeek-V4.1-Flash on identical inputs, followed by BibTeX citation.

## Assets

| path | contents |
|---|---|
| `index.html` | complete static page with inline CSS and progressive-enhancement JS (zero external CDN or runtime dependencies) |
| `assets/videos/` | web-optimized `.webm` + `.mp4` gameplay clips and poster frames around annotated windows |
| `assets/games/` | 16:9 first-person gameplay thumbnails across all nine benchmark games |
| `assets/figures/` | paper figures and HUD detail insets (`PNG`/`JPEG` + `WebP`) with `figures.manifest.json` |
| `assets/icons/` | model family icons used in the interactive leaderboard and live companion cards |
| `assets/fonts/` | subset `WOFF2` fonts (`Outfit`, `Work Sans`, `JetBrains Mono`) and licences |
| `assets/brand/` | CoPlay project mark / favicon |
| `assets/social-card.png` | `1200x630` Open Graph preview card |
