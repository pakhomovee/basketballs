# Clip data

Each clip lives in its own folder here (`nba-03/`, `nba-17/`, …) and is listed
in `manifest.json`. A folder is expected to contain:

- `annotations.json` — the pipeline's exported overlay data (committed).
- `video.mp4` — the muted source clip (**not committed**).

The broadcast `video.mp4` files are intentionally kept out of the repository
because they are third-party copyrighted footage. Supply them at deploy time by
placing each clip's `video.mp4` into its folder on the host (e.g. upload to
`public_html/data/<clip>/video.mp4`). The viewer resolves the paths listed in
`manifest.json`, so no code changes are needed.
