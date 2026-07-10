# Basketballs — Demo Web Viewer

A **zero-build, read-only** website that plays pre-annotated basketball clips with
an interactive canvas overlay of the analysis pipeline's outputs — bounding boxes,
segmentation masks, pose skeletons, the ball, track/jersey numbers, player IDs —
plus a game-state sidebar (possession, passes, shots). Every overlay layer can be
toggled on and off.

There is **no backend and no build step** — it is plain HTML/CSS/JS served as
static files.

> This `demo-website` branch contains only the standalone viewer. The full
> computer-vision pipeline that produces the annotations lives on the `main`
> branch.

## Run it

Because the page fetches JSON, open it through any static file server (not
`file://`):

```bash
python3 -m http.server 8080
# open http://localhost:8080/
```

That's it — no npm, no application server.

## Layout

```
.
  index.html          # landing: overview + disclaimer + clip gallery
  player.html         # the viewer (video + canvas + toggles + sidebar)
  css/style.css
  js/
    draw.js           # canvas rendering of every overlay layer
    home.js           # builds the gallery from data/manifest.json
    player.js         # video↔annotation frame sync, controls, game state
  data/
    manifest.json     # list of clips shown on the landing page
    nba-03/
      video.mp4         # muted broadcast clip
      annotations.json  # pipeline output for that clip
    nba-17/ …
    nba-20/ …
```

## Adding a clip

Drop a muted `video.mp4` and its `annotations.json` into a new folder under
`data/`, then add an entry to `data/manifest.json`:

```json
{
  "id": "nba-42",
  "title": "NBA Clip 42",
  "description": "…",
  "video": "data/nba-42/video.mp4",
  "annotations": "data/nba-42/annotations.json",
  "available": true
}
```

The `annotations.json` uses the pipeline's export schema: a `metadata` block,
a `frames` map keyed by frame index (`players`, `balls`), and `pass_events` /
`shot_events` arrays. Set `"available": false` to show a clip as *Coming soon*
(not clickable) until its files are in place.

## Disclaimer

The clips are provided strictly for research/educational purposes to showcase the
pipeline. They are muted, used non-commercially, and must not be redistributed.
