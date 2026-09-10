# CloudView

A lightweight, dependency-free (Three.js only) 3D point cloud viewer and annotation tool, driven live from Python — no Jupyter, no notebook widgets, just a local browser tab your script updates in place.

[![Donate](https://img.shields.io/badge/Donate-PayPal-00457C?logo=paypal&logoColor=white)](https://paypal.com/paypalme/SAMBITMOHAPATRA1206)

## Features

- Live viewer: creating a `CloudViewPy()` opens one browser tab per instance; every `add_point_cloud()` / `add_boxes()` / `select_box()` call updates that same tab in place, so you can stream frames straight from a loop.
- 3D bounding-box annotation: add, move, rotate, and scale boxes; assign a class label to the points currently inside a box; export both box-wise and point-wise CSV labels.
- Per-box glow highlighting (solid colors or an animated rainbow "liquid glass" effect) to visually track which points belong to which box.
- Per-layer coloring by any feature channel (Turbo / Gradient / Rainbow colormaps), with a live Min/Max filter.
- Click-to-inspect: click any point to see its full feature vector.
- Built-in Scale/Measure tool: click two points to get an on-screen arrow and the Δx/Δy/Δz + total distance.
- Ego-pose trajectory plotting: feed it a list of 4x4 poses and it draws the corresponding axis triads along the path.
- Handles large clouds (500k+ points) via a binary (base64 float32) transfer instead of JSON.
- `export_html()` bakes everything added so far into a single standalone, offline HTML file.

## Install

Copy the `cloud_view/` package into your project (it's just two files: `cloud_view.py` + `cloud_view.html` — keep them together, the `.py` loads the `.html` as its template).

Requires `numpy`. No other dependencies (Three.js is loaded from a CDN inside the HTML template).

## Quick start

```python
import numpy as np
from cloud_view import CloudViewPy

viewer = CloudViewPy()  # opens a browser tab on first call below

# points: (N, 3+) array — first 3 columns must be x, y, z
points = np.random.uniform(-5, 5, size=(10000, 4)).astype(np.float32)
viewer.add_point_cloud(points, names="frame_0", field_names=["x", "y", "z", "intensity"])

# add a labeled 3D box: (x, y, z, l, w, h, rx, ry, rz)
viewer.add_boxes([(0, 0, 0, 3, 3, 3, 0, 0, 0)], uids=[1], labels=["car"])
viewer.select_box(1, color="orange")
```

Call `add_point_cloud()` / `add_boxes()` again (e.g. in a loop, one call per frame) to update the same tab live.

To save a standalone, shareable snapshot instead of / in addition to the live viewer:

```python
viewer.export_html("frame_001.html")
```

## Support

I built and maintain CloudView on my own time, and it's taken a lot of evenings and weekends to get it to where it is now. If it's useful to you, please consider supporting its development — donations go directly toward the time I spend adding user-requested features, fixing bugs, and keeping it maintained.

👉 [Donate via PayPal](https://paypal.com/paypalme/SAMBITMOHAPATRA1206)

Even if you can't donate, starring the repo, opening issues, or suggesting features helps too — thank you for using CloudView!

## License

MIT — see [LICENSE](LICENSE).
