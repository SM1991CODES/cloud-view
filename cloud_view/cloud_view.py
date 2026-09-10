import base64
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

import numpy as np

# Common named colors accepted by select_box(), mapped to the hex values used by the viewer's
# glow-color picker. A "#rrggbb" string can also be passed directly and bypasses this table.
NAMED_COLORS = {
    "red": "#ff3333",
    "green": "#33ff33",
    "blue": "#3388ff",
    "orange": "#ff8800",
    "yellow": "#ffe600",
    "magenta": "#ff00ff",
    "cyan": "#00e5ff",
    "purple": "#8800ff",
    "pink": "#ff66cc",
    "white": "#ffffff",
    "black": "#000000",
    "gray": "#888888",
    "grey": "#888888",
    "lime": "#33ff33",
    "teal": "#008080",
    "brown": "#8b4513",
    "navy": "#000080",
    "gold": "#ffd700",
    "violet": "#ee82ee",
    "indigo": "#4b0082",
    # Not a real hex color: the viewer recognizes this sentinel and renders the box's glowing
    # points with an animated, shimmering liquid-glass rainbow effect instead of a flat color.
    "rainbow": "rainbow",
}


def _encode_float32_b64(array_like):
    """(N, K) array-like -> (base64 string, N, K), stored as row-major float32 bytes."""
    arr = np.asarray(array_like, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D (N, K) array, got shape {arr.shape}.")
    n, k = arr.shape
    b64 = base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")
    return b64, n, k


def _make_request_handler(viewer):
    """Builds a BaseHTTPRequestHandler bound to this specific CloudViewPy instance/server."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep the user's console quiet; nothing here is an error

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._serve_html()
            elif path == "/poll":
                self._serve_poll()
            else:
                self.send_error(404)

        def _serve_html(self):
            body = viewer._template_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_poll(self):
            # Plain request/response, not a long-lived stream: the page calls this every ~1.5s
            # asking "anything past event #N?" — deliberately simple and stateless per-request so
            # there's nothing persistent for a proxy/antivirus/extension to interfere with.
            query = parse_qs(urlsplit(self.path).query)
            try:
                since = int(query.get("since", ["0"])[0])
            except ValueError:
                since = 0

            with viewer._lock:
                history_snapshot = list(viewer._history)

            since = max(0, min(since, len(history_snapshot)))
            new_events = history_snapshot[since:]

            first_poll = not viewer._client_connected.is_set()
            viewer._client_connected.set()
            if first_poll:
                viewer._log(f"first /poll received; {len(history_snapshot)} event(s) available")
            if new_events:
                viewer._log(f"/poll?since={since} returning {len(new_events)} new event(s)")

            body = json.dumps({"events": new_events, "total": len(history_snapshot)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


class CloudViewPy:
    """
    A live 3D point cloud / box-annotation viewer. Creating an instance opens one browser tab
    (one "figure") backed by a small local server; every add_point_cloud()/add_boxes()/select_box()
    call on that instance updates the SAME tab live, in place — no new tabs, no page reloads, no
    re-sending data that's already there. Create a second CloudViewPy() for a second, independent
    figure/tab.
    """

    def __init__(self, host="127.0.0.1", port=0, auto_open=True, template_path=None, connect_timeout=10,
                 post_connect_delay=2.0, debug=False):
        """
        Cheap and side-effect-free: no server is started and no browser tab is opened yet. Both
        happen lazily on your first add_point_cloud()/add_boxes() call (see _ensure_started()) —
        so the viewer appears exactly when you first give it something to show, not before, and
        every entry point (not just __init__) benefits from the same "wait for the tab to
        actually connect" logic below.

        :param host: Interface to bind the local server to. Default: loopback only.
        :param port: Port to listen on. 0 (default) lets the OS pick a free one, so multiple
                      instances never collide.
        :param auto_open: If True, opens the viewer in the default web browser on first use, and
                           that first call blocks (up to `connect_timeout` seconds) until the tab
                           actually connects. Without this wait, a script that calls
                           add_point_cloud()/add_boxes() and then exits shortly after can race the
                           browser: opening a tab, loading Three.js, and connecting all take real
                           time, and if the script (and its server thread) is gone before that
                           finishes, the earliest calls are lost.
        :param template_path: Path to the viewer's HTML template. Defaults to 'cloud_view.html'
                               in the same directory as this script.
        :param connect_timeout: Max seconds to wait for the browser tab to connect when
                                 auto_open=True. Only a cap on startup latency, not a hard
                                 requirement — if it's exceeded (slow machine, blocked popup,
                                 etc.) execution proceeds anyway; events broadcast before a late
                                 connection still reach it via history replay once it does.
        :param post_connect_delay: Extra seconds to block after the browser tab's connection is
                                    detected, before returning control (default 2.0). The socket
                                    connecting doesn't guarantee the page's JS has finished
                                    initializing (scene/camera/etc.) and is actually ready to
                                    handle the first broadcast — this is a blunt safety margin for
                                    that gap. Set to 0 to disable.
        :param debug: If True, prints diagnostics about client connections, live pushes, and
                       disconnects (with the reason) — useful when updates aren't reaching the tab.
        """
        if template_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.template_path = os.path.join(current_dir, "cloud_view.html")
        else:
            self.template_path = template_path

        if not os.path.exists(self.template_path):
            raise FileNotFoundError(
                f"Template HTML file not found at: {self.template_path}. "
                "Ensure 'cloud_view.html' is placed in the same directory as this script."
            )

        with open(self.template_path, "r", encoding="utf-8") as f:
            self._template_html = f.read()

        self._host = host
        self._port_requested = port
        self._auto_open = auto_open
        self._connect_timeout = connect_timeout
        self._post_connect_delay = post_connect_delay
        self._debug = debug

        self.layers = []
        self.boxes = []
        self._next_box_uid = 1
        self.ego_poses = []
        self._next_pose_index = 0

        # Every add_point_cloud()/add_boxes()/select_box() call is recorded here (as the exact
        # JSON line the page will eventually poll for), so any tab — fresh, reconnecting, or
        # refreshed — can always catch up to full current state via GET /poll?since=<n>.
        self._history = []
        self._lock = threading.Lock()
        self._client_connected = threading.Event()  # set on the first /poll a tab ever makes

        # Server/URL state — filled in by _ensure_started() on first use.
        self._server = None
        self._server_thread = None
        self._start_lock = threading.Lock()
        self._started = False
        self.port = None
        self.url = None

    def _ensure_started(self):
        """Starts the server (once) and, on that same first call, opens the browser and waits for
        it to connect. Safe to call every time — a no-op after the first time."""
        if self._started:
            return

        with self._start_lock:
            if self._started:
                return

            handler_cls = _make_request_handler(self)
            self._server = ThreadingHTTPServer((self._host, self._port_requested), handler_cls)
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()

            self.port = self._server.server_address[1]
            self.url = f"http://{self._host}:{self.port}/"
            self._started = True

        if self._auto_open:
            webbrowser.open(self.url)
            connected = self._client_connected.wait(timeout=self._connect_timeout)
            if not connected:
                print(
                    f"CloudViewPy: no browser tab connected to {self.url} within "
                    f"{self._connect_timeout}s - continuing anyway, but any add_point_cloud()/"
                    f"add_boxes() calls made before a tab connects will only show up once one "
                    f"does (open {self.url} manually if it didn't appear)."
                )
            elif self._post_connect_delay > 0:
                time.sleep(self._post_connect_delay)

    def _log(self, message):
        """Diagnostic logging, off unless the instance was created with debug=True."""
        if self._debug:
            print(f"[CloudViewPy] {message}", flush=True)

    def _broadcast(self, event: dict):
        line = json.dumps(event)
        with self._lock:
            self._history.append(line)
            total = len(self._history)
        self._log(f"recorded '{event.get('type')}' (history now has {total} event(s), picked up on next poll)")

    def add_point_cloud(self, clouds, names, initial_cols=None, initial_point_sizes=None, field_names=None):
        """
        Adds one or more point cloud layers to the already-open viewer, live.

        :param clouds: A single (N, K) NumPy array/list, or a list of such arrays — one per layer.
                        Columns 0, 1, 2 must be X, Y, Z, followed by any number of feature channels.
        :param names: A single label string, or a list of label strings (one per cloud in `clouds`,
                       used as the layer name and as the frame/point identifier shown in the UI).
        :param initial_cols: Index of the feature channel to color-map by default. Either a single
                              int applied to every layer, or a list matching `clouds` one-to-one.
        :param initial_point_sizes: Initial point rendering size in Three.js units. Either a single
                                     float applied to every layer, or a list matching `clouds`.
        :param field_names: Names for each column (e.g. ["x", "y", "z", "intensity", "range"]),
                             shown in the "Color By Channel" dropdown instead of generic labels.
                             Either a single flat list applied to every cloud (each must then have
                             that many columns), or a list of lists — one names-list per cloud,
                             matching that cloud's own column count. Omit to keep the default
                             X/Y/Z + "Feature Channel N" labels.
        """
        self._ensure_started()

        if isinstance(clouds, np.ndarray) or (
            len(clouds) > 0 and not isinstance(clouds[0], (list, tuple, np.ndarray))
        ):
            clouds = [clouds]
        if isinstance(names, str):
            names = [names]

        if len(clouds) != len(names):
            raise ValueError(
                f"clouds and names must have the same length (got {len(clouds)} clouds, {len(names)} names)."
            )

        n = len(clouds)

        def _broadcast_value(value, default):
            if value is None:
                return [default] * n
            if isinstance(value, (list, tuple)):
                if len(value) != n:
                    raise ValueError(f"Expected {n} values, got {len(value)}.")
                return list(value)
            return [value] * n

        def _resolve_field_names(value):
            if value is None:
                return [None] * n
            is_nested = (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and isinstance(value[0], (list, tuple))
            )
            if is_nested:
                if len(value) != n:
                    raise ValueError(
                        f"field_names must have {n} entries (one per cloud) when given as a list "
                        f"of lists, got {len(value)}."
                    )
                return [list(fn) if fn is not None else None for fn in value]
            # A single flat list of names, broadcast to every cloud (they must all share that many columns).
            return [list(value)] * n

        cols = _broadcast_value(initial_cols, 3)
        sizes = _broadcast_value(initial_point_sizes, 0.08)
        field_names_list = _resolve_field_names(field_names)

        for data, name, col, size, names_for_cloud in zip(clouds, names, cols, sizes, field_names_list):
            data_b64, num_points, num_channels = _encode_float32_b64(data)

            if names_for_cloud is not None and len(names_for_cloud) != num_channels:
                raise ValueError(
                    f"field_names for cloud '{name}' has {len(names_for_cloud)} names but the "
                    f"cloud has {num_channels} columns."
                )

            layer_record = {
                "name": name,
                "n": num_points,
                "k": num_channels,
                "initialCol": col,
                "initialPointSize": size,
                "dataB64": data_b64,
                "fieldNames": names_for_cloud,
            }
            self.layers.append(layer_record)
            self._broadcast({"type": "add_point_cloud", **layer_record})

    def add_boxes(self, boxes, uids=None, labels=None):
        """
        Adds one or more 3D bounding boxes to the already-open viewer, live, pre-placed and pre-labeled.

        :param boxes: List of boxes, each given as a 9-value sequence
                       (x, y, z, l, w, h, rx, ry, rz) — center position, size (length/width/height
                       along the box's local X/Y/Z), and rotation in radians about the box's local
                       X (rx, roll), Y (ry, pitch), and Z (rz, yaw) axes.
        :param uids: Optional list of unique integer ids, one per box. Auto-assigned
                     (continuing from any previously added boxes) if omitted.
        :param labels: Optional list of class-label strings, one per box. Defaults to "Box_<uid>".
        """
        self._ensure_started()

        n = len(boxes)

        if uids is None:
            uids = list(range(self._next_box_uid, self._next_box_uid + n))
        elif len(uids) != n:
            raise ValueError(f"uids must have the same length as boxes (got {len(uids)}, expected {n}).")

        if labels is None:
            labels = [f"Box_{uid}" for uid in uids]
        elif len(labels) != n:
            raise ValueError(f"labels must have the same length as boxes (got {len(labels)}, expected {n}).")

        new_boxes = []
        for box, uid, label in zip(boxes, uids, labels):
            x, y, z, l, w, h, rx, ry, rz = box
            box_record = {
                "id": uid,
                "label": label,
                "x": x, "y": y, "z": z,
                "l": l, "w": w, "h": h,
                "rx": rx, "ry": ry, "rz": rz,
            }
            new_boxes.append(box_record)
            self.boxes.append(box_record)
            self._next_box_uid = max(self._next_box_uid, uid + 1)

        self._broadcast({"type": "add_boxes", "boxes": new_boxes})

    def select_box(self, uid, color="orange"):
        """
        Marks a box (by uid) to glow in the given color, live, in the already-open viewer.

        :param uid: The unique id of a box previously added via add_boxes().
        :param color: A named color (see cloud_view.NAMED_COLORS) or a "#rrggbb" hex string.
        :return: True if a box with that uid exists and was updated, False if it isn't
                 present in the current visualization (nothing is changed in that case).
        """
        self._ensure_started()

        if isinstance(color, str) and color.startswith("#"):
            hex_color = color
        else:
            hex_color = NAMED_COLORS.get(str(color).lower())
            if hex_color is None:
                raise ValueError(
                    f"Unknown color name '{color}'. Use a hex string like '#ff8800' or one of: "
                    f"{', '.join(sorted(NAMED_COLORS))}"
                )

        for box in self.boxes:
            if box["id"] == uid:
                box["glowColor"] = hex_color
                self._broadcast({"type": "select_box", "id": uid, "glowColor": hex_color})
                return True

        return False

    def add_ego_poses(self, poses, names=None, axis_length=1.0):
        """
        Adds one or more ego-vehicle poses to the already-open viewer, live, plotted as small 3D
        axis triads (red=X, green=Y, blue=Z) joined by a line through their positions. Each call
        APPENDS to the existing trajectory rather than replacing it, so you can call this once
        per frame in a loop (e.g. alongside add_point_cloud) and watch the trajectory grow, or
        once with a full list of K poses to plot them all at once.

        :param poses: A single 4x4 transformation matrix, or a list/array of K of them
                       (shape (K, 4, 4)) -- vehicle-local -> world, exactly what
                       EgoOdometryLoader.get_pose()/get_next_pose()/get_frame_pose() return. Only
                       the rotation (top-left 3x3) and translation (top-right 3x1) are used.
        :param names: Optional label per pose (e.g. the matching Step Id), shown in the viewer.
                      Defaults to an auto-incrementing index continuing from any previously added
                      poses.
        :param axis_length: Length of each plotted axis triad, in the same units as your point
                             cloud. Applies to every pose added so far (not just this call).
        """
        self._ensure_started()

        poses_arr = np.asarray(poses, dtype=np.float64)
        if poses_arr.ndim == 2:
            poses_arr = poses_arr[None, :, :]
        if poses_arr.ndim != 3 or poses_arr.shape[1:] != (4, 4):
            raise ValueError(
                f"poses must be a single 4x4 matrix or a (K, 4, 4) array/list, got shape {poses_arr.shape}."
            )

        k = poses_arr.shape[0]

        if names is None:
            names = [str(self._next_pose_index + i) for i in range(k)]
        else:
            if len(names) != k:
                raise ValueError(f"names must have the same length as poses (got {len(names)}, expected {k}).")
            names = [str(n) for n in names]

        self._next_pose_index += k

        matrices_b64, _, _ = _encode_float32_b64(poses_arr.reshape(k, 16))

        chunk = {"n": k, "matricesB64": matrices_b64, "names": names, "axisLength": axis_length}
        self.ego_poses.append(chunk)
        self._broadcast({"type": "add_ego_poses", **chunk})

    def export_html(self, output_path: str = "index.html", auto_open: bool = True):
        """
        Bakes everything added so far into a single standalone HTML file — useful for sharing or
        archiving a frame, independent of the live server/viewer this instance also runs.

        :param output_path: Path where the generated HTML file will be saved.
        :param auto_open: If True, automatically opens the exported file in the default web browser.
        """
        html_content = self._template_html

        payload_js = f"let pointCloudDataList = {json.dumps(self.layers)};"
        html_content = html_content.replace("let pointCloudDataList = null;", payload_js)

        box_payload_js = f"let initialBoxDataList = {json.dumps(self.boxes)};"
        html_content = html_content.replace("let initialBoxDataList = null;", box_payload_js)

        ego_pose_payload_js = f"let egoPoseDataList = {json.dumps(self.ego_poses)};"
        html_content = html_content.replace("let egoPoseDataList = null;", ego_pose_payload_js)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        try:
            print(f"✨ Successfully generated viewer: {output_path}")
        except UnicodeEncodeError:
            # Some Windows consoles default to a non-UTF-8 codepage that can't print the emoji.
            print(f"Successfully generated viewer: {output_path}")

        if auto_open:
            abs_output_path = os.path.abspath(output_path)
            webbrowser.open(f"file://{abs_output_path}")

    def close(self):
        """Stops this instance's local server (a no-op if nothing was ever added, so the server
        never started). The already-open browser tab keeps showing its last state, but won't
        receive further live updates and a fresh reload of it will fail."""
        if self._server is not None:
            self._server.shutdown()
