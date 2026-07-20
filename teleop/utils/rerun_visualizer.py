#!/usr/bin/env python3
"""Visualize recorded episode images, state, action, and alignment in Rerun."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

os.environ.setdefault("RUST_LOG", "error")


class RerunEpisodeReader:
    def __init__(self, task_dir: str | Path = ".", json_file: str = "data.json"):
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.json_file = json_file

    def episode_dir(self, episode_idx: int) -> Path:
        return self.task_dir / f"episode_{episode_idx:04d}"

    def iter_episode_data(self, episode_idx: int) -> Iterator[dict[str, Any]]:
        episode_dir = self.episode_dir(episode_idx)
        json_path = episode_dir / self.json_file
        if not json_path.exists():
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as handle:
            episode = json.load(handle)

        for item_data in episode["data"]:
            yield {
                "idx": item_data.get("idx", item_data.get("frame_index", 0)),
                "timestamp": item_data.get("timestamp"),
                "colors": self._resolve_paths(item_data, "colors", episode_dir),
                "depths": self._process_depths(item_data, episode_dir),
                "states": item_data.get("states", {}),
                "actions": item_data.get("actions", {}),
                "alignment": item_data.get("alignment", {}),
                "tactiles": item_data.get("tactiles", {}),
                "audios": {},
            }

    def return_episode_data(self, episode_idx: int) -> list[dict[str, Any]]:
        return list(self.iter_episode_data(episode_idx))

    @staticmethod
    def _resolve_paths(
        item_data: dict[str, Any], data_type: str, episode_dir: Path
    ) -> dict[str, str]:
        paths = {}
        for key, file_name in (item_data.get(data_type, {}) or {}).items():
            if not file_name:
                continue
            path = episode_dir / file_name
            if path.exists():
                paths[key] = str(path)
        return paths

    @staticmethod
    def _process_depths(
        item_data: dict[str, Any], episode_dir: Path
    ) -> dict[str, np.ndarray]:
        depths = {}
        for key, file_name in (item_data.get("depths", {}) or {}).items():
            if not file_name:
                continue
            path = episode_dir / file_name
            if path.exists():
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    depths[key] = image
        return depths


class RerunLogger:
    def __init__(
        self,
        prefix: str = "",
        IdxRangeBoundary: int | None = 30,
        memory_limit: str | None = None,
        *,
        image_keys: list[str] | None = None,
        mode: str = "native",
        web_port: int = 9090,
        grpc_port: int = 9876,
        viewer_host: str = "127.0.0.1",
        save_path: str | Path | None = None,
        open_browser: bool = True,
    ):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self.image_keys = image_keys or []
        self.mode = mode
        self.web_url = None
        app_id = datetime.now().strftime("tele_robot_%Y%m%d_%H%M%S")
        rr.init(app_id)
        blueprint = self.build_blueprint()
        server_memory_limit = memory_limit or "1GB"

        if mode == "web":
            rr.serve_grpc(
                grpc_port=grpc_port,
                default_blueprint=blueprint,
                server_memory_limit=server_memory_limit,
            )
            connect_to = f"rerun+http://{viewer_host}:{grpc_port}/proxy"
            self.web_url = (
                f"http://{viewer_host}:{web_port}/"
                f"?url={quote(connect_to, safe='')}"
            )
            rr.serve_web_viewer(
                web_port=web_port,
                open_browser=open_browser,
                connect_to=connect_to,
            )
        elif mode == "save":
            if save_path is None:
                raise ValueError("save_path is required when mode='save'")
            rr.save(str(Path(save_path).expanduser().resolve()), default_blueprint=blueprint)
        elif mode == "native":
            rr.spawn(
                memory_limit=memory_limit or "75%",
                hide_welcome_screen=True,
                default_blueprint=blueprint,
            )
        else:
            raise ValueError(f"Unsupported Rerun mode: {mode}")

        rr.send_blueprint(blueprint)

    def _visible_time_range(self):
        if not self.IdxRangeBoundary:
            return None
        return [
            rrb.VisibleTimeRange(
                "idx",
                start=rrb.TimeRangeBoundary.cursor_relative(
                    seq=-self.IdxRangeBoundary
                ),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            )
        ]

    def build_blueprint(self) -> rrb.Blueprint:
        time_range = self._visible_time_range()
        image_names = {
            "color_0": "Head camera",
            "color_1": "Left wrist camera",
            "color_2": "Right wrist camera",
        }
        image_views = [
            rrb.Spatial2DView(
                origin=f"{self.prefix}colors/{key}",
                name=image_names.get(key, key),
                time_ranges=time_range,
            )
            for key in self.image_keys
        ]
        image_row = (
            rrb.Horizontal(*image_views, name="Cameras")
            if image_views
            else None
        )

        plot_views = [
            rrb.TimeSeriesView(
                origin=f"{self.prefix}{part}",
                name=name,
                time_ranges=time_range,
                plot_legend=rrb.PlotLegend(visible=True),
            )
            for part, name in (
                ("left_arm", "Left arm state/action"),
                ("right_arm", "Right arm state/action"),
                ("left_ee", "Left end effector"),
                ("right_ee", "Right end effector"),
                ("alignment", "Alignment quality"),
            )
        ]
        plot_grid = rrb.Grid(*plot_views, grid_columns=2, name="Trajectories")
        contents = [part for part in (image_row, plot_grid) if part is not None]
        return rrb.Blueprint(
            rrb.Vertical(*contents, row_shares=[1, 2] if image_row else None),
            rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
            rrb.TimePanel(state=rrb.PanelState.Expanded),
            collapse_panels=False,
        )

    def log_item_data(self, item_data: dict[str, Any]) -> None:
        rr.set_time("idx", sequence=int(item_data.get("idx", 0)))
        timestamp = item_data.get("timestamp")
        if timestamp is not None:
            rr.set_time("episode_time", duration=float(timestamp))

        for part, state_info in (item_data.get("states", {}) or {}).items():
            if part == "body" or not state_info:
                continue
            for index, value in enumerate(state_info.get("qpos", [])):
                rr.log(
                    f"{self.prefix}{part}/states/qpos/{index}",
                    rr.Scalars(value),
                )

        for part, action_info in (item_data.get("actions", {}) or {}).items():
            if part == "body" or not action_info:
                continue
            for index, value in enumerate(action_info.get("qpos", [])):
                rr.log(
                    f"{self.prefix}{part}/actions/qpos/{index}",
                    rr.Scalars(value),
                )

        for color_key, color in (item_data.get("colors", {}) or {}).items():
            entity = f"{self.prefix}colors/{color_key}"
            if isinstance(color, (str, Path)):
                rr.log(entity, rr.EncodedImage(path=color))
            elif color is not None:
                rr.log(entity, rr.Image(color))

        for depth_key, depth in (item_data.get("depths", {}) or {}).items():
            if depth is not None:
                rr.log(f"{self.prefix}depths/{depth_key}", rr.DepthImage(depth))

        alignment = item_data.get("alignment", {}) or {}
        rr.log(
            f"{self.prefix}alignment/valid",
            rr.Scalars(1.0 if alignment.get("valid", True) else 0.0),
        )
        for name, value in (alignment.get("camera_delta_ms", {}) or {}).items():
            if value is not None:
                rr.log(
                    f"{self.prefix}alignment/camera_delta_ms/{name}",
                    rr.Scalars(value),
                )
        for key in ("lowstate_gap_ms", "pico_gap_ms", "lowcmd_age_ms"):
            value = alignment.get(key)
            if value is not None:
                rr.log(f"{self.prefix}alignment/{key}", rr.Scalars(value))

    def log_episode_data(self, episode_data) -> None:
        for item_data in episode_data:
            self.log_item_data(item_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Directory containing episode_XXXX folders",
    )
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("web", "native", "save"),
        default="web",
        help="Open a browser viewer, native viewer, or save an .rrd file",
    )
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument(
        "--viewer-host",
        default="127.0.0.1",
        help="Host/IP used by the browser to reach this computer",
    )
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--time-window", type=int, default=120)
    parser.add_argument("--output-rrd", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = RerunEpisodeReader(args.task_dir)
    iterator = reader.iter_episode_data(args.episode)
    try:
        first_item = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"Episode {args.episode} contains no frames") from exc

    save_path = args.output_rrd
    if args.mode == "save" and save_path is None:
        save_path = reader.episode_dir(args.episode) / f"episode_{args.episode:04d}.rrd"

    logger = RerunLogger(
        prefix="episode/",
        IdxRangeBoundary=args.time_window,
        memory_limit=args.memory_limit,
        image_keys=sorted(first_item.get("colors", {})),
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        viewer_host=args.viewer_host,
        save_path=save_path,
        open_browser=not args.no_open_browser,
    )
    logger.log_episode_data(chain((first_item,), iterator))

    if args.mode == "save":
        print(f"Saved Rerun recording: {Path(save_path).expanduser().resolve()}")
        return

    if args.mode == "web":
        print(f"Rerun web viewer: {logger.web_url}")
        print("Keep this process running; press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
