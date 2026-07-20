import json
from pathlib import Path

from teleop.utils.rerun_visualizer import RerunEpisodeReader


def test_episode_reader_resolves_images_and_keeps_alignment(tmp_path):
    episode_dir = tmp_path / "episode_0001"
    colors_dir = episode_dir / "colors"
    colors_dir.mkdir(parents=True)
    image_path = colors_dir / "000000_color_0.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    (episode_dir / "data.json").write_text(
        json.dumps({
            "data": [{
                "idx": 0,
                "timestamp": 0.0,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "depths": {},
                "states": {"left_arm": {"qpos": [0.1]}},
                "actions": {"left_arm": {"qpos": [0.2]}},
                "alignment": {"valid": True, "lowstate_gap_ms": 20.0},
            }]
        }),
        encoding="utf-8",
    )

    frame = next(RerunEpisodeReader(tmp_path).iter_episode_data(1))

    assert frame["colors"]["color_0"] == str(image_path.resolve())
    assert frame["states"]["left_arm"]["qpos"] == [0.1]
    assert frame["actions"]["left_arm"]["qpos"] == [0.2]
    assert frame["alignment"]["lowstate_gap_ms"] == 20.0
