# Timestamp-aligned teleoperation recording

The control loop remains latest-frame and non-blocking. Recording also writes
independent raw streams, which are aligned offline before LeRobot export.

## Current alignment contract (v2)

The offline aligner uses a true fixed-rate target grid. A camera frame is
matched *to* a 20 Hz grid timestamp; its jittered capture timestamp never
replaces that target timestamp. Invalid frames are therefore not silently
renumbered into a shorter, apparently continuous trajectory.

Short acquisition gaps are handled without changing the 20 Hz timeline. A
nearby camera frame may be held for at most 75 ms and two consecutive grid
targets; LowState may be interpolated only across at most 150 ms; LowCmd may be
held for at most 150 ms. Every such frame is marked with `alignment.imputed`
and named `imputation_reasons`. The default quality gate allows at most 10%
imputed frames and 5% missing source sequence numbers.

Those are bounded recovery rules, not silent repair. A longer gap remains
invalid. The default `--gap-policy trim-edges` may remove an incomplete prefix
or suffix, but rejects a remaining invalid run inside an episode.
`--gap-policy compress` exists only for inspecting legacy data and must not be
used for formal long-horizon training.

Pico poses are diagnostic metadata by default. Add `--require-pico` only when a
downstream model actually consumes Pico data. RGB, LowState, and the published
LowCmd remain required training streams.

Action pairing is explicit: `--action-match` selects `latest-before`, `nearest`,
or `first-after`; `--action-offset-ms` applies a calibrated observation-to-action
offset before matching. Keep the offset at `0` until it has been measured.

The raw writer records queue delay, ignores duplicate camera packets, and starts
a new writer epoch when a camera sequence counter resets. This prevents a
restarted camera from overwriting an earlier JPEG with the same sequence number.

## 1. Install and configure Chrony

`chronyc` is provided by the Ubuntu `chrony` package. Install it on both the
camera host (`192.168.31.3`) and control host (`192.168.31.4`):

```bash
sudo apt update
sudo apt install -y chrony
sudo systemctl enable --now chrony
```

For a stable local relationship, use the control host as the LAN time source.
Append this line to `/etc/chrony/chrony.conf` on `192.168.31.4`:

```text
allow 192.168.31.0/24
```

Append this line on the camera host `192.168.31.3`:

```text
server 192.168.31.4 iburst prefer
```

Then restart Chrony on both hosts:

```bash
sudo systemctl restart chrony
```

If UFW is enabled on `192.168.31.4`, allow NTP from the camera host:

```bash
sudo ufw allow from 192.168.31.3 to any port 123 proto udp
```

Do not add `local stratum` while `192.168.31.4` has a usable upstream NTP
source. It is only appropriate for a deliberately isolated LAN.

## 2. Verify synchronization

Run on both the camera host (`192.168.31.3`) and control host
(`192.168.31.4`):

```bash
chronyc tracking
chronyc sources -v
python teleop/utils/check_clock_sync.py --max-offset-ms 2
```

Use PTP only when the wired NICs and switch support it. Check with:

```bash
sudo ethtool -T <wired-interface>
```

Do not compare monotonic timestamps from different machines. Cross-host
alignment uses synchronized wall time; monotonic time is retained for local
latency measurements.

For production recording, require `Leap status: Normal`, a selected source
marked `^*`, and `System time` below 2 ms. `Last offset` is retained for
diagnosis but is the previous clock sample, not the current wall-clock
correction. Each new raw episode also stores the control host's Chrony audit in
`manifest.json`.

## 3. Record

The existing command enables raw recording by default whenever `--record` is
present. Press `s` to start and again to stop:

```bash
cd /home/ai/lium/tele_robot/teleop
python teleop_hand_and_arm.py \
  --frequency 20 \
  --input-mode controller \
  --robot TOPSTAR_H1 \
  --control-mode arms_only \
  --ee suction_cup \
  --img-server-ip 192.168.31.3 \
  --arm-scale 1.2 \
  --record \
  --task-name topstar_h1_sync_001
```

Raw data is written under:

```text
teleop/utils/data/topstar_h1_sync_001/raw/episode_XXXX/
```

The camera host must run the updated `teleimager` server so camera packets
contain device timestamps and sequence numbers.

## 4. Align one episode

The aligner removes repeated camera sequence/device timestamp entries before
fitting each camera clock. The original duplicate and missing sequence counts
remain in `alignment_report.json`; only fitting and nearest-frame matching use
the deduplicated stream.

```bash
cd /home/ai/lium/tele_robot
PYTHONPATH=. python teleop/utils/align_raw_session.py \
  --input-episode teleop/utils/data/topstar_h1_sync_001/raw/episode_0001 \
  --reference-camera head_camera \
  --fps 20 \
  --overwrite
```

After confirming that the installed Vuer and Orbbec SDK versions expose source
timestamps, use strict mode for production data:

```bash
  --require-source-clocks \
  --max-clock-fit-residual-ms 2 \
  --require-quality \
  --max-invalid-ratio 0.05
```

Without strict mode the aligner records `source_clock_issues` and falls back to
host receipt time. That fallback is useful for diagnostics, but should not be
presented as hardware-level synchronization.

For the current TOPSTAR H1 pipeline, `/lowstate` has no publisher-side wall
timestamp and Pico browser events have normal scheduling jitter. Keep the
strict report for diagnosis, then generate a practical slow-manipulation
alignment with explicit interpolation limits:

```bash
PYTHONPATH=. python teleop/utils/align_raw_session.py \
  --input-episode teleop/utils/data/topstar_h1_sync_001/raw/episode_0001 \
  --reference-camera head_camera \
  --fps 20 \
  --max-camera-error-ms 20 \
  --max-camera-hold-ms 75 \
  --max-camera-hold-frames 2 \
  --max-state-gap-ms 60 \
  --max-state-interpolation-gap-ms 150 \
  --max-pico-gap-ms 50 \
  --max-action-age-ms 100 \
  --max-action-hold-ms 150 \
  --max-imputed-ratio 0.10 \
  --max-sequence-missing-ratio 0.05 \
  --overwrite
```

The `*-error/gap/age` values are the native-sample targets; the corresponding
`*-hold/interpolation` values are hard recovery bounds. Values between them are
kept but explicitly counted as imputed. These settings do not change raw data
or robot control. Do not use this practical profile for fast contact/collision
tasks without adding a publisher-side LowState timestamp.

Optional measured camera latency can be subtracted before matching:

```bash
  --camera-latency-ms head_camera=42 \
  --camera-latency-ms left_wrist_camera=38 \
  --camera-latency-ms right_wrist_camera=39
```

The aligned episode and quality report are written to
`.../topstar_h1_sync_001/aligned/episode_0001/`.

## 5. Inspect robot response delay

Use an episode containing safe, sufficiently varied arm motion:

```bash
PYTHONPATH=. python teleop/utils/analyze_actuation_delay.py \
  --input-episode teleop/utils/data/topstar_h1_sync_001/raw/episode_0001 \
  --output teleop/utils/data/topstar_h1_sync_001/raw/episode_0001/actuation_delay.json
```

This reports velocity cross-correlation delay per joint. It is response delay,
not the time required to settle exactly at the target.

## 6. Export aligned episodes to LeRobot

```bash
PYTHONPATH=. python teleop/utils/export_lerobot_dataset.py \
  --input-dir teleop/utils/data/topstar_h1_sync_001/aligned \
  --output-dir teleop/utils/data/topstar_h1_sync_001/lerobot \
  --repo-id local/topstar_h1_sync_001 \
  --overwrite
```

Review `alignment_report.json` before training. Do not export an episode whose
clock check failed or whose invalid/missing frame rate is unexpectedly high.

## 7. Finalize a complete task directory

The MuJoCo incremental live teleop intentionally records only asynchronous raw
streams plus the background legacy episode while the robot is moving. It does
not call `LeRobotDataset.add_frame()` in the control loop. After collection,
batch-align every raw episode, write an aggregate quality report, and optionally
export the accepted aligned set to LeRobot:

```bash
cd /home/ai/lium/tele_robot
PYTHONPATH=. python -m teleop.utils.finalize_raw_dataset \
  --task-dir teleop/utils/data/topstar_h1_box_new_004
```

This writes:

```text
topstar_h1_box_new_004/aligned/episode_XXXX/
topstar_h1_box_new_004/offline_quality_summary.json
```

Only request LeRobot export after reviewing the report:

```bash
PYTHONPATH=. python -m teleop.utils.finalize_raw_dataset \
  --task-dir teleop/utils/data/topstar_h1_box_new_004 \
  --export-lerobot \
  --video-codec h264
```

The exporter refuses episodes whose alignment report fails its quality gate;
do not bypass that refusal without reviewing the named episode and failure.
