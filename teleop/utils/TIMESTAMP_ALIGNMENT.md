# Timestamp-aligned teleoperation recording

The control loop remains latest-frame and non-blocking. Recording also writes
independent raw streams, which are aligned offline before LeRobot export.

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
