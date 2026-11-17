# CountSense Edge

Offline vehicle and pedestrian detection and counting on a **Raspberry Pi 5**
(built and first measured on a Pi 4; the same code, models and packages run on both).
YOLO11n (COCO, 80 classes) exported to **PyTorch ExecuTorch** (XNNPACK backend, FP32),
with all post-processing in plain **NumPy**. Give it a video, a folder of images or one
image; it reports per-frame counts per category, a density level, annotated frames,
and a timing report.

- **Runtime dependencies on the Pi:** `executorch`, `torch` (CPU), `numpy`, `opencv-python`. Nothing else is imported.
- **Fully offline:** no downloads, uploads or cloud calls at runtime.
- **Counts are per frame.** There is no tracking, so a video gives the average and maximum per frame, never "unique vehicles".

```
image / folder / video
  -> letterbox to S x S (pad 114)             countsense/preprocess.py
  -> ExecuTorch "forward" (XNNPACK, FP32)     countsense/io.py
  -> NumPy decode + class-aware NMS           countsense/decode.py
  -> per-frame counts + density               countsense/counting.py
  -> annotated frames, CSV, JSON, REPORT.md   countsense/annotate.py, report.py, timing.py
```

---

## Problem Statement 2: deliverables checklist

*AI for Safer Communities - Edge-Based Object Detection and Counting.* The challenge:
affordable ways to understand vehicle or pedestrian movement without expensive computing
or continuous cloud connectivity. Our three presets are the three applications the
statement names: traffic observation (`traffic`), pedestrian monitoring (`pedestrian`)
and public-space utilisation (`public-space`).

| Mandatory deliverable | Where it is |
|---|---|
| Process one 10-20 s clip or 10-20 images with a pretrained object-detection model on ExecuTorch and Raspberry Pi | `run_poc.py --source <clip or folder>`, run on a Raspberry Pi 5. Model: Ultralytics YOLO11n (COCO-pretrained, not retrained), exported to ExecuTorch `.pte` |
| Detect at least one category and display counts for each image or frame | One terminal line per frame, `counts.csv`, and the header bar on every image in `annotated/` |
| Submit category and counts plus at least two sample detection outputs | `REPORT.md`: "Summary" and "Sample annotated frames" (at least two); all frames in `annotated/` |
| Report average inference time, model used and approximate throughput | `REPORT.md` "Summary" and "Speed"; the same numbers in `summary.json` |
| Explain the societal issue, how counting helps, and one application (2 minutes) | The pitch; measured numbers to quote are in "Picking a speed setting at the venue" below |

Not required by the statement, and not built: live camera, real-time surveillance, tracking,
dashboards, cloud connectivity, mobile apps.

---

## Project layout

| Path | What it is |
|---|---|
| `run_poc.py` | Command-line entry point: normal runs and `--benchmark` |
| `countsense/io.py` | Load the `.pte` and `metadata.yaml`, read frames (video / folder / image), write images and video |
| `countsense/preprocess.py` | Letterbox: resize keeping aspect ratio, pad to S x S, BGR->RGB, CHW, /255 |
| `countsense/decode.py` | Raw output `(1, 84, N)` -> boxes: confidence + class filter, xywh->xyxy, undo letterbox, clip, NMS |
| `countsense/counting.py` | Categories, presets, per-frame counts, density, average/max statistics |
| `countsense/annotate.py` | Boxes, labels and the header bar on each frame; `--blur-people`: pixelates head areas |
| `countsense/tracking.py` | `--count-line`: links boxes between frames and counts each object once when it crosses a line |
| `countsense/timeline.py` | `--timeline`: the chart of counts over the run, the busiest moment and the busy periods |
| `countsense/telemetry.py` | `--telemetry`: the Pi's temperature, CPU load, memory, board power and throttle state during the run |
| `countsense/sharedspace.py` | `--shared-space`: the map of ground used by both pedestrians and vehicles |
| `countsense/items.py` | `--watch-items`: notices for a knife, baseball bat or scissors next to a person |
| `countsense/timing.py` | Mean / p50 / p95 timing statistics |
| `countsense/report.py` | `counts.csv`, `detections.json`, `summary.json`, `REPORT.md`, device info |
| `scripts/verify_against_ultralytics.py` | Laptop only: checks our counts equal Ultralytics' on the same `.pte` |
| `scripts/setup_pi.sh` | Pi only: offline install + checks + one test detection |
| `requirements-pi.txt` | Exact pinned package list for the Pi (torch is installed by name, see the file's header) |
| `yolo11n_320/`, `yolo11n_416/`, `yolo11n_640/`, `yolo11n_960/`, `yolo11n_1280/` | Exported models (`model.pte` + `metadata.yaml`) for each input size. Larger = slower, but finds smaller and more distant objects |
| `data/` | Test inputs (`bus.jpg`, `zidane.jpg`; put your clips here) |
| `pi_wheels/` | Laptop only: the Pi's packages as wheel files, for the offline install |

---

## Usage

On the laptop use `.venv/Scripts/python.exe`; on the Pi use `.venv/bin/python`.

```bash
.venv/bin/python run_poc.py --source data/clip.mp4 --preset traffic
.venv/bin/python run_poc.py --source data/clip.mp4 --preset traffic --model yolo11n_320 --stride 2 --save-video
.venv/bin/python run_poc.py --source data/judge_images/ --preset pedestrian
.venv/bin/python run_poc.py --source data/clip.mp4 --preset traffic --count-line --timeline --blur-people --save-video
.venv/bin/python run_poc.py --source data/clip.mp4 --benchmark
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | (required) | A video (`.mp4 .avi .mov .mkv .m4v .webm`), a folder of images, or one image |
| `--preset` | `traffic` | Which categories to count (see below) |
| `--model` | `yolo11n_640` | Model folder (`yolo11n_320`, `yolo11n_416`, `yolo11n_640`, `yolo11n_960`, `yolo11n_1280`) or a `.pte` file |
| `--imgsz` | from `metadata.yaml` | Input size S. Must match the model; a mismatch stops with a clear message |
| `--conf` | `0.25` | Confidence threshold |
| `--stride` | `1` | Process every Nth frame. True frame numbers and timestamps are kept |
| `--max-frames` | all | Stop after N processed frames |
| `--density LOW HIGH` | `8 16` | LOW if total < LOW, HIGH if total >= HIGH, MODERATE in between |
| `--threads` | `4` | CPU threads for XNNPACK (the Pi 5 has 4 cores, like the Pi 4). Shown in `REPORT.md` |
| `--save-video` | off | Also write `annotated.mp4` (plays at real speed) |
| `--shared-space` | off | Also write `shared_space.jpg`: where pedestrians and vehicles used the same ground. Fixed-camera video with the `traffic` preset only |
| `--watch-items` | off | Also print a notice when a knife, baseball bat or scissors is detected next to a person. Unverified object detections for a human to check (see below) |
| `--count-line [LINE]` | off | Count each object once when it crosses a line, with its direction. Fixed-camera video only. `LINE` is `h` or `v` plus a percentage: `h50` (the default) is a horizontal line at half the height, `v30` a vertical line at 30% of the width (see below) |
| `--timeline` | off | Also write `timeline.png`, a chart of the counts over the run, and report the busiest moment and the busy periods (see below) |
| `--blur-people` | off | Privacy: pixelate the head area of every detected person in all saved images and video (see below) |
| `--telemetry` | off | Print the Pi's temperature, CPU load, memory, board power and throttle state every 2 seconds during the run, and add their averages and the energy used to `REPORT.md` (see below) |
| `--benchmark` | off | Run the first 20 frames (or `--max-frames`) through every model size and print a comparison table. Writes no files |
| `--out` | `outputs` | Root folder for results |

### Presets and categories (COCO class ids)

| Category | COCO classes |
|---|---|
| pedestrian | person (0) |
| cyclist | bicycle (1) |
| vehicle | car (2), motorcycle (3), bus (5), truck (7) |

| Preset | Counts |
|---|---|
| `traffic` | vehicle, cyclist, pedestrian |
| `pedestrian` | pedestrian |
| `public-space` | pedestrian, cyclist |

### Outputs: `outputs/<input name>_<preset>_<timestamp>/`

For example `outputs/knife_traffic_20261114_142122/` for `data/knife.mp4`: the input's file
or folder name (letters, digits and `_` only, at most 30 characters), the preset, and the
date and time the run started.

| File | Contents |
|---|---|
| `REPORT.md` | One-page report: model and size, image size, threads, device (from `/proc/cpuinfo`), frames read / processed / skipped, timing (mean, p50, p95), FPS, wall clock, per-category average and max, density distribution, sample frames |
| `summary.json` | The same numbers as data |
| `counts.csv` | One row per processed frame: true frame index, timestamp, counts, total, density, the confident-only counts, a borderline flag, timing. With `--count-line` also `passed_so_far` |
| `detections.json` | Every box of every frame: class, category, confidence, coordinates. With `--count-line` also its track number |
| `annotated/` | One JPG per processed frame, with boxes and a header bar (counts, total, density, inference ms) |
| `annotated.mp4` | With `--save-video` only (mp4v: plays in VLC and the Windows player) |
| `timeline.png` | With `--timeline` only: the counts over the run as a line chart |
| `shared_space.jpg` | With `--shared-space` only: the scene in grey, pedestrian ground in green, vehicle ground in blue, ground used by both in red |

### Counts that show their uncertainty

Every count is given as **confident-possible**, for example `vehicle=6-8`:

- **Confident:** detections scoring 0.50 or more.
- **Possible:** all detections above `--conf` (0.25), so it includes the less certain ones. On annotated frames these have a thin box and a `?` after the score.
- **Borderline frame:** the density level would be lower if only confident detections were counted. The console marks it `MODERATE?`, the header bar shows `LOW-MODERATE`, and `REPORT.md` counts how many frames were borderline.

The existing count columns (`vehicle`, `total`, `density`) still use every detection, as before.
Objects the model misses completely are in neither number, so this is a confidence band, not a guarantee.

### Passing counts (`--count-line`)

Per-frame counts cannot say how many vehicles went by: a car seen in 30 frames is 30
sightings. With `--count-line`, each object is counted **once**, when it crosses a line,
and the count is split by direction.

- **The line:** `--count-line` alone draws a horizontal line at half the height (`h50`), for traffic moving up or down the picture. For traffic moving sideways use a vertical line, for example `--count-line v50`. The number is the position in percent.
- **How it works:** each box is linked to the nearest box of the same category in the previous processed frame, at the place the object was heading. An object is counted when the centre of its box moves from one side of the line to the other.
- **Only confident objects count:** a crossing counts if the object scored 0.50 or more in at least one frame. This keeps flickering low-score boxes out of the total.
- **Only real movement counts:** the object must have moved in the crossing direction by at least half its own size since it was first seen. A vehicle waiting on the line with a wobbling box is not counted until it drives off.
- **Objects already on the line when the clip starts are not counted:** they never cross it during the clip.
- **Terminal:** `PASSED: vehicle #12 crossed the line going down` at each crossing, `passed N` on every frame line, and a final `Passed` line with the split per category and the rate per minute.
- **Files:** the line, the running tally and each object's track number (`#12`) are drawn on the annotated frames; `REPORT.md` gets a table per category and direction and a list of every crossing.

Read this before relying on it:

- **Fixed camera only.** If the camera moves, everything in the picture "crosses" the line.
- **Use a small stride** (`--stride 1` or `2`). The longer the gap between processed frames, the further objects move and the more links go wrong.
- **It is a simple tracker.** An object hidden while it crosses, or lost and found again on the other side, is missed. Two objects swapping places can be confused. No identities are kept.
- **Riders count twice here too:** a motorcycle with a rider is one vehicle and one pedestrian crossing.

Checked against a count made by eye on a real road-camera clip: see "Accuracy check on real footage".

### Timeline (`--timeline`)

`timeline.png` is a line chart of the counts over the run: seconds along the bottom for a
video, image number for a folder, one line per category plus the total, and grey lines
where the density level changes. `REPORT.md` shows the chart and adds:

- **Busiest moment:** the frame with the highest total, and its annotated image.
- **Busy periods:** the stretches spent at the highest density level the run reached. In a video, a dip shorter than 1 second does not end a stretch, and the table says how many of its frames were really at that level.

The terminal prints one `Busiest` line with the same facts. It needs at least two processed frames.

### Privacy blur (`--blur-people`)

With this flag the head area of every detected person is pixelated in everything the run
saves: `annotated/`, `annotated.mp4` and `shared_space.jpg`. The model still sees the real
frame, so counts and timing do not change. The terminal confirms it at the start and
reports how many head areas were pixelated at the end.

- **There is no face detector.** The head is taken to be the top of the person's box (as tall as 70% of the box's width, at least a quarter of the box).
- **A person the model misses is not blurred**, and a head that is not at the top of its box (someone bending down) may be only partly covered.
- The input video or images are never copied into the results folder.

### Pi telemetry (`--telemetry`)

With this flag a line like this appears between the frame lines every 2 seconds:

```
    PI  temp 66.7 C  cpu 82% at 2400 MHz  ram 1.1/7.9 GB (this run 0.5)  power 6.8 W  throttled: no
```

- **temp:** the chip's temperature. **cpu:** how busy the four cores were since the last line, and their clock speed. **ram:** memory in use on the whole Pi, and by this run. **power:** what the board is drawing.
- **throttled:** whether the Pi is slowing itself down.
  - `no`: full speed.
  - `YES (heat)` or `YES (low voltage)`: it is slowed down right now, so the timings of this run are worse than the Pi can do. Cool it, or use the 5 A power supply.
  - `no (was earlier, heat)`: full speed now, but it was slowed down at some time since the Pi was switched on. The Pi remembers that until it is restarted, so this can come from an earlier run.
- The run ends with a `Telemetry` line and a `Power` line (average and highest readings, energy used, joules per frame), and `REPORT.md` gets a "Pi telemetry during the run" table with the same numbers.
- Everything is read from Linux's `/proc` and `/sys` files and the Pi's own `vcgencmd` tool (`countsense/telemetry.py`), so nothing extra is installed. A reading the machine cannot give shows as `n/a`.
- **Board power needs a Pi 5.** It comes from the Pi 5's power chip and covers the board only: USB devices and the losses in the power supply are not in it, so a wall-socket meter reads higher.
- The readings are taken by a small background thread while the model is running. That matters for power: read between two frames, the cores are idle for a moment and the reading comes out far too low. The cost is two short `vcgencmd` calls every 2 seconds.
- On Windows there is nothing to read and the flag only prints a note.

### Shared-space map (`--shared-space`)

For a fixed-camera clip, one image shows where people and vehicles mix:

- Each frame marks the bottom 20% of every box (feet, wheels) on an 8-pixel grid; the map adds the frames up.
- Green = ground pedestrians stood on, blue = ground vehicles stood on, red = ground both used at some point. Brighter = used in more frames.
- `REPORT.md` states one number: the share of pedestrian ground that vehicles also used.
- No tracking and no identities. A person whose feet are inside a bicycle or motorcycle box is treated as a rider and left out of the pedestrian layer.
- It shows presence, not risk, and it is meaningless if the camera moves.

### Item notices (`--watch-items`)

Off by default. With the flag, the same model also looks for three of its 80 object types:
**knife, baseball bat, scissors**. When one is detected with a score of 0.50 or more **and**
its box touches a person's box (within half the item's size), the run:

- prints one line for that frame, with the second and the frame: `POSSIBLE THREAT at t=3.40s, frame_000085: knife next to a person (score 0.62), unverified - check annotated/frame_000085.jpg`, and a closing `Threats` line with how many there were and where the first one is;
- draws a red box labelled `possible knife 0.62` on that frame;
- adds an `item_notices` column to `counts.csv` and a table of every notice to `REPORT.md`.

The watched items are never added to the vehicle or pedestrian counts, and the flag adds
no inference time (it only stops discarding those three types).

Read this before relying on it:

- **It is object detection, not behaviour detection.** It cannot tell a threat from a cook, a tailor or a cricketer, and "next to" is box geometry, not "in the hand". Every notice needs a human to look at the image.
- **It misses most small or distant items.** A knife a few pixels wide is not detected.
- **There is no rod, stick or gun type** in the model. A rod is occasionally detected as a baseball bat; mostly it is not detected.
- **It failed on real CCTV footage.** On a shop-camera clip of a person holding a machete (444 x 390 pixels, 314 frames) there was no alert. The model's best "knife" score in any frame was 0.05, at 640, 960 and 1280 alike, against the 0.50 an alert needs, so no setting would have found it. The model learned "knife" from close-up photos of kitchen and table knives; a blurred blade on a security camera does not look like those.

### Picking a speed setting at the venue

Resolution, stride and threads are where the time goes. Measured on our **Pi 5**
(Raspberry Pi 5 Model B Rev 1.1, 4 x Cortex-A76, 4 threads) on 2026-11-14, over the
first 20 frames of the traffic test clip (1280 x 720):

| Model | Inference avg | Frame avg (pre + infer + post) | FPS | Vehicles found in the 20 frames |
|---|---:|---:|---:|---:|
| `yolo11n_320` | 31 ms | 34 ms | 29.6 | 58 |
| `yolo11n_416` | 57 ms | 61 ms | 16.4 | 73 |
| `yolo11n_640` | 157 ms | 163 ms | 6.2 | 115 |
| `yolo11n_960` | 394 ms | 405 ms | 2.5 | 119 |
| `yolo11n_1280` | 778 ms | 801 ms | 1.2 | 140 |

A complete run of the 48-second clip (1192 frames) at 640 with `--count-line --timeline
--save-video` took **235 s** on the Pi 5: 160 ms average inference and 5.2 frames per
second end to end.

For comparison, the Pi 4 we started on (4 x Cortex-A72) measured 102, 175 and 445 ms
inference at 320, 416 and 640, so the Pi 5 is about three times as fast.

Processing time for a clip = processed frames x "frame avg", plus reading the video and
saving the annotated images. For a 20 s clip at 30 fps (600 frames) on the Pi 5 that is
about **2 minutes** at 640 with every frame, **4 minutes** at 960, and **8 minutes** at
1280 (under 3 minutes with `--stride 3`).

- **Videos:** use `--model yolo11n_640` with every frame (no `--stride`), plus `--count-line` and `--save-video`, and `--shared-space` if the camera does not move.
- **With `--count-line`:** keep `--stride 1` or `2`. Linking objects between frames needs them close together in time.
- **Short on time:** `--stride 3` cuts the time to a third. Every 3rd frame is still 10 frames per second of video, so per-frame averages and maxima barely change.
- **Image folders:** never use `--stride` on a folder, or images get skipped. For 10 to 20 images time hardly matters (20 images at 1280 are about 16 seconds of inference), so `yolo11n_1280` is worth it when the objects are small.
- **Check on the judges' clip:** `--benchmark` runs all five sizes on the first 20 frames and takes about a minute on the Pi 5.

---

## Accuracy check on real footage

Checked on 2026-11-14, on the laptop, against counts made by eye. Two inputs: a
48-second clip from a fixed road camera (1280 x 720, 25 fps, 1192 frames) and a small
parking-lot picture (349 x 198).

### Vehicles passing the line (`--count-line`, whole clip)

By eye, 38 cars and buses cross the line (33 going up, 5 coming down) and so do 2
scooters. The eye count was made on a "time picture": the line's pixel row from every
frame, stacked, where each passing vehicle is one blob.

| Model | Cars and buses counted | Counted in error | Passing cars labelled truck or bus | Inference on the laptop |
|---|---:|---:|---:|---:|
| `yolo11n_640` | 37 of 38 | 0 | 0 of 34 | 53 ms |
| `yolo11n_960` | 38 of 38 | 0 | 4 of 35 | 129 ms |
| `yolo11n_1280` | 38 of 38 | 0 | 8 of 35 | 249 ms |

- The one car missed at 640 was half outside the left edge of the picture when it crossed.
- **Neither scooter was counted as a vehicle at any size.** The model saw each rider as a "person" and counted a pedestrian instead.
- Two vehicles were already on the line in the first frame, so they are not in the 38. The truck was not counted at any size; the car was counted at 960 only, in its true direction.
- A wrong label does not change a count: car, truck and bus are all "vehicle".
- **On the Pi 5 itself**, the 640 run counted 38 of 38: it also caught the car the laptop missed at the edge. Nothing was counted in error and every passing car was labelled "car".

### Vehicles per frame (four frames counted by eye)

| Part of the picture | By eye | `640` | `960` | `1280` |
|---|---:|---:|---:|---:|
| Lower 85%: cars, buses, trucks | 15 | 15 | 15 | 15 |
| Lower 85%: scooters | 2 | 0 | 0 | 0 |
| Top 15% (the far end of the road) | about 54 | 17 | 28 | 33 |

- Everything from the middle distance to the camera is found at every size. The far end, where a car is only a few pixels wide, is under-counted, and a larger model sees further.
- In the frames checked, no box was on something that is not a vehicle.
- The far-end figure "about 54" is approximate: those vehicles are hard to tell apart even by eye.

### Parking-lot picture

By eye: 3 cars and 11 people (one cut off by the picture's edge).

| Model | Cars found | People found |
|---|---:|---:|
| `yolo11n_320` | 3 | 5 |
| `yolo11n_416` | 3 | 6 |
| `yolo11n_640` | 3 | 8 |
| `yolo11n_960` | 3 | 9 |
| `yolo11n_1280` | 3 | 10 |

The people missed are those standing behind someone else (the model draws one box around
both) and the one cut off at the edge. No box was on something that is not a person or a car.
The Pi 5 gave the same result at 1280 as the laptop: 3 cars and 10 people.

### What to use

- **Fastest, cleanest labels:** `yolo11n_640`. This stays the default, because it was the only size that gave no passing car a wrong label.
- **More of the distance, labels still mostly right:** `yolo11n_960`, at about 2.5 times the time on the Pi 5.
- **The most distant and smallest objects, and small pictures of people:** `yolo11n_1280`, at about 5 times the time on the Pi 5. It calls more nearby cars "truck" or "bus".

---

## Raspberry Pi 5 (offline)

The Pi never needs internet. The laptop downloads the Pi's packages once, then
everything goes over a **direct Ethernet cable** with fixed addresses: laptop
`192.168.50.1`, Pi `192.168.50.2`. The same setup works at the venue with no router.

Pi used: a **Raspberry Pi 5 Model B Rev 1.1** (4 x Cortex-A76) with 64-bit Raspberry Pi
OS, **Python 3.13.5**, hostname `raspberrypi`, user `test`. The wheels are built for
Python 3.13; another Python version needs new wheels. This Pi's kernel uses 16 KB memory
pages (the Pi 4's used 4 KB); the same wheels install and run on it unchanged.

### Moving from the Pi 4 to the Pi 5

The project was first set up on a Pi 4. The models (`.pte`), the wheels in `pi_wheels/`
and the code are the same for both boards: nothing is re-exported or rebuilt.

- **Same SD card (fastest):** shut the Pi 4 down, put its card in the Pi 5, power on. The 64-bit Raspberry Pi OS boots on both boards, and the fixed address, `~/countsense`, `.venv` and `~/pi_wheels` come along. Do step 3 to copy the latest code, then run step 4 again as a check (it reuses `.venv`).
- **A different card or Pi** (fresh install, or a Pi handed out at the venue): do steps 2 to 4. This is what we did on 2026-11-14: a new Pi 5 with its own card, set up over the direct cable. Check `python3 --version` first. It must be 3.13 (Trixie). An older OS such as Bookworm has Python 3.11: the wheels in `pi_wheels/` do not fit it, and the package list has to be resolved and downloaded again for that Python, which needs internet on the laptop. If the user name is not `test`, replace `test@` in every command.


### 1. On the laptop: the Pi's packages (`pi_wheels/`, needs internet once)

`pi_wheels/` is already prepared (53 files, 308 MB) and checked. To rebuild it,
run these in PowerShell from `D:\AI`:

```powershell
.venv\Scripts\python.exe -m pip wheel --no-deps antlr4-python3-runtime==4.9.3 -w pi_wheels
.venv\Scripts\python.exe -m pip download --no-deps -r requirements-pi.txt -d pi_wheels --find-links pi_wheels --only-binary=:all: --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --python-version 3.13 --implementation cp --abi cp313 --retries 20 --timeout 60 --resume-retries 50
.venv\Scripts\python.exe -m pip download --no-deps torch==2.14.1+cpu -d pi_wheels --index-url https://download.pytorch.org/whl/cpu --only-binary=:all: --platform manylinux_2_28_aarch64 --python-version 3.13 --implementation cp --abi cp313 --retries 20 --timeout 60 --resume-retries 50
.venv\Scripts\python.exe -m pip download --no-deps pip opencv-python-headless==5.0.0.93 -d pi_wheels --only-binary=:all: --platform manylinux2014_aarch64 --python-version 3.13 --implementation cp --abi cp313 --retries 20 --timeout 60
```

1. The first command builds `antlr4`, which only exists as source code.
2. The second gets the 49 packages of `requirements-pi.txt` from PyPI.
3. The third gets torch **separately** from the PyTorch CPU index. The file must be named `torch-2.14.1+cpu-...aarch64.whl` (with `+cpu`).
4. The fourth gets `pip` (for the venv fallback) and `opencv-python-headless` (for the libGL fallback).

If a download breaks, run the same command again: finished files are kept. Expect 53 files.

### 2. Connect laptop and Pi (one time)

1. **First login** (the Pi has no fixed address yet). Either:
   - **Option 1:** cable laptop to Pi. On the laptop: `Win + R` -> `ncpa.cpl` -> right-click **Wi-Fi** -> Properties -> **Sharing** -> tick "Allow other network users..." -> choose **Ethernet**. Power on the Pi, wait 3 minutes, then `ssh test@raspberrypi.local`. If not found, run `arp -a` and look under `192.168.137.1` for a hardware ID starting `2c-cf-67`, `d8-3a-dd` or `88-a2-9e` (a Pi 4 starts `dc-a6-32` or `e4-5f-01`).
   - **Option 2:** use a monitor and keyboard on the Pi, and open Terminal.
2. **On the Pi**, switch SSH on (`sudo systemctl enable --now ssh`) and give it the fixed address, then reboot:
   ```bash
   sudo nmcli connection add type ethernet ifname eth0 con-name direct-pc ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv6.method link-local connection.autoconnect yes connection.autoconnect-priority 100
   sudo reboot
   ```
3. **On the laptop**, set up the Ethernet adapter:
   1. If you used Option 1, untick the Wi-Fi sharing again.
   2. In `ncpa.cpl`, right-click **Ethernet** (Realtek PCIe GbE, *not* a phone-tethering adapter like "Ethernet 3") -> Properties -> **Internet Protocol Version 4** -> Use the following IP address: `192.168.50.1`, mask `255.255.255.0`, no gateway.
4. Test: `ping 192.168.50.2`, then `ssh test@192.168.50.2`. **From now on always use this address.** If an earlier Pi used it, run `ssh-keygen -R 192.168.50.2` on the laptop first.

### 3. Copy everything to the Pi

From `D:\AI` on the laptop:

```bash
ssh test@192.168.50.2 mkdir -p countsense
scp -r pi_wheels test@192.168.50.2:
scp -r run_poc.py countsense scripts data yolo11n_320 yolo11n_416 yolo11n_640 yolo11n_960 yolo11n_1280 requirements-pi.txt README.md test@192.168.50.2:countsense/
```

Never copy `.venv` (it's the laptop's Windows environment), `.tmp`, `.pipcache`,
`.ultralytics` or `outputs`. Run the last command again whenever the code changes.

### 4. Install on the Pi (one command)

```bash
cd ~/countsense
bash scripts/setup_pi.sh
```

It prints the board (`Raspberry Pi 5 Model B ...`), checks the system (64-bit,
Python 3.13, at least 2 GB free), creates `.venv` (or reuses it),
installs all packages offline from `~/pi_wheels`, verifies them, and finishes by
detecting objects in `data/bus.jpg`. Setup is good when you see `executorch: 1.5.1`,
`XNNPACK available: True` and **5 detections** (1 bus + 4 persons).

### 5. Run on the Pi and fetch the results

```bash
cd ~/countsense
.venv/bin/python run_poc.py --source data/clip.mp4 --benchmark
.venv/bin/python run_poc.py --source data/clip.mp4 --preset traffic --model yolo11n_320
```

Back on the laptop (from `D:\AI`), copy a judge's clip over, or bring the results home:

```bash
scp data/clip.mp4 test@192.168.50.2:countsense/data/
scp -r test@192.168.50.2:countsense/outputs outputs_pi
```

### At the venue

1. **Same cable, same address:** `ssh test@192.168.50.2`. Nothing to set up.
2. **Power and cooling.** The Pi 5 needs more of both than the Pi 4. Use the official 27 W (5 V, 5 A) USB-C supply; a 3 A phone charger can sag under load. Fit the Active Cooler or a fan case: a bare Pi 5 running 4 threads reaches 80 °C and slows itself down.
3. **Set the clock.** The Pi 5 has a clock chip, but it only keeps time with the optional clock battery fitted, and results folders are named by date and time:
   ```bash
   sudo date -s "2026-11-14 10:00"
   ```
4. **Before timed runs** (resets on reboot), lock the CPU at full speed:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
5. **During a run**, watch the temperature in a second SSH window: `watch -n 1 vcgencmd measure_temp`. Under 75 °C is fine; 80 °C means throttling.
6. **After a run**, check `vcgencmd get_throttled`. It must be `throttled=0x0`; anything else means the power supply or cooling slowed the Pi down.
7. **Optional, steadier timing:** boot to the console instead of the desktop (`sudo raspi-config` -> System Options -> Boot -> Console).
8. **Shut down safely:** run `sudo shutdown -h now` and unplug only after the LED turns red (on a Pi 4: after the green LED stops).

---

## Laptop setup from scratch (Windows, D: drive)

Python 3.10-3.12. Keep the venv, caches and temp files on `D:`. In PowerShell, from `D:\AI`:

```powershell
python -m venv .venv
$env:TMP='D:\AI\.tmp'; $env:TEMP='D:\AI\.tmp'; $env:PIP_CACHE_DIR='D:\AI\.pipcache'
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install ultralytics executorch opencv-python numpy
```

CPU torch goes in first, so nothing pulls the 3 GB CUDA build. torchvision and
ultralytics are needed only on the laptop, for exporting and for the verify script.

**Exporting** each size (320, 416, 640, 960, 1280): run from `D:\AI`, then rename the
output folder `yolo11n_executorch_model` to `yolo11n_<size>`:

```powershell
$env:YOLO_CONFIG_DIR='D:\AI\.ultralytics'; $env:FLATC_EXECUTABLE='D:\AI\.venv\Lib\site-packages\executorch\data\bin\flatc.exe'
.venv\Scripts\yolo.exe export model=yolo11n.pt format=executorch imgsz=640
```

`FLATC_EXECUTABLE` works around an ExecuTorch 1.5.1 bug on Windows. Without it the
export fails with `FileNotFoundError: [WinError 2]`, because ExecuTorch looks for
`flatc` instead of the bundled `flatc.exe`. Exporting only happens on the laptop; the
Pi never needs `flatc`.

**Verify** that our pipeline matches Ultralytics (laptop only):

```bash
.venv/Scripts/python.exe scripts/verify_against_ultralytics.py
.venv/Scripts/python.exe scripts/verify_against_ultralytics.py --source data/clip.mp4 --preset traffic
```

This runs both pipelines on the **same `.pte`** and the same frames, for every model
size, with the same NMS setting on both sides. It ends with `PASS` and exit code 0 if
every per-class count is identical. Last result (NMS IoU 0.5): 6 of 6 frame-model pairs
identical on the sample images and 60 of 60 on `data/test_clip.mp4`.

---

## How it works (in one paragraph per step)

- **Model loading** (`io.py`): `Runtime.get().load_program(pte).load_method("forward")`. Class names and the input size S come from `metadata.yaml`, read line by line so the Pi needs no YAML library; if it's missing, the COCO-80 names are the fallback. Each `.pte` has a fixed input shape, so S is checked against it. The XNNPACK thread pool is sized with `--threads` before loading.
- **Preprocessing** (`preprocess.py`): resize so the long side is S, pad with grey (114) to S x S, BGR->RGB, HWC->CHW, /255, float32 `(1, 3, S, S)`. The scale and padding are kept so boxes map back to the original pixels.
- **Decoding** (`decode.py`, pure NumPy):
  1. Transpose `(1, 84, N)` to N rows; N is read from the output (2100 / 3549 / 8400 for 320 / 416 / 640).
  2. Take the best class per row.
  3. Filter by confidence and by the preset's classes **before** NMS, so NMS sees dozens of boxes instead of thousands.
  4. Convert xywh to xyxy, undo the letterbox, clip to the image.
  5. Run greedy NMS at IoU 0.5, separately per class. Ultralytics' default is 0.7, but at 0.7 one person often kept two or three boxes (a tight box inside a loose one) and was counted more than once. On four hand-counted frames of `data/test_clip.mp4`, 0.7 gave 5, 6, 7 and 4 boxes for 4, 4, 4 and 3 people; 0.5 gives 4, 4, 4 and 2. The cost: two real objects that overlap by more than half (one almost hidden behind the other) become one.
- **Counting** (`counting.py`): each detection's class maps to a category, giving counts per frame. The total gives the density level. Before counting, one object detected as two classes is reduced to one box: a van is often detected as "car" and "truck", a coach as "bus" and "truck", and the per-class NMS keeps both. So NMS runs once more per category and keeps the better-scored box. On the traffic test video this removed 268 double counts, spread over 225 of 1192 frames.
- **Timing** (`run_poc.py: process_frame`, `timing.py`):
  - `perf_counter` around preprocess, inference and postprocess separately.
  - One warm-up inference on a blank image runs before timing starts, because the first run sets up XNNPACK.
  - Reported as mean, p50 and p95, plus processing FPS, end-to-end frames/s, and whole-run wall clock (measured from before torch is imported).
- **Stride** (`io.py: FrameSource`): skipped frames are only grabbed, not converted. Every record keeps its true frame number and timestamp (frame ÷ fps).

## Known limitations

- **Per-frame counts by default.** The same car in 30 frames is 30 sightings, so videos are summarised as average and maximum per frame. `--count-line` adds how many objects passed a line, using a simple tracker with the limits listed in its section.
- **Riders are counted twice.** A person on a bicycle or motorcycle counts as a pedestrian plus a cyclist or vehicle. Telling riders from walkers would need tracking or a model with a "rider" class.
- **Only the COCO classes the model knows.** There's no auto-rickshaw class: autos usually come out as car, motorcycle or truck, which still count as vehicles. Trams usually come out as `train`, which is not counted.
- **Small or distant objects are missed, more so at the smaller model sizes.** A vehicle that is only a few pixels wide in the model's input is not detected, so on a long road the far end is under-counted. `yolo11n_960` and `yolo11n_1280` see further (see "Accuracy check on real footage"); `--benchmark` shows the difference on your own clip.
- **Scooters and motorcycles seen from above are mostly not detected.** On the traffic test video the model found the rider as a "person" and missed the scooter itself, at every model size. So a scooter passing the line shows up as one pedestrian, not one vehicle.
- **The verify script proves equivalence, not accuracy.** It shows our pipeline equals Ultralytics on the same model, not that the model is right.
- **The shared-space map is approximate.** It uses the bottom of each box as the ground position, which is rough for large or partly hidden objects; the rider rule is a simple geometric guess; and a single false detection can colour a patch of ground.

## Model and licence

The detector is Ultralytics YOLO11n with its published COCO-pretrained weights; we did not
train or fine-tune it. The weights and the exported `.pte` files are licensed **AGPL-3.0**
by Ultralytics (each `metadata.yaml` records this), so a project that distributes them has
to be open-sourced under AGPL-3.0 as well, or use an Ultralytics Enterprise licence.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open video` | Re-export the clip as H.264 `.mp4` (some phone formats such as HEVC `.mov` may not decode) |
| `--imgsz 640 does not match ...` | Each `.pte` has a fixed size: use `--model yolo11n_640` instead of `--imgsz` |
| `setup_pi.sh`: `wheels are built for Python 3.13` | The Pi's Python differs: rebuild `pi_wheels` for that version |
| `setup_pi.sh`: `No matching distribution found for X` | X is missing from `pi_wheels`: download it on the laptop (step 1), copy it into `~/pi_wheels`, run setup again |
| `ImportError: libGL.so.1` | `setup_pi.sh` switches to `opencv-python-headless` automatically (same OpenCV, no windows; we never open any) |
| `ping 192.168.50.2` fails | Check the laptop's Ethernet IPv4 is `192.168.50.1` / `255.255.255.0` and Wi-Fi sharing is off; replug, wait a minute. Still failing: monitor + keyboard on the Pi, `ip addr show eth0` |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | On the laptop: `ssh-keygen -R 192.168.50.2` |
| Inference gets slower during a long run, or the Pi 5 is no faster than the Pi 4 | Heat or a weak power supply. Check `vcgencmd measure_temp` and `get_throttled`; improve cooling, use the 5 A supply, set the `performance` governor, use a smaller model or `--stride` |
| Lines like `[cpuinfo_utils.cpp...]`, `[threadpool.cpp...]`, `W... KernelPreference` | Harmless log messages from ExecuTorch and torch |
