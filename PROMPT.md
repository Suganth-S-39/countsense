Build "CountSense Edge": an offline vehicle and pedestrian detection and
counting pipeline using YOLO11n on PyTorch ExecuTorch (XNNPACK, FP32).
Developed on a laptop, then deployed to a Raspberry Pi 4 (64-bit OS,
4x Cortex-A72, no GPU). Hackathon in 2 days. At the event a judge hands us
a 10-20 s clip or 10-20 images; we must process it ON THE PI and report
per-category counts, sample annotated outputs, average inference time per
frame, the model used and throughput.

HARD CONSTRAINTS
- Runtime deps on the Pi: executorch, torch (CPU), numpy, opencv-python
  ONLY. No ultralytics, no torchvision in run_poc.py or anything it
  imports. Ultralytics is allowed ONLY in scripts/verify_against_ultralytics.py.
- All post-processing in pure NumPy.
- Fully offline. No cloud calls, no uploads, no downloads at runtime.
- pathlib for all paths (must work on Windows, Linux and the Pi).
- Simple, readable code. I have to explain every part to judges.
- Do not add features I did not ask for.

WINDOWS / DISK CONSTRAINT (important)
- The project lives on D: (D:\AI). The C: drive is nearly full. Nothing
  large may be written to C:.
- Create the venv here: D:\AI\.venv
- Each bash command runs in a fresh shell, so activation does not
  persist. Always call the venv explicitly:
    .venv\Scripts\python.exe -m pip install ...
    .venv\Scripts\python.exe run_poc.py ...
- Before installing, create D:\AI\.tmp and D:\AI\.pipcache, then run pip
  with the cache and temp redirected to them, e.g. in cmd:
    set TMP=D:\AI\.tmp
    set TEMP=D:\AI\.tmp
    set PIP_CACHE_DIR=D:\AI\.pipcache
    .venv\Scripts\python.exe -m pip install ...
  Set these in the SAME command as the install, since the shell resets.
- Install CPU-only torch so pip does not pull ~3 GB of CUDA libraries:
    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  Do this BEFORE installing ultralytics or executorch, so neither pulls
  the default CUDA build.
- Run the yolo export from D:\AI so yolo11n.pt downloads here, not to C.
  If Ultralytics still writes weights or settings to C, set
  YOLO_CONFIG_DIR=D:\AI\.ultralytics in the same command.
- After Stage 0, report free space on D: and the total size of .venv.

STAGE 0 - ENVIRONMENT AND EXPORT (do this first, then stop and report)
- Check `python --version`. If not 3.10-3.12, stop and tell me.
- Create the project structure and the .venv here.
- Install, in this order: CPU torch (as above), then ultralytics,
  executorch, opencv-python, numpy.
- Export: yolo export model=yolo11n.pt format=executorch imgsz=640
  Rename the output folder to yolo11n_640/.
- Also export imgsz=320 and imgsz=416 into yolo11n_320/ and yolo11n_416/.
  (The Pi 4 is slow; I need smaller inputs ready to swap in.)
- Confirm each folder has a .pte and metadata.yaml; print each .pte size in MB.
- Create data/ and fetch a couple of sample images for testing
  (e.g. Ultralytics bus.jpg, zidane.jpg). I will add my own clips myself.
- If anything fails, show me the full error and propose a fix. Do NOT
  silently work around it by switching runtimes or formats - ExecuTorch
  is required by the brief.

PIPELINE
1. Load the .pte via executorch.runtime.Runtime.get() -> load_program ->
   load_method("forward"). Read class names from metadata.yaml, falling
   back to COCO-80 names.
2. Input: a video file, a folder of images, or a single image.
3. Preprocess: letterbox to S x S keeping aspect ratio (pad 114),
   BGR->RGB, HWC->CHW, /255, float32, shape (1,3,S,S). Keep scale and
   padding so boxes map back to original pixels. Read S from the model
   metadata or --imgsz; never hardcode 640.
4. Inference: run the method. Output is (1, 84, N): 4 box values + 80
   class scores. Derive N from the output; do not assume 8400.
5. Decode in NumPy: transpose, max class score per anchor, filter by
   confidence (default 0.25) and by the preset's class IDs, xywh->xyxy,
   undo letterbox, clip to bounds, class-aware NMS at IoU 0.7.
   Filter by class BEFORE NMS so we never run NMS over 8400 boxes.
6. Categories (COCO ids): pedestrian = person(0); cyclist = bicycle(1);
   vehicle = car(2) + motorcycle(3) + bus(5) + truck(7).
7. Density from total count: LOW < 8, MODERATE 8-15, HIGH >= 16.
   Thresholds configurable via CLI or a small JSON.
8. Annotated output: boxes, label + confidence, and a header bar showing
   per-category counts, total, density and inference ms. Save to
   annotated/*.jpg. Optional --save-video mp4.
9. Timing: time.perf_counter around preprocess, inference and
   postprocess separately. Exclude a warm-up frame. Report mean, p50,
   p95 in ms, plus FPS. ALSO report wall-clock time for the whole run
   and how many frames were actually processed vs skipped by stride.
10. Outputs in outputs/<preset>_<timestamp>/: counts.csv (per frame),
    detections.json, summary.json, annotated/, REPORT.md (model name and
    size, imgsz, thread count, device info from /proc/cpuinfo, avg and
    p95 inference ms, FPS, per-category avg and max counts, density
    distribution, stride used).

SPEED (important - the Pi 4 is the bottleneck)
- Set XNNPACK/torch threads to 4 by default, with --threads to override.
  Make the chosen value visible in REPORT.md.
- --stride N (default 1) to process every Nth video frame. Counts and
  CSV must record true frame indices and timestamps so stride does not
  corrupt the per-frame record.
- --max-frames N to cap a run.
- Make swapping between the 320/416/640 .pte a single --model flag.
- Add --benchmark MODE: runs the first ~20 frames at each available
  imgsz and prints a comparison table of avg ms, FPS and total counts,
  so I can pick a setting at the venue in under a minute.
- Do not micro-optimise elsewhere. Resolution, stride and threads are
  where the time is.

CLI
python run_poc.py --source data/clip.mp4 --preset traffic
Presets: traffic (vehicle, cyclist, pedestrian), pedestrian, public-space.
Flags: --model, --imgsz, --conf, --stride, --max-frames, --threads,
--save-video, --benchmark, --out.

STRUCTURE
run_poc.py, countsense/ (io.py, preprocess.py, decode.py, counting.py,
annotate.py, timing.py, report.py), scripts/verify_against_ultralytics.py,
scripts/setup_pi.sh, requirements-pi.txt, README.md.

HOW TO WORK
Build in stages. STOP after each one and give me a command to verify it.
Do not write everything at once.
 Stage 0: environment + exports (above).
 Stage 1: load .pte, run one image, print output shape and top detections.
 Stage 2: preprocess + decode + NMS, draw boxes on one image.
 *** After Stage 2 I will copy the project to the Pi 4 and run it there.
     WAIT for my result before starting Stage 3. ***
 Stage 3: counting, density, per-frame CSV, video and folder input, stride.
 Stage 4: timing, REPORT.md, summary.json, annotated header bar, --benchmark.
 Stage 5: verify script comparing my decoder's per-class counts against
          Ultralytics on the same images, for each .pte size.
 Stage 6: setup_pi.sh, requirements-pi.txt, README with Pi instructions.
At each stage explain briefly WHY the code is written that way and what
could go wrong.
