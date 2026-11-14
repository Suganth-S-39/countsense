"""CountSense Edge: offline vehicle and pedestrian counting with YOLO11n on ExecuTorch.

Video / image folder / single image -> per-frame detection and counting ->
counts.csv, detections.json, summary.json, REPORT.md and annotated images.
--benchmark compares every exported input size on the first frames instead.
"""
import time

RUN_START = time.perf_counter()  # the whole-run wall clock starts before the slow imports (torch, ExecuTorch)

import argparse  # noqa: E402
import re  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from countsense.annotate import (add_header, blur_people, draw_count_line, draw_detections,  # noqa: E402
                                 draw_notices)
from countsense.counting import (CONFIDENT_SCORE, DEFAULT_DENSITY, PRESETS, count_frame,  # noqa: E402
                                 density_level, one_box_per_object, preset_class_ids, range_text)
from countsense.decode import IOU_THRES, decode  # noqa: E402
from countsense.io import (FrameSource, VideoOut, find_pte, load_model, model_description,  # noqa: E402
                           model_input_size, read_metadata, write_image)
from countsense.items import WATCH_NAMES, check_items, watch_class_ids  # noqa: E402
from countsense.preprocess import letterbox  # noqa: E402
from countsense.report import (build_summary, detections_to_list, write_counts_csv,  # noqa: E402
                               write_detections_json, write_report_md, write_summary_json)
from countsense.sharedspace import SharedSpace  # noqa: E402
from countsense.timeline import draw_timeline, timeline_stats  # noqa: E402
from countsense.timing import timing_stats  # noqa: E402
from countsense.tracking import LineCounter, parse_line  # noqa: E402

BENCHMARK_FRAMES = 20


def parse_args():
    p = argparse.ArgumentParser(description="CountSense Edge - offline vehicle and pedestrian counting")
    p.add_argument("--source", type=Path, required=True, help="a video file, a folder of images, or one image")
    p.add_argument("--preset", choices=PRESETS, default="traffic", help="which categories to count")
    p.add_argument("--model", type=Path, default=Path("yolo11n_640"),
                   help="model folder (yolo11n_320 / yolo11n_416 / yolo11n_640) or a .pte file")
    p.add_argument("--imgsz", type=int, default=None,
                   help="input size S (default: read from the model's metadata.yaml)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--stride", type=int, default=1,
                   help="process every Nth frame; true frame numbers and timestamps are kept")
    p.add_argument("--max-frames", type=int, default=None, help="stop after processing N frames")
    p.add_argument("--density", type=int, nargs=2, default=list(DEFAULT_DENSITY), metavar=("LOW", "HIGH"),
                   help="thresholds on the total count: LOW if total < LOW, HIGH if total >= HIGH, "
                        "MODERATE in between (default: 8 16)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for XNNPACK (the Pi 5 has 4 cores)")
    p.add_argument("--save-video", action="store_true", help="also save the annotated frames as annotated.mp4")
    p.add_argument("--shared-space", action="store_true",
                   help="also save shared_space.jpg: where pedestrians and vehicles used the same ground "
                        "(fixed-camera video with the traffic preset only)")
    p.add_argument("--watch-items", action="store_true",
                   help=f"also print a notice when one of these is detected next to a person: "
                        f"{', '.join(WATCH_NAMES)}. Unverified object detections for a human to check")
    p.add_argument("--blur-people", action="store_true",
                   help="privacy: pixelate the head area of every detected person in all saved images and video")
    p.add_argument("--timeline", action="store_true",
                   help="also save timeline.png (the counts over the run) and report the busiest moment "
                        "and the busy periods")
    p.add_argument("--count-line", nargs="?", const="h50", default=None, metavar="LINE",
                   help="count each confident object once when it crosses a line, with its direction "
                        "(fixed-camera video only). LINE is h or v plus a percentage: h50 (the default) is a "
                        "horizontal line at half the height, v30 a vertical line at 30%% of the width")
    p.add_argument("--benchmark", action="store_true",
                   help=f"compare every model folder next to --model on the first {BENCHMARK_FRAMES} "
                        f"frames (or --max-frames) and print a table; writes no files")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="root folder for results")
    args = p.parse_args()
    if args.stride < 1:
        p.error("--stride must be 1 or more")
    if args.max_frames is not None and args.max_frames < 1:
        p.error("--max-frames must be 1 or more")
    if args.density[0] > args.density[1]:
        p.error("--density: LOW must not be greater than HIGH")
    if args.count_line and not parse_line(args.count_line):
        p.error("--count-line: use h or v plus a percentage from 1 to 99, for example h50 or v30")
    return args


def prepare_model(model_arg, imgsz_arg, threads):
    """Load the .pte and work out the input size S. Returns (pte_path, method, imgsz, names)."""
    pte_path = find_pte(model_arg)
    meta_imgsz, names = read_metadata(pte_path.parent)
    method = load_model(pte_path, threads)

    # The .pte only accepts the size it was exported at, so S must agree with it.
    pte_imgsz = model_input_size(method)
    imgsz = imgsz_arg or meta_imgsz or pte_imgsz
    if imgsz != pte_imgsz:
        raise SystemExit(f"--imgsz {imgsz} does not match {pte_path} (exported at {pte_imgsz}). "
                         f"Use --model yolo11n_{imgsz} instead.")
    return pte_path, method, imgsz, names


def warm_up(method, imgsz):
    """Run one inference on a blank image before timing anything. The first run
    sets up XNNPACK and is much slower, so it would distort the averages."""
    method.execute([torch.zeros(1, 3, imgsz, imgsz)])


def process_frame(method, image, imgsz, conf, class_ids):
    """Detect objects in one image, timing the three stages separately.
    Returns (boxes, scores, class_ids, {"pre", "infer", "post"} in ms)."""
    t0 = time.perf_counter()
    blob, scale, pad = letterbox(image, imgsz)
    t1 = time.perf_counter()
    output = method.execute([torch.from_numpy(blob)])[0].numpy()
    t2 = time.perf_counter()
    boxes, scores, cls = decode(output, scale, pad, image.shape, conf_thres=conf, class_ids=class_ids)
    t3 = time.perf_counter()
    timing = {"pre": (t1 - t0) * 1000, "infer": (t2 - t1) * 1000, "post": (t3 - t2) * 1000}
    return boxes, scores, cls, timing


def run(args):
    source = FrameSource(args.source, stride=args.stride, max_frames=args.max_frames)
    categories = PRESETS[args.preset]
    class_ids = preset_class_ids(args.preset)
    if args.shared_space and (source.kind != "video" or not {"pedestrian", "vehicle"} <= set(categories)):
        raise SystemExit("--shared-space needs a video and a preset with pedestrians and vehicles (traffic).")
    if args.count_line and source.kind != "video":
        raise SystemExit("--count-line needs a video: a crossing is only seen across consecutive frames.")
    pte_path, method, imgsz, names = prepare_model(args.model, args.imgsz, args.threads)
    # --watch-items: also let the watched items through the class filter (they are never counted).
    watch_ids = watch_class_ids(names) if args.watch_items else []
    if args.watch_items and not watch_ids:
        raise SystemExit(f"--watch-items: this model has none of these classes: {', '.join(WATCH_NAMES)}")

    started = datetime.now()
    # Results folder = the input's name + preset + start time, e.g. knife_traffic_20261114_142122.
    # The input's name is cut to letters, digits and "_" so it is a safe folder name everywhere.
    source_name = re.sub(r"[^A-Za-z0-9]+", "_", args.source.stem).strip("_")[:30] or "input"
    run_dir = args.out / f"{source_name}_{args.preset}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fps_note = f", {source.fps:.1f} fps" if source.fps else ""
    print(f"Model  : {pte_path} ({pte_path.stat().st_size / 2**20:.2f} MB)  imgsz={imgsz}  threads={args.threads}")
    print(f"Source : {args.source} ({source.kind}{fps_note})  stride={args.stride}  max_frames={args.max_frames}")
    print(f"Preset : {args.preset} -> {', '.join(categories)}  conf > {args.conf}  "
          f"density LOW < {args.density[0]} <= MODERATE < {args.density[1]} <= HIGH")
    if args.blur_people:
        print("Privacy: the head area of every detected person is pixelated in all saved images")
    if args.count_line:
        print(f"Line   : {args.count_line} - each confident object is counted once when its centre crosses the line")
    print()

    warm_up(method, imgsz)
    # Saved video plays at real speed: with stride N we keep every Nth frame, so fps / N.
    # Image folders have no fps, so show one image per second.
    video_fps = source.fps / args.stride if source.fps else 1.0
    video = VideoOut(run_dir / "annotated.mp4", video_fps) if args.save_video else None

    records = []
    shared_space = None
    line_counter = None
    loop_start = time.perf_counter()
    for frame in source:
        boxes, scores, cls, timing = process_frame(method, frame.image, imgsz, args.conf, class_ids + watch_ids)
        notices = None                                   # None = not watching; [] = watching, nothing found
        if args.watch_items:
            notices, boxes, scores, cls = check_items(boxes, scores, cls, watch_ids, names)
        # One vehicle detected as two classes (car + truck) is one object, not two.
        boxes, scores, cls = one_box_per_object(boxes, scores, cls)
        # Every count is kept twice: all detections, and only the confident ones (score >= 0.5).
        counts = count_frame(cls, categories)
        confident = count_frame(cls[scores >= CONFIDENT_SCORE], categories)
        total, total_confident = sum(counts.values()), sum(confident.values())
        density = density_level(total, args.density)
        density_confident = density_level(total_confident, args.density)
        borderline = density != density_confident      # the level depends on low-confidence detections
        # --count-line: link the boxes to the previous frame's and see what crossed the line.
        track_ids = crossings = None                    # None = not counting at a line
        if args.count_line:
            line_counter = line_counter or LineCounter(*parse_line(args.count_line), frame.image.shape, categories,
                                                       source.fps / args.stride if source.fps else None)
            track_ids, crossings = line_counter.update(boxes, scores, cls)
        records.append({"frame_index": frame.index, "time_s": frame.time_s, "name": frame.name,
                        "counts": counts, "counts_confident": confident, "total": total,
                        "total_confident": total_confident, "density": density, "borderline": borderline,
                        "notices": notices, "crossings": crossings,
                        "passed_so_far": line_counter.total() if line_counter else None, "timing_ms": timing,
                        "detections": detections_to_list(boxes, scores, cls, names, track_ids)})

        # --blur-people: the model has seen the real frame; everything saved from here on is blurred.
        picture = blur_people(frame.image, boxes, cls) if args.blur_people else frame.image
        annotated = draw_notices(draw_detections(picture, boxes, scores, cls, names, track_ids), notices or [])
        if line_counter:
            draw_count_line(annotated, line_counter.endpoints, line_counter.tally_text())
        annotated = add_header(annotated, counts, confident, density, density_confident, timing["infer"])
        write_image(run_dir / "annotated" / f"{frame.name}.jpg", annotated)
        if video:
            video.write(annotated)
        if args.shared_space:
            shared_space = shared_space or SharedSpace(frame.image.shape)
            shared_space.add(picture, boxes, cls)

        when = f"t={frame.time_s:.2f}s" if frame.time_s is not None else ""
        counts_str = "  ".join(f"{cat}={range_text(confident[cat], n)}" for cat, n in counts.items())
        level = f"{density}?" if borderline else density
        passed = f"  passed {line_counter.total()}" if line_counter else ""
        print(f"{frame.name:<16} {when:<10} {counts_str}  total={range_text(total_confident, total):<5} "
              f"{level:<9} infer {timing['infer']:.0f} ms{passed}")
        for n in notices or []:
            # The alert line stands on its own: when (for a video), which frame, what, and where to look.
            at = f"t={frame.time_s:.2f}s, " if frame.time_s is not None else ""
            print(f"    POSSIBLE THREAT at {at}{frame.name}: {n['item']} next to a person "
                  f"(score {n['conf']:.2f}), unverified - check annotated/{frame.name}.jpg")
        for c in crossings or []:
            print(f"    PASSED: {c['category']} #{c['track_id']} crossed the line going {c['direction']}")
    loop_s = time.perf_counter() - loop_start
    if video:
        video.close()
    if not records:
        raise SystemExit("No frames were processed.")
    if shared_space:
        write_image(run_dir / "shared_space.jpg", shared_space.render())
    timeline = None
    if args.timeline and len(records) < 2:
        print("\nTimeline skipped: it needs at least two processed frames.")
    elif args.timeline:
        write_image(run_dir / "timeline.png", draw_timeline(records, categories, args.density))
        timeline = timeline_stats(records)
    passing = None
    if line_counter:
        passing = line_counter.stats(args.count_line, source.frames_read / source.fps if source.fps else None)
        passing["events"] = [{"frame": r["name"], "time_s": r["time_s"], **c}
                             for r in records for c in r["crossings"]]

    write_counts_csv(run_dir / "counts.csv", records, categories)
    write_detections_json(run_dir / "detections.json", records, source=str(args.source),
                          model=str(pte_path), imgsz=imgsz, preset=args.preset, conf=args.conf,
                          stride=args.stride, fps=source.fps)
    summary = build_summary(
        records, categories,
        run_info={"started": f"{started:%Y-%m-%d %H:%M:%S}", "source": str(args.source),
                  "source_kind": source.kind, "source_fps": source.fps, "preset": args.preset,
                  "categories": categories, "conf": args.conf, "iou": IOU_THRES, "stride": args.stride,
                  "max_frames": args.max_frames, "density_thresholds": list(args.density),
                  "save_video": args.save_video, "blur_people": args.blur_people,
                  "watch_items": list(WATCH_NAMES) if args.watch_items else None},
        model_info={"name": pte_path.parent.name, "file": str(pte_path),
                    "description": model_description(pte_path.parent),
                    "size_mb": round(pte_path.stat().st_size / 2**20, 2), "num_classes": len(names),
                    "runtime": "ExecuTorch, XNNPACK backend, FP32", "imgsz": imgsz, "threads": args.threads},
        frames_info={"read": source.frames_read, "processed": len(records),
                     "skipped_by_stride": source.frames_read - len(records)},
        wall_info={"frame_loop_s": round(loop_s, 2),
                   "whole_run_s": round(time.perf_counter() - RUN_START, 2)},
        shared_space=shared_space.stats() if shared_space else None,
        passing=passing, timeline=timeline,
    )
    write_summary_json(run_dir / "summary.json", summary)
    write_report_md(run_dir / "REPORT.md", summary, records)

    t, fps, cps = summary["timing_ms"], summary["fps"], summary["counts_per_frame"]
    print(f"\nFrames     : {len(records)} processed, {summary['frames']['skipped_by_stride']} skipped by stride "
          f"(of {source.frames_read} read)")
    print(f"Inference  : mean {t['inference']['mean']} ms, p50 {t['inference']['p50']} ms, "
          f"p95 {t['inference']['p95']} ms")
    print(f"Speed      : {fps['processing']} FPS processing, {fps['end_to_end']} frames/s end-to-end, "
          f"whole run {summary['wall_clock_s']['whole_run_s']} s")
    print("Per frame  : " + "  ".join(f"{k} avg {range_text(v['avg_confident'], v['avg'])} / max {v['max']}"
                                      for k, v in cps.items()) + "   (confident-possible)")
    print("Density    : " + "  ".join(f"{k} {v}" for k, v in summary["density_frames"].items())
          + f"   borderline frames: {summary['borderline_frames']}")
    if args.blur_people:
        blurred = sum(r["counts"].get("pedestrian", 0) for r in records)
        print(f"Privacy    : {blurred} head areas pixelated in the saved images of {len(records)} frames")
    if passing:
        fwd, back = passing["directions"]
        rate = f"   about {passing['per_minute']} per minute" if passing["per_minute"] is not None else ""
        print(f"Passed     : {passing['total']} crossed the line  ("
              + "  ".join(f"{cat} {n[fwd]} {fwd} / {n[back]} {back}" for cat, n in passing["passed"].items())
              + f"){rate}")
    if timeline:
        peak = timeline["peak"]
        at = peak["name"] if peak["time_s"] is None else f"t={peak['time_s']:.2f}s"
        periods = [b["from"] + " to " + b["to"] if b["from_s"] is None else f"{b['from_s']:.2f}-{b['to_s']:.2f}s"
                   for b in timeline["busy_periods"]]
        more = f" and {len(periods) - 3} more" if len(periods) > 3 else ""
        busy = (f"busy periods ({timeline['top_level']}): " + ", ".join(periods[:3]) + more if periods
                else "no busy period, the density stayed LOW")
        print(f"Busiest    : {peak['total']} objects at {at}   {busy}   (timeline.png)")
    if shared_space:
        print(f"Shared     : {summary['shared_space']['shared_pct_of_pedestrian_space']}% of the ground "
              f"pedestrians used was also used by vehicles (shared_space.jpg)")
    if args.watch_items:
        found = summary["item_notices"]
        first = next((r for r in records if r["notices"]), None)       # the first frame with an alert
        where = "" if first is None else (f", first at {first['name']}" if first["time_s"] is None
                                          else f", first at t={first['time_s']:.2f}s ({first['name']})")
        print(f"Threats    : {found['notices']} possible, in {found['frames']} frame(s){where} - "
              f"{' / '.join(WATCH_NAMES)} next to a person, unverified (listed in REPORT.md)")
    print(f"Results    : {run_dir}  (REPORT.md, summary.json, counts.csv, detections.json, annotated/"
          f"{', annotated.mp4' if video else ''}{', timeline.png' if timeline else ''}"
          f"{', shared_space.jpg' if shared_space else ''})")


def find_models(model_arg):
    """Every model folder next to --model (e.g. yolo11n_320 / _416 / _640)."""
    model_path = Path(model_arg)
    root = model_path.parent.parent if model_path.is_file() else model_path.parent
    return sorted(root.glob("*/*.pte"))


def benchmark(args):
    """Same frames through every available model; print one comparison row per model."""
    n = args.max_frames or BENCHMARK_FRAMES
    frames = list(FrameSource(args.source, stride=args.stride, max_frames=n))  # decode once, reuse for each model
    models = find_models(args.model)
    if not models:
        raise SystemExit(f"No .pte models found next to {args.model}")
    categories = PRESETS[args.preset]
    class_ids = preset_class_ids(args.preset)

    print(f"Benchmark: {len(frames)} frames of {args.source} (stride {args.stride}), preset {args.preset}, "
          f"threads {args.threads}, warm-up excluded\n")
    header = (f"{'model':<14} {'imgsz':>5}  {'infer avg':>9} {'infer p95':>9} {'frame avg':>9} {'FPS':>6}  "
              + " ".join(f"{c:>10}" for c in categories) + f" {'total':>6}")
    print(header)
    print("-" * len(header))
    for pte in models:
        pte_path, method, imgsz, _ = prepare_model(pte, None, args.threads)
        warm_up(method, imgsz)
        timings, sums = [], {cat: 0 for cat in categories}
        for frame in frames:
            boxes, scores, cls, timing = process_frame(method, frame.image, imgsz, args.conf, class_ids)
            _, _, cls = one_box_per_object(boxes, scores, cls)      # same counting rule as a normal run
            timings.append(timing)
            for cat, count in count_frame(cls, categories).items():
                sums[cat] += count
        s = timing_stats(timings)
        print(f"{pte_path.parent.name:<14} {imgsz:>5}  {s['infer']['mean']:>6.1f} ms {s['infer']['p95']:>6.1f} ms "
              f"{s['total']['mean']:>6.1f} ms {s['fps']:>6.1f}  "
              + " ".join(f"{sums[c]:>10}" for c in categories) + f" {sum(sums.values()):>6}")
    print(f"\nms = per frame. 'frame avg' = preprocess + inference + postprocess. FPS = 1000 / frame avg.\n"
          f"Counts are summed over the same {len(frames)} frames: lower counts at a smaller size = missed objects.")


def main():
    args = parse_args()
    if args.benchmark:
        benchmark(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
