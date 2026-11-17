"""Write the run's result files: counts.csv, detections.json, summary.json and REPORT.md."""
import csv
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path

from countsense.counting import CLASS_TO_CATEGORY, CONFIDENT_SCORE, count_stats, density_distribution
from countsense.timing import timing_stats

# ARM "CPU part" codes from /proc/cpuinfo -> core name (Pi 5 = 0xd0b, Pi 4 = 0xd08).
CPU_PARTS = {"0xd03": "Cortex-A53", "0xd08": "Cortex-A72", "0xd0b": "Cortex-A76"}


def detections_to_list(boxes, scores, class_ids, names, track_ids=None):
    """One frame's detections as plain Python values, ready for JSON.
    With --count-line each detection also carries its track number."""
    return [
        {
            "class_id": int(cid),
            "class_name": names[cid],
            "category": CLASS_TO_CATEGORY[int(cid)],
            "conf": round(float(score), 4),
            "box_xyxy": [round(float(v), 1) for v in box],
            **({} if track_ids is None else {"track_id": int(track_ids[i])}),
        }
        for i, (box, score, cid) in enumerate(zip(boxes, scores, class_ids))
    ]


def write_counts_csv(path, records, categories):
    """One row per processed frame, with its TRUE frame index and timestamp (stride-safe)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # "<category>" counts every detection; "<category>_confident" only those scoring >= 0.5.
        # "borderline" = the density level would change if the low-confidence detections were ignored.
        # "item_notices" (only with --watch-items) = watched items detected next to a person in this frame.
        # "passed_so_far" (only with --count-line) = objects that have crossed the line up to this frame.
        watching = records[0]["notices"] is not None
        line_counting = records[0]["passed_so_far"] is not None
        writer.writerow(["frame_index", "time_s", "name", *categories, "total", "density",
                         *(f"{c}_confident" for c in categories), "total_confident", "borderline",
                         *(["item_notices"] if watching else []), *(["passed_so_far"] if line_counting else []),
                         "pre_ms", "infer_ms", "post_ms"])
        for r in records:
            time_s = "" if r["time_s"] is None else f"{r['time_s']:.3f}"
            t = r["timing_ms"]
            writer.writerow([r["frame_index"], time_s, r["name"],
                             *(r["counts"][c] for c in categories), r["total"], r["density"],
                             *(r["counts_confident"][c] for c in categories), r["total_confident"],
                             "yes" if r["borderline"] else "no",
                             *([len(r["notices"])] if watching else []),
                             *([r["passed_so_far"]] if line_counting else []),
                             f"{t['pre']:.2f}", f"{t['infer']:.2f}", f"{t['post']:.2f}"])


def write_detections_json(path, records, **run_info):
    """Every detection of every processed frame, plus a few lines saying what was run."""
    frames = [
        {"frame_index": r["frame_index"], "time_s": r["time_s"], "name": r["name"],
         "detections": r["detections"],
         **({"item_notices": r["notices"]} if r["notices"] is not None else {})}
        for r in records
    ]
    path.write_text(json.dumps({**run_info, "frames": frames}, indent=2), encoding="utf-8")


def device_info():
    """Describe the machine. On the Pi the model and CPU come from /proc/cpuinfo."""
    info = {
        "model": "unknown",
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "arch": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": md.version("torch"),
        "executorch": md.version("executorch"),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        fields = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = (s.strip() for s in line.split(":", 1))
                fields.setdefault(key, value)               # keep the first core's values
        info["model"] = fields.get("Model") or fields.get("model name") or "unknown"
        info["cpu"] = CPU_PARTS.get(fields.get("CPU part"), fields.get("model name", info["cpu"]))
    return info


def build_summary(records, categories, run_info, model_info, frames_info, wall_info, shared_space=None,
                  passing=None, timeline=None, telemetry=None):
    """Everything REPORT.md shows, as one dict (also saved as summary.json)."""
    timing = timing_stats([r["timing_ms"] for r in records])
    return {
        "run": run_info,
        "model": model_info,
        "device": device_info(),
        "frames": frames_info,
        "timing_ms": {"preprocess": timing["pre"], "inference": timing["infer"],
                      "postprocess": timing["post"], "total": timing["total"]},
        "fps": {
            "processing": timing["fps"],
            "end_to_end": round(frames_info["processed"] / wall_info["frame_loop_s"], 2),
        },
        "wall_clock_s": wall_info,
        "counts_per_frame": count_stats(records, categories),
        "density_frames": density_distribution(records),
        "borderline_frames": sum(r["borderline"] for r in records),
        "shared_space": shared_space,            # None unless --shared-space was used
        "passing": passing,                      # None unless --count-line was used
        "timeline": timeline,                    # None unless --timeline was used
        "telemetry": telemetry,                  # None unless --telemetry was used (and readable here)
        "item_notices": None if run_info["watch_items"] is None else {   # None unless --watch-items was used
            "watched": run_info["watch_items"],
            "notices": sum(len(r["notices"]) for r in records),
            "frames": sum(bool(r["notices"]) for r in records),
        },
    }


def write_summary_json(path, summary):
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report_md(path, summary, records):
    """A one-page, human-readable report of the run."""
    run, model, dev = summary["run"], summary["model"], summary["device"]
    frames, t, fps, wall = summary["frames"], summary["timing_ms"], summary["fps"], summary["wall_clock_s"]
    low, high = run["density_thresholds"]
    fps_note = f", {run['source_fps']:.1f} fps" if run["source_fps"] else ""

    cps = summary["counts_per_frame"]
    avg_counts = ", ".join(f"{c} {cps[c]['avg']}" for c in run["categories"])
    lines = [
        "# CountSense Edge - run report",
        "",
        f"**Preset:** {run['preset']} ({', '.join(run['categories'])})  ",
        f"**Source:** `{run['source']}` ({run['source_kind']}{fps_note})  ",
        f"**Started:** {run['started']}"
        + ("  \n**Privacy:** the head area of every detected person is pixelated in all saved images"
           if run["blur_people"] else ""),
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Categories counted | {', '.join(run['categories'])} |",
        f"| Frames / images processed | {frames['processed']} |",
        f"| Average count per frame | {avg_counts} (total {cps['total']['avg']}) |",
        f"| Model used | {model['description']} (pretrained, not retrained), `{model['name']}`, {model['runtime']} |",
        f"| Device | {dev['model']}, {dev['cpu_count']} x {dev['cpu']} |",
        f"| Average inference time | **{t['inference']['mean']} ms** per frame |",
        f"| Throughput | **{fps['end_to_end']} frames/s** end-to-end ({fps['processing']} FPS for processing alone) |",
        "| Counts for every frame | `counts.csv` |",
        "| Sample detection outputs | below, under \"Sample annotated frames\"; every frame is in `annotated/` |",
        "",
        "## Model and device",
        "",
        "| | |",
        "|---|---|",
        f"| Model | {model['name']} (`{model['file']}`, {model['size_mb']} MB, {model['num_classes']} classes) |",
        f"| Pretrained model | {model['description']} |",
        f"| Runtime | {model['runtime']} |",
        f"| Input size | {model['imgsz']} x {model['imgsz']} |",
        f"| Threads | {model['threads']} |",
        f"| Device | {dev['model']} |",
        f"| CPU | {dev['cpu_count']} x {dev['cpu']} ({dev['arch']}) |",
        f"| OS / Python | {dev['os']} / {dev['python']} |",
        f"| torch / executorch | {dev['torch']} / {dev['executorch']} |",
        f"| Confidence / NMS IoU | {run['conf']} / {run['iou']} |",
        "",
        "## Frames",
        "",
        "| Read | Processed | Skipped by stride | Stride | Max frames |",
        "|---:|---:|---:|---:|---:|",
        f"| {frames['read']} | {frames['processed']} | {frames['skipped_by_stride']} | {run['stride']} "
        f"| {run['max_frames'] or '-'} |",
        "",
        "## Speed (per processed frame, warm-up excluded)",
        "",
        "| Stage | mean ms | p50 ms | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Preprocess", "preprocess"), ("Inference", "inference"),
                       ("Postprocess", "postprocess"), ("**Total**", "total")):
        lines.append(f"| {label} | {t[key]['mean']} | {t[key]['p50']} | {t[key]['p95']} |")
    lines += [
        "",
        f"- **Average inference: {t['inference']['mean']} ms** per frame (p95 {t['inference']['p95']} ms)",
        f"- Processing speed: **{fps['processing']} FPS** (1000 / mean total ms)",
        f"- End-to-end throughput: **{fps['end_to_end']} frames/s** ({frames['processed']} frames in "
        f"{wall['frame_loop_s']} s, including video decoding and saving annotated images)",
        f"- Wall-clock for the whole run: **{wall['whole_run_s']} s** (startup, model load, all frames, outputs)",
        "",
        "## Counts per frame",
        "",
        "Counts are per frame (no tracking): the same car seen in 30 frames is counted in each frame.",
        f"*Confident* = detections scoring {CONFIDENT_SCORE} or more; *all* also includes the less certain "
        f"ones (score between {run['conf']} and {CONFIDENT_SCORE}). Objects the model misses are in neither.",
        "",
        "| Category | Average (confident) | Average (all) | Max (all) |",
        "|---|---:|---:|---:|",
    ]
    for key, s in summary["counts_per_frame"].items():
        label = "**Total**" if key == "total" else key
        lines.append(f"| {label} | {s['avg_confident']} | {s['avg']} | {s['max']} |")
    borderline = summary["borderline_frames"]
    lines += [
        "",
        f"## Density (LOW < {low}, MODERATE {low}-{high - 1}, HIGH >= {high})",
        "",
        "| Level | Frames | Share |",
        "|---|---:|---:|",
    ]
    for level, n in summary["density_frames"].items():
        lines.append(f"| {level} | {n} | {100 * n / frames['processed']:.0f}% |")
    lines += ["",
              f"**Borderline frames: {borderline} of {frames['processed']}** "
              f"({100 * borderline / frames['processed']:.0f}%). In these, the density level would be lower "
              f"if only confident detections were counted, so treat their level as uncertain."]

    def when(name, time_s):
        """A moment as seconds into the video, or as the image name when there is no clock."""
        return name if time_s is None else f"{time_s:.2f} s"

    if summary["passing"]:
        p = summary["passing"]
        fwd, back = p["directions"]
        rate = f", about **{p['per_minute']} per minute**" if p["per_minute"] is not None else ""
        lines += [
            "",
            "## Passing counts (fixed camera only)",
            "",
            f"**{p['total']} objects crossed the line**{rate} "
            f"({p['orientation']} line `{p['line']}`, {p['duration_s']} s of video).",
            "",
            f"| Category | {fwd} | {back} | Total |",
            "|---|---:|---:|---:|",
        ]
        for cat, n in p["passed"].items():
            lines.append(f"| {cat} | {n[fwd]} | {n[back]} | {n[fwd] + n[back]} |")
        lines += [
            "",
            "How to read it: each box is linked to the nearest box of the same category in the previous "
            "processed frame, and an object is counted once, when the centre of its box crosses the yellow "
            "line. Unlike the per-frame counts, a car seen in 30 frames is one car here. Only objects the "
            f"model was sure about at least once (score {CONFIDENT_SCORE} or more) are counted, and no "
            "identities are kept. An object that is hidden while it crosses, or lost and found again, can be "
            "missed or counted wrongly, and the result is only meaningful when the camera does not move.",
        ]
        if p["events"]:
            lines += ["", "| Time | Frame | Object | Direction |", "|---:|---|---|---|"]
            lines += [f"| {when('-', e['time_s'])} | {e['frame']} | {e['category']} #{e['track_id']} "
                      f"| {e['direction']} |" for e in p["events"]]

    if summary["timeline"]:
        tl = summary["timeline"]
        lines += [
            "",
            "## Timeline",
            "",
            "![timeline](timeline.png)",
            "",
            f"**Busiest moment:** {tl['peak']['total']} objects at "
            f"{when(tl['peak']['name'], tl['peak']['time_s'])} (`annotated/{tl['peak']['name']}.jpg`).",
            "",
        ]
        if tl["busy_periods"]:
            lines += [f"**Busy periods** (stretches at {tl['top_level']}, the highest level reached; in a video "
                      "a dip shorter than 1 s does not end a stretch):", "",
                      f"| From | To | Frames at {tl['top_level']} | Highest count |", "|---:|---:|---:|---:|"]
            lines += [f"| {when(b['from'], b['from_s'])} | {when(b['to'], b['to_s'])} | {b['frames']} of "
                      f"{b['span']} | {b['max_total']} |" for b in tl["busy_periods"]]
        else:
            lines.append("**Busy periods:** none, the density stayed LOW for the whole run.")

    if summary["telemetry"]:
        tm = summary["telemetry"]

        def row(label, key, unit):
            v = tm[key]
            return f"| {label} | n/a | n/a |" if v is None else f"| {label} | {v['avg']} {unit} | {v['max']} {unit} |"

        code = tm["throttled_code"]
        if code is None:
            slowed = "not readable on this machine"
        elif tm["throttled_during_run"]:
            slowed = (f"**{tm['throttled']}** in at least one reading (code `{code}`): the timings in this "
                      "report are slower than this Pi manages when it is cool and well powered")
        elif code != "0x0":
            slowed = (f"{tm['throttled']} (code `{code}`). It was not slowed down in any reading of this run; "
                      "the Pi remembers an earlier slowdown until it is restarted")
        else:
            slowed = "no (code `0x0`)"
        lines += [
            "",
            "## Pi telemetry during the run",
            "",
            "| | Average | Highest |",
            "|---|---:|---:|",
            row("Temperature", "temp_c", "°C"),
            row("CPU load (all cores)", "cpu_pct", "%"),
            row("CPU clock", "cpu_mhz", "MHz"),
            row(f"Memory in use, whole Pi (of {tm['ram_total_gb']} GB)", "ram_used_gb", "GB"),
            row("Memory used by this run", "run_ram_gb", "GB"),
            row("Board power", "power_w", "W"),
            "",
            f"- Slowed down by heat or low voltage (throttled): {slowed}.",
        ]
        if tm["energy_j"] is not None:
            lines.append(f"- Energy for the frame loop: **{tm['energy_j']} J** "
                         f"({tm['energy_j'] / 3600:.3f} Wh), **{tm['energy_j_per_frame']} J per processed frame**")
        lines.append(f"- {tm['readings']} readings, one every {tm['interval_s']:g} s over {tm['seconds']} s, "
                     "taken while the model was running. Board power is read from the Pi 5's own power "
                     "chip and covers the board only: it leaves out USB devices and the losses in the "
                     "power supply.")

    if summary["shared_space"]:
        s = summary["shared_space"]
        lines += [
            "",
            "## Shared space (fixed camera only)",
            "",
            f"**{s['shared_pct_of_pedestrian_space']}% of the ground that pedestrians used was also used by "
            f"vehicles** at some point in the clip ({s['frames']} frames).",
            "",
            "![shared space](shared_space.jpg)",
            "",
            f"Green = ground pedestrians stood on ({s['pedestrian_area_pct']}% of the picture), blue = ground "
            f"vehicles stood on ({s['vehicle_area_pct']}%), red = both ({s['shared_area_pct']}%). Brighter = "
            "used in more frames.",
            "",
            "How to read it: each frame marks the bottom 20% of every box (feet, wheels) on a grid; the map adds "
            "the frames up. No tracking, no identities. A person whose feet are inside a bicycle or motorcycle "
            "box counts as a rider, not a pedestrian. The map shows presence, not risk, and is only meaningful "
            "when the camera does not move.",
        ]

    if summary["item_notices"]:
        found = summary["item_notices"]
        lines += [
            "",
            "## Item notices (unverified)",
            "",
            f"Watched items: {', '.join(found['watched'])}. A notice means the model detected one of these, "
            f"with a score of {CONFIDENT_SCORE} or more, next to a detected person. **It is an object "
            "detection for a human to check, not a judgement about the person**: the model cannot see "
            "intent, it also detects harmless items (a cook's knife, a cricket bat), and it misses small "
            "or distant ones.",
            "",
        ]
        if found["notices"]:
            lines += ["| Frame | Time | Item | Score | Image to check |", "|---|---:|---|---:|---|"]
            for r in records:
                for n in r["notices"]:
                    when = "-" if r["time_s"] is None else f"{r['time_s']:.2f} s"
                    lines.append(f"| {r['name']} | {when} | possible {n['item']} | {n['conf']:.2f} "
                                 f"| `annotated/{r['name']}.jpg` |")
        else:
            lines.append(f"No watched item was detected next to a person in the {frames['processed']} "
                         "processed frames.")

    # At least two sample outputs whenever there are two frames: the busiest, the first and the last.
    samples = []
    for r in (max(records, key=lambda r: r["total"]), records[0], records[-1]):
        if all(r is not s for s in samples):
            samples.append(r)
    samples.sort(key=lambda r: r["frame_index"])
    lines += ["", "## Sample annotated frames", ""]
    for r in samples:
        when = f", t = {r['time_s']:.2f} s" if r["time_s"] is not None else ""
        lines += [f"**{r['name']}** (total {r['total']}, {r['density']}{when})", "",
                  f"![{r['name']}](annotated/{r['name']}.jpg)", ""]
    lines += ["## Files", "",
              "`counts.csv` (one row per processed frame), `detections.json` (every box), "
              "`summary.json` (this report as data), `annotated/` (one image per processed frame)"
              + (", `annotated.mp4`" if run["save_video"] else "")
              + (", `timeline.png`" if summary["timeline"] else "")
              + (", `shared_space.jpg`" if summary["shared_space"] else "") + "."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
