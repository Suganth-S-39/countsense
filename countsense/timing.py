"""Timing statistics.

Each frame is timed in three parts with time.perf_counter() (see process_frame in
run_poc.py): preprocess (letterbox), inference (ExecuTorch) and postprocess
(decode + NMS). The warm-up inference is never part of these numbers.
"""
import numpy as np

STAGES = ("pre", "infer", "post")


def ms_stats(values):
    """Mean, median (p50) and 95th percentile of a list of milliseconds."""
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
    }


def timing_stats(frame_timings):
    """frame_timings: one {"pre", "infer", "post"} dict (ms) per processed frame.

    Returns stats per stage, for their sum ("total"), and the processing FPS
    (frames per second the three stages alone could sustain).
    """
    stats = {stage: ms_stats([t[stage] for t in frame_timings]) for stage in STAGES}
    stats["total"] = ms_stats([sum(t[s] for s in STAGES) for t in frame_timings])
    stats["fps"] = round(1000 / stats["total"]["mean"], 2)
    return stats
