
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

INTERVAL_S = 2.0       # take and print a reading this often


def parse_cpu_times(stat_text):
    """First line of /proc/stat -> (busy, total) time counters, added up over all cores.
    The line is: cpu user nice system idle iowait irq softirq steal ..."""
    fields = [int(v) for v in stat_text.splitlines()[0].split()[1:9]]
    idle = fields[3] + fields[4]                      # idle + waiting for the disk
    return sum(fields) - idle, sum(fields)


def parse_meminfo(text):
    """/proc/meminfo -> (used GB, total GB). "Used" = total minus what is still available."""
    kb = {line.split(":")[0]: int(line.split()[1]) for line in text.splitlines() if ":" in line}
    return (kb["MemTotal"] - kb["MemAvailable"]) / 2**20, kb["MemTotal"] / 2**20


def parse_own_memory(status_text):
    """/proc/self/status -> GB of memory this program is using (the VmRSS line), or None."""
    found = re.search(r"VmRSS:\s+(\d+) kB", status_text)
    return int(found.group(1)) / 2**20 if found else None


def parse_board_power(pmic_text):
    """`vcgencmd pmic_read_adc` (Pi 5 only) -> watts: volts x amps, added up over the board's
    power rails. Its lines look like "VDD_CORE_A current(7)=0.71A" and "VDD_CORE_V volt(15)=0.72V"."""
    amps, volts = {}, {}
    for rail, kind, value in re.findall(r"(\S+)_([AV])\s+\w+\(\d+\)=([0-9.]+)[AV]", pmic_text):
        (amps if kind == "A" else volts)[rail] = float(value)
    watts = [amps[rail] * volts[rail] for rail in amps if rail in volts]
    return sum(watts) if watts else None


def parse_throttled(text):
    """`vcgencmd get_throttled` ("throttled=0x50005") -> the number, or None."""
    found = re.search(r"=(0x[0-9a-fA-F]+)", text)
    return int(found.group(1), 16) if found else None


def describe_throttled(value):
    """The Pi's throttle number in words: is the Pi slowing itself down, and why?

    Bits 0-3 mean "right now"; bits 16-19 mean "at some time since the Pi was switched on"
    (those stay set until a reboot). In each group the first bit is low voltage, the other
    three (clock capped, throttled, temperature limit) come with heat."""
    if value is None:
        return "n/a"

    def cause(bits):
        return " and ".join((["low voltage"] if bits & 0b0001 else []) + (["heat"] if bits & 0b1110 else []))

    now, earlier = value & 0xF, (value >> 16) & 0xF
    if now:
        return f"YES ({cause(now)})"
    return f"no (was earlier, {cause(earlier)})" if earlier else "no"


class Telemetry:
    def __init__(self, root="/"):
        self.root = Path(root)                        # "/" on the Pi; a folder of sample files in tests
        self.vcgencmd = shutil.which("vcgencmd")
        self.available = (self.root / "proc/stat").is_file()
        self.readings = []
        self.shown = 0                                # how many readings fresh() has handed out
        self.energy_j = 0.0                           # board power added up over time
        self.start_time = self.last_time = time.monotonic()
        self.last_cpu = self._number("proc/stat", parse_cpu_times)
        self.last_power = None
        self.stopping = threading.Event()
        self.thread = None

    def _number(self, path, parse):
        """Read a text file and parse it; None if the file is missing or looks different."""
        try:
            return parse((self.root / path).read_text())
        except (OSError, ValueError, KeyError, IndexError):
            return None

    def _tool(self, *args):
        """Output of `vcgencmd <args>`, or "" when the tool is missing or fails."""
        if not self.vcgencmd:
            return ""
        try:
            done = subprocess.run([self.vcgencmd, *args], capture_output=True, text=True, timeout=2)
            return done.stdout if done.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def read(self):
        """Take one reading now and keep it."""
        now = time.monotonic()
        elapsed = now - self.last_time
        cpu = self._number("proc/stat", parse_cpu_times)
        cpu_pct = None
        if cpu and self.last_cpu and cpu[1] > self.last_cpu[1]:       # share of the time the cores were busy
            cpu_pct = 100 * (cpu[0] - self.last_cpu[0]) / (cpu[1] - self.last_cpu[1])
        memory = self._number("proc/meminfo", parse_meminfo) or (None, None)
        power = parse_board_power(self._tool("pmic_read_adc"))
        if power is not None:
            self.energy_j += power * elapsed                          # watts x seconds = joules
        reading = {
            "time_s": round(now - self.start_time, 1),
            "temp_c": self._number("sys/class/thermal/thermal_zone0/temp", lambda t: int(t) / 1000),
            "cpu_pct": cpu_pct,
            "cpu_mhz": self._number("sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", lambda t: int(t) / 1000),
            "ram_used_gb": memory[0],
            "ram_total_gb": memory[1],
            "run_ram_gb": self._number("proc/self/status", parse_own_memory),
            "power_w": power,
            "throttled": parse_throttled(self._tool("get_throttled")),
        }
        self.last_time, self.last_cpu, self.last_power = now, cpu, power
        self.readings.append(reading)
        return reading

    def start(self):
        """Take a reading every INTERVAL_S in the background until stop().

        It has to be a background thread: power must be read WHILE the model is running.
        Read between two frames, the cores are idle for that moment and the power comes out
        far too low."""
        def loop():
            while not self.stopping.wait(INTERVAL_S):
                self.read()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        """End the background readings. A run shorter than INTERVAL_S still gets one reading."""
        self.stopping.set()
        if self.thread:
            self.thread.join()
        if not self.readings:
            self.read()
        elif self.last_power is not None:             # the stretch after the last reading
            self.energy_j += self.last_power * (time.monotonic() - self.last_time)
            self.last_time = time.monotonic()

    def fresh(self):
        """The readings taken since the last call, for printing between frame lines."""
        new = self.readings[self.shown:]
        self.shown += len(new)
        return new

    @staticmethod
    def line(r):
        """One reading as a terminal line, e.g.
        PI  temp 61.5 C  cpu 98% at 2400 MHz  ram 1.9/7.9 GB (this run 0.6)  power 6.8 W  throttled: no"""
        def show(value, text):
            return "n/a" if value is None else text.format(value)

        return (f"    PI  temp {show(r['temp_c'], '{:.1f} C')}  cpu {show(r['cpu_pct'], '{:.0f}%')}"
                f" at {show(r['cpu_mhz'], '{:.0f} MHz')}  ram {show(r['ram_used_gb'], '{:.1f}')}"
                f"/{show(r['ram_total_gb'], '{:.1f} GB')} (this run {show(r['run_ram_gb'], '{:.1f}')})"
                f"  power {show(r['power_w'], '{:.1f} W')}  throttled: {describe_throttled(r['throttled'])}")

    def summary(self, frames):
        """Average and highest of every reading, and the energy used, for REPORT.md and summary.json."""
        def stats(key, digits=1):
            values = [r[key] for r in self.readings if r[key] is not None]
            return None if not values else {"avg": round(sum(values) / len(values), digits or None),
                                            "max": round(max(values), digits or None)}

        flags = [r["throttled"] for r in self.readings if r["throttled"] is not None]
        any_flag = 0
        for flag in flags:
            any_flag |= flag                          # every bit that was set in any reading
        power = stats("power_w", 2)
        return {
            "readings": len(self.readings),
            "interval_s": INTERVAL_S,
            "seconds": round(self.last_time - self.start_time, 1),
            "temp_c": stats("temp_c"),
            "cpu_pct": stats("cpu_pct", 0),
            "cpu_mhz": stats("cpu_mhz", 0),
            "ram_used_gb": stats("ram_used_gb", 2),
            "ram_total_gb": next((round(r["ram_total_gb"], 2) for r in self.readings if r["ram_total_gb"]), None),
            "run_ram_gb": stats("run_ram_gb", 2),
            "power_w": power,
            "energy_j": round(self.energy_j, 1) if power else None,
            "energy_j_per_frame": round(self.energy_j / frames, 2) if power and frames else None,
            "throttled_code": f"{any_flag:#x}" if flags else None,      # the Pi's own number; 0x0 = never
            "throttled_during_run": bool(any_flag & 0xF) if flags else None,
            "throttled": describe_throttled(any_flag if flags else None),
        }
