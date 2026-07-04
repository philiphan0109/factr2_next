import argparse
from pathlib import Path

import h5py
import numpy as np
from termcolor import colored

ROOT_ATTRS = ("schema", "session_name", "created_at")

def c(text, color):
    return colored(str(text), color)

def attr_text(h5, name):
    value = h5.attrs.get(name)
    if value is None:
        return c("missing", "yellow")
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

def read_stream(ep_name, key, obj):
    errors = []
    path = f"{ep_name}/{key}"
    if not isinstance(obj, h5py.Group):
        return None, None, [f"{path}: not a group"]
    if "data" not in obj or "timestamps" not in obj:
        return None, None, [f"{path}: missing data or timestamps"]

    data = np.asarray(obj["data"])
    stamps = np.asarray(obj["timestamps"])
    rows = data.shape[0] if data.ndim > 0 else None
    time_rows = stamps.shape[0] if stamps.ndim > 0 else None

    if rows is None or time_rows is None:
        errors.append(f"{path}: scalar data/timestamps")
    elif rows == 0:
        errors.append(f"{path}: empty")
    elif rows != time_rows:
        errors.append(f"{path}: rows {rows} != timestamps {time_rows}")

    try:
        if data.size and not np.isfinite(data).all():
            errors.append(f"{path}: NaN or inf in data")
    except TypeError:
        errors.append(f"{path}: nonnumeric data")

    if stamps.ndim != 1:
        errors.append(f"{path}: timestamps not 1-D")
        stamps = np.asarray([], dtype=np.int64)
    else:
        try:
            if len(stamps) > 1 and np.any(np.diff(stamps) < 0):
                errors.append(f"{path}: timestamps decrease")
        except TypeError:
            errors.append(f"{path}: nonnumeric timestamps")
            stamps = np.asarray([], dtype=np.int64)
    return rows, stamps, errors

def dt_stats(stamps):
    if len(stamps) < 2:
        return 0.0, 0.0, None, 0
    duration = float(stamps[-1] - stamps[0]) * 1e-9
    dt = np.diff(stamps.astype(np.float64)) * 1e-6
    hz = (len(stamps) - 1) / duration if duration > 0 else 0.0
    median = float(np.median(dt))
    large = int(np.sum(dt > 2.0 * median)) if median > 0 else 0
    return duration, hz, dt, large

def check_episode(name, ep):
    keys = sorted(ep.keys())
    rows, errors = [], []
    ref_stamps = None
    for key in keys:
        row, stamps, stream_errors = read_stream(name, key, ep[key])
        errors.extend(stream_errors)
        if row is not None:
            rows.append(row)
        if ref_stamps is None and stamps is not None and len(stamps):
            ref_stamps = stamps

    warnings = []
    if not keys:
        errors.append(f"{name}: no streams")
    if rows and len(set(rows)) > 1:
        errors.append(f"{name}: stream row counts differ {sorted(set(rows))}")

    row_text = rows[0] if rows and len(set(rows)) == 1 else "mixed"
    duration, hz, dt, large = dt_stats(ref_stamps if ref_stamps is not None else [])
    if 0.0 < duration < 5.0:
        warnings.append(f"{name}: duration under 5s")
    if dt is not None and large:
        warnings.append(f"{name}: {large} gaps > 2x median")

    color = "red" if errors else "yellow" if warnings else "green"
    status = "FAIL" if errors else "WARN" if warnings else "OK"
    print(c(f"\n{name} {status}", color))
    print(f"  rows: {row_text}")
    print(f"  duration: {duration:.2f}s / {duration / 60.0:.2f} min")
    print(f"  effective_hz: {hz:.2f}")
    if dt is not None:
        print(
            "  gaps ms: "
            f"min={dt.min():.3f}, mean={dt.mean():.3f}, "
            f"median={np.median(dt):.3f}, max={dt.max():.3f}, large={large}"
        )
    print("  keys: " + ", ".join(keys))
    for message in warnings:
        print("  " + c("WARN: " + message, "yellow"))
    for message in errors:
        print("  " + c("ERROR: " + message, "red"))
    return len(errors), len(warnings)

def check_h5(path):
    print(c(f"Checking: {path}", "cyan"))
    try:
        with h5py.File(Path(path), "r") as h5:
            warnings = [f"missing root attr: {name}" for name in ROOT_ATTRS if name not in h5.attrs]
            print("Schema: " + attr_text(h5, "schema"))
            print("Session: " + attr_text(h5, "session_name"))
            print("Created: " + attr_text(h5, "created_at"))
            episodes = [(k, v) for k, v in sorted(h5.items()) if isinstance(v, h5py.Group)]
            print(f"Episodes: {len(episodes)}")
            errors = 0 if episodes else 1
            if not episodes:
                print("  " + c("ERROR: no episodes", "red"))
            for name, ep in episodes:
                ep_errors, ep_warnings = check_episode(name, ep)
                errors += ep_errors
                warnings.extend([None] * ep_warnings)
            for warning in warnings:
                if warning:
                    print("  " + c("WARN: " + warning, "yellow"))
    except OSError as exc:
        print(c(f"ERROR: cannot open file: {exc}", "red"))
        print(c("\nFAIL", "red"))
        return False

    if errors:
        print(c(f"\nFAIL ({errors} errors, {len(warnings)} warnings)", "red"))
        return False
    color = "yellow" if warnings else "green"
    suffix = f" ({len(warnings)} warnings)" if warnings else ""
    print(c(f"\nPASS{suffix}", color))
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a FACTR2 NEXT H5 recording.")
    parser.add_argument("path")
    return 0 if check_h5(parser.parse_args(argv).path) else 1

if __name__ == "__main__":
    raise SystemExit(main())
