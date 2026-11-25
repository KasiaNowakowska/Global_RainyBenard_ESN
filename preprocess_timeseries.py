"""Convert Dedalus global time-series output to the NumPy inputs used by the ESN."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_scalar_task(file: h5py.File, task_name: str) -> np.ndarray:
    dataset_path = f"tasks/{task_name}"
    if dataset_path not in file:
        raise ValueError(f"Missing HDF5 dataset: {dataset_path}")

    values = np.asarray(file[dataset_path])
    if values.ndim == 0 or any(size != 1 for size in values.shape[1:]):
        raise ValueError(
            f"Expected {dataset_path} to contain one scalar per time, "
            f"got shape {values.shape}"
        )
    return values.reshape(values.shape[0])


def read_timeseries(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as file:
        required_paths = ("scales/sim_time", "tasks/KE", "tasks/moisture")
        for dataset_path in required_paths:
            if dataset_path not in file:
                raise ValueError(f"{path} is missing HDF5 dataset: {dataset_path}")

        times = np.asarray(file["scales/sim_time"])
        if times.ndim != 1:
            raise ValueError(
                f"Expected scales/sim_time to be one-dimensional, got {times.shape}"
            )
        ke = read_scalar_task(file, "KE")
        moisture = read_scalar_task(file, "moisture")

    if not (len(times) == len(ke) == len(moisture)):
        raise ValueError(
            f"Time and task lengths differ in {path}: "
            f"{len(times)}, {len(ke)}, {len(moisture)}"
        )
    return times, ke, moisture


def load_base_arrays(base_dir: Path, suffix: str) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        np.load(base_dir / f"{name}{suffix}.npy")
        for name in ("total_time", "KE", "q")
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Base time, KE, and q arrays must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Base time, KE, and q arrays must have matching lengths")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        required=True,
        type=Path,
        help="Dedalus timeseries_s*.h5 file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for NumPy arrays",
    )
    parser.add_argument(
        "--output-suffix",
        required=True,
        help="Suffix used in ESN filenames, e.g. 5000_48000",
    )
    parser.add_argument(
        "--base-suffix",
        help="Suffix of existing total_time/KE/q arrays to append to",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Directory containing base arrays (defaults to --output-dir)",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Initial HDF5 rows to discard when appending a restarted run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output arrays if they already exist",
    )
    args = parser.parse_args()

    if args.skip_rows < 0:
        parser.error("--skip-rows must be non-negative")
    if args.skip_rows and not args.base_suffix:
        parser.error("--skip-rows can only be used together with --base-suffix")
    if args.base_dir and not args.base_suffix:
        parser.error("--base-dir can only be used together with --base-suffix")

    times, ke, moisture = read_timeseries(args.input_h5)
    if args.skip_rows >= len(times):
        parser.error(f"--skip-rows must be less than the HDF5 row count ({len(times)})")
    times = times[args.skip_rows:]
    ke = ke[args.skip_rows:]
    moisture = moisture[args.skip_rows:]

    if args.base_suffix:
        base_dir = args.base_dir or args.output_dir
        base_times, base_ke, base_moisture = load_base_arrays(base_dir, args.base_suffix)
        times = np.concatenate((base_times, times))
        ke = np.concatenate((base_ke, ke))
        moisture = np.concatenate((base_moisture, moisture))

    if len(times) > 1 and np.any(np.diff(times) <= 0):
        parser.error("Combined simulation times must be strictly increasing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "KE": ke,
        "q": moisture,
        "total_time": times,
    }
    output_paths = {
        name: args.output_dir / f"{name}{args.output_suffix}.npy"
        for name in outputs
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "Output file(s) already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    for name, array in outputs.items():
        np.save(output_paths[name], array)
        print(f"Saved {output_paths[name]} ({len(array)} samples)")


if __name__ == "__main__":
    main()
