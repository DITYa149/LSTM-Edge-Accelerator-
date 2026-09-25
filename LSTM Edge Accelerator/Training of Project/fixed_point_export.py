"""Export a bit-accurate fixed-point golden reference for this LSTM model.

Run from the ``Training of Project`` directory:

    py fixed_point_export.py

The generated ``golden_reference`` directory contains:
  * RTL-friendly weight, bias, input, and activation-LUT .mem files;
  * expected intermediate values for every LSTM timestep; and
  * manifest.json, which is the single source of truth for numeric formats
    and memory address ordering.

The HDL must implement the same rounding, saturation, LUT index calculation,
and memory ordering recorded in manifest.json for comparisons to be exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


GATE_NAMES = ("forget", "input", "candidate", "output")
SEQUENCE_LENGTH = 10
INPUT_SIZE = 4
HIDDEN_SIZE = 16
DENSE_SIZE = 16
OUTPUT_SIZE = 4


class FixedPoint:
    """Signed fixed-point arithmetic used by both the exporter and RTL."""

    def __init__(self, width: int, frac_bits: int) -> None:
        if width < 2 or not 0 <= frac_bits < width:
            raise ValueError("Require width >= 2 and 0 <= frac_bits < width.")
        self.width = width
        self.frac_bits = frac_bits
        self.scale = 1 << frac_bits
        self.minimum = -(1 << (width - 1))
        self.maximum = (1 << (width - 1)) - 1
        self.saturation_count = 0

    @staticmethod
    def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
        """Round to nearest integer; exact half values go away from zero."""
        values = np.asarray(values, dtype=np.float64)
        return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(np.int64)

    def saturate(self, values: np.ndarray | int) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        self.saturation_count += int(np.count_nonzero((values < self.minimum) | (values > self.maximum)))
        return np.clip(values, self.minimum, self.maximum).astype(np.int64)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        return self.saturate(self._round_away_from_zero(np.asarray(values) * self.scale))

    def rounded_shift(self, values: np.ndarray | int) -> np.ndarray:
        """Rescale a signed 2F result to F using nearest, ties-away rounding."""
        values = np.asarray(values, dtype=np.int64)
        offset = 1 << (self.frac_bits - 1) if self.frac_bits else 0
        magnitudes = np.abs(values)
        shifted = (magnitudes + offset) >> self.frac_bits
        return np.where(values < 0, -shifted, shifted).astype(np.int64)

    def mul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.saturate(self.rounded_shift(np.asarray(left, dtype=np.int64) * np.asarray(right, dtype=np.int64)))

    def hex_lines(self, values: np.ndarray) -> Iterable[str]:
        mask = (1 << self.width) - 1
        digits = (self.width + 3) // 4
        for value in np.asarray(values, dtype=np.int64).reshape(-1):
            yield f"{int(value) & mask:0{digits}X}"


class ActivationLUT:
    """1025-entry sigmoid/tanh tables over [-8, +8] in fixed-point."""

    def __init__(self, fxp: FixedPoint, size: int = 1025, limit: float = 8.0) -> None:
        if size != 1025:
            raise ValueError("This implementation uses 1025 entries (1024 equal intervals).")
        self.fxp = fxp
        self.size = size
        self.limit_code = int(limit * fxp.scale)
        if self.limit_code <= 0 or self.limit_code % 1024:
            raise ValueError("LUT range must have an integral 1024-way step in fixed-point codes.")
        self.step_code = self.limit_code // 512  # 2*limit / 1024
        x = np.arange(size, dtype=np.float64) * (self.step_code / fxp.scale) - limit
        self.sigmoid = fxp.quantize(1.0 / (1.0 + np.exp(-x)))
        self.tanh = fxp.quantize(np.tanh(x))

    def index(self, codes: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(codes, dtype=np.int64), -self.limit_code, self.limit_code)
        # Nearest table point, ties upward.  This is simple RTL arithmetic:
        # index = clamp((clamped_code + LIMIT_CODE + STEP_CODE/2) / STEP_CODE).
        return np.clip((clipped + self.limit_code + self.step_code // 2) // self.step_code, 0, self.size - 1)

    def apply(self, codes: np.ndarray, kind: str) -> np.ndarray:
        table = self.sigmoid if kind == "sigmoid" else self.tanh
        return table[self.index(codes)]


def write_mem(path: Path, fxp: FixedPoint, values: np.ndarray) -> None:
    path.write_text("\n".join(fxp.hex_lines(values)) + "\n", encoding="ascii")


def quantized_matmul(left: np.ndarray, right: np.ndarray, bias: np.ndarray, fxp: FixedPoint) -> np.ndarray:
    """Matrix product where operands/bias are QF and the accumulator is Q2F.

    RTL equivalent: accumulate all products at full precision, add (bias <<< F),
    then round-shift by F and saturate once.
    """
    accumulator = np.asarray(left, dtype=np.int64) @ np.asarray(right, dtype=np.int64)
    accumulator += np.asarray(bias, dtype=np.int64) << fxp.frac_bits
    return fxp.saturate(fxp.rounded_shift(accumulator))


def export_parameters(weights: Dict[str, np.ndarray], mem_dir: Path, fxp: FixedPoint) -> Dict[str, np.ndarray]:
    q = {name: fxp.quantize(value) for name, value in weights.items()}
    w, u, b = q["W_lstm"], q["U_lstm"], q["b_lstm"]

    for gate_index, gate_name in enumerate(GATE_NAMES):
        start = gate_index * HIDDEN_SIZE
        stop = start + HIDDEN_SIZE
        # Address order: neuron-major, then input/state index.
        write_mem(mem_dir / f"w_{gate_name}.mem", fxp, w[:, start:stop].T)
        write_mem(mem_dir / f"u_{gate_name}.mem", fxp, u[:, start:stop].T)
        write_mem(mem_dir / f"b_{gate_name}.mem", fxp, b[start:stop])

    # Same output-neuron-major order for dense layers.
    write_mem(mem_dir / "w_dense.mem", fxp, q["W_dense"].T)
    write_mem(mem_dir / "b_dense.mem", fxp, q["b_dense"])
    write_mem(mem_dir / "alpha_prelu.mem", fxp, q["alpha_prelu"])
    write_mem(mem_dir / "w_output.mem", fxp, q["W_output"].T)
    write_mem(mem_dir / "b_output.mem", fxp, q["b_output"])
    return q


def fixed_lstm_case(x: np.ndarray, q: Dict[str, np.ndarray], fxp: FixedPoint, lut: ActivationLUT) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    h = np.zeros(HIDDEN_SIZE, dtype=np.int64)
    c = np.zeros(HIDDEN_SIZE, dtype=np.int64)
    trace = {name: [] for name in ("z", "f", "i", "g", "o", "c", "h")}

    for x_t in x:
        # Keep both MAC terms at Q2F, add the bias at Q2F, then rescale once.
        # This matches a conventional full-precision accumulator datapath.
        gate_accumulator = x_t @ q["W_lstm"] + h @ q["U_lstm"]
        gate_accumulator += q["b_lstm"] << fxp.frac_bits
        z = fxp.saturate(fxp.rounded_shift(gate_accumulator))
        zf, zi, zg, zo = np.split(z, 4)
        f, i = lut.apply(zf, "sigmoid"), lut.apply(zi, "sigmoid")
        g, o = lut.apply(zg, "tanh"), lut.apply(zo, "sigmoid")
        c = fxp.saturate(fxp.mul(f, c) + fxp.mul(i, g))
        h = fxp.mul(o, lut.apply(c, "tanh"))
        for name, value in zip(("z", "f", "i", "g", "o", "c", "h"), (z, f, i, g, o, c, h)):
            trace[name].append(value.copy())

    dense = quantized_matmul(h[None, :], q["W_dense"], q["b_dense"], fxp)[0]
    prelu = np.where(dense >= 0, dense, fxp.mul(q["alpha_prelu"], dense))
    prediction = quantized_matmul(prelu[None, :], q["W_output"], q["b_output"], fxp)[0]
    trace.update({"dense": dense, "prelu": prelu, "prediction": prediction})
    return {name: np.asarray(value, dtype=np.int64) for name, value in trace.items()}, prediction


def validate_shapes(weights: Dict[str, np.ndarray], inputs: np.ndarray) -> None:
    expected = {
        "W_lstm": (INPUT_SIZE, 4 * HIDDEN_SIZE), "U_lstm": (HIDDEN_SIZE, 4 * HIDDEN_SIZE),
        "b_lstm": (4 * HIDDEN_SIZE,), "W_dense": (HIDDEN_SIZE, DENSE_SIZE), "b_dense": (DENSE_SIZE,),
        "alpha_prelu": (DENSE_SIZE,), "W_output": (DENSE_SIZE, OUTPUT_SIZE), "b_output": (OUTPUT_SIZE,),
    }
    for name, shape in expected.items():
        if name not in weights or weights[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {weights.get(name, np.empty(0)).shape}.")
    if inputs.ndim != 3 or inputs.shape[1:] != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise ValueError(f"Inputs must have shape (cases, {SEQUENCE_LENGTH}, {INPUT_SIZE}); got {inputs.shape}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-point LSTM golden-reference .mem files.")
    parser.add_argument("--model", type=Path, default=Path("trained_model_original/best_model_weights.npz"))
    parser.add_argument("--inputs", type=Path, default=Path("windowed_data/X_test.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("golden_reference"))
    parser.add_argument("--cases", type=int, default=1, help="Number of test sequences to export (default: 1).")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--frac-bits", type=int, default=12, help="Default is signed Q4.12.")
    args = parser.parse_args()

    if args.cases < 1:
        raise ValueError("--cases must be at least one.")
    with np.load(args.model) as source:
        weights = {name: source[name] for name in source.files}
    inputs = np.load(args.inputs).astype(np.float64)
    validate_shapes(weights, inputs)
    inputs = inputs[:args.cases]

    mem_dir = args.output_dir / "mem"
    trace_dir = args.output_dir / "expected"
    mem_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    fxp = FixedPoint(args.width, args.frac_bits)
    lut = ActivationLUT(fxp)
    q = export_parameters(weights, mem_dir, fxp)
    write_mem(mem_dir / "sigmoid_lut.mem", fxp, lut.sigmoid)
    write_mem(mem_dir / "tanh_lut.mem", fxp, lut.tanh)

    all_predictions = []
    for case_index, sample in enumerate(inputs):
        q_input = fxp.quantize(sample)
        trace, prediction = fixed_lstm_case(q_input, q, fxp, lut)
        write_mem(mem_dir / f"input_case_{case_index:03d}.mem", fxp, q_input)
        for name, values in trace.items():
            write_mem(trace_dir / f"case_{case_index:03d}_{name}.mem", fxp, values)
        all_predictions.append(prediction)

    write_mem(trace_dir / "all_predictions.mem", fxp, np.asarray(all_predictions))
    manifest = {
        "numeric_format": {
            "signed": True, "width_bits": args.width, "fraction_bits": args.frac_bits,
            "real_range": [fxp.minimum / fxp.scale, fxp.maximum / fxp.scale],
            "rounding": "nearest, exact halves away from zero", "overflow": "saturate",
        },
        "architecture": {"sequence_length": 10, "input_size": 4, "hidden_size": 16, "dense_size": 16, "output_size": 4},
        "gate_order": list(GATE_NAMES),
        "lstm_math": "z=round_shift(x*W + h*U + (bias<<F)); c=saturate(f*c + i*g); h=round_shift(o*tanh(c))",
        "lut": {"entries": 1025, "input_range_real": [-8.0, 8.0], "input_clamp": True,
                "index_formula": "clamp((clamped_code + LIMIT_CODE + STEP_CODE/2) // STEP_CODE, 0, 1024)"},
        "memory_order": {
            "inputs": "time-major then feature", "lstm_weights": "gate file: neuron-major then input/state index",
            "dense_weights": "output-neuron-major then input index", "traces": "time-major then neuron; z is forget,input,candidate,output blocks",
        },
        "files": {"parameters": "mem/", "test_inputs": "mem/input_case_###.mem", "expected_values": "expected/case_###_*.mem"},
        "cases_exported": int(len(inputs)), "saturations_during_export": fxp.saturation_count,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(inputs)} case(s) to: {args.output_dir.resolve()}")
    print(f"Format: signed Q{args.width - args.frac_bits}.{args.frac_bits}; saturation events: {fxp.saturation_count}")


if __name__ == "__main__":
    main()
