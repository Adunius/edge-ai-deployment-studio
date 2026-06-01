"""Generate TFLite ConvBlock proxy models for LG G8X real-device validation.

This script requires TensorFlow in a Python 3.10-3.12 environment. The main
project currently uses Python 3.14, where TensorFlow wheels may be unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = EXPERIMENT_DIR / "reports" / "real_device_candidates_lg_g8x.csv"
DEFAULT_OUTPUT_DIR = (
    EXPERIMENT_DIR / "android_benchmark" / "app" / "src" / "main" / "assets" / "models"
)
DEFAULT_INDEX_PATH = (
    EXPERIMENT_DIR / "android_benchmark" / "app" / "src" / "main" / "assets" / "model_index.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TFLite models for selected FBNet validation blocks."
    )
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required to generate .tflite models. Install it in a "
            "Python 3.10-3.12 environment, for example:\n"
            "  py -3.11 -m venv .venv-tflite\n"
            "  .\\.venv-tflite\\Scripts\\python.exe -m pip install tensorflow pandas\n"
            "  .\\.venv-tflite\\Scripts\\python.exe real_device_validation\\scripts\\generate_fbnet_tflite_models.py"
        ) from exc
    return tf


def build_conv_block_model(tf, row: pd.Series):
    input_h = int(row["input_h"])
    input_w = int(row["input_w"])
    cin = int(row["cin"])
    cout = int(row["cout"])
    expansion = int(row["expansion"])
    kernel = int(row["kernel"])
    stride = int(row["stride"])
    group = int(row["group"])

    inputs = tf.keras.Input(shape=(input_h, input_w, cin), batch_size=1, name="input")
    x = inputs

    expanded_channels = cin * expansion
    if expansion > 1:
        x = tf.keras.layers.Conv2D(
            filters=expanded_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="expand_conv",
        )(x)
        x = tf.keras.layers.ReLU(name="expand_relu")(x)

    groups = group if expanded_channels % group == 0 and cout % group == 0 else 1
    x = tf.keras.layers.Conv2D(
        filters=cout,
        kernel_size=kernel,
        strides=stride,
        padding="same",
        groups=groups,
        use_bias=False,
        name="spatial_grouped_conv",
    )(x)
    outputs = tf.keras.layers.ReLU(name="output_relu")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def convert_to_tflite(tf, model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    return converter.convert()


def main() -> None:
    args = parse_args()
    tf = import_tensorflow()

    candidates = pd.read_csv(args.candidates)
    output_dir = Path(args.output_dir)
    index_path = Path(args.index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    index = []
    for _, row in candidates.iterrows():
        block_id = int(row["block_id"])
        model_name = f"block_{block_id}.tflite"
        model = build_conv_block_model(tf, row)
        tflite_model = convert_to_tflite(tf, model)
        (output_dir / model_name).write_bytes(tflite_model)

        index.append(
            {
                "block_id": block_id,
                "model_asset": f"models/{model_name}",
                "benchmark_device": row["benchmark_device"],
                "real_device": row["real_device"],
                "predicted_latency": float(row["predicted_latency"]),
                "latency_group": row["latency_group"],
                "input_h": int(row["input_h"]),
                "input_w": int(row["input_w"]),
                "cin": int(row["cin"]),
                "cout": int(row["cout"]),
                "expansion": int(row["expansion"]),
                "kernel": int(row["kernel"]),
                "stride": int(row["stride"]),
                "group": int(row["group"]),
                "block_name": row["block_name"],
            }
        )

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Generated {len(index)} TFLite models in {output_dir}")
    print(f"Wrote model index to {index_path}")


if __name__ == "__main__":
    main()
