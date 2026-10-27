#!/usr/bin/env python
"""Extract hidden layers from an existing GraN-DAG model without training."""
import argparse
import json
import os

from gran_dag.hidden import extract_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="Trusted model.pkl/best-model.pkl with adjacent opt.pkl")
    parser.add_argument("--data-path", help="Original training data directory; overrides opt.pkl")
    parser.add_argument("--output-dir", help="Default: <checkpoint directory>/hidden")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-data", help="Optional new observed-only (N,D) .npy; uses training statistics")
    parser.add_argument("--variable-names", help="JSON list of variable names in original column order")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing export files")
    args = parser.parse_args()
    names = None
    if args.variable_names:
        with open(args.variable_names) as stream:
            names = json.load(stream)
    info = extract_from_checkpoint(
        args.model_path, args.data_path, args.output_dir, args.split,
        args.batch_size, args.input_data, names, args.overwrite)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.model_path), "hidden")
    for layer in info["layers"]:
        print("{}: {} {}".format(os.path.join(output_dir, layer["file"]),
                                  tuple(layer["shape"]), layer["dtype"]))
    print("Variable/row/preprocessing mapping: " + os.path.join(output_dir, "metadata.json"))


if __name__ == "__main__":
    main()
