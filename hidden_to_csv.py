#!/usr/bin/env python3
"""Make exported (sample, variable, unit) hidden layers readable as CSV."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def convert_hidden_directory(directory, overwrite=False):
    directory = Path(directory)
    with (directory / 'metadata.json').open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    row_ids = np.load(str(directory / 'sample_indices.npy'), allow_pickle=False)
    splits = np.load(str(directory / 'sample_split.npy'), allow_pickle=False)
    variables = metadata['variables']
    split_names = {-1: 'new_input', 0: 'train', 1: 'validation', 2: 'final_test'}
    destinations = [directory / Path(layer['file']).with_suffix('.csv')
                    for layer in metadata['layers']]
    if not overwrite:
        for destination in destinations:
            if destination.exists():
                raise FileExistsError('{} already exists; use --overwrite'.format(destination))
    for layer, destination in zip(metadata['layers'], destinations):
        values = np.load(str(directory / layer['file']), mmap_mode='r', allow_pickle=False)
        if (values.ndim != 3 or values.shape[1] != len(variables) or
                row_ids.shape != (len(values),) or splits.shape != (len(values),)):
            raise ValueError('Inconsistent activation shape or row/variable mapping')
        columns = ['{}_unit{:03d}'.format(variable['name'], unit + 1)
                   for variable in variables for unit in range(values.shape[2])]
        with destination.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'split'] + columns)
            for row, value in enumerate(values):
                writer.writerow([int(row_ids[row]), split_names[int(splits[row])]] +
                                value.reshape(-1).tolist())
        print('{}: {} rows, {} columns'.format(destination, len(values), len(columns) + 2))
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', help='Hidden export directories with metadata.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for directory in args.directories:
        convert_hidden_directory(directory, args.overwrite)


if __name__ == '__main__':
    main()
