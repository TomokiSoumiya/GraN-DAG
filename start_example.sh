#!/usr/bin/env bash
set -euo pipefail

# Resolve all paths relative to this repository so the script can be launched
# from any working directory.
CODE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_PATH="${CODE_PATH}/experiments/p10-run1"
DATA_PATH="${CODE_PATH}/data/unpacked/data_p10_e10_n1000_GP"
CONTAINER_PATH="${CODE_PATH}/container.simg"
ROOTFS_PATH="${CODE_PATH}/.tools/grandag-rootfs"
MODEL="NonLinGaussANM"  # or NonLinGauss
DATA_INDEX=1  # Choose which dataset to use. Program will use (data${DATA_INDEX}.npy, DAG${DATA_INDEX}.npy, CPDAG${DATA_INDEX}.npy)
NUM_VARS=10  # should match the data provided

if [[ -d "${ROOTFS_PATH}" ]] && command -v bwrap >/dev/null 2>&1; then
    CONTAINER_RUNTIME="bwrap"
elif [[ -x "${CODE_PATH}/.tools/apptainer/bin/apptainer" ]]; then
    CONTAINER_RUNTIME="${CODE_PATH}/.tools/apptainer/bin/apptainer"
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="singularity"
else
    echo "Error: no usable Bubblewrap, Apptainer, or Singularity runtime found." >&2
    exit 1
fi

if [[ ! -f "${CONTAINER_PATH}" ]]; then
    echo "Error: container image not found: ${CONTAINER_PATH}" >&2
    exit 1
fi

if [[ ! -f "${DATA_PATH}/data${DATA_INDEX}.npy" || ! -f "${DATA_PATH}/DAG${DATA_INDEX}.npy" ]]; then
    echo "Error: dataset ${DATA_INDEX} is incomplete under ${DATA_PATH}" >&2
    exit 1
fi

mkdir -p "${EXP_PATH}"

GRANDAG_COMMAND="python main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX} --model ${MODEL} --train --to-dag --cam-pruning --cam-pruning-cutoff 0.001 --num-vars ${NUM_VARS} --jac-thresh"

# Run the augmented Lagrangian procedure without PNS, convert the result to a
# DAG, and apply CAM pruning at the paper's p-value cutoff of 0.001. Existing
# train and to-dag results are loaded from disk instead of being recomputed.
if [[ "${CONTAINER_RUNTIME}" == "bwrap" ]]; then
    bwrap --die-with-parent --new-session \
        --ro-bind "${ROOTFS_PATH}" / \
        --dev /dev \
        --proc /proc \
        --tmpfs /tmp \
        --ro-bind "${DATA_PATH}" /dataset \
        --bind "${EXP_PATH}" /final_log \
        --ro-bind "${CODE_PATH}" /code \
        --setenv HOME /tmp \
        --setenv MPLCONFIGDIR /tmp/matplotlib \
        --chdir /code \
        /bin/bash -c "${GRANDAG_COMMAND}"
else
    "${CONTAINER_RUNTIME}" exec --containall \
        -B "${DATA_PATH}:/dataset/" \
        -B "${EXP_PATH}:/final_log/" \
        -B "${CODE_PATH}:/code/" \
        "${CONTAINER_PATH}" \
        bash -c "cd /code && ${GRANDAG_COMMAND}"
fi

# To use the baselines methods, replace main.py by the right path (e.g. for notears, it is notears/main.py)
# singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python cam/main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX}"
