#!/usr/bin/env bash
# Execute a command using the prepared GraN-DAG environment.
set -euo pipefail
CODE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTFS_PATH="${CODE_PATH}/.tools/grandag-rootfs"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash run_in_container.sh python extract_hidden.py [options]" >&2
    exit 2
fi
if [[ -d "${ROOTFS_PATH}" ]] && command -v bwrap >/dev/null 2>&1; then
    # Bind onto mount points already present in the prepared root filesystem.
    # /final_log contains ALL experiment directories; /dataset contains data/.
    exec bwrap --die-with-parent --new-session \
        --ro-bind "${ROOTFS_PATH}" / \
        --dev /dev --proc /proc --tmpfs /tmp \
        --ro-bind "${CODE_PATH}/data" /dataset \
        --bind "${CODE_PATH}/experiments" /final_log \
        --ro-bind "${CODE_PATH}" /code \
        --setenv HOME /tmp --setenv MPLCONFIGDIR /tmp/matplotlib \
        --chdir /code "$@"
fi
if command -v apptainer >/dev/null 2>&1; then
    RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME="singularity"
elif [[ -x "${CODE_PATH}/.tools/apptainer/bin/apptainer" ]]; then
    RUNTIME="${CODE_PATH}/.tools/apptainer/bin/apptainer"
else
    echo "Error: no prepared container environment found." >&2
    exit 1
fi
exec "${RUNTIME}" exec --containall \
    -B "${CODE_PATH}/data:/dataset" \
    -B "${CODE_PATH}/experiments:/final_log" \
    -B "${CODE_PATH}:/code" --pwd /code "${CODE_PATH}/container.simg" "$@"
