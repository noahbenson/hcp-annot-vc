#! /bin/bash

# This script assumes that we have initialized the appropriate python
# environment ahead of time and that we just need to execute the various Python
# scripts in order.

# The raters we are running over:
RATERS=(bogengsong BrendaQiu JiyeongHa lindazelinzhao nourahboujaber
        jennifertepan)
MEANRATER=mean
# The input and output directories:
SAVE_PATH=/data/crcns2021/results/data_branch/save
PROC_PATH=/data/crcns2021/results/proc

# Die function for errors:
function die {
    echo "$*"
    exit 1
}

# Make sure we're in the repository directory above the scripts directory.
SCRIPT_DIR=`cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd`
cd "$SCRIPT_DIR"/..
[ -d ./hcpannot ] && [ -d ./scripts ] \
    || die "script must be in the hcp-annot-vc repo when run"

# Perform each step in turn:
# (1) Process the contours into traces.
python scripts/proc_traces.py \
    "$SAVE_PATH" "$PROC_PATH"/traces \
    --raters ${RATERS[@]} "$@"
 (2) Use the traces to make mean contours.
python scripts/proc_means.py \
    "$PROC_PATH"/traces "$PROC_PATH"/means \
    --raters ${RATERS[@]} "$@"
# (3) Process the mean contours into traces.
python scripts/proc_meantraces.py \
    "$PROC_PATH"/means "$PROC_PATH"/traces \
    --raters ${RATERS[@]} "$@"
# (4) Process the traces into paths.
python scripts/proc_paths.py \
    "$PROC_PATH"/traces "$PROC_PATH"/paths \
    --raters ${RATERS[@]} $MEANRATER "$@"
# (5) Process the paths into labels.
python scripts/proc_labels.py \
    "$PROC_PATH"/paths "$PROC_PATH"/labels \
    --raters ${RATERS[@]} $MEANRATER "$@"
# (6) Process the paths into surface areas.
python scripts/proc_sarea.py \
    "$PROC_PATH"/paths "$PROC_PATH"/surface_areas.csv \
    --raters ${RATERS[@]} $MEANRATER "$@"


# That's it!
exit 0
