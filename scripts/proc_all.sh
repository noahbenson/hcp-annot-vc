#! /bin/bash

# This script assumes that we have initialized the appropriate python
# environment ahead of time and that we just need to execute the various Python
# scripts in order.

# The raters we are running over:
VENTRAL_RATERS=(R1 R2 R3 R4 R5)
DORSAL_RATERS=(R1 R6 R7 R8 R9)
# The input and output directories!
# If DATA_PATH is already defined, it is used.
DATA_PATH="${DATA_PATH:-$PWD}"

# Die function for errors:
function die {
    echo "$*"
    exit 1
}

# Make sure we're in the repository directory above the scripts directory.
SCRIPT_DIR=`cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd`
cd "$SCRIPT_DIR"
[ -r ./proc_raters.py ] && [ -r ./proc_means.py ] \
    || die "scripts must be in the same directory together when run"

# Figure out which region we are processing.
region="$1"
if [ "$1" = "ventral" ]
then RATERS=(${VENTRAL_RATERS[@]})
elif [ "$1" = "dorsal" ]
then RATERS=(${DORSAL_RATERS[@]})
else die "Syntax: proc_all.sh [ventral|dorsal] [options]"
fi
shift

# (1) Process all the individual raters.
python proc_raters.py \
    "$region" \
    "$DATA_PATH" \
    "$@" --raters ${RATERS[@]}
# (2) Process the means.
python proc_means.py \
    "$region" \
    "$DATA_PATH" \
    "$@" --raters ${RATERS[@]}

# That's it!
exit 0
