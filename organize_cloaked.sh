#!/bin/bash
# organize_cloaked.sh
# Moves fawkes cloaked output files into a cloaked_version subfolder.
# Called by core_module after fawkes_module returns "done".
#
# Usage: ./organize_cloaked.sh /path/to/folder

#this should be placed in the same folder as the fawkes_module.py script,
#and the core_module.py script should be run from the same folder as well.
#The folder passed to this script should be the same folder that fawkes_module.py was run in."""

FOLDER="$1"

if [ -z "$FOLDER" ] || [ ! -d "$FOLDER" ]; then
    echo "failed"
    exit 1
fi

DEST="$FOLDER/cloaked_version"
mkdir -p "$DEST"

shopt -s nullglob
FILES=("$FOLDER"/*_cloaked*)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "failed"
    exit 1
fi

mv "${FILES[@]}" "$DEST"/
echo "done"