#!/bin/bash
set -euo pipefail

# Delete empty subdirectories in out/
find "./out" -mindepth 1 -type d -empty -delete 2>/dev/null && \
    echo "Deleted empty subdirectories in out/" || \
    echo "No empty subdirectories found in out/"

# Delete generated files from root level
rm -f "profile.h5" "dump_*.h5" "constants.txt"
echo "Deleted leftover output from root"