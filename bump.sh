#!/bin/bash
# Keep this lite-lib tree in sync with the library files of a source branch.
# The source branch (master by default) stays authoritative; this orphan branch
# carries only the minimum set needed to consume GSPIKE as a CMake submodule.
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
    echo "No arguments supplied, using default branch"
    BRANCH_SOURCE=master
elif [ "$#" -eq 1 ]; then
    BRANCH_SOURCE=$1
else
    echo "Usage: $0 [source-branch]"
    exit 2
fi

if ! command -v git &> /dev/null; then
    echo "git not found"
    exit 1
fi

cd "$SCRIPT_DIR"

git fetch origin

# Only these paths make up the library.  Tests, benchmarks, the Fortran
# reference, docs, CI and build trees stay in the source branch.
LIB_PATHS=(CMakeLists.txt include src)

# Drop the previous copies first so deletions in the source branch propagate.
for path in "${LIB_PATHS[@]}"; do
    if [ -e "$path" ]; then
        git rm -r -q --ignore-unmatch -- "$path"
        rm -rf -- "$path"
    fi
done

# Check out whatever still exists in the source branch.
paths=()
for path in "${LIB_PATHS[@]}"; do
    if git cat-file -e "origin/$BRANCH_SOURCE:$path" 2>/dev/null; then
        paths+=("$path")
    fi
done
if [ ${#paths[@]} -gt 0 ]; then
    git checkout "origin/$BRANCH_SOURCE" -- "${paths[@]}"
fi
git restore --staged .

if ! git diff --quiet; then
    echo "Changes detected in library files, syncing lite-lib with $BRANCH_SOURCE..."
    git add --all -- "${LIB_PATHS[@]}"
    git commit -m "Automatic sync with $BRANCH_SOURCE"
fi
