#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Format source code with clang-format (or cf when clang-format is unavailable).
#
# Formats C/C++/CUDA sources across the project, excluding build directories
# and the original Fortran/C reference solver in ref/.
#
# Usage: ./scripts/format.sh [--check]
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR=$(dirname "$(dirname "$(realpath "$0")")")

CHECK=0
if [[ ${1:-} == --check ]]; then
  CHECK=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

EXCLUDE_PATTERNS=("cmake*" "build*" "ref")
EXTENSIONS=("cpp" "c" "h" "hpp" "cuh" "cu")

FORMATTER=()
if command -v clang-format >/dev/null 2>&1; then
  FORMATTER=(clang-format)
elif command -v cf >/dev/null 2>&1; then
  FORMATTER=(cf)
else
  echo "Error: neither 'clang-format' nor 'cf' is available." >&2
  exit 1
fi

# .clang-format uses keys that need clang-format 16 or newer.
if [[ ${FORMATTER[0]} == clang-format ]]; then
  version=$(clang-format --version | grep -oE '[0-9]+' | head -1)
  if [[ -z $version || $version -lt 16 ]]; then
    echo "Error: clang-format >= 16 is required (found: $(clang-format --version))." >&2
    exit 1
  fi
elif [[ $CHECK -eq 1 ]]; then
  echo "Error: --check requires clang-format (cf has no dry-run mode)." >&2
  exit 1
fi

PRUNE_ARGS=()
for PATTERN in "${EXCLUDE_PATTERNS[@]}"; do
  # shellcheck disable=SC2086
  for DIR in "$ROOT_DIR"/$PATTERN; do
    [ -d "$DIR" ] && PRUNE_ARGS+=(-path "$DIR" -prune -o)
  done
done

NAME_ARGS=()
for EXT in "${EXTENSIONS[@]}"; do
  NAME_ARGS+=(-name "*.$EXT" -o)
done
unset 'NAME_ARGS[${#NAME_ARGS[@]}-1]'

if [[ $CHECK -eq 1 ]]; then
  STATUS=0
  while IFS= read -r -d '' FILE; do
    "${FORMATTER[@]}" --dry-run --Werror "$FILE" || STATUS=1
  done < <(find "$ROOT_DIR" "${PRUNE_ARGS[@]}" \( "${NAME_ARGS[@]}" \) -type f -print0)
  exit "$STATUS"
fi

find "$ROOT_DIR" "${PRUNE_ARGS[@]}" \( "${NAME_ARGS[@]}" \) -type f -print0 |
  xargs -0 -r -n1 -P "$(nproc)" "${FORMATTER[@]}" -i
