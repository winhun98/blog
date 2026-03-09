#!/usr/bin/env bash
# Renders mermaid code blocks in markdown files to PNG images.
# Writes rendered images to static/images/ and replaces code blocks with image refs.
#
# Usage: ./render-mermaid.sh <markdown-file> [--dry-run]
# The original file is backed up to <file>.bak before modification.

set -euo pipefail

MMDC="${MMDC:-$(command -v mmdc 2>/dev/null || echo "$HOME/.npm-global/bin/mmdc")}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IMAGES_DIR="$REPO_DIR/static/images"
PUPPETEER_CONFIG="$SCRIPT_DIR/puppeteer-config.json"
DRY_RUN=false
INPUT_FILE=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) INPUT_FILE="$arg" ;;
  esac
done

if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 <markdown-file> [--dry-run]"
  exit 1
fi
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File not found: $INPUT_FILE"
  exit 1
fi

if ! command -v "$MMDC" &>/dev/null; then
  echo "Error: mmdc (mermaid-cli) not found. Install: npm i -g @mermaid-js/mermaid-cli"
  exit 1
fi

mkdir -p "$IMAGES_DIR"

BLOCK_NUM=0
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Extract mermaid blocks to individual files
perl -0ne 'while (/```mermaid\n(.*?)\n```/sg) {
  $i++;
  open(F, ">", "'"$TEMP_DIR"'/block_$i.mmd");
  print F $1;
  close F;
}' "$INPUT_FILE"

TOTAL=$(ls "$TEMP_DIR"/block_*.mmd 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 0 ]; then
  echo "No mermaid blocks found in $(basename "$INPUT_FILE")"
  exit 0
fi

# Back up original before any modification
if [ "$DRY_RUN" = false ]; then
  cp "$INPUT_FILE" "${INPUT_FILE}.bak"
fi

RESULT_FILE="$TEMP_DIR/result.md"
cp "$INPUT_FILE" "$RESULT_FILE"

SLUG=$(basename "$INPUT_FILE" .md)

for mmd_file in "$TEMP_DIR"/block_*.mmd; do
  BLOCK_NUM=$((BLOCK_NUM + 1))
  BLOCK_CONTENT=$(cat "$mmd_file")
  HASH=$(echo -n "$BLOCK_CONTENT" | sha256sum | cut -c1-12)
  PNG_NAME="${SLUG}-fig${BLOCK_NUM}-${HASH}.png"
  PNG_PATH="$IMAGES_DIR/$PNG_NAME"

  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would render: $PNG_NAME"
    continue
  fi

  PPCONFIG_ARGS=()
  if [ -f "$PUPPETEER_CONFIG" ]; then
    PPCONFIG_ARGS=(-p "$PUPPETEER_CONFIG")
  fi

  if timeout 30 "$MMDC" -i "$mmd_file" -o "$PNG_PATH" "${PPCONFIG_ARGS[@]}" --scale 2 -b transparent 2>/dev/null; then
    IMG_REF="![Figure ${BLOCK_NUM}](/images/${PNG_NAME})"

    # Use awk for safe replacement (no regex special char issues)
    awk -v img="$IMG_REF" -v target="$BLOCK_NUM" '
      /^```mermaid$/ { in_block=1; block++; next }
      in_block && /^```$/ {
        in_block=0
        if (block == target) print img
        next
      }
      in_block && block == target { next }
      { print }
    ' "$RESULT_FILE" > "$TEMP_DIR/swap.md"
    mv "$TEMP_DIR/swap.md" "$RESULT_FILE"

    echo "  Rendered: $PNG_NAME"
  else
    echo "  WARNING: Failed to render block #${BLOCK_NUM}, keeping original"
  fi
done

if [ "$DRY_RUN" = false ] && [ "$BLOCK_NUM" -gt 0 ]; then
  cp "$RESULT_FILE" "$INPUT_FILE"
  echo "Processed $BLOCK_NUM mermaid block(s) in $(basename "$INPUT_FILE")"
  echo "Backup saved: ${INPUT_FILE}.bak"
fi
