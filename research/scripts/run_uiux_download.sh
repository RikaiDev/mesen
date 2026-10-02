#!/usr/bin/env bash
set -u

research_root=/home/gloomcheng/Workspace/RikaiDev/mesen/research
cd "$research_root" || exit 1
python3 scripts/download_uiux_teacher.py > logs/uiux-download.log 2>&1
exit_code=$?
printf '%s\n' "$exit_code" > logs/uiux-download.exit
exit "$exit_code"
