#!/usr/bin/env bash
# Format all notebooks in a directory
# Usage: sh format_notebook.sh a_directory

DIR=$1
if [[ -z $1 ]]; then
	DIR=.
fi

find . -type f -name '*.ipynb' -print0 | xargs -0 -I file python3 scripts/format_notebook.py file
