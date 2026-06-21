#!/usr/bin/env bash

# List of directories to format
DIRECTORIES=("include" "tests" "benchmarks")

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

# Loop through each directory
for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Formatting files in directory: $DIR"

        # Find and format all .cpp and .hpp files in one command
	find "$DIR" \( -iname "*.cpp" -o -iname "*.hpp" \) -exec ${CLANG_FORMAT} -i {} \; -print
    else
        echo "Directory $DIR does not exist."
    fi
done

echo "Formatting complete."
