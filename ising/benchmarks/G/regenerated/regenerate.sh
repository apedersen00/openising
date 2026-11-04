#!/bin/sh
# Regenerate the Gset graphs with rudy, using the arguments listed in commands.txt.
# Output: one <name>_<nodes>_<edges>_<type>_<best known cut>.txt per graph in this folder, in
# the format of the parent folder. The name fields are read from the table of ../README.md.
set -e
cd "$(dirname "$0")"
make -C ../ccode rudy

grep -v '^#' commands.txt | while IFS='|' read -r name args; do
    # $args is split on purpose: it is a list of rudy arguments
    # README row: | G39 | 2,000 | 11,778 | +1, -1 | planar | 2,407 |
    suffix=$(awk -F'|' -v g="$name" '{gsub(/[ ,]/, "")}
        $2 == g {print $3 "_" $4 "_" $6 "_" $7}' ../README.md)
    [ -n "$suffix" ] || { echo "$name: not found in ../README.md" >&2; exit 1; }
    ../ccode/rudy $args > "${name}_${suffix}.txt"
    echo "${name}_${suffix}.txt: rudy $args"
done
