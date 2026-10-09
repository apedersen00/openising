"""Compare the regenerated Gset graphs with the files in the parent folder.

What is tested, and why
-----------------------
`regenerate.sh` rebuilds every graph of the Gset README (G1..G81 and K2000) with rudy and names each file
<name>_<nodes>_<edges>_<type>_<best known cut>.txt after the README table. The rudy sources are
1990s C compiled on a modern 64-bit machine, and the arguments in `commands.txt` were
recovered by search, so each regenerated graph is checked against the original file.

For each graph of `commands.txt`:
  0. name:    exactly one file <name>_*.txt exists and it is named after the README row.
  1. header:  the regenerated N and E equal the N and E of the README table.
  2. size:    the regenerated file holds exactly E edge lines.
  3. edges:   the regenerated edge list (node_i, node_j, weight, in file order) equals
              the original one. Some original files are cut short (fewer edge lines
              than their header announces); for those the original edges must be a
              prefix of the regenerated ones, and the file is listed as TRUNCATED.

A graph passes when the four checks hold. Line endings and the trailing text of the
header line (e.g. "800 19176 J") are ignored.

Run `./regenerate.sh` first, then `python 01_compare_with_original.py`.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent


def read_graph(path: Path):
    """Return (N, E, edges) where edges are the complete 3-field lines, in file order."""
    lines = path.read_text().replace("\r", "").split("\n")
    n, e = (int(x) for x in lines[0].split()[:2])
    edges = [tuple(line.split()) for line in lines[1:] if len(line.split()) == 3]
    return n, e, edges


def readme_sizes():
    """Return {name: (N, E, type, best known cut)} from the table of the Gset README."""
    rows = re.findall(
        r"^\|\s*(G\d+|K2000)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|[^|]*\|\s*(\w+)\s*\|\s*([\d,]+)\s*\|",
        (ORIGINAL / "README.md").read_text(),
        re.M,
    )
    num = lambda x: int(x.replace(",", ""))
    return {name: (num(n), num(e), typ, num(best)) for name, n, e, typ, best in rows}


def main():
    sizes = readme_sizes()
    names = [line.split("|")[0] for line in (HERE / "commands.txt").read_text().splitlines() if line and line[0] != "#"]

    missing = sorted(set(sizes) - set(names), key=lambda s: int(s[1:]))
    print(f"README graphs: {len(sizes)}, graphs in commands.txt: {len(names)}, missing: {missing or 'none'}\n")

    passed, truncated = 0, []
    for name in names:
        n_readme, e_readme, typ, best = sizes[name]
        expected = f"{name}_{n_readme}_{e_readme}_{typ}_{best}.txt"
        candidates = sorted(f.name for f in HERE.glob(f"{name}_*.txt"))
        if candidates != [expected]:
            print(f"[FAIL] {name}: expected only {expected}, found {candidates or 'nothing'}, run ./regenerate.sh")
            continue
        new_file = HERE / expected
        n, e, edges = read_graph(new_file)
        n_ref, e_ref, edges_ref = read_graph(ORIGINAL / f"{name}.txt")

        header_ok = (n, e) == (n_readme, e_readme)
        size_ok = len(edges) == e
        is_truncated = len(edges_ref) < e_ref
        edges_ok = edges[: len(edges_ref)] == edges_ref if is_truncated else edges == edges_ref

        ok = header_ok and size_ok and edges_ok
        passed += ok
        note = ""
        if is_truncated:
            truncated.append(name)
            note = f"  (original TRUNCATED: {len(edges_ref)}/{e_ref} edges, compared as prefix)"
        print(
            f"[{'PASS' if ok else 'FAIL'}] {expected}: N={n} E={e} "
            f"header={'ok' if header_ok else 'BAD'} size={'ok' if size_ok else 'BAD'} "
            f"edges={'ok' if edges_ok else 'BAD'}{note}"
        )

    print(f"\n{passed}/{len(names)} passed")
    if truncated:
        print(f"Original files that are truncated in the parent folder: {', '.join(truncated)}")
    return 0 if passed == len(names) and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
