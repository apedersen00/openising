# MKP Benchmark: OR-Library

The multidimensional knapsack problem (MKP): pick the items with maximum total profit such that every resource (constraint) stays within its capacity. All files are from J.E. Beasley's OR-Library and can be found [here](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html).

The main set, `mknapcb1` to `mknapcb9`, is from P.C. Chu and J.E. Beasley, "A genetic algorithm for the multidimensional knapsack problem", Journal of Heuristics 4 (1998) 63-86. It is the standard MKP benchmark, and the one used by Delacour et al. (2025) for SAIM.

| **File** | **Constraints** | **Items** | **Instances** | **Source** |
|----------|-----------------|-----------|---------------|------------|
mknapcb1 | 5  | 100 | 30 | Chu & Beasley (1998)
mknapcb2 | 5  | 250 | 30 | Chu & Beasley (1998)
mknapcb3 | 5  | 500 | 30 | Chu & Beasley (1998)
mknapcb4 | 10 | 100 | 30 | Chu & Beasley (1998)
mknapcb5 | 10 | 250 | 30 | Chu & Beasley (1998)
mknapcb6 | 10 | 500 | 30 | Chu & Beasley (1998)
mknapcb7 | 30 | 100 | 30 | Chu & Beasley (1998)
mknapcb8 | 30 | 250 | 30 | Chu & Beasley (1998)
mknapcb9 | 30 | 500 | 30 | Chu & Beasley (1998)
mknap1   | 5-10 | 6-50 | 7 | C.C. Petersen, Management Science 13(9) (1967) 736-750
mknap2   | various | various | 48 | Classic instances from the literature, listed at the top of the file

In every `mknapcb` file, the first ten instances have a tightness ratio of 0.25, the second ten 0.50 and the last ten 0.75. The tightness ratio is the capacity of a constraint divided by the sum of all item weights in that constraint.

## Benchmark format
`mknap1` and `mknapcb1` to `mknapcb9` share one format. The numbers wrap over lines at random, so read the file as one stream of numbers:
<ol>
    <li> The number of instances in the file,</li>
    <li> per instance: the number of items n, the number of constraints m and the optimal value (0 if not given),</li>
    <li> the n profits,</li>
    <li> the m x n weights, one constraint after the other, and</li>
    <li> the m capacities.</li>
</ol>

`mknap2` has its own format, which is described at the top of the file.

## Best known values
`mkcbres.txt` lists, for all 270 `mknapcb` instances, the best value found by the genetic algorithm of Chu & Beasley and the value of the LP relaxation. The GA values are not proven optimal. For the first ten instances of `mknapcb1` and `mknapcb4` they equal the exact optima from `scipy.optimize.milp`. The optimal values of `mknap1` are in the file itself, and those of `mknap2` are listed with each instance.
