import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from lagrange import BENCH, knapsack_to_qubo, load_qkp, paper_rule, qubo_to_ising

    return BENCH, knapsack_to_qubo, load_qkp, np, paper_rule, pd, qubo_to_ising


@app.cell
def _(knapsack_to_qubo, load_qkp, np, paper_rule, pd, qubo_to_ising):
    def coefficient_range(x):
        """
        Smallest and largest nonzero |value|, and the signed bits needed so the smallest value is exactly one step.
        """
        x = np.abs(x[x != 0])
        ratio = x.max() / x.min()
        return x.min(), x.max(), np.log2(ratio), int(np.ceil(np.log2(ratio + 1))) + 1


    def row_bits(J):
        """
        Signed bits needed if every row gets its own scale: the worst row decides.
        """
        return max(coefficient_range(row)[3] for row in J if row.any())


    def precision_table(name, P_factor=8, lam=25.0):
        w, profit, C = load_qkp(name)
        Q_obj, Q_pen, _, a = knapsack_to_qubo(w, profit, C)
        J_obj, h_obj, _ = qubo_to_ising(Q_obj)
        J_pen, h_pen, _ = qubo_to_ising(Q_pen)
        P_rule, _ = paper_rule(profit, a, C)
        P = P_factor * P_rule

        parts = {
            ('A', 'J'):                 J_obj + P * J_pen,              # penalty in the grid
            ('A', 'h'):                 h_obj + P * h_pen - lam * a / 2,
            ('B', 'J (profits)'):       J_obj,                          # constraint outside the grid
            ('B', 'h (profits)'):       h_obj,
            ('B', 'a (items)'):         a[:len(w)],                     # extra row/column cells
            ('B', 'a (items + slack)'): a,
        }
        rows = []
        for (option, stored), x in parts.items():
            lo, hi, log_ratio, bits = coefficient_range(x)
            rows.append({'option': option, 'stored': stored, 'min': lo, 'max': hi, 'log2(max/min)': log_ratio,
                         'bits': bits, 'bits per row': row_bits(x) if x.ndim == 2 else None})
        return pd.DataFrame(rows)

    return (precision_table,)


@app.cell
def _(BENCH, pd, precision_table):
    BAD = {'jeu_100_100_4', 'jeu_200_25_3', 'jeu_300_25_3'}
    names = sorted(p.stem for p in BENCH.glob('jeu_*.txt') if p.stem not in BAD)

    tables = pd.concat([precision_table(name).assign(instance=name) for name in names], ignore_index=True)
    tables
    return (tables,)


@app.cell
def _(tables):
    A_bits = tables[(tables.option == 'A') & (tables.stored == 'J')].bits
    counts = A_bits.value_counts().reindex(range(A_bits.min(), A_bits.max() + 1), fill_value=0)
    counts.plot.bar(color='#2a78d6', xlabel='bits needed for J', ylabel='# benchmarks')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
