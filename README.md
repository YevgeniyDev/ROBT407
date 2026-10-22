# ROBT 407 - Linear Regression

This folder contains one notebook for Part 2 and one shared module with reusable functions.

## Files

- `part2_linear_regression.ipynb` contains Sections 2.2 to 2.7.
- `mlcore.py` contains the common regression methods, preprocessing, and classification helpers.
- `test_mlcore.py` checks the helper functions.
- `figures/` is used for exported plots.
- `results/` is used for numerical results.

## Run

Install the packages in `requirements.txt`. Open the notebook in VS Code or Jupyter and run the cells in order.

Set `FAST = True` for shorter test runs. Set `FAST = False` to produce the report experiments. Section 2.3 then uses 100 runs per delta, Section 2.4 uses 500 runs per training size, and Section 2.6 uses 20 repeats per timing case.

California Housing is downloaded by scikit-learn on first use. The file is cached for later runs. The benchmark with 1,000,000 rows and 1,000 features can take a long time and use a lot of RAM on a laptop. Save the output files from the full run before writing down final results.

The digit feature split in Section 2.7 follows the assignment, but the Part 1 teammate should use exactly the same split and scaling for a direct comparison.

Use a single shared `mlcore.py` for the final team submission. If other members add functions to it, merge them instead of creating separate versions.
