# Using Intermarket Data to Evaluate the Efficient Market Hypothesis with Machine Learning

### Authors: [N'yoma Diamond](https://github.com/nyoma-diamond), [Grant Perkins](https://github.com/GrantPerkins)

#### Worcester Polytechnic Institute, Departments of Computer Science & Data Science

Code for experiments conducted in the paper of the same name.


## Requirements

### Python Version: 3.9.12

### Dependencies

 - [NumPy](https://numpy.org/) (ver. 1.23.3)
 - [pandas](https://pandas.pydata.org/) (ver. 1.5.0)
 - [scikit-learn](https://scikit-learn.org/) (ver. 1.1.2)
 - [statsmodels](https://www.statsmodels.org/) (ver. 0.13.5)
 - [more-itertools](https://github.com/more-itertools/more-itertools) (ver. 8.14.0)


### Input Data

#### `main.py`

By default, `main.py` expects input data in the form of CSVs using the following directory structure: 

```
./data/{asset type}/{asset name}.csv
```

Expected asset type/name combinations are as follows:

 - `stock`
   - `SPY.US` 
 - `forex`
   - `USDGBP.FOREX`
   - `USDEUR.FOREX`
   - `USDCAD.FOREX`
   - `USDJPY.FOREX`
   - `EURGBP.FOREX`
 - `bond`
   - `US10Y.GBOND`
   - `US5Y.GBOND`
   - `UK5Y.GBOND`
   - `JP5Y.GBOND`
 - `future`
   - `US.COMM` 
   - `ES.COMM`
   - `NK.COMM`
   - `HSI.COMM`
   - `FESX.COMM`
   - `VIX.COMM`
   - `GC.COMM`
   - `NG.COMM`
   - `ZC.COMM`
   - `ZS.COMM`

#### `stats.ipynb`

By default, `stats.ipynb` expects input data in the form of CSVs with the following directory/name structure:
```
./out/**/*results.csv
``` 

Note that `**/*` enables the selection of any and all files ending in `results.csv` from arbitrary subdirectories of `./out`

If desired, this can be changed by editing the `data_dir` and `concat_results` variables.


## File Organization

The files in this project are organized as follows:

 - `main.py` contains code for running experiments, including model hyperparameter setting and initialization.
 - `stats.ipynb` contains code for evaluating experiment results.
 - `trainer.py` contains code for the model training and hyperparameter search pipelines.
 - `dataset.py` contains code for initializing datasets.
 - `utils.py` contains general-purpose utility code.
 - `constants.py` contains useful constants, settings, and typing information.

## Running Experiments

The code can be run from command-line using the following command:

```bash
python main.py <optional arguments>
```

This will create an output directory (if one does not already exist) and save the experimental results into it. If no arguments are provided, this will run a single replication where all models are trained in serial with results saved into a file named `results.csv`.

### Optional Arguments

If desired, optional command-line arguments may be provided to `main.py`:

 - `-p <value (optional)>`/`--processes=<value (optional)>`: Enable the use of multiple parallel processes when training models. Providing a number specifies the maximum number of processes to run concurrently **_per replication_**. E.g., `main.py -p 5` sets a maximum of 5 concurrent processes per replication. Unlimited processes are used if a numeric value is not provided (I.e., `main.py -p`). 
    
    **NOTE:** The `n_jobs` parameter for supported scikit-learn operations will be set to 1 due to incompatibility with multiprocessing. Otherwise, `n_jobs` is set to -1 (unlimited jobs) unless `-r`/`--replications` is used.

    **_WARNING:_** Allowing unlimited processes may use large amounts of resources and cause systems to freeze or crash.

 - `-r <value>`/`--replications=<value>`: Run experiment using the provided number of replications. 

    **NOTE:** Each replication is given a dedicated process. As a result, the `n_jobs` parameter for supported scikit-learn operations will be set to 1 due to incompatibility with multiprocessing. Therefore, combining this argument with `-p`/`--processes` is highly recommended.

    **NOTE:** The results of each replication are saved to a dedicated file with the name `{replication number}_results.csv`.

 - `-m <model name>`/`--model=<model name>`: Run experiment using only the provided model. Acceptable inputs are as follows:
   - DecisionTree
   - RandomForest
   - LogisticRegression
   - LinearSVM
   - KNN
   - RandomBaseline
   - ConstantBaseline
   - PreviousBaseline
   - ConsensusBaseline

    **NOTE:** This changes the name of the `results.csv` output file to contain the specified model name (i.e., `{model name}_results.csv`).

 - `-o <output path>`/`--out_dir=<output path>`: Use the specified output directory to save experiment results to. 

    **NOTE:** The provided directory and any parent directories will be created if not already present.

 - `-u`/`--use-uuid`: Append a unique identifier (UUID) to the end of the output directory. This argument _can_ be used in conjunction with `-o`/`--out_dir`. This is useful to prevent accidentally overwriting results files when running multiple experiments without using `-r`/`--replications`


## Analyzing Results

In order to analyze results, run the code in `stats.ipynb`. This can be done using any Jupyter Notebook environment (e.g. Jupyter Notebook IDE, Jupyter Lab, PyCharm, Google Colab). This will perform the same analysis described in the paper. 

**NOTE:** By default, this code assumes that the files to analyze will end in `results.csv`, and will not work otherwise.

### Editable Parameters

`stats.ipynb` has a number of parameters which can be changed if desired:

 - `data_dir`: The directory from which to load results data. `./out` by default.

 - `concat_results`: Whether to combine any and all available results files in `data_dir`. `True` by default. If `False`, use `<data_dir>/results.csv`.

   **NOTE:** When True, requires desired files for joining to end in `results.csv`. Files in subdirectories of `data_dir` will also be used.

 - `alpha`: $\alpha$ threshold to use when computing reduced ANOVA models (1 - confidence level). `0.05` by default.

 - `only_reduced`: Only display reduced ANOVA models. `False` by default.

 - `latex_output`: Change table output to print $\LaTeX$ formatted tables instead of DataFrames. `False` by default. 

 - `combine_anova_latex`: Combine full and reduced ANOVA model printouts into one table. 

   **NOTE:** **_Requires_** `latex_output` to be `True`, does nothing otherwise. Overrides `only_reduced` if `True`