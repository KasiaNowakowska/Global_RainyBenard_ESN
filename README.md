# Global Rainy-Benard ESN

This repository contains scripts to generate global Rainy-Benard simulation
time series, prepare them for the Echo State Network (ESN), train and test the
ESN, and run further analyses on saved model results.

## ESN framework and citation

The ESN framework in this repository is based on the implementation in
[alberacca/Echo-State-Networks](https://github.com/alberacca/Echo-State-Networks),
which implements the methods described in:

> A. Racca and A. Magri, "Robust Optimization and Validation of Echo State
> Networks for learning chaotic dynamics," *Neural Networks*, vol. 142,
> pp. 252-268, 2021. https://doi.org/10.1016/j.neunet.2021.05.004

Please cite the paper when using the ESN optimization and validation framework.
The Rainy-Benard simulation setup, data preparation, and application-specific
analysis in this repository are separate from that upstream implementation.

## Environment setup

Create the Conda environment used for preprocessing and the ESN scripts:

```bash
conda env create -f environment.yml
conda activate rainy-esn
```

The simulation generator also requires Dedalus and MPI, which are not included
in `environment.yml`. Run it in a Linux/HPC environment where Dedalus v3 is
installed and configured. The commands below use one MPI rank; the parallel
simulation script is not required for this workflow.

## 1. Generate simulation output (optional)

Run `data_generation.py` from the repository root. For example:

```bash
mpiexec -n 1 python data_generation.py \
  --Rayleigh 2e8 \
  --beta 1.201 \
  --gamma 0.5 \
  --alpha 3.0 \
  --nx 256 \
  --nz 64 \
  --Lx 20 \
  --RH0 0.75 \
  --RHb 0.75 \
  --RHt 0.75 \
  --simtime 50000 \
  --max_dt 1e-2
```

The script writes output under a `scratch/RainyBenard_...` directory relative
to the working directory. Its `timeseries` output contains the global
volume-averaged kinetic energy (`KE`) and moisture (`moisture`) used by the
ESN; `checkpoints` contains the solver state for restarting a simulation.
Simulation output is not included in this repository.

If you already have the NumPy time series used for the paper, skip this step.

## 2. Convert Dedalus HDF5 output to NumPy

The ESN scripts read three one-dimensional arrays: `KE`, `q`, and
`total_time`. In Dedalus output, the moisture task is called `moisture`; the
preprocessor saves it under the ESN input name `q`.

Convert a single Dedalus timeseries file:

```bash
python preprocess_timeseries.py \
  --input-h5 path/to/timeseries_s1.h5 \
  --output-dir path/to/input_data \
  --output-suffix 5000_30000
```

This creates `KE5000_30000.npy`, `q5000_30000.npy`, and
`total_time5000_30000.npy` in `input_data`.

To append a restarted run to an existing NumPy dataset, skip the duplicated
initial rows from the restart. For example, to discard 200 rows:

```bash
python preprocess_timeseries.py \
  --input-h5 path/to/restarted/timeseries_s1.h5 \
  --output-dir path/to/input_data \
  --output-suffix 5000_48000 \
  --base-suffix 5000_40000 \
  --skip-rows 200
```

Use `--base-dir path/to/existing_arrays` if the original arrays are in a
different directory. Existing output files are not overwritten unless
`--overwrite` is specified. The script checks the array lengths and requires
the merged simulation times to be strictly increasing.

## 3. Train and test the ESN

Run `ESN_main.py` from the repository root (or another working directory; its
function files are located relative to the script):

```bash
python ESN_main.py \
  --input_path path/to/input_data \
  --output_path path/to/results \
  --hyperparam_file ESN_params.json \
  --config_number paper \
  --data_suffix 5000_48000
```

`--data_suffix` selects the shared suffix in the three NumPy filenames; it
defaults to `5000_30000`. The input directory does not need a trailing slash.
The `--config_number` value is used to distinguish the output folder.

`ESN_params.json` is the example configuration. It controls reservoir size,
washout/training/validation/test lengths (in Lyapunov times), ensemble size,
hyperparameter search grid, validation objective, and input noise. The example
uses 100 ensemble members and performs a Bayesian-optimization search, so a
full run may take considerable time. To do a quick trial, make a copy of the
JSON and reduce `ens`, `ensemble_test`, grid dimensions, and added points.
Keep `ensemble_test` no larger than `ens`; grid dimensions should be at least
2 when using a nonzero grid.

The data must be long enough for the configured washout, training, and test
intervals. With the example configuration (`t_lyap=500`, one washout LT, 40
training LTs, and three 3-LT test intervals), `ESN_main.py` requires at least
25,000 samples.

The script writes plots, hyperparameters, trained matrices, and results below
`path/to/results/Run_n_units...`, using the configuration values and
`--config_number` in the folder name.

## 4. Predict on unseen data with a saved ESN

For straightforward inference, use `predict_esn.py`. It loads the model
matrices saved by `ESN_main.py`, warms up the reservoir from the measured
history before the forecast, then saves closed-loop predictions.

The input arrays should contain the training segment followed by the unseen
segment. Set `--start-index` to the first unseen sample and `--horizon` to the
number of samples to predict. For the example configuration, training ends at
sample `N_washout * t_lyap + N_train * t_lyap = 20,500`; a 3-Lyapunov-time
forecast has 1,500 samples:

```bash
python predict_esn.py \
  --input_path path/to/input_data \
  --output_path path/to/results/Run_n_units250_ensemble100_normalisationon_washout1_configpaper \
  --hyperparam_file ESN_params.json \
  --config_number paper \
  --data_suffix 5000_48000 \
  --start_index 20500 \
  --horizon 1500
```

The output array has shape `(ensemble, horizon, 2)`; the last dimension is
`[KE, q]`. The model is loaded from the run directory using the objective name
and reservoir size in the JSON file (for this example,
`RVC_Noise250.mat`). Predictions are saved to that directory as
`predictions_paper_5000_48000_20500_1500.npy`. `--start_index` must be at or
after the configured training end, and the input arrays must include the
warmup history before it.

## Data availability

Large simulation outputs and NumPy arrays are kept outside the code
repository.
