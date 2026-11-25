"""
Generate closed-loop ESN predictions from a saved training model.

Usage: predict_esn.py --input_path=<input_path> --output_path=<output_path> --hyperparam_file=<hyperparam_file> --config_number=<config_number> --data_suffix=<data_suffix> --start_index=<start_index> --horizon=<horizon>

Options:
    --input_path=<input_path>            Directory containing KE, q, and time NumPy files
    --output_path=<output_path>          ESN_main run directory containing the saved model
    --hyperparam_file=<hyperparam_file>  ESN hyperparameter JSON file
    --config_number=<config_number>      Label used in the predictions filename
    --data_suffix=<data_suffix>          Shared suffix used by the NumPy input files
    --start_index=<start_index>          First sample of the unseen forecast interval
    --horizon=<horizon>                  Number of forecast samples
"""

import json
from pathlib import Path

import numpy as np
from docopt import docopt
from scipy.io import loadmat


def load_inputs(input_dir: Path, suffix: str) -> np.ndarray:
    ke = np.load(input_dir / f"KE{suffix}.npy")
    moisture = np.load(input_dir / f"q{suffix}.npy")
    times = np.load(input_dir / f"total_time{suffix}.npy")
    if any(array.ndim != 1 for array in (ke, moisture, times)):
        raise ValueError("KE, q, and total_time arrays must be one-dimensional")
    if not (len(ke) == len(moisture) == len(times)):
        raise ValueError("KE, q, and total_time arrays must have matching lengths")
    if not all(np.all(np.isfinite(array)) for array in (ke, moisture, times)):
        raise ValueError("Input arrays must contain only finite values")
    return np.column_stack((ke, moisture))


def load_ensemble(
    model_path: Path,
    expected_units: int,
) -> tuple[list, list, np.ndarray, np.ndarray, float]:
    model = loadmat(model_path)
    required_keys = ("Win", "W", "Wout", "opt_hyp", "fix_hyp")
    missing = [key for key in required_keys if key not in model]
    if missing:
        raise ValueError(f"{model_path} is missing model arrays: {', '.join(missing)}")

    win_matrices = list(model["Win"].reshape(-1))
    recurrent_matrices = list(model["W"].reshape(-1))
    readouts = np.asarray(model["Wout"])
    hyperparameters = np.asarray(model["opt_hyp"])
    fixed_parameters = np.asarray(model["fix_hyp"]).reshape(-1)
    if len(fixed_parameters) < 2:
        raise ValueError("Model fix_hyp must contain input bias and washout length")
    if readouts.ndim != 3 or readouts.shape[1] != expected_units + 1 or readouts.shape[2] != 2:
        raise ValueError(
            f"Expected Wout shape (ensemble, {expected_units + 1}, 2), "
            f"got {readouts.shape}"
        )
    if hyperparameters.shape != (readouts.shape[0], 2):
        raise ValueError(
            f"Expected opt_hyp shape ({readouts.shape[0]}, 2), "
            f"got {hyperparameters.shape}"
        )
    if len(win_matrices) != readouts.shape[0] or len(recurrent_matrices) != readouts.shape[0]:
        raise ValueError("Win, W, Wout, and opt_hyp must contain the same ensemble size")
    return (
        win_matrices,
        recurrent_matrices,
        readouts,
        hyperparameters,
        float(fixed_parameters[0]),
    )


def advance(
    state: np.ndarray,
    input_value: np.ndarray,
    win,
    recurrent,
    sigma_in: float,
    rho: float,
    alpha0: float,
    normalisation: str,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    bias_in: float,
) -> np.ndarray:
    if normalisation in ("on", "standard"):
        normalized = (input_value - input_mean) / input_scale
    elif normalisation == "off":
        normalized = input_value
    else:
        raise ValueError(
            "predict_esn.py supports the global ESN normalisations: "
            "'on', 'standard', and 'off'"
        )

    augmented_input = np.concatenate((normalized, [bias_in]))
    candidate = np.tanh(win.dot(augmented_input * sigma_in) + recurrent.dot(rho * state))
    next_state = (1 - alpha0) * state + alpha0 * candidate
    return np.concatenate((next_state, [1.0]))


def predict_member(
    washout: np.ndarray,
    horizon: int,
    win,
    recurrent,
    readout: np.ndarray,
    sigma_in: float,
    rho: float,
    alpha0: float,
    normalisation: str,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    bias_in: float,
) -> np.ndarray:
    n_units = readout.shape[0] - 1
    state = np.zeros(n_units)
    for input_value in washout:
        augmented_state = advance(
            state,
            input_value,
            win,
            recurrent,
            sigma_in,
            rho,
            alpha0,
            normalisation,
            input_mean,
            input_scale,
            bias_in,
        )
        state = augmented_state[:-1]

    augmented_state = np.concatenate((state, [1.0]))
    prediction = np.empty((horizon, readout.shape[1]))
    prediction[0] = augmented_state @ readout
    for step in range(1, horizon):
        augmented_state = advance(
            augmented_state[:-1],
            prediction[step - 1],
            win,
            recurrent,
            sigma_in,
            rho,
            alpha0,
            normalisation,
            input_mean,
            input_scale,
            bias_in,
        )
        prediction[step] = augmented_state @ readout
    return prediction


def main() -> None:
    args = docopt(__doc__)
    input_dir = Path(args["--input_path"])
    output_dir = Path(args["--output_path"])
    hyperparam_file = Path(args["--hyperparam_file"])
    config_number = args["--config_number"]
    data_suffix = args["--data_suffix"]
    start_index = int(args["--start_index"])
    horizon = int(args["--horizon"])

    with hyperparam_file.open(encoding="utf-8") as file:
        config = json.load(file)
    n_units = int(config["Nr"])
    washout_length = int(config["N_washout"]) * int(config["t_lyap"])
    train_length = int(config["N_train"]) * int(config["t_lyap"])
    normalisation = config["normalisation"]
    alpha0 = float(config.get("alpha0", 1.0))

    data = load_inputs(input_dir, data_suffix)
    if horizon <= 0:
        raise ValueError("--horizon must be a positive number of samples")
    if start_index < washout_length:
        raise ValueError(
            f"--start_index must be at least {washout_length} to provide "
            "the model's washout history"
        )
    if start_index + horizon > len(data):
        raise ValueError(
            f"Prediction interval ends at sample {start_index + horizon}, "
            f"but the input contains only {len(data)} samples"
        )

    training_end = int(config["N_washout"]) * int(config["t_lyap"]) + train_length
    if training_end > len(data):
        raise ValueError(
            f"Input has {len(data)} samples but training normalization requires "
            f"{training_end} samples"
        )
    training_data = data[:training_end]
    input_mean = training_data.mean(axis=0)
    if normalisation == "on":
        input_scale = np.ptp(training_data, axis=0)
    elif normalisation == "standard":
        input_scale = training_data.std(axis=0)
    elif normalisation == "off":
        input_scale = np.ones(training_data.shape[1])
    else:
        raise ValueError(
            "predict_esn.py supports the global ESN normalisations: "
            "'on', 'standard', and 'off'"
        )
    if np.any(input_scale == 0):
        raise ValueError("Training data has a zero normalization scale for an input")

    model_path = output_dir / f"{config['val']}{n_units}.mat"
    win_matrices, recurrent_matrices, readouts, opt_hyp, bias_in = load_ensemble(
        model_path, n_units
    )
    if start_index < training_end:
        raise ValueError(
            f"--start_index must be at or after the end of training "
            f"(sample {training_end}) to predict unseen data"
        )

    washout = data[start_index - washout_length : start_index]
    predictions = np.empty((readouts.shape[0], horizon, 2))
    for member in range(readouts.shape[0]):
        predictions[member] = predict_member(
            washout,
            horizon,
            win_matrices[member],
            recurrent_matrices[member],
            readouts[member],
            float(opt_hyp[member, 1]),
            float(opt_hyp[member, 0]),
            alpha0,
            normalisation,
            input_mean,
            input_scale,
            bias_in,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"predictions_{config_number}_{data_suffix}_"
        f"{start_index}_{horizon}.npy"
    )
    np.save(output_path, predictions)
    print(f"Saved predictions with shape {predictions.shape} to {output_path}")


if __name__ == "__main__":
    main()
