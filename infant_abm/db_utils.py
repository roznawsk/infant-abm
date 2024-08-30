import json
import numpy as np
import pandas as pd


from os import path
from pathlib import Path


def load_run(run_path):
    description_df = pd.read_csv(path.join(run_path, "description.csv"), index_col=0)

    def load_partial(index):
        partial_path = get_partial_path(run_path, index)
        partial_dir = get_partial_dir(run_path, index)

        with open(partial_path, "r") as file:
            result = json.load(file)

        for repeat, rep_result in result.items():
            for k, v in rep_result.items():
                if isinstance(v, list) and v[0] == "ndarray":
                    data_path = path.join(partial_dir, str(repeat), str(k))

                    shape = [int(s) for s in v[2]]

                    result[repeat][k] = np.fromfile(data_path, dtype=v[1]).reshape(
                        shape
                    )

        return result

    return description_df, load_partial


def partial_exists(run_path, index):
    partial_path = get_partial_path(run_path, index)
    return path.exists(partial_path)


def save_partial(run_path: str, index: int, result: dict):
    partial_path = get_partial_path(run_path, index)
    partial_dir = get_partial_dir(run_path, index)

    arrays_stored = any([isinstance(v, np.ndarray) for v in result["0"].values()])
    if arrays_stored:
        Path(partial_dir).mkdir(parents=False, exist_ok=False)

    for repetition, rep_result in result.items():
        repetition_dir = path.join(partial_dir, str(repetition))
        if arrays_stored:
            Path(repetition_dir).mkdir(parents=False, exist_ok=False)

        for k, v in rep_result.items():
            if isinstance(v, np.ndarray):
                arr_path = path.join(repetition_dir, str(k))
                v.tofile(arr_path)

                result[repetition][k] = [("ndarray"), str(v.dtype), v.shape]

    with open(partial_path, "w") as file:
        json.dump(result, file)


def get_partial_dir(run_path, index):
    return path.join(run_path, str(index))


def get_partial_path(run_path, index):
    return path.join(run_path, f"{index}.json")
