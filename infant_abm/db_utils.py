import json

from os import path

import pandas as pd


def load_run(run_path):
    description_df = pd.read_csv(path.join(run_path, "description.csv"), index_col=0)

    def load_partial(index):
        partial_path = get_partial_path(run_path, index)

        with open(partial_path, "r") as file:
            return json.load(file)

    return description_df, load_partial


def save_partial(run_path, index, result):
    partial_path = get_partial_path(run_path, index)

    with open(partial_path, "w") as file:
        json.dump(result, file)


def get_partial_path(run_path, index):
    return path.join(run_path, f"{index}.json")
