import multiprocessing
import os
import pandas as pd
import tqdm

from copy import deepcopy

from infant_abm.model import InfantModel

from infant_abm.db_utils import save_partial


class Simulation:
    def __init__(
        self,
        model_param_sets: list[dict],
        iterations: int,
        repeats: int,
        output_dir=None,
        display=False,
    ):
        self.parameter_sets: dict = model_param_sets
        self.iterations: int = iterations
        self.repeats: int = repeats
        self.display = display

        if (
            output_dir is None
            or not os.path.exists(output_dir)
            or not os.path.isdir(output_dir)
        ):
            raise ValueError("Output path must point to an existing directory")

        self.output_dir = output_dir
        self._save_description()

    def run(self):
        n_runs = len(self.parameter_sets)

        file_size = 8 * self.iterations * 3 * n_runs / 1024 / 1024 + 1
        if self.display:
            print(f"Runs no: {n_runs}, estimated output size: {file_size:.2f}MB")

        pool = multiprocessing.Pool()

        for _ in tqdm.tqdm(
            pool.imap(self._run_param_set, enumerate(self.parameter_sets), chunksize=1),
            total=len(self.parameter_sets),
            disable=not self.display,
        ):
            pass

    def _save_description(self):
        parameter_sets = []

        for d in self.parameter_sets:
            d = deepcopy(d)
            infant_params = d.pop("infant_params")
            config = d.pop("config")

            parameter_sets.append({**infant_params.to_dict(), **config.to_dict(), **d})

        columns = parameter_sets[0].keys()

        data = [s.values() for s in parameter_sets]
        out_df = pd.DataFrame(data, columns=columns)

        out_path = os.path.join(self.output_dir, "description.csv")
        out_df.to_csv(out_path)

    def _run_param_set(self, param_set):
        index, param_set = param_set

        result = dict()

        for repetition in range(self.repeats):
            result[repetition] = self._single_run_param_set(
                param_set, index, repetition
            )

        save_partial(self.output_dir, index, result)

    def _single_run_param_set(self, param_set, index, repetition):
        model = InfantModel(**param_set)

        goal_dist = []
        infant_positions = []

        for _ in range(self.iterations):
            model.step()

            goal_dist.append(model.get_middle_dist())
            infant_positions.append(model.infant.pos.tolist())

        return {
            "index": index,
            "repetition": repetition,
            "iterations": self.iterations,
            "goal_dist": goal_dist,
            "infant_positions": infant_positions,
        }
