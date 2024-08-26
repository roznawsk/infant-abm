import multiprocessing
import os
import pandas as pd
import tqdm
import importlib
import logging

from copy import deepcopy

from infant_abm.config import Config
from infant_abm.model import InfantModel
from infant_abm.agents.infant import Params as InfantParams

from infant_abm.db_utils import partial_exists, save_partial

SUCCESS_DIST = 10


class Simulation:
    def __init__(
        self,
        model_param_sets: list[dict],
        iterations: int,
        repeats: int,
        output_dir=None,
        display=False,
        processes=None,
    ):
        self.parameter_sets: dict = model_param_sets
        self.iterations: int = iterations
        self.repeats: int = repeats
        self.display = display
        self.processes = processes

        if (
            output_dir is None
            or not os.path.exists(output_dir)
            or not os.path.isdir(output_dir)
        ):
            raise ValueError("Output path must point to an existing directory")

        self.output_dir = output_dir
        self._maybe_save_description()

    def run(self):
        n_runs = len(self.parameter_sets)

        file_size = 8 * self.iterations * 3 * n_runs / 1024 / 1024 + 1
        if self.display:
            print(f"Runs no: {n_runs}, estimated output size: {file_size:.2f}MB")

        pool = multiprocessing.Pool(processes=self.processes)

        for _ in tqdm.tqdm(
            pool.imap(self._run_param_set, enumerate(self.parameter_sets), chunksize=1),
            total=len(self.parameter_sets),
            disable=not self.display,
        ):
            pass

    @staticmethod
    def from_description(
        output_dir,
        iterations,
        repeats,
        display=False,
        processes=None,
    ):
        csv_path = os.path.join(output_dir, "description.csv")
        out_df = pd.read_csv(csv_path)

        parameter_sets = []
        for i, row in out_df.iterrows():
            infant_params = InfantParams.from_array(
                [row["perception"], row["persistence"], row["coordination"]]
            )

            config = Config(
                row["persistence_boost_value"], row["coordination_boost_value"]
            )

            parameter_sets.append(
                {
                    "infant_params": infant_params,
                    "config": config,
                    "infant_class": Simulation._resolve_class(row["infant_class"]),
                    "parent_class": Simulation._resolve_class(row["parent_class"]),
                }
            )

        return Simulation(
            model_param_sets=parameter_sets,
            iterations=iterations,
            repeats=repeats,
            output_dir=output_dir,
            display=display,
            processes=processes,
        )

    def _maybe_save_description(self):
        out_path = os.path.join(self.output_dir, "description.csv")
        if os.path.exists(out_path):
            logging.info("Skipping description, file exists")
            return

        parameter_sets = []

        for d in self.parameter_sets:
            d = deepcopy(d)
            infant_params = d.pop("infant_params")
            config = d.pop("config", Config())

            parameter_sets.append({**infant_params.to_dict(), **config.to_dict(), **d})

        columns = parameter_sets[0].keys()

        data = [s.values() for s in parameter_sets]
        out_df = pd.DataFrame(data, columns=columns)
        out_df.to_csv(out_path)

    def _run_param_set(self, param_set):
        index, param_set = param_set

        if partial_exists(self.output_dir, index):
            return

        result = dict()

        for repetition in range(self.repeats):
            result[repetition] = self._single_run_param_set(
                param_set, index, repetition
            )

        save_partial(self.output_dir, index, result)

    def _single_run_param_set(self, param_set, index, repetition):
        model = InfantModel(**param_set)

        goal_dist_iteration = None
        # infant_positions = []
        # boosts = []

        for iteration in range(self.iterations):
            model.step()

            if model.get_middle_dist() < SUCCESS_DIST:
                goal_dist_iteration = iteration
                break

            # goal_dist.append(model.get_middle_dist())
            # boosts.append("")
            # infant_positions.append(model.infant.pos.tolist())

            # action = model.infant.next_action

            # match action:
            #     case Crawl():
            #         if action.metadata == "persistence_boost":
            #             boosts[-1] = "persistence"
            #         elif action.metadata == "no_boost":
            #             boosts[-1] = "no_persistence"

            #     case InteractWithToy():
            #         if action.metadata == "coordination_boost":
            #             boosts[-1] = "coordination"
            #         else:
            #             boosts[-1] = "no_coordination"

        return {
            "index": index,
            "repetition": repetition,
            "iterations": self.iterations,
            "goal_dist": goal_dist_iteration,
            # "boosts": boosts,
        }

    def _resolve_class(name):
        raw_string = name.split("'")[1]

        module_name = ".".join(raw_string.split(".")[:-1])
        class_name = raw_string.split(".")[-1]

        module = importlib.import_module(module_name)
        return getattr(module, class_name)
