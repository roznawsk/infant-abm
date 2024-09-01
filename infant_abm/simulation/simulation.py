import multiprocessing
import os
from pathlib import Path
import uuid
import pandas as pd
import tqdm
import logging

from abc import ABC, abstractmethod

from copy import deepcopy

from infant_abm.config import Config
from infant_abm.model import InfantModel
from infant_abm.agents.infant import Params as InfantParams
from infant_abm.agents import (
    NoVisionInfant,
    NoVisionParent,
    SpatialVisionInfant,
    SpatialVisionParent,
    AbstractVisionInfant,
    AbstractVisionParent,
    QLearnDetachedInfant,
    QLearnDetachedParent,
    QLearnPairedInfant,
    QLearnPairedParent,
)

from infant_abm.db_utils import partial_exists, save_partial


class Model_0_1_0:
    infant_class = NoVisionInfant
    parent_class = NoVisionParent
    output_dir = "v0.1.0"


class Model_0_1_1:
    infant_class = SpatialVisionInfant
    parent_class = SpatialVisionParent
    output_dir = "v0.1.1"


class Model_0_1_2:
    infant_class = AbstractVisionInfant
    parent_class = AbstractVisionParent
    output_dir = "v0.1.2"


class Model_0_2_0:
    infant_class = QLearnDetachedInfant
    parent_class = QLearnDetachedParent
    output_dir = "v0.2.0"


class Model_0_2_1:
    infant_class = QLearnPairedInfant
    parent_class = QLearnPairedParent
    output_dir = "v0.2.1"


class DataCollector(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def after_step(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class Simulation:
    def __init__(
        self,
        model,
        model_param_sets: list[dict],
        iterations: int,
        repeats: int,
        datacollector: DataCollector,
        run_name=None,
        output_dir="results",
        display=False,
        processes=None,
    ):
        self.model = model
        self.parameter_sets: dict = model_param_sets
        self.iterations: int = iterations
        self.repeats: int = repeats
        self.datacollector: DataCollector = datacollector
        self.display = display
        self.processes = processes
        self.base_dir = output_dir

        if run_name is None:
            run_name = str(uuid.uuid4())[:7]
        self.run_name = run_name

        self._validate_output_path()
        self.output_dir = self._get_results_dir()
        Path(self.output_dir).mkdir(parents=True, exist_ok=False)
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
        model,
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
                    "infant_class": model.infant_class,
                    "parent_class": model.parent_class,
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

    def _validate_output_path(self):
        if not os.path.exists(self.base_dir) or not os.path.isdir(self.base_dir):
            raise ValueError("Output path must point to an existing directory")

        results_dir = self._get_results_dir()

        if os.path.exists(results_dir):
            raise ValueError("Output path already exists")

    def _get_results_dir(self):
        return os.path.join(self.base_dir, self.model.output_dir, self.run_name)

    def _maybe_save_description(self):
        desc_path = os.path.join(self.output_dir, "description.csv")
        if os.path.exists(desc_path):
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
        out_df.to_csv(desc_path)

    def _run_param_set(self, param_set):
        index, param_set = param_set

        if partial_exists(self.output_dir, index):
            return

        result = dict()

        for repetition in range(self.repeats):
            result[str(repetition)] = self._single_run_param_set(
                param_set, index, repetition
            )

        save_partial(self.output_dir, index, result)

    def _single_run_param_set(self, param_set, index, repetition):
        model = InfantModel(
            infant_class=self.model.infant_class,
            parent_class=self.model.parent_class,
            **param_set,
        )

        collector = self.datacollector(model)

        for _ in range(self.iterations):
            model.step()

            if not collector.after_step():
                break

        return {
            "iterations": self.iterations,
            "index": index,
            "repetition": repetition,
            **collector.to_dict(),
        }
