from pathlib import Path
from infant_abm import Simulation, InfantParams
from infant_abm.config import Config

from infant_abm.agents import (
    SpatialVisionInfant,
    NoVisionInfant,
    AbstractVisionInfant,
    SpatialVisionParent,
    AbstractVisionParent,
    NoVisionParent,
)


def run_basic_scenario(
    infant_class, parent_class, infant_params: InfantParams, config=Config()
):
    output_dir = Path("test/output")
    filename = output_dir.joinpath("basic_scenario.hdf")

    Path(filename).unlink(missing_ok=True)
    Path.mkdir(output_dir, exist_ok=True)

    parameter_sets = [
        {
            "infant_class": infant_class,
            "parent_class": parent_class,
            "infant_params": infant_params,
            "config": config,
        }
    ]
    iterations = 10000
    repeats = 2

    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_dir=output_dir,
    )

    simulation.run()

    return simulation


def test_model_v0_1_0():
    simulation = run_basic_scenario(
        NoVisionInfant,
        NoVisionParent,
        infant_params=InfantParams.from_array([0, 0, 0]),
    )
    _output_dir = simulation.output_dir


def test_model_v0_1_1():
    simulation = run_basic_scenario(
        SpatialVisionInfant,
        SpatialVisionParent,
        infant_params=InfantParams.from_array([0, 0, 0]),
    )
    _output_dir = simulation.output_dir


def test_model_v0_1_2():
    simulation = run_basic_scenario(
        AbstractVisionInfant,
        AbstractVisionParent,
        infant_params=InfantParams.from_array([0, 0, 0]),
    )
    _output_dir = simulation.output_dir


def test_v0_1_1_changing_global_params():
    infant_class = SpatialVisionInfant
    parent_class = SpatialVisionParent
    infant_params = InfantParams.new(0.5, 0.05, 0.05)

    config = Config(persistence_boost_value=0.0)
    _simulation = run_basic_scenario(
        infant_class, parent_class, infant_params=infant_params, config=config
    )

    config = Config(persistence_boost_value=1.0, coordination_boost_value=1.0)
    _simulation_2 = run_basic_scenario(
        infant_class, parent_class, infant_params=infant_params, config=config
    )
