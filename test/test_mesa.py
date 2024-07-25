from pathlib import Path
from infant_abm import Simulation, InfantParams
from infant_abm.config import Config


def run_basic_scenario(infant_params: InfantParams = None, **base_params):
    output = Path("test/output")
    filename = output.joinpath("basic_scenario.hdf")

    Path(filename).unlink(missing_ok=True)
    Path.mkdir(output, exist_ok=True)

    if infant_params is None:
        infant_params = InfantParams.new(0.5, 0.5, 0.5)

    parameter_sets = [{**base_params, **{"infant_params": infant_params}}]
    iterations = 10000
    repeats = 2

    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_path=filename,
    )

    simulation.run()
    simulation.save()

    assert len(simulation.results) == 1
    assert simulation.results[0].iterations == iterations
    assert simulation.results[0].repeats == repeats

    saved_parameter_set = simulation.results[0].parameter_set
    saved_parameter_set["infant_params"].reset()

    assert saved_parameter_set == parameter_sets[0]
    assert Path.is_file(filename)

    return simulation


def test_basic_simulation():
    simulation = run_basic_scenario()
    filename = simulation.output_path

    assert Path.is_file(filename)
    file_size = Path.stat(filename).st_size
    assert 10**6 < file_size < 2 * 10**6

    Path.unlink(filename)


def test_changing_global_params():
    infant_params = InfantParams.new(0.5, 0.0, 0.0)

    config = Config(persistence_boost_value=0.0)
    simulation = run_basic_scenario(infant_params=infant_params, config=Config)

    _goal_1 = min(simulation.results[0].goal_dist)

    config = Config(persistence_boost_value=1.0, coordination_boost_value=1.0)
    simulation_2 = run_basic_scenario(infant_params=infant_params, config=config)

    _goal_2 = min(simulation_2.results[0].goal_dist)
