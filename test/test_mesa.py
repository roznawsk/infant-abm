from pathlib import Path
from infant_abm import Simulation, InfantParams


def run_basic_scenario(output_path):
    parameter_sets = [{"infant_params": InfantParams.new(0.5, 0.5, 0.5)}]
    iterations = 10000
    repeats = 2

    simulation = Simulation(
        model_param_sets=parameter_sets,
        iterations=iterations,
        repeats=repeats,
        output_path=output_path,
        # display=False,
    )

    simulation.run()
    simulation.save()

    assert len(simulation.results) == 1
    assert simulation.results[0].iterations == iterations
    assert simulation.results[0].repeats == repeats

    saved_parameter_set = simulation.results[0].parameter_set
    saved_parameter_set["infant_params"].reset()

    assert saved_parameter_set == parameter_sets[0]


def test_basic_simulation():
    output = Path("test/output")
    filename = output.joinpath("basic_scenario.hdf")

    Path(filename).unlink(missing_ok=True)
    Path.mkdir(output, exist_ok=True)

    run_basic_scenario(filename)

    assert Path.is_file(filename)

    file_size = Path.stat(filename).st_size
    assert 10**6 < file_size < 2 * 10**6

    Path.unlink(filename)
