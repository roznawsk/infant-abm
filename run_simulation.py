from infant_abm.simulation import (
    v1Collector,  # noqa: F401
    v2Collector,  # noqa: F401
    Model_0_1_0,  # noqa: F401
    Model_0_1_1,  # noqa: F401
    Model_0_1_2,  # noqa: F401
    Model_0_2_1,  # noqa: F401
    run_comparative_boost_simulation,
)


if __name__ == "__main__":
    model = Model_0_2_1()
    collector = v2Collector

    grid = 8
    boost = 1
    repeats = 20
    iterations = 50_000
    run_name = "test_q_learnv2_50k"
    q_learn_params = [0.13, 0.5, 0.005]

    linspace = (0.05, 0.95, grid)
    boost_linspace = (0, 1, boost)

    run_comparative_boost_simulation(
        model=model,
        iterations=iterations,
        collector=collector,
        run_name=run_name,
        repeats=repeats,
        linspace=linspace,
        q_learn_params=q_learn_params,
    )
