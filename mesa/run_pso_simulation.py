import numpy as np
import itertools
import time
import copy

from simulation import Simulation
from infant_abm.agents.infant import Params as InfantParams
from infant_abm.pso.infant_particle import InfantParticle


def get_model_param_sets(default_params):
    prec = np.linspace(0.2, 1, 2)
    exp = np.linspace(0, 1, 2)
    coord = np.linspace(0, 1, 2)

    params = []

    for param_set in itertools.product(*[prec, exp, coord, resp, rel]):
        p, e, c, rs, rl = param_set

        i_params = InfantParams(precision=p, coordination=c, exploration=e)

        params.append(
            {
                **default_params,
                **{"infant_params": i_params, "responsiveness": rs, "relevance": rl},
            }
        )

    return params


def get_particles(momentum, particle_number):
    return [
        InfantParticle(
            momentum=momentum,
            cognitive_r=np.random.random(),
            social_r=np.random.random(),
            params=InfantParams.random(),
        )
        for _ in range(particle_number)
    ]


if __name__ == "__main__":
    repeats = 17
    max_iter = 1500
    iterations = 8

    momentum = 0.6
    particle_number = 24 * 3
    average_steps = 500

    metric = {
        "metric": "infant_tps",
        "average_steps": average_steps,
    }

    max_function = np.nanmax
    max_arg_function = np.nanargmax

    output_path = "../results/pso/test_run_temp.hdf"

    # parent = {'responsiveness': 100, 'relevance': 0 }
    # 11 Fitness: 234.0 at Params(precision=0.84, coordination=1.0, exploration=0.28), 32.56 s

    # parent = {'responsiveness': 20, 'relevance': 20}
    parent = {"responsiveness": 80, "relevance": 20}
    # parent = {'responsiveness': 20, 'relevance': 80}
    # parent = {'responsiveness': 80, 'relevance': 80}

    # parent = {'responsiveness': 20, 'relevance': 85}
    # 11 Fitness: 468.0 at Params(precision=0.62, coordination=1.0, exploration=1.0), 29.80 s

    for resp, rel in [[20, 20], [20, 80], [80, 20], [80, 80]]:
        parent = {"responsiveness": resp, "relevance": rel}
        print(f"\n\nparent: {parent}\n")

        default_model_params = {
            **{"toy_count": 4},
            **parent,
        }

        particles = get_particles(momentum=momentum, particle_number=particle_number)

        parameter_sets = [
            {"infant_params": InfantParams.from_numpy(p.pos), **default_model_params}
            for p in particles
        ]
        simulation = Simulation(
            model_param_sets=parameter_sets,
            max_iterations=max_iter,
            repeats=repeats,
            output_path=output_path,
        )
        simulation.run()

        fitness = [result.fitness(**metric) for result in simulation.results]

        best_global_fitness = max_function(fitness)
        best_global_pos = particles[max_arg_function(fitness)].pos

        for particle, fitness in zip(particles, fitness):
            particle.set_best_fitness(fitness)

        print(f"Initial Fitness: {best_global_fitness:.4f} at {best_global_pos}")

        for i in range(iterations):
            start = time.time()

            for particle in particles:
                particle.move(best_global_pos)

            parameter_sets = [
                {
                    "infant_params": InfantParams.from_numpy(p.pos),
                    **default_model_params,
                }
                for p in particles
            ]
            simulation = Simulation(
                model_param_sets=parameter_sets,
                max_iterations=max_iter,
                repeats=repeats,
                output_path=output_path,
            )
            simulation.run()

            fitness = [result.fitness(**metric) for result in simulation.results]

            current_ev_fitness = max_function(fitness)
            if current_ev_fitness > best_global_fitness:
                best_global_fitness = current_ev_fitness
                best_global_pos = copy.copy(particles[max_arg_function(fitness)].pos)

            for particle, fitness in zip(particles, fitness):
                if fitness < particle.best_fitness:
                    particle.set_best_fitness(fitness)

            elapsed = time.time() - start
            disp_params = InfantParams.from_numpy(np.round(best_global_pos, 2))

            print(
                f"{i:02d} Fitness: {best_global_fitness:.4f} at {disp_params}, {elapsed:.2f} s"
            )
