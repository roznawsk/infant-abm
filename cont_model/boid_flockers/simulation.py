import mesa

import itertools
import numpy as np
import time
from matplotlib import pyplot as plt


from model import ToddlerModel
from SimpleContinuousModule import SimpleCanvas
from agents.toddler import Toddler
from agents.parent import Parent
from agents.toy import Toy


# model_params = {
#     "title": mesa.visualization.StaticText(f'Grid size: {grid_size}'),
#     "width": grid_size,
#     "height": grid_size,
#     "speed": 2,
#     "lego_count": mesa.visualization.Slider("Brick count", 5, 1, 15),
#     "precision": mesa.visualization.Slider("Toddler Precision", 50, 0, 100),
#     "exploration": mesa.visualization.Slider("Toddler Exploration", 50, 0, 100),
#     "coordination": mesa.visualization.Slider("Toddler Coordination", 50, 0, 100),
#     "responsiveness": mesa.visualization.Slider("Parent responsiveness", 50, 0, 100),
#     "relevance": mesa.visualization.Slider("Parent relevance", 50, 0, 100)
# }


if __name__ == '__main__':
    grid_size = 300
    success_dist = 40
    max_iterations = 1000
    repeats = 100

    default_params = {
        'lego': [4],
        'precision': [50],
        'exploration': [50],
        'coordination': [60],
        'responsiveness': [75],
        'relevance': [70]
    }

    for current_param in ['precision', 'coordination', 'responsiveness', 'relevance']:
        # for current_param in ['coordination']:
        params = dict(default_params)

        x = np.arange(0, 101, 10)
        params[current_param] = x

        params = list(params.values())
        print(params)

        r_steps = []
        r_parent = []
        r_toddler = []

        for param_set in itertools.product(*params):
            lego, prec, exp, coord, resp, rele = param_set

            params_results = {'steps': [], 'parent': [], 'toddler': []}

            start = time.time()

            for _ in range(repeats):
                model = ToddlerModel(
                    width=grid_size,
                    height=grid_size,
                    speed=2,
                    lego_count=lego,
                    exploration=exp,
                    precision=prec,
                    coordination=coord,
                    responsiveness=resp,
                    relevance=rele,
                    success_dist=success_dist
                )

                run_steps = None

                for i in range(max_iterations):
                    if run_steps is None and model.success():
                        run_steps = i

                    model.step()

                if run_steps is None:
                    run_steps = max_iterations

                # print(run_steps)

                params_results['steps'].append(run_steps)
                params_results['parent'].append(model.get_parent_satisfaction())
                params_results['toddler'].append(model.get_toddler_satisfaction())

            print('t = {:2f} ms'.format((time.time() - start) * 1000))

            r_steps.append(np.average(params_results['steps']))
            r_parent.append(np.average(params_results['parent']))
            r_toddler.append(np.average(params_results['toddler']))

        fig, ax1 = plt.subplots()

        ax1.plot(x, r_steps, linestyle='dashed', marker='s', color='r')
        ax1.set_xlabel(current_param)
        ax1.set_ylim(bottom=0, top=1000)
        ax1.set_ylabel('steps to goal', color='r')

        ax2 = ax1.twinx()
        ax2.plot(x, r_parent, linestyle='dashed', marker='s', color='b')
        ax2.set_ylabel('satisfaction')

        ax2.plot(x, r_toddler, linestyle='dashed', marker='s', color='orange')
        ax2.legend(['parent', 'toddler'])
        ax2.set_ylim(bottom=0)

        fig.tight_layout()
        plt.savefig(f'../../plots/big_{current_param}.png', dpi=300)
        # plt.show()
