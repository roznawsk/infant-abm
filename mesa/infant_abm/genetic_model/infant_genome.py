import numpy as np


class InfantGenome:
    def __init__(self, precision=None, coordination=None, exploration=None):

        self.genome = {
            'precision': precision if precision is not None else np.random.uniform(0, 1),
            'coordination': coordination if coordination is not None else np.random.uniform(0, 1),
            'exploration': exploration if exploration is not None else np.random.uniform(0, 1)
        }

        self.gene_names = list(self.genome.keys())

        # Absolute value by which one of the genes can change during a mutation
        self.mutation_intensity = 0.15

    def crossover(self, other):
        offspring_genome = dict()

        for gene in self.gene_names:
            value = None
            if np.random.rand() < 0.5:
                value = getattr(self, gene)
            else:
                value = getattr(other, gene)

            offspring_genome[gene] = value

        return InfantGenome(**offspring_genome)

    def mutate(self):
        gene = np.random.choice(self.gene_names)[0]

        new_val = getattr(self, gene) + np.random.uniform(-1, 1) * self.mutation_intensity
        new_val = max(0, min(1, new_val))
        setattr(self, gene, new_val)

    def __getattr__(self, name: str):
        print('getting attr', name)

        if 'genome' not in self.__dict__:
            raise RuntimeError('InfantGenome has no attribute genome')

        if name in self.genome:
            return self.genome[name]
        else:
            raise AttributeError(f'InfantGenome has no attribute \'{name}\'')
