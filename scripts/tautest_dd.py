from reactionmodel.model import Species, Reaction, Model
import numpy as np

S1 = Species('S1')
S2 = Species('S2')
S3 = Species('S3')

reactions = []
reactions.append(Reaction([S1], [], k='c1'))
reactions.append(Reaction([(S1, 2)], [S2], k='c2'))
reactions.append(Reaction([S2], [(S1, 2)],  k='c3'))
reactions.append(Reaction([S2], [S3],  k='c4'))

m = Model([S1, S2, S3], reactions)

rathinam_parameters = {
    'c1': 1,
    'c2': 10,
    'c3': 1000,
    'c4': 0.1,
}

rathinam_initial = {
    'S1': 400,
    'S2': 798,
    'S3': 0
}

from hybrid.tau import TauLeapSimulator

t_s = TauLeapSimulator(
    m.get_k(parameters=rathinam_parameters, jit=True),
    m.stoichiometry(),
    m.kinetic_order(),
)

result = t_s.simulate(
    [0, 0.1],
    m.make_initial_condition(rathinam_initial, parameters=rathinam_parameters),
    np.random.default_rng()
)

result.plot(m.legend())
import matplotlib.pyplot as plt
plt.show()