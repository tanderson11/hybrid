from hybrid.tau import TauLeapSimulator
import numpy as np
from reactionmodel.model import Species, Reaction, Model
from dataclasses import dataclass

@dataclass
class MutantEmergenceParameters():
    d: float
    a: float
    l: float
    y0: float
    f_rep: float
    mu: float
    R_mu: float

    def __post_init__(self):
        print(f"Mutant generation rate: {self.a*self.f_rep*self.mu*self.y0:0.3f}")
        print(f"Mutant infection rate at eq: {self.a*self.R_mu}")
        print(f"Maximum implied s: {self.to_standard()['s']:0.3f}")

    def initial(self):
        initial = {
            'x' : self.l / self.d,
            'y_s': self.y0,
            'y_r': 0,
        }
        return initial

    def to_standard(self):
        initial = self.initial()
        standard = {
            'd' : self.d,
            'a' : self.a,
            'l' : self.l,
            'mu': self.mu
        }
        standard['b_r'] = self.a * self.R_mu / initial['x']
        # we manage to reduce our # of parameters by 1, because s * b_s always appear in combination
        standard['b_s'] = standard['b_r']
        s_b_s_product = self.a * self.f_rep / initial['x']
        standard['s'] = s_b_s_product / standard['b_r']
        standard['eta'] = self.y0 * self.a * (1 - self.f_rep)

        return standard

    def to_gillespie_compatible_parameters(self):
        pass

x  = Species('x')
ys = Species('y_s')
yr = Species('y_r')

reactions = []
reactions.append(Reaction([x], [], k='d'))
reactions.append(Reaction([], [x], k='l'))
reactions.append(Reaction([ys], [], k='a'))
reactions.append(Reaction([yr], [], k='a'))
reactions.append(Reaction([x, ys], [(ys, 2)], k='s*b_s'))
reactions.append(Reaction([x, yr], [(yr, 2)], k='b_r'))
reactions.append(Reaction([x, ys], [ys, yr], k='s*b_s*mu'))

m_v_steady_with_drug = Model([x, ys, yr], reactions)

reactions.append(Reaction([], [ys], k='eta'))

m_v_steady_with_drug_and_reactivation = Model([x, ys, yr], reactions)

m = m_v_steady_with_drug_and_reactivation
p = MutantEmergenceParameters(**{
    'd': 0.05,
    'a': 1.0,
    'l': 1e10,
    'mu': 1e-6,
    'R_mu': 1.5,
    'y0': 1e5,
    'f_rep': 0.5
})
params = p.to_standard()
initial = m.make_initial_condition(p.initial(), parameters=params)

s = TauLeapSimulator(
    m.get_k(parameters=params, jit=True),
    m.stoichiometry(),
    m.kinetic_order(),
    #species_creation_is_critical=True
)
t_eval = np.linspace(0, 120, 121)
result = s.simulate([0.0, 120.0], initial, np.random.default_rng(), t_eval=t_eval)