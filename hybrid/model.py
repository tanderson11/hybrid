from dataclasses import dataclass
import numpy as np
from reactionmodel.model import Species, Reaction, Model, ReactionRateFamily

@dataclass(frozen=True)
class SimulationAwareReaction(Reaction):
    poisson_products: bool = False
    assume_quasi_equilibrium: bool = False

    def __post_init__(self):
        assert isinstance(self.poisson_products, bool)
        assert isinstance(self.assume_quasi_equilibrium, bool)
        return super().__post_init__()

class SimulationAwareModel(Model):
    def equilibrium_mask(self):
        equilibrium_mask = np.array([r.assume_quasi_equilibrium for r in self.all_reactions])
        return equilibrium_mask
    
    def poisson_products_mask(self):
        poisson_products_mask = np.array([r.poisson_products for r in self.all_reactions])
        return poisson_products_mask

    @classmethod
    def from_dict(cls, dictionary, functions_by_name=None, species_class=Species, reaction_class=SimulationAwareReaction, reaction_rate_family_class=ReactionRateFamily):
        return super().from_dict(dictionary, functions_by_name, species_class=species_class, reaction_class=reaction_class, reaction_rate_family_class=reaction_rate_family_class)