import numpy as np
from enum import Enum
from typing import NamedTuple
from types import FunctionType as function

class Species():
    def __init__(self, name, abbreviation) -> None:
        self.name = name
        self.abbreviation = abbreviation
    
    def __hash__(self) -> int:
        return hash(self.name)

class MultiplicityType(Enum):
    reacants = 'reactants'
    products = 'products'
    stoichiometry = 'stoichiometry'
    rate_involvement = 'rate involvement'

class Reaction():
    def __init__(self, description, reactants, products, rate_involvement=None, k=None, reversible=False) -> None:
        assert reversible == False
        self.description = description

        self.k = k

        self.reactants = set([(r[0] if isinstance(r, tuple)  else r) for r in reactants])
        self.products = set([(p[0] if isinstance(p, tuple)  else p) for p in products])
        self.reactant_data = reactants
        self.product_data  = products

        self.rate_involvement = self.reactants if rate_involvement is None else rate_involvement

    def multiplicities(self, mult_type):
        multplicities = {}
        
        positive_multplicity_data = []
        negative_multiplicity_data = []
        if mult_type == MultiplicityType.reacants:
            positive_multplicity_data = self.reactant_data
        elif mult_type == MultiplicityType.products:
            positive_multplicity_data = self.product_data
        elif mult_type == MultiplicityType.stoichiometry:
            positive_multplicity_data = self.product_data
            negative_multiplicity_data = self.reactant_data
        elif mult_type == MultiplicityType.rate_involvement:
            positive_multplicity_data = self.rate_involvement
        else:
            raise ValueError(f"bad value for type of multiplicities to calculate: {mult_type}.")

        for species in negative_multiplicity_data:
            if isinstance(species, tuple):
                species, multiplicity = species
            else:
                multiplicity = 1
            multplicities[species] = -1 * multiplicity
        
        for species in positive_multplicity_data:
            if isinstance(species, tuple):
                species, multiplicity = species
            else:
                multiplicity = 1
            try:
                multplicities[species] += multiplicity
            except KeyError:
                multplicities[species] = multiplicity

        return multplicities

    def stoichiometry(self):
        return self.multiplicities(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicities(MultiplicityType.rate_involvement)

class RateConstantCluster(NamedTuple):
    k: function
    slice_bottom: int
    slice_top: int

class Model():
    def __init__(self, species: list[Species], reactions: list[Reaction]) -> None:
        self.species = species
        self.reactions = []
        for r in reactions:
            if isinstance(r, Reaction):
                self.reactions.append(r)
            elif isinstance(r, ReactionRateFamily):
                self.reactions.extend(r.reactions)
            else:
                raise TypeError(f"bad type for reaction in model {type(r)}. Expected Reaction or ReactionRateFamily")
        

        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)

        # ReactionRateFamilies allow us to calculate k(t) for a group of reactions all at once
        self.base_k = np.zeros(self.n_reactions)
        self.k_of_ts = []
        i = 0
        for r in self.reactions:
            if isinstance(r, Reaction):
                if isinstance(r.k, float):
                    self.base_k[i] = r.k
                assert(isinstance(r.k, function), "a reaction's rate constant should be a float or function with signature k(t) --> float")
                self.k_of_ts.append(RateConstantCluster(r.k, i, i+1))
                i+=1
                continue
            # reaction rate family
            self.k_of_ts.append(RateConstantCluster(r.k, i, i+len(r.reactions)+1))
            i += len(r.reactions)

        self.species_indices = {s:i for i,s in enumerate(self.species)}
        self.reaction_indices = {r:i for i,r in enumerate(self.reactions)}

    def multiplicity_matrix(self, mult_type):
        matrix = np.zeros((self.n_species, self.n_reactions))
        for column, reaction in enumerate(self.reactions):
            multiplicity_column = np.zeros(self.n_species)
            reaction_info = reaction.multiplicities(mult_type)
            for species, multiplicity in reaction_info.items():
                multiplicity_column[self.species_indices[species]] = multiplicity
            
            matrix[:,column] = multiplicity_column
        
        return matrix

    def k(self, t):
        k = self.base_k.copy()
        for family in self.k_of_ts:
            k[family.slice_bottom:family.slice_top] = family.k(t)
        return k

    def stoichiometry(self):
        return self.multiplicity_matrix(MultiplicityType.stoichiometry)

    def rate_involvement(self):
        return self.multiplicity_matrix(MultiplicityType.rate_involvement)

    @staticmethod
    def pad_equally_until(string, length, tie='left'):
        missing_length = length - len(string)
        if tie == 'left':
            return " " * int(np.ceil(missing_length/2)) + string + " " * int(np.floor(missing_length/2))
        return " " * int(np.floor(missing_length/2)) + string + " " * int(np.ceil(missing_length/2))

    def pretty_side(self, reaction, side, absentee_value):
        prior_species_flag = False
        pretty_side = ""
        reactant_multiplicities = reaction.multiplicities(side)
        for i,s in enumerate(self.species):
            mult = reactant_multiplicities.get(s, absentee_value)
            if i == 0:
                species_piece = ''
            else:
                species_piece = ' +' if prior_species_flag else '  '
            if mult is None:
                pretty_side += species_piece + " " * 5
                prior_species_flag = False
                continue
            prior_species_flag = True           
            species_piece += self.pad_equally_until(f"{str(int(mult)) if mult < 10 else '>9':.2}{s.abbreviation:.2}", 5)
            #print(f"piece: |{species_piece}|")
            pretty_side += species_piece
        return pretty_side

    def pretty(self, hide_absent=True) -> str:
        absentee_value = None if hide_absent else 0
        pretty = ""
        for reaction in self.reactions:
            pretty_reaction = f"{reaction.description:.22}" + " " * max(0, 22-len(reaction.description)) + ":"
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.reacants, absentee_value)
            pretty_reaction += ' --> '
            pretty_reaction += self.pretty_side(reaction, MultiplicityType.products, absentee_value)

            pretty += pretty_reaction + '\n'
        return pretty

class ReactionRateFamily():
    def __init__(self, reactions, k) -> None:
        self.reactions = reactions
        self.k = k
