import yaml
import json
from dataclasses import dataclass, asdict

from hybrid.gillespie import GillespieSimulator, GillespieOptions
from hybrid.hybrid import HybridSimulator, HybridSimulationOptions, PartitionScheme
from hybrid.tau import TauLeapSimulator, TauLeapOptions

@dataclass
class SimulatorFactory():
    simulator = ''
    simulator_klass = None

    version: str

    def to_dict(self):
        d = asdict(self)
        d['simulator'] = self.simulator
        return d

    @classmethod
    def from_dict(cls, dict):
        assert dict.pop('simulator') == cls.simulator
        return cls(**dict)

    @classmethod
    def load_dictionary(cls, file, format='yaml'):
        if format=='yaml':
            import yaml
            with open(file, 'r') as f:
                d = yaml.load(f, Loader=yaml.SafeLoader)
        elif format=='json':
            import json
            with open(file, 'r') as f:
                d = json.load(f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")
        
        return d

    @classmethod
    def load(cls, file, format='yaml'):
        d = cls.load_dictionary(file, format=format)
        return cls.from_dict(d)
        
    def save(self, file, format='yaml'):
        if format=='yaml':
            with open(file, 'w') as f:
                yaml.dump(self.to_dict(), f, Dumper=yaml.SafeDumper, sort_keys=False)
        elif format=='json':
            with open(file, 'w') as f:
                json.dump(self.to_dict(), f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")

    def make_simulator(self, *args, **kwargs):
        self_dict = self.to_dict()
        self_dict.pop('version')
        self_dict.pop('description')
        options = self_dict.pop('options')
        self.simulator_klass(*args, **kwargs, **self_dict, **options)

@dataclass
class GillespieSimulatorFactory(SimulatorFactory):
    simulator = 'gillespie'
    simulator_klass = GillespieSimulator

    options: GillespieOptions
    description: str = ""

@dataclass
class TauLeapSimulatorFactory(SimulatorFactory):
    simulator = 'tau'
    simulator_klass = TauLeapSimulator

    options: TauLeapOptions
    description: str = ""

@dataclass
class HybridSimulatorFactory(SimulatorFactory):
    simulator = 'haseltinerawlings'
    simulator_klass = HybridSimulator

    options: HybridSimulationOptions
    partition_scheme: PartitionScheme
    description: str = ""