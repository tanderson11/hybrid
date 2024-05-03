import os
import yaml
import json
from dataclasses import dataclass, asdict

from reactionmodel.parser import ConfigParser

from hybrid.gillespie import GillespieSimulator, GillespieOptions
from hybrid.hybrid import HybridSimulator, HybridSimulationOptions, PartitionScheme, FixedThresholdPartitioner, NThresholdPartitioner
from hybrid.tau import TauLeapSimulator, TauLeapOptions

@dataclass
class SimulatorFactory(ConfigParser):
    simulator = ''
    simulator_klass = None

    version: str

    def to_dict(self):
        d = asdict(self)
        return self.tune_dictionary(d)

    def tune_dictionary(self, selfdictionary):
        selfdictionary['simulator'] = self.simulator
        return selfdictionary

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

        handled_d = {}
        for k,v in d.items():
            handled_d[k] = cls.handle_field(k, v)
        return cls.from_dict(handled_d)
        
    def save(self, file, format='yaml'):
        if format=='yaml':
            with open(file, 'w') as f:
                yaml.dump(self.to_dict(), f, Dumper=yaml.SafeDumper, sort_keys=False)
        elif format=='json':
            with open(file, 'w') as f:
                json.dump(self.to_dict(), f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")

    @classmethod
    def handle_field(cls, key, value):
        return value

    def make_simulator(self, *args, **kwargs):
        self_dict = self.to_dict()
        self_dict.pop('simulator')
        self_dict.pop('version')
        self_dict.pop('description')
        options = self_dict.pop('options')

        return self.simulator_klass(*args, **kwargs, **self_dict, **options)

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

schemes = [FixedThresholdPartitioner, NThresholdPartitioner]

SCHEMES_BY_NAME = {
    x.__name__: x for x in schemes
}

@dataclass
class HybridSimulatorFactory(SimulatorFactory):
    simulator = 'haseltinerawlings'
    simulator_klass = HybridSimulator

    options: HybridSimulationOptions
    partition_scheme: PartitionScheme
    description: str = ""

    @classmethod
    def handle_field(cls, key, value):
        if key != 'partition_scheme':
            return super().handle_field(key, value)
        
        scheme_type = value.pop('partition_scheme_type')
        return SCHEMES_BY_NAME[scheme_type](**value)
    
    def tune_dictionary(self, selfdictionary):
        selfdictionary['partition_scheme']['partition_scheme_type'] = type(self.partition_scheme).__name__
        return super().tune_dictionary(selfdictionary)

FACTORIES = {
    'gillespie': GillespieSimulatorFactory,
    'haseltinerawlings': HybridSimulatorFactory,
    'tau': TauLeapSimulatorFactory,
}

def load_simulator_factory(file, format='yaml'):
    d = SimulatorFactory.load_dictionary(file, format=format)
    factory_klass = FACTORIES[d['simulator']]
    return factory_klass.from_dict(d)

@dataclass
class SimulatorFactoryPathParser(ConfigParser):
    """Parses a simulator config that points to one of a subset of pre-defined simulators."""
    path: str
    format: str = 'yaml'

    @classmethod
    def from_dict(cls, config_dictionary):
        parser = cls(**config_dictionary)
        return load_simulator_factory(parser.path, format=parser.format)

@dataclass
class PreconfiguredSimulatorLoader(ConfigParser):
    preconfigured_root = os.path.join(os.path.dirname(__file__), 'simulators')
    format = 'yaml'

    name: str

    @classmethod
    def from_dict(cls, config_dictionary):
        parser = cls(**config_dictionary)
        return load_simulator_factory(os.path.join(cls.preconfigured_root, f'{parser.name}.{cls.format}'), format=cls.format)