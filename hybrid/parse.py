import os
import yaml
import json
from dataclasses import dataclass, asdict, fields

from reactionmodel.parser import ConfigParser

from hybrid.gillespie import GillespieSimulator, GillespieOptions
from hybrid.hybrid import HybridSimulator, HybridSimulationOptions, PartitionScheme, FixedThresholdPartitioner, NThresholdPartitioner
from hybrid.tau import TauLeapSimulator, TauLeapOptions

@dataclass
class SimulatorFactory(ConfigParser):
    simulator = ''
    simulator_klass = None

    version: str

    def to_dict(self, for_saving=False):
        if for_saving:
            d = asdict(self)
            return self.tune_dictionary(d)
        d = {field.name:getattr(self, field.name) for field in fields(self)}
        return d

    def tune_dictionary(self, selfdictionary):
        selfdictionary['simulator'] = self.simulator
        return selfdictionary

    @classmethod
    def from_dict(cls, dict):
        handled_d = {}
        for k,v in dict.items():
            _k, _v = cls.handle_field(k, v)
            handled_d[_k] = _v
        assert handled_d.pop('simulator') == cls.simulator
        return cls(**handled_d)

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
                yaml.dump(self.to_dict(for_saving=True), f, Dumper=yaml.SafeDumper, sort_keys=False)
        elif format=='json':
            with open(file, 'w') as f:
                json.dump(self.to_dict(for_saving=True), f)
        else:
            raise ValueError(f"format should be one of yaml or json was {format}")

    @classmethod
    def handle_field(cls, key, value):
        return key, value

    def make_simulator(self, *args, **kwargs):
        self_dict = self.to_dict()
        #self_dict.pop('simulator')
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
    partition_function: PartitionScheme
    description: str = ""

    @classmethod
    def handle_field(cls, key, value):
        if key != 'partition_function':
            return super().handle_field(key, value)
        
        scheme_type = value.pop('partition_function_type')
        return key, SCHEMES_BY_NAME[scheme_type](**value)
    
    def tune_dictionary(self, selfdictionary):
        selfdictionary['partition_function']['partition_function_type'] = type(self.partition_function).__name__
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

    @classmethod
    def load_preconfigured(cls, name):
        return load_simulator_factory(os.path.join(cls.preconfigured_root, f'{name}.{cls.format}'), format=cls.format)