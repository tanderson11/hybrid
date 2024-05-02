import unittest
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pathlib

from hybrid.simulate import SIMULATORS

@dataclass
class SimulatorArguments():
    t_span: tuple
    t_eval: tuple

class FilesystemTestMeta(type):
    test_collection = None
    def __new__(mcs, names, bases, dct):
        def gen_test(test_name, specification, check_file):
            def test(self):
                self.test_name = test_name
                self.specification = specification
                self.check_file = check_file
                # load the csv of analytic/high quality simulation results
                self.check_data = pd.read_csv(self.check_file)
                print("About to run test.")
                self._test_single()
            return test

        for root, spec_name, specification, check_file in mcs.test_collection:
            test_name = f'{os.path.basename(os.path.normpath(root))}_{spec_name}'
            dct[f'test_{test_name}'] = gen_test(test_name, specification, check_file)
        return type.__new__(mcs, names, bases, dct)

class TestSpec(unittest.TestCase):
    TEST_ARGUMENTS = SimulatorArguments((0.0, 50.0), np.linspace(0, 50, 51))
    n = 2

    # wherever we are, save test output to test_output folder
    test_out = './test_output/'

    # must define _test_single()

    def run_simulations(self, end_routine):
        processed_results = []
        rng = np.random.default_rng()
        initial_condition = self.specification.model.make_initial_condition(self.specification.initial_condition)
        simulator_config = self.specification.simulator_config.copy()
        simulator_class = simulator_config.pop('simulator')
        simulator_class = SIMULATORS[simulator_class]

        k = self.specification.model.get_k(parameters=self.specification.parameters, jit=True)
        print(simulator_config)
        simulator = simulator_class(k, self.specification.model.stoichiometry(), self.specification.model.kinetic_order(), **simulator_config)

        processed_results = simulator.run_simulations(self.n, self.TEST_ARGUMENTS.t_span, initial_condition, rng=rng, t_eval=self.TEST_ARGUMENTS.t_eval, end_routine=end_routine)

        return processed_results

class EndpointTest(TestSpec):
    """A test of a configuration that relies only on the final y value."""
    def end_routine_factory(self):
        def end_routine(result):
            return self.specification.model.y_to_dict(result.y)
        return end_routine

    def _test_single(self):
        results = self.do_simulations(self.end_routine_factory())
        df = pd.DataFrame(results)
        self.df = df

    def tearDown(self):
        # save results
        out = os.path.join(self.test_out, self.test_name)
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        self.df.to_csv(os.path.join(out, f'n={self.n}_tend_results.csv'))