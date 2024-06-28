import unittest
import os
import numpy as np
import pandas as pd
import pathlib
import time

class FilesystemTestMeta(type):
    test_collection = None
    def __new__(mcs, names, bases, dct):
        def gen_test(test_name, specification, check_file):
            def test(self):
                self.test_name = test_name
                self.specification = self.apply_overrides(specification)
                self.check_file = check_file
                # load the csv of analytic/high quality simulation results
                if check_file is not None:
                    self.check_data = pd.read_csv(self.check_file)
                print("About to run test.")
                self._test_single()
            return test

        for root, spec_name, specification, check_file in mcs.test_collection:
            test_name = f'{os.path.basename(os.path.normpath(root))}_{spec_name}'
            dct[f'test_{test_name}'] = gen_test(test_name, specification, check_file)
        return type.__new__(mcs, names, bases, dct)

class TestSpec(unittest.TestCase):
    # wherever we are, save test output to test_output folder
    test_out = './test_output/'
    reaction_to_k = None
    n = 10000

    # subclasses must define _test_single()

    def apply_overrides(self, specification):
        return specification

    def run_simulations(self, end_routine):
        processed_results = []
        rng = np.random.default_rng()
        initial_condition = self.specification.model.make_initial_condition(self.specification.initial_condition)
        factory = self.specification.simulator_config

        simulator = factory.make_simulator_from_model(self.specification.model, reaction_to_k=self.reaction_to_k, parameters=self.specification.parameters)

        start = time.time()
        processed_results = simulator.run_simulations(self.n, self.specification.t.t_span, initial_condition, rng=rng, t_eval=self.specification.t.t_eval, end_routine=end_routine)
        end = time.time()
        self.elapsed_time = end-start

        return processed_results

    def tearDown(self):
        self.out = os.path.join(self.test_out, self.test_name)
        pathlib.Path(self.out).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.out, 'time.txt'), 'w') as f:
            f.write(str(self.elapsed_time))

class TEvalTest(TestSpec):
    t_eval = None

    def apply_overrides(self, specification):
        specification = super().apply_overrides(specification)
        specification.t.t_eval = self.t_eval
        return specification

class TrajectoryTest(TEvalTest):
    def end_routine(self, result):
        t_history, y_history = result.restricted_values(self.t_eval)
        legend = self.specification.model.legend()
        df = pd.DataFrame({'time': t_history})
        for i,s in enumerate(legend):
            df[s] = y_history[i, :]
        df = df.set_index('time')
        return df

    def _test_single(self):
        dfs = self.run_simulations(self.end_routine)
        for i,df in enumerate(dfs):
            df['trial'] = i
        df = pd.concat(dfs)
        self.df = df

    def tearDown(self):
        super().tearDown()
        # save results
        self.df.to_csv(os.path.join(self.out, f'n={self.n}_simulation_results.csv'))

class MeanTest(TrajectoryTest):
    def consolidate_data(self, dfs):
        for _df in dfs:
            _df.index = _df.index.round(4)
            _df = _df.loc[~_df.index.duplicated(keep='first')]

        df = pd.concat(dfs, axis=1)
        means = df.T.groupby(by=df.columns).mean().T
        means.columns = [c + '-mean' for c in means.columns]
        stds = df.T.groupby(by=df.columns).std().T
        stds.columns = [c + '-sd' for c in stds.columns]
        df = pd.concat([means, stds], axis=1)
        df = df.round(4)
        df.index = df.index.round(4)

        _df = pd.concat(dfs, axis=1)
        return df

    def _test_single(self, end_routine=None):
        end_routine = end_routine if end_routine is not None else self.end_routine
        dfs = self.run_simulations(self.end_routine)
        self.df = self.consolidate_data(dfs)
        return dfs

class EndpointTest(TestSpec):
    """A test of a configuration that relies only on the final y value."""
    def end_routine(self, result):
        return self.specification.model.y_to_dict(result.y)

    def _test_single(self):
        results = self.run_simulations(self.end_routine)
        df = pd.DataFrame(results)
        self.df = df

    def tearDown(self):
        super().tearDown()
        # save results
        self.df.to_csv(os.path.join(self.out, f'n={self.n}_tend_results.csv'))