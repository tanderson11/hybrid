import unittest
import os
import numpy as np
import pandas as pd
import pathlib
from typing import NamedTuple

from tests.discover import discover_tests
from tests.filesystem_test import FilesystemTestMeta, TEvalTest

sbml_tests = discover_tests(os.path.dirname(__file__), 'sbml-*', include_check=True, simulators_share_checks=True)

class SBMLCollection(FilesystemTestMeta):
    test_collection = sbml_tests

class ZScoreTest(TEvalTest):
    t_eval = None
    do_yscores = True

    class ResultsAndCheck(NamedTuple):
        results_df: pd.DataFrame
        check_df: pd.DataFrame
        z_scores_for_mean_by_species: pd.DataFrame
        y_scores_for_std_by_species: pd.DataFrame = None

    def align_results_factory(self, time, targets, desired_species):
        def align_single_result(r):
            t_history, y_history = r.restricted_values(time)

            indexed_results = pd.DataFrame({'time':t_history})
            for species, target_index in zip(desired_species, targets):
                indexed_results[species] = y_history[target_index, :]
            indexed_results.set_index('time')

            return indexed_results

        return align_single_result

    def tearDown(self):
        # save results
        results_table = self.test_result.results_df
        z_ts = self.test_result.z_scores_for_mean_by_species
        out = os.path.join(self.test_out, self.test_name)
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        results_table.to_csv(os.path.join(out, f'n={self.n}_simulation_results.csv'))
        z_ts.to_csv(os.path.join(out, f'n={self.n}_simulation_zscores.csv'))

        if self.do_yscores:
            y_ts = self.test_result.y_scores_for_std_by_species
            y_ts.to_csv(os.path.join(out, f'n={self.n}_simulation_yscores.csv'))

    def _test_single(self):
        desired_species = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_species = [s.name for s in self.specification.model.species]
        targets = [all_species.index(s) for s in desired_species]
        align_results = self.align_results_factory(self.check_data['time'], targets, desired_species)

        results = self.run_simulations(end_routine=align_results)
        for df in results:
            df.set_index('time', inplace=True)
        df = pd.concat(results, axis=1)
        all_results = pd.concat([df.T.groupby(by=df.columns).mean().T, df.T.groupby(by=df.columns).std().T], axis=1)

        check_targets = set([c.split('-')[0] for c in self.check_data.columns if len(c.split('-')) > 1])
        all_results.columns = [c + '-mean' if i < len(check_targets) else c + '-sd' for i,c in enumerate(all_results.columns)]

        z_ts = self.z_score_for_mean(all_results, check_targets, self.check_data, self.n)
        y_ts = None
        if self.do_yscores:
            y_ts = self.y_score_for_std(all_results, check_targets, self.check_data, self.n)

        self.test_result = self.ResultsAndCheck(all_results, self.check_data, z_ts, y_ts)

    @staticmethod
    def z_score_for_mean(all_results, target_species, check_data, n):
        # https://github.com/sbmlteam/sbml-test-suite/blob/release/cases/stochastic/DSMTS-userguide-31v2.pdf
        z_ts = {}
        for species in target_species:
            z_t = (all_results[f'{species}-mean'] - check_data[f'{species}-mean'])/(check_data[f'{species}-sd']) * np.sqrt(n)
            z_ts[species] = z_t

        z_ts = pd.DataFrame(z_ts)

        return z_ts

    @staticmethod
    def y_score_for_std(all_results, target_species, check_data, n):
        y_ts = {}
        for species in target_species:
            y_t = (all_results[f'{species}-sd']**2/check_data[f'{species}-sd']**2 - 1) * np.sqrt(n/2)
            y_ts[species] = y_t

        y_ts = pd.DataFrame(y_ts)

        return y_ts

class TestSBML(ZScoreTest, metaclass=SBMLCollection):
    t_eval = np.linspace(0.0, 50.0, 51)

if __name__ == '__main__':
    unittest.main()