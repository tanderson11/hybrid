import sys
import unittest
import os
import subprocess
import shlex

import unit_tests

if __name__ == '__main__':
    assert len(sys.argv) <= 2, "Usage: 'python submit_sbml.py maxjobs' where maxjobs == maximum jobs to run at once"
    if len(sys.argv) == 2:
        maxjobs = int(sys.argv[1])
    else:
        maxjobs = 16
    ntests = len(list(unittest.TestLoader().loadTestsFromTestCase(unit_tests.TestSBML)))
    subprocess.run(shlex.split(f'sbatch -o slurm-%A_%a.out --array=0-{ntests-1}%{maxjobs} submit_sbml.sh'))
