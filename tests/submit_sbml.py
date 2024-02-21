import sys
import unittest
import os
import subprocess

import unit_tests

if __name__ == '__main__':
    assert len(sys.argv) <= 2, "Usage: 'python generate_submission_script maxjobs' where maxjobs == maximum jobs to run at once"
    if len(sys.argv) == 2:
        maxjobs = int(sys.argv[1])
    else:
        maxjobs = 16
    ntests = len(unittest.TestLoader().loadTestsFromTestCase(unit_tests.TestSBML))
    subprocess.run(['sbatch', '-o', 'slurm-%A_%a.out', f'--array=0-{ntests-1}%{maxjobs}' 'submit_sbml.sh'])