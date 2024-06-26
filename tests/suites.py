import tests.sbml.sbml_tests as sbml
import tests.hybrid.run_hybrid as run_hybrid
import tests.time_dependent.run_time_dependent as run_time_dependent

suites = {
    'sbml': sbml.TestSBML,
    'decayingisom': run_hybrid.TestDecayingIsomerization,
    'schlogl': run_hybrid.TestSchlogl,
    'timedependent': run_time_dependent.TestTimeDependent,
    'laczlacy': run_hybrid.TestLacZLacY,
}