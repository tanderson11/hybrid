import tests.sbml.sbml_tests as sbml
import tests.hybrid.run_hybrid as run_hybrid

suites = {
    'sbml': sbml.TestSBML,
    'decayingisom': run_hybrid.TestDecayingIsomerization,
    'schlogl': run_hybrid.TestSchlogl,
}