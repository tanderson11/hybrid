# Overview

The value of a simulator is determined by three features:

1. Correctness of implementation
2. Suitability of approach for problem
3. Speed / accuracy relationship

A simulator that is not correctly implemented (maybe due to a programming or mathematical error) has no value. But a correct simulator may still be unsuitable for a particular problem. For instance, a perfect implementation of Gillespie's SSA is intolerably slow when simulating chemical species interacting near the thermodynamic limit. Alternatively, a hybrid SSA/ODE approach is unsuitable for simulating the limiting behavior of system that switches between two stable equilibria due to noise from reactions that the algorithm will treat as deterministic. If a simulator is correct and its approach is suitable for a given problem, then its value against other simulators is determined by its speed / accuracy trade-off. Well-honed approximate simulators have control parameters that can be tuned for greater accuracy or greater speed. The best algorithm for a problem is typically the one that produces results the fastest to within the desired error tolerance.

Corresponding to these three features of simulators, we can divide "testing" into three methods:

1. Testing for correctness
2. Analyzing suitability
3. Measuring performance

# Testing

## Unit tests

## SBML discrete model test suite

The Systems Biology Markup Language (SBML) has been a popular standard for specifying biological networks since its development in 2003. In 2008, developers on the team produced a group of test scenarios used to benchmark the correctness of an exact or approximate stochastic simulation algorithm. This group of test cases, called the SBML discrete model test suite, specifies variations of three basic models — the birth-death process, the reversible dimerization process, and the batch immigration-death process — where analytic results are known for the first two moments of the time-evolving populations. To use the test suite, algorithm developers realize many trajectories of each model and compare the means and standard deviations of those realizations to the analytically expected results (with a suggestion to test the z-scores for the mean and to use an approximate statistical test for the standard deviation).

Many of the test cases provided in the suite check that the markup language has been correctly implemented rather than testing the behavior of stochastic simulation algorithm. For instance, tests 01-08 through 01-11 exclusively test that the `Cell` compartment command and `hasOnlySubstanceUnits` tag are properly parsed. Of the thirty-nine test cases presented in the latest version of the discrete model test suite at the time of writing (version 3.4.0), we selected the eleven test cases that test the behavior of the stochastic simulation algorithm and implemented them in our modeling and simulation framework.

The SBML discrete model test suite is primarily intended for use with an exact stochastic simulation algorithm. Developers of approximate algorithms are invited to plot the value of the means and standard deviations as percentages of their true values.

The discrete model test suite is not a comprehensive test of the correctness of a stochastic simulation algorithm. First, no tests in the suite probe the simulator’s treatment of a rate law with explicit time dependence. (While some tests have time dependence in the form of “triggered events,” such as test 002-09 where the population of X is set to 50 at time t=25, these events are either parsed and executed correctly or they are not. They do not affect the behavior of the dynamical system.) This limits the test of Gillespie’s SSA to the time homogenous case. Second, in the case of approximate algorithms, hard test problems are not implemented. For instance, the reversible dimerization problem tests the basic correctness of fluctuations in two interchanging species, but it would be the reversible dimerization with decay problem to pit the algorithm against a dynamically stiff system. Third, most hybrid algorithms consciously sacrifice the second and higher moments of abundant populations for the sake of efficiency. Therefore, using the discrete model test suite as recommended may not be best way to test the behavior of hybrid algorithms. 

## Nitty-gritty toy problems

Hybrid stochastic simulation is a territory full of land mines: subtle implementation problems that can bias or invalidate results. Where we have found and solved these problems, we have also developed tests to ensure that simulator behavior never regresses and reintroduces these easy-to-miss errors.

### Biasing shortfalls/windfalls

*[Applies to algorithms with deterministic non-integer updates (ODE partitioning algorithms).]*

If we use an approximating scheme that allows real number updates to populations, then we will also need to round those populations periodically (typically after every update but at the very least whenever a population becomes small and representing it as a real number becomes un-physical). As noted by Vasudeva and Bhalla in their survey of hybrid algorithms (2004), conventional rounding can produce biased outcomes if there is a consistent shortfall/windfall in a population's derivative. For example, suppose we have a system with one population, $x$, where

0. $x$ is initialized at $x_0$ copies
1. $x$ is updated and rounded at intervals of $0.1$ seconds
2. $dx/dt$ is $4$ copies / second.

In such a system, every $0.1 \text{seconds}$ $x$ is updated $x = x_0 + dx/dt \cdot 0.1 = x_0 + 0.4$ but then rounded to $\text{round}(x_0 + 0.4) = x_0$. Due to the persistent "shortfall" of $x$, its value never changes despite its positive derivative. In partitioning algorithms, this behavior sometimes manifests as a species whose population "clings" to a critical threshold where a relevant reaction switches between the fast and the slow sets.

Such behavior is obviously undesirable. The recommended solution is to round *randomly* in proportion to the decimal.

We implement a test case that is likely to produce undesirable threshold clinging to verify that our simulator avoids this bias.

### Unrealizable rounding

*[Applies to algorithms with stochastic non-integer updates (CLE partitioning algorithms, implicit tau-leaping).]*

When species quantities are rounded, care needs to be take that the rounding does not move the system to a stoichiometrically unreachable state.

In the isomerization of X and Y

```math
\text{X} \leftrightarrow \text{Y}
```

the sum $n_{\text{tot}} = n_\text{X} + n_\text{Y}$ is conserved. But when populations are rounded (particularly when they are rounded *randomly*, see the previous section), it may be possible to break the lockstep relationship between the creation of X and the destruction of Y (or vice versa). If so, then $n_{\text{tot}}$ will seem to random walk and mass will not be conserved.

As Rathinam et al. suggest in their work on implicit tau-leaping algorithms (2003), to prevent this unwanted behavior, rounding should be done in a "stoichiometrically realizable" fashion: that is, we should round populations to states that are reachable by integer progress of the available reactions. This is equivalent to rounding the firings of reaction pathways before calculating population updates (ie before calculating the matrix multiplication of the stoichiometry matrix and the reaction firings) rather than rounding the population update itself.

We implement a reversible isomerization system and check that $n_{\text{tot}}$ does not vary as the simulation proceeds in order to verify that our simulator conserves mass.

**NOTE TK: this should also apply to the transient of ODE/SSA partitioning. Implement a test!**

### Partition breaking

*[Applies to partitioning algorithms.]*

In a partitioned system, forward simulation may eventually "break" the partition (cause a rate or population to cross the critical threshold). If the partition is not redrawn after it breaks, then continued forward simulation will no longer be a valid approximation of the system's dynamics. Consider, for example, the death process unfolding with first-order rate constant $k$

```math
\text{X} \to \emptyset
```

initialized with large $x_0$. A natural question to ask of the death process is: what is the distribution of times until extinction? A partitioning approach will begin by treating the death of $\text{X}$ as a "fast" process, potentially using differential equations. As $\text{X}$ exponentially decays, it will eventually cross a critical threshold below which its behavior should be treated as a "slow" process using the SSA. If the algorithm fails to notice $\text{X}$ crossing the threshold, then the decline of $\text{X}$ will be treated deterministically forever and the time to extinction for $\text{X}$ will have zero variance.

In their original hybrid partitioning algorithm, Haseltine and Rawlings (2002) propose introducing a contrived rate of no reaction: a fictitious process occurring with finite rate, which will interrupt simulation periodically and ensure no timestep becomes too large. While this will not stop integration at exactly the point that the partition breaks, if the contrived rate is large enough, simulation will be stopped "not long" afterwards. An alternative approach that we prefer is to use numerical integration routines with event finding methods that can detect when the partition breaks and halt integration.

We implement a death process that begins with an abundant population and check that the variance in extinction time matches the analytic result to determine if a simulator is a victim of this problem.

# Citations

SBML
1. Hucka, M., A. Finney, H. M. Sauro, H. Bolouri, J. C. Doyle, H. Kitano, A. P. Arkin, et al. 2003. “The Systems Biology Markup Language (SBML): A Medium for Representation and Exchange of Biochemical Network Models.” Bioinformatics 19 (4): 524–31. https://doi.org/10.1093/bioinformatics/btg015.
2. Evans, Thomas W., Colin S. Gillespie, and Darren J. Wilkinson. 2008. “The SBML Discrete Stochastic Models Test Suite.” Bioinformatics 24 (2): 285–86. https://doi.org/10.1093/bioinformatics/btm566.

Shortfall problem
1. Vasudeva, Karan, and Upinder S. Bhalla. 2004. “Adaptive Stochastic-Deterministic Chemical Kinetic Simulations.” Bioinformatics 20 (1): 78–84. https://doi.org/10.1093/bioinformatics/btg376.

Stoichiometrically realizable rounding
1. Rathinam, Muruhan, Linda R. Petzold, Yang Cao, and Daniel T. Gillespie. 2003. “Stiffness in Stochastic Chemically Reacting Systems: The Implicit Tau-Leaping Method.” The Journal of Chemical Physics 119 (24): 12784–94. https://doi.org/10.1063/1.1627296.

Contrived rate of no reaction
1. Haseltine, Eric L., and James B. Rawlings. 2002. “Approximate Simulation of Coupled Fast and Slow Reactions for Stochastic Chemical Kinetics.” The Journal of Chemical Physics 117 (15): 6959–69. https://doi.org/10.1063/1.1505860.

