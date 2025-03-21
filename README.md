# Multi-scale simulation for chemical and biological processes

## Installation

For smaller projects:
1. Clone the repository: `git clone https://github.com/tanderson11/hybrid.git`
2. Use [Poetry](https://python-poetry.org) to install the package and its dependencies: `poetry install --with dev`

For larger projects:
1. Use [Poetry](https://python-poetry.org) to create a project:
```bash
cd project/
poetry init --python ">=3.9,<3.11"
```
2. Install hybrid as a project dependency by adding running the following commands:
```bash
poetry add git+https://github.com/tanderson11/hybrid.git
poetry update
```

## Explanation

For a simple interactive tutorial based on the SIR model in infectious disease epidemiology, visit [Tutorial.ipynb](notebooks/Tutorial.ipynb).

This package implements [tau-leaping](https://doi.org/10.1063/1.2745299) (originally described by Gillespie, *The Journal of Chemical Physics*, 2001) and [partioned-based forward simulation](https://pubs.aip.org/aip/jcp/article-abstract/117/15/6959/447100/Approximate-simulation-of-coupled-fast-and-slow) (Haseltine and Rawlings, *The Journal of Chemical Physics* in 2002).

For any system, the reaction pathways are defined by two unchanging matrices: `N`, the stoichiometry matrix (`N_ij` = net change in species `i` after unit progress in reaction `j`), and `O` the kinetic order matrix (`O_ij` = kinetic intensity (exponent) for species `i` in reaction `j`). The rate constants are defined by the vector `k`, the `j`th entry of which is the rate constant of reaction `j`.
Explicit time dependence is possible by specifying `k` as a function of time `t`.

## Simulators

This module supports three different methods of simulation: Gillespie simulation, tau-leaping, and Haseltine-Rawlings hybrid simulation. To access the simulator you want, import them from the appropriate module:

1. `from hybrid.hybrid import HybridSimulator`
2. `from hybrid.tau import TauLeapSimulator`
3. `from hybrid.gillespie import GillespieSimulator`

The recommended pattern for using this package is:
1. Import your desired simulator: `from hybrid.hybrid import HybridSimulator as Simulator` or `from hybrid.gillespie import GillespieSimulator as Simulator`.
2. Initialize with simulator by specifying your system:
```python
# given k rate constant vector, N stoichiometry matrix, and O kinetic order matrix
simulator = GillespieSimulator(
    k=k,
    N=N,
    kinetic_order=O,
)
```
3. Simulate
```python
# given t_span, the window between which to simulate, and y0 the initial condition
simulator.simulate(
    t_span=t_span,
    y0=y0,
    rng=np.random.default_rng()
)
```

To see the kinds of arguments involved, let's consider two examples.

### Gillespie simulation

First, suppose we want to simulate a simple birth death process for one species using Gillespie's algorithm. We can represent the system as

```math
\text{A} \to \emptyset
```
```math
\text{A} \to 2 \text{A}
```
and let us suppose that the deaths unfold at a rate of 1.1 deaths / person * time whereas the births unfold at a rate 1 birth / person * time. Then we could simulate this system as follows:

```python
import numpy as np
from hybrid.gillespie import GillespieSimulator

k = np.array([1.0, 1.1]) # 1 birth / person * time. 1.1 deaths / person * time.

N = np.array([
    [1, -1], # birth, death
])

kinetic_orders = np.array([
    [1, 1], # both birth and death are first order in the # of individuals
])

simulator = GillespieSimulator(k, N, kinetic_orders)

result = simulator.simulate(
    t_span = (0.0, 100.0),
    y0 = [10.0],
    rng = np.random.default_rng(),
)

plt.plot(result.t_history, result.y_history.T)
plt.title("Birth death process")
plt.legend(['A'])
plt.xlabel('time')
plt.ylabel('# A')
```
![One realization of the simple birth death process](notebooks/birthdeath.png)

### Hybrid simulation

Second, suppose we had two possible crystallizations of a reactant $\text{A}$:

```math
2 \text{A} \to \text{B}
```
and
```math
\text{A} + \text{C} \to \text{D}
```

where both reactions proceed with a rate constant $k = 10^{-7}$ but the initial quantity of $\text{A}$ is large: $10^{6}$ whereas the initial quantity of $\text{C}$ is small: $10$. In this case, the second reaction proceeds very slowly compared to the first, and may be better treated stochastically, but the first reaction proceeds very quickly compared to the second and would be prohibitively time-consuming to simulate as with Gillespie's Stochastic Simulation Algorithm (SSA). In that case, we need to employ a hybrid simulation technique:

```python
import numpy as np
from hybrid.hybrid import HybridSimulator, FixedThresholdPartitioner

k = np.array([1e-7, 1e-7])

# it's often more intuitive to work with N transpose
# so that rows correspond to reactions
N = np.array([ # A, B, C, D
    [-2, 1, 0, 0],
    [-1, 0, -1, 1],
]).T

# and V transpose for the same reason
kinetic_orders = np.array([
    [2, 0, 0, 0],
    [1, 0, 1, 0]
]).T


simulator = HybridSimulator(k, N, kinetic_orders, partition_function=FixedThresholdPartitioner(100.0))

result = simulator.simulate(
    t_span = (0.0, 100.0),
    y0 = [1e6, 0, 10, 0],
    rng = np.random.default_rng(),
)

print(result.y)

# array([9.09940e+04, 4.54501e+05, 1.00000e+00, 9.00000e+00])
# 9/10 C were converted to D in this realization
```

Note that in order to run a hybrid simulation, we had to specify a means of partitioning the system into fast and slow reactions. In this example, we used a `FixedThresholdPartitioner` to treat any reaction with propensity of 100 firings / unit time or less as stochastic. This is the scheme orginally proposed by Haseltine and Rawlings. For more information, read their [original paper](https://pubs.aip.org/aip/jcp/article-abstract/117/15/6959/447100/Approximate-simulation-of-coupled-fast-and-slow).

## Combining `hybrid` and `reactionmodel`

This package is designed to be compatible with the [`reactionmodel`](https://github.com/tanderson11/reactionmodel), a package used to specify systems of reactions. For example, if we revisit the simple birth death model, we could also build a `simulator` like so:

```python
from reactionmodel.model import Species, Reaction, Model
from hybrid.gillespie import GillespieSimulator

S = Species('S')

birth = Reaction([S], [(S, 2)], description='birth', k='l')
death = Reaction([S], [], description='death', k='mu')

m = Model([S], [birth, death])

parameters = {
    'l': 1.0,
    'mu': 1.01,
}

simulator = m.get_simulator(GillespieSimulator, parameters=parameters)
simulator.simulate(
    [0.0, 100.0],
    m.make_initial_condition({'S': 10.0}),
    rng = np.random.default_rng(),
)
```

If we revisit the crystallization model, here is how we could specify it with `reactionmodel`:

```python
from reactionmodel.model import Species, Reaction, Model
from hybrid.hybrid import HybridSimulator, FixedThresholdPartitioner

A = Species('A')
B = Species('B')
C = Species('C')
D = Species('D')

path1 = Reaction([(A, 2)], [B], k='k')
path2 = Reaction([A, C], [D], k='k')

m = Model([A,B,C,D], [path1, path2])

m.get_simulator(HybridSimulator, parameters={'k':1e-7}, partition_function=FixedThresholdPartitioner(100.0))

m.simulate(
    [0.0, 100.0],
    m.make_initial_condition({'A':1e6, 'C': 10})
    rng=np.random.defaultrng(),
)
```

## As a drop in replacement for `solve_ivp`

I recommend using the `Simulator` class and its subclasses to execute forward simulations: this avoids the overhead of creating a simulator every time you run a simulation and allows for the isolation of a model from its parameters. But if you want, you can also use this package as a replacement for Scipy's [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function like so:

```python
import hybrid.simulate as simulate
result = simulate.simulate(*args)
```

Instead of specifying the time derivative of the system with a `fun` argument, we specify the system with a stoichiometry matrix, a kinetic order matrix, and a vector of rate constants (or function of time that produces a vector of rate constants). The full signature of the `simulate` function is provided below:

```python
def simulate(t_span: ArrayLike, y0: ArrayLike, k: Callable[[float], ArrayLike], N: ArrayLike, kinetic_order_matrix: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, method='haseltinerawlings', **simulator_kwargs):
    """Simulate a system of reactions over a span of time given an initial state using `method`.

    Parameters
    ----------
    t_span : ArrayLike
        A tuple of times `(t0, t_end)` to simulate between.
    y0 : ArrayLike
        A vector y_i of the quantity of species i at time 0.
    k : ArrayLike | Callable
        Either a vector of unchanging rate constants or a function of time that returns a vector of rate constants.
    N : ArrayLike
        The stoichiometry matrix N such that N_ij is the change in species `i` after unit progress in reaction `j`.
    kinetic_order_matrix : ArrayLike
        The kinetic order matrix such that the _ij entry is the kinetic intensity of species i in reaction j.
    rng : np.random.Generator
        The random number generator to use for all random numbers needed during simulation.
    t_eval : ArrayLike, optional
        A vector of time points at which to evaluate the system and return in the final results.
        If None, evaluate at points chosen by the simulator, by default None.
    method : str, optional
        The method to use for simulation. Options include `'gillespie'` and `'haseltinerawlings'`, by default 'haseltinerawlings'
    **simulator_kwargs
        Options that are passed to the specified simulator class. To see valid configurations, inspect the class that you are using.

    Returns
    -------
    History
        The result and record of simulation.

        t: float
        The time at the end of simulation.
        y: ArrayLike
            The state vector at t. `y_i` is the quantity of the `i`th species at time `t`.
        t_history: ArrayLike
            The vector of all times where the state was recorded.
        y_history: ArrayLike
            An array of state vectors where the `i`th state vector corresponds to the `i`th entry in `t_history`.
        status_counter: Counter
            A counter object that records all the status of the simulator at the end of each simulation step.
    """
```

## MIDAS talk

This package was presented at MIDAS 2024. The slides are available in `midas2024.pptx`. Here is an index of which notebooks were used to generate which figures:

1. SIR curves: [VirusDynamics.ipynb](notebooks/VirusDynamics.ipynb)
2. Interventions: [HillTimeDependence.ipynb](tests/HillTimeDependence.ipynb) and [LinearTimeDependence.ipynb](tests/LinearTimeDependence.ipynb)
3. Mutant emergence curves: [VirusDynamics.ipynb](notebooks/VirusDynamics.ipynb)
4. Schögl system realizations: [Schlogl.ipynb](tests/hybrid/Schlogl.ipynb)
5. Biased conventional rounding: [PartitionClinging.ipynb](tests/PartitionClinging.ipynb)
6. Decaying isomerization problem: [DecayingIsomerization.ipynb](tests/hybrid/DecayingIsomerization.ipynb)
