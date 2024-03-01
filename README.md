# Hybrid stochastic-deterministic simulation for chemical and biological processes

## Installation

1. Clone the repository: `git clone https://github.com/tanderson11/hybrid-algorithm.git`
2. Use `poetry` to install dependencies: `poetry install`

## Usage

### Forward simulation

`hybrid.py` uses numerical integration to solve a system of differential equations corresponding to a biological/chemical system given an initial state.

The reaction pathways are defined by two unchanging matrices: `N`, the stoichiometry matrix (`N_ij` = net change in species i after unit progress in reaction j), and `A` the rate involvement matrix (`A_ij` = kinetic intensity (power) for species i in reaction j). The rate constants are defined by the vector `k`, which has a number of entries equal to the number of reaction pathways.

Explicit time dependence is possible by specifying `k` as a function of time `t`.

The full signature and output of the `forward_time` function is given below:

```python
def forward_time(y0: np.ndarray, t_span: list[float], partition_function: Callable[[np.ndarray], Partition], k: Callable[[float], np.ndarray], N: np.ndarray, kinetic_order_matrix: np.ndarray, rng: np.random.Generator, discontinuities=[], events=[], **kwargs) -> SimulationResult:
    """Evolve system of irreversible reactions forward in time using hybrid deterministic-stochastic approximation.

    Args:
        y0 (np.ndarray): initial state of the system.
        t_span (list[float]): [t0, t_upper_limit].
        partition_function (Callable[[np.ndarray], Partition]): function that takes rates at time t and outputs a partition of the system.
        k (f: float -> np.ndarray or np.ndarray): either a callable that gives rate constants at time t or a list of unchanging rate constants.
        N (np.ndarray): the stoichiometry matrix for the system. N_ij = net change in i after unit progress in reaction j.
        kinetic_order_matrix (np.ndarray): A_ij = kinetic intensity (power) for species i in reaction j.
        rng (np.random.Generator): rng to use for stochastic simulation (and rounding).
        discontinuities (list[float], optional): a list of time points where k(t) is discontinuous. Defaults to [].
        events (list[Callable[[np.ndarray], float]], optional): a list of continuous functions of the state that have a 0 when an event of interest occurs. Defaults to [].
        **kwargs (SimulationOptions): configuration of the simulation. Defaults to SimulationOptions().

    Returns:
        SimulationResult: simulation result object:
            result.t: t_end, should be approximately equal to t_span[-1],
            result.y: system state at t_end,
            result.n_stochastic: number of stochastic events that occured,
            result.nfev: number of evaluations of the derivative.
            result.t_history: every time point evaluated.
            result.y_history: state of the system at each point in t_history.

    """
```