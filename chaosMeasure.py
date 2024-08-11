import numpy as np
from system import System
from utils import timer

def lyapunov(stan_state, total_time=20, delta_t=0.01, divisor=10):
    time = np.arange(0, total_time, delta_t*divisor)
    lyapunov_exponents = []

    stan_system = System(state=stan_state)
    perturb_state = stan_state + np.random.normal(0, 1e-10, stan_state.shape)
    perturb_system = System(state=perturb_state)

    distance_initial = np.linalg.norm(stan_state - perturb_state)

    for step, t in enumerate(time):
        stan_state = stan_system.integrate(delta_t, divisor)
        perturb_state = perturb_system.integrate(delta_t, divisor)
        distance_final = np.linalg.norm(stan_state - perturb_state)
        lyapunov_exponent = np.log(abs(distance_final / distance_initial))
        lyapunov_exponents.append(lyapunov_exponent)

        perturb_state = stan_state + distance_initial * (perturb_state - stan_state) / distance_final
        perturb_system = System(state=perturb_state)
        distance_initial = distance_final

    return np.sum(lyapunov_exponents) * (1/total_time)

def proximity(stan_state, total_time=20, delta_t=0.01):
    time = np.arange(0, total_time, delta_t)
    initial_state = stan_state
    stan_system = System(state=stan_state)
    min = float("inf")
    di = 0 # inital distance
    for t in time:
        stan_state = stan_system.integrate(delta_t,1)
        df = np.linalg.norm(stan_state-initial_state) # new distance
        if df < di and df < min:
            min = df
        di = df
    if min == float("inf"):
        min = df
    return min
