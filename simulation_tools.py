import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize._optimize import OptimizeResult
from dataclasses import dataclass

V_true: float = 10.0

TSOLVE_START: float = 0.0
TSOLVE_END: float = 96.0

T_OBSERVATION: np.ndarray[float] = np.linspace(TSOLVE_START, TSOLVE_END, 100)

@dataclass
class pk_SyntheticPatientData:

    # Initial condition
    C0: np.ndarray[float]

    # Constant
    Ke: float

    # Times
    t_train: np.ndarray[float]

    # Observed concentrations
    C: np.ndarray[float]

    # Time -> Amount
    dose_events: dict[float, float]

    # (t_start, t_end)
    tspan_train: tuple[float, float]


@dataclass
class pkpd_SyntheticPatientData:

    # Initial condition
    C0: np.ndarray[float]

    # Initial condition
    E0: np.ndarray[float]

    # Constant
    Ke: float

    # Constant
    kin: float

    # Constant
    Emax: float

    # Constant
    EC50: float

    # Times
    t_train: np.ndarray[float]

    # Observed concentrations
    C: np.ndarray[float]

    # Observed effects
    E: np.ndarray[float]

    # Time -> Amount
    dose_events: dict[float, float]

    # (t_start, t_end)
    tspan_train: tuple[float, float]


def pk_ode_python(t: float, C: np.ndarray[float], Ke_true: float) -> np.ndarray[float]:
    """
    The true one-compartment elimination rate (dC/dt = -Ke * C)
    """
    dCdt: np.ndarray[float] = -Ke_true*C
    return dCdt


def pkpd_ode_python(t: float, y: np.ndarray[float], Ke: float, kin: float, Emax: float, EC50: float) -> np.ndarray[float]:
    """
    """
    C: float = y[0]
    E: float = y[1]
    dCdt: float = -Ke * C
    dEdt: float = kin * (Emax * C / (EC50 + C) - E)
    return [dCdt, dEdt]


def pk_generate_synthetic_data_python(Ke: float, SEED: int = 12, scale: float = 0.1) -> pk_SyntheticPatientData:
    """
    Generate PK data for a patient using a one-compartment model with first-order elimination and random dosing events.

    Code from SanofiMPetrizzelli/KANODE_PKPD/code/PKtoy_example.jl but translated to Python.
    """
    np.random.seed(SEED)

    # Generate random dose times and amounts
    local_dose_times: list[float] = [i for i in range(int(TSOLVE_START) + 1, int(TSOLVE_END)) if np.random.binomial(1, 0.05)]
    local_dose_amounts: list[float] = [100 + np.random.normal(0, 10) for _ in local_dose_times]

    # Dose events: time -> amount
    dose_events: dict[float, float] = dict(zip(local_dose_times, local_dose_amounts))

    # # C(t=0) before dose
    current_C: np.ndarray[float] = (C0_initial := np.array([0.0], dtype=np.float32))
    current_time: float = (start_time := 0.0)

    all_times: list[float] = []
    all_concentrations: list[float] = []

    # Loop over dose events
    for i, (t_dose, dose_amount) in enumerate(zip(local_dose_times, local_dose_amounts)):

        # Elimination phase until next dose
        if t_dose > current_time:

            # Time between the beginning and the dose
            t_eval: np.ndarray[float] = np.linspace(current_time, t_dose, 1000)

            # Solve ODE for each time segment
            sol: OptimizeResult = solve_ivp(pk_ode_python, (current_time, t_dose), current_C, t_eval=t_eval, args=(Ke,))

            noise: np.ndarray[float] = np.random.normal(size=sol.y.shape, scale=scale)

            # Skip the first time point to avoid duplication
            if i > 0:
                # Add all time points except the first
                all_times.extend(sol.t[1:])
                # Add all concentrations except the first
                all_concentrations.extend(sol.y[0, 1:] + noise[0, 1:])

            # For the first dose include all points
            else:
                # Add all time points
                all_times.extend(sol.t)
                # Add all concentrations
                all_concentrations.extend(sol.y[0] + noise[0])

            # Concentration just before dose (almost at the dose time)
            current_C: np.ndarray[float] = sol.y[:, -1] + noise[:, -1]

        # Apply the impulse dose
        C_jump = dose_amount / V_true
        current_C = current_C + C_jump

        # Record immediately after dose
        all_times.append(t_dose)
        all_concentrations.append(current_C[0])

        # Update
        current_time: float = t_dose

    # Final elimination phase
    t_eval: np.ndarray[float] = np.linspace(current_time, TSOLVE_END, 1000)
    sol: OptimizeResult = solve_ivp(pk_ode_python, (current_time, TSOLVE_END), current_C, t_eval=t_eval, args=(Ke,))

    noise: np.ndarray[float] = np.random.normal(size=sol.y.shape, scale=scale)
    all_times.extend(sol.t[1:])
    all_concentrations.extend(sol.y[0, 1:] + noise[0, 1:])

    # Sort time–concentration pairs
    combined: list[tuple[float, float]] = sorted(zip(all_times, all_concentrations), key=lambda x: x[0])
    T_data: list[tuple[float, float]] = np.array([t for t, _ in combined], dtype=np.float32)
    C_data: list[tuple[float, float]] = np.array([c for _, c in combined], dtype=np.float32)

    # Nearest-neighbor mapping
    C: list[float] = []
    for t_obs in T_OBSERVATION:
        idx: int = np.argmin(np.abs(T_data - t_obs))
        C.append(C_data[idx])
    C: np.ndarray[float] = np.array(C, dtype=np.float32)

    # No negative concentrations
    C[C < 0] = 0

    # Initial condition AFTER first dose
    C0: np.ndarray[float] = np.array([C[0]], dtype=np.float32)

    # Span for the train
    tspan_train: tuple[float, float] = (T_OBSERVATION[0], T_OBSERVATION[-1])

    return pk_SyntheticPatientData(C0, Ke, T_OBSERVATION, C, dose_events, tspan_train)


def simulate_true_pk_python(patient: pk_SyntheticPatientData) -> np.ndarray[float]:
    """
    Simulate the true PK trajectory for a patient.
    
    Code from SanofiMPetrizzelli/KANODE_PKPD/code/PKtoy_example.jl but translated to Python.
    """
    all_times: list[float] = []
    all_concentrations: list[float] = []
    current_C: np.ndarray[float] = (C0_initial := np.array([0.0], dtype=np.float32))
    current_time: float = (start_time := 0.0)

    # Dose events
    dose_events: dict[float, float] = patient.dose_events
    dose_times_sorted: list[float] = sorted(dose_events.keys())

    # Extended time set: observation times + dose times + final time
    T_plot_ext: np.ndarray[float] = np.unique(np.sort(np.concatenate([patient.t_train, dose_times_sorted, [TSOLVE_END]])))

    # Loop over doses
    for t_dose in dose_times_sorted:
        dose_amount: float = dose_events[t_dose]

        # Elimination until next dose
        if t_dose >= current_time:

            # Only keep required times in this segment
            t_span_segment: np.ndarray[float] = T_plot_ext[(T_plot_ext > current_time) & (T_plot_ext <= t_dose)]
            t_to_solve: np.ndarray[float] = np.unique(np.sort(np.concatenate([[current_time], t_span_segment])))

            if len(t_to_solve) > 1:
                sol: OptimizeResult = solve_ivp(pk_ode_python, (current_time, t_dose), current_C, t_eval=t_to_solve, args=(patient.Ke,) )

                all_times.extend(sol.t[1:])
                all_concentrations.extend(sol.y[0, 1:])
                current_C: np.ndarray[float] = sol.y[:, -1]

        C_jump: float = dose_amount / V_true
        current_C: np.ndarray[float] = current_C + C_jump

        # Record immediately after dose only if needed
        if (t_dose in patient.t_train) or np.isclose(t_dose, patient.t_train[0]):
            all_times.append(t_dose)
            all_concentrations.append(current_C[0])

        current_time: float = t_dose

    # Final elimination phase
    end_time: float = TSOLVE_END

    if end_time > current_time:

        # Only keep required times in this segment
        t_span_final: np.ndarray[float] = T_plot_ext[(T_plot_ext > current_time) & (T_plot_ext <= end_time)]
        t_to_solve: np.ndarray[float] = np.unique(np.sort(np.concatenate([[current_time], t_span_final])))

        if len(t_to_solve) > 1:
            sol: OptimizeResult = solve_ivp(pk_ode_python, (current_time, end_time), current_C, t_eval=t_to_solve, args=(patient.Ke,) )

            all_times.extend(sol.t[1:])
            all_concentrations.extend(sol.y[0, 1:])

    # Convert to arrays and sort
    all_times = np.array(all_times)
    all_concentrations = np.array(all_concentrations)

    perm = np.argsort(all_times)
    all_times = all_times[perm]
    all_concentrations = all_concentrations[perm]

    # Nearest-neighbor interpolation to match T_plot
    C_interp = []
    for t_obs in patient.t_train:
        idx = np.argmin(np.abs(all_times - t_obs))
        C_interp.append(all_concentrations[idx])

    C_interp = np.array(C_interp, dtype=np.float32)

    return C_interp


def pkpd_generate_synthetic_data_python(Ke: float, kin: float, Emax: float, EC50: float, SEED: int = 12, scale_c: float = 0.1, scale_e: float = 0.01) -> pkpd_SyntheticPatientData:
    """
    Generate PK/PD data for a patient using a one-compartment model with Emax PD and random dosing events.

    Code from SanofiMPetrizzelli/KANODE_PKPD/code/PKPDtoy_example.jl but translated to Python and extended to PK/PD.
    """
    np.random.seed(SEED)

    local_dose_times: list[float] = [i for i in range(int(TSOLVE_START) + 1, int(TSOLVE_END)) if np.random.binomial(1, 0.05)]
    local_dose_amounts: list[float] = [100 + np.random.normal(0, 10) for _ in local_dose_times]

    # Dose events: time -> amount
    dose_events: dict[float, float] = dict(zip(local_dose_times, local_dose_amounts))

    # # C(t=0) before dose
    current_state: np.ndarray[float] = (state_initial := np.array([0.0, 0.0], dtype=np.float32))
    current_time: float = (start_time := 0.0)

    all_times: list[float] = []
    all_concentrations: list[float] = []
    all_effects: list[float] = []

    # Loop over dose events
    for i, (t_dose, dose_amount) in enumerate(zip(local_dose_times, local_dose_amounts)):

        # Elimination phase until next dose
        if t_dose > current_time:

            # Time between the beginning and the dose
            t_eval: np.ndarray[float] = np.linspace(current_time, t_dose, 1000)

            # Solve ODE for each time segment
            sol: OptimizeResult = solve_ivp(pkpd_ode_python, (current_time, t_dose), current_state, t_eval=t_eval, args=(Ke, kin, Emax, EC50) )

            C_sol = sol.y[0]
            E_sol = sol.y[1]

            noise_c: np.ndarray[float] = np.random.normal(size=C_sol.shape, scale=scale_c)
            noise_e: np.ndarray[float] = np.random.normal(size=E_sol.shape, scale=scale_e)

            # Skip the first time point to avoid duplication
            if i > 0:
                # Add all time points except the first
                all_times.extend(sol.t[1:])
                # Add all concentrations except the first
                all_concentrations.extend(C_sol[1:] + noise_c[1:])
                # Add all effects except the first
                all_effects.extend(E_sol[1:] + noise_e[1:])

            # For the first dose include all points
            else:
                # Add all time points
                all_times.extend(sol.t)
                # Add all concentrations
                all_concentrations.extend(C_sol + noise_c)
                # Add all effects
                all_effects.extend(E_sol + noise_e)

            # Concentration just before dose (almost at the dose time)
            current_state: np.ndarray[float] = np.array([
                C_sol[-1] + noise_c[-1], E_sol[-1] + noise_e[-1]
            ])

        # Apply the impulse dose
        C_jump = dose_amount / V_true
        current_state[0] += C_jump

        # Record immediately after dose
        all_times.append(t_dose)
        all_concentrations.append(current_state[0])
        all_effects.append(current_state[1])

        current_time: float = t_dose

    # Final elimination phase
    t_eval: np.ndarray[float] = np.linspace(current_time, TSOLVE_END, 1000)

    sol: OptimizeResult = solve_ivp(pkpd_ode_python, (current_time, TSOLVE_END), current_state, t_eval=t_eval, args=(Ke, kin, Emax, EC50) )

    C_sol = sol.y[0]
    E_sol = sol.y[1]

    noise_c: np.ndarray[float] = np.random.normal(size=C_sol.shape, scale=scale_c)
    noise_e: np.ndarray[float] = np.random.normal(size=E_sol.shape, scale=scale_e)

    all_times.extend(sol.t[1:])
    all_concentrations.extend(C_sol[1:] + noise_c[1:])
    all_effects.extend(E_sol[1:] + noise_c[1:])

    # Sort time–concentration pairs
    combined: list[tuple[float, float]] = sorted(zip(all_times, all_concentrations, all_effects), key=lambda x: x[0])
    T_data: list[tuple[float, float]] = np.array([t for t, _, _ in combined], dtype=np.float32)
    C_data: list[tuple[float, float]] = np.array([c for _, c, _ in combined], dtype=np.float32)
    E_data: list[tuple[float, float]] = np.array([e for _, _, e in combined], dtype=np.float32)

    # Nearest-neighbor mapping
    C: list[float] = []
    E: list[float] = []
    for t_obs in T_OBSERVATION:
        idx: int = np.argmin(np.abs(T_data - t_obs))
        C.append(C_data[idx])
        E.append(E_data[idx])

    C: np.ndarray[float] = np.array(C, dtype=np.float32)
    E: np.ndarray[float] = np.array(E, dtype=np.float32)

    C[C < 0] = 0
    #E[E < 0] = 0

    # Initial condition AFTER first dose
    tspan_train: tuple[float, float] = (T_OBSERVATION[0], T_OBSERVATION[-1])
    C0_train: np.ndarray[float] = np.array([C[0]], dtype=np.float32)
    E0_train: np.ndarray[float] = np.array([E[0]], dtype=np.float32)

    return pkpd_SyntheticPatientData(C0_train, E0_train, Ke, kin, Emax, EC50, T_OBSERVATION, C, E, dose_events, tspan_train)


def simulate_true_pkpd_python(patient: pk_SyntheticPatientData) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Simulate the true PK/PD trajectory for a patient.

    Code from SanofiMPetrizzelli/KANODE_PKPD/code/PKPDtoy_example.jl but translated to Python and extended to PK/PD.
    """
    # Keep tracks
    all_times: list[float] = []
    all_concentrations: list[float] = []
    all_effects: list[float] = []

    # Initial conditions
    current_state: np.ndarray[float] = (state_initial := np.array([0.0, 0.0], dtype=np.float32))
    current_time: float = (start_time := 0.0)

    # Dose events
    dose_events: dict[float, float] = patient.dose_events
    dose_times_sorted: list[float] = sorted(dose_events.keys())

    # Extended time set: observation times + dose times + final time
    T_plot_ext: np.ndarray[float] = np.unique(np.sort(np.concatenate([patient.t_train, dose_times_sorted, [TSOLVE_END]])))

    # Loop over doses
    for t_dose in dose_times_sorted:
        dose_amount: float = dose_events[t_dose]

        # Elimination until next dose
        if t_dose >= current_time:

            # Only keep required times in this segment
            t_span_segment: np.ndarray[float] = T_plot_ext[(T_plot_ext > current_time) & (T_plot_ext <= t_dose)]
            t_to_solve: np.ndarray[float] = np.unique(np.sort(np.concatenate([[current_time], t_span_segment])))

            if len(t_to_solve) > 1:

                sol: OptimizeResult = solve_ivp(pkpd_ode_python, (current_time, t_dose), current_state, t_eval=t_to_solve, args=(patient.Ke, patient.kin, patient.Emax, patient.EC50))
                C_sol: np.ndarray[float] = sol.y[0]
                E_sol: np.ndarray[float] = sol.y[1]

                all_times.extend(sol.t[1:])
                all_concentrations.extend(C_sol[1:])
                all_effects.extend(E_sol[1:])

                current_state: np.ndarray[float] = sol.y[:, -1]

        C_jump: float = dose_amount / V_true
        current_state[0] += C_jump

        # Record immediately after dose only if needed
        if (t_dose in patient.t_train) or np.isclose(t_dose, patient.t_train[0]):
            all_times.append(t_dose)
            all_concentrations.append(current_state[0])
            all_effects.append(current_state[1])

        current_time: float = t_dose

    # Final elimination phase
    end_time: float = TSOLVE_END

    if end_time > current_time:

        # Only keep required times in this segment
        t_span_final: np.ndarray[float] = T_plot_ext[(T_plot_ext > current_time) & (T_plot_ext <= end_time)]
        t_to_solve: np.ndarray[float] = np.unique(np.sort(np.concatenate([[current_time], t_span_final])))

        if len(t_to_solve) > 1:
            sol: OptimizeResult = solve_ivp(pkpd_ode_python, (current_time, end_time), current_state, t_eval=t_to_solve, args=(patient.Ke, patient.kin, patient.Emax, patient.EC50) )

            C_sol = sol.y[0]
            E_sol = sol.y[1]

            all_times.extend(sol.t[1:])
            all_concentrations.extend(C_sol[1:])
            all_effects.extend(E_sol[1:])

    # Convert to arrays and sort
    all_times = np.array(all_times)
    all_concentrations = np.array(all_concentrations)
    all_effects = np.array(all_effects)

    perm = np.argsort(all_times)
    all_times = all_times[perm]
    all_concentrations = all_concentrations[perm]
    all_effects = all_effects[perm]

    # Nearest-neighbor interpolation to match T_plot
    C_interp = []
    E_interp = []

    for t_obs in patient.t_train:
        idx = np.argmin(np.abs(all_times - t_obs))
        C_interp.append(all_concentrations[idx])
        E_interp.append(all_effects[idx])

    C_interp = np.array(C_interp, dtype=np.float32)
    E_interp = np.array(E_interp, dtype=np.float32)

    return C_interp, E_interp