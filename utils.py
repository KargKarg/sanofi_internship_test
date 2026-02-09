import matplotlib.pyplot as plt
import numpy as np
import simulation_tools as simulation_tools
from src.models import KAN, KANLinear
from typing import Callable
import torch
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import copy


def C_E_t_matrices(PATIENTS: np.ndarray[simulation_tools.pk_SyntheticPatientData]) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Create matrices for C, E, and t from the PATIENTS data.
    
    Each matrix has dimensions (Ke_type, N, T).
    """
    C: np.ndarray[float] = np.array([[PATIENTS[n][i].C for i in range(PATIENTS.shape[-1])] for n in range(PATIENTS.shape[0])])
    E: np.ndarray[float] = np.array([[PATIENTS[n][i].E for i in range(PATIENTS.shape[-1])] for n in range(PATIENTS.shape[0])])
    t: np.ndarray[float] = np.array([[PATIENTS[n][i].t_train for i in range(PATIENTS.shape[-1])] for n in range(PATIENTS.shape[0])])
    return C, E, t

def pk_plot_true_vs_observed(PATIENTS: np.ndarray[simulation_tools.pk_SyntheticPatientData], limit: int = 1) -> None:
    """
    Plot the true PK curve against the observed PK curve for patients.
    """
    for n in range(PATIENTS.shape[0]):

        for pid, patient in enumerate(PATIENTS[n][:limit]):

            Ctrue: np.ndarray[float] = simulation_tools.simulate_true_pk_python(patient)

            plt.plot(patient.t_train, Ctrue, label="True PK")
            plt.plot(patient.t_train, patient.C, linestyle=":", label=f"Patient {pid} Observed PK")

        plt.xlabel("Time (hours)")
        plt.ylabel("Concentration (mg/L)")
        plt.title(f"Patients with Ke = {round(patient.Ke, 2)}")
        plt.legend()
        plt.show()
        plt.close()
    return None

def pkpd_plot(patient: simulation_tools.pk_SyntheticPatientData) -> None:
    """
    Plot the PK and PD curves for a given patient.
    """
    plt.plot(patient.t_train, patient.C, label=f"Concentration")
    plt.plot(patient.t_train, patient.E, label=f"Effect")
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.show()
    plt.close()
    return None


def train_KAN_for_pkpd(KAN_model: KAN, hyperparams: dict[str, float], PATIENTS_Ke: np.ndarray[simulation_tools.pk_SyntheticPatientData], CE_train: torch.Tensor, CE_test: torch.Tensor, t_train: torch.Tensor, t_test: torch.Tensor, device: torch.device, gamma_c: float = 1, gamma_e: float = 1, n: int = 0, plot: bool = False) -> KAN:
    """
    Train the KAN model for PK/PD data.

    Return the best model based on test loss. Optionally plot the training and test loss curves.
    """
    if plot:
        EPOCHS: list[int] = []
        loss_train: list[float] = []
        loss_test: list[float] = []

    KAN_model: KAN = KAN_model.to(device)

    CE_train: torch.Tensor = CE_train.to(device)
    CE_test: torch.Tensor = CE_test.to(device)
    t_train: torch.Tensor = t_train.to(device)
    t_test: torch.Tensor = t_test.to(device)

    optimizer: torch.optim.Adam = torch.optim.Adam(KAN_model.parameters(), lr=hyperparams["lr"])

    # Define the function to compute dC/dt and dE/dt using the KAN model
    dCdt_dEdt: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda t, state: KAN_model(state)
    
    # Initialize the best model and loss
    best_: tuple[float, KAN] = (float("inf"), copy.deepcopy(KAN_model))

    for epoch in tqdm(range(hyperparams["epochs"])):
    
        optimizer.zero_grad()
        KAN_model.train()

        # Initialize total loss for the epoch
        total_loss: torch.Tensor = torch.zeros(1, dtype=torch.float32).to(device)

        for n in range(CE_train.shape[0]):

            # Initialize the previous time and dose for the current patient
            t_prev: float = simulation_tools.TSOLVE_START
            t_prev_dose: float = CE_train[n, 0]
            
            for bolus_time in PATIENTS_Ke[n].dose_events:

                # Create a mask to select the time points of interest
                mask: torch.Tensor = (t_train[n] > t_prev) & (t_train[n] < bolus_time)

                # Create the time span
                t_span: torch.Tensor = t_train[n][mask] - t_prev

                # Compute the predicted trajectory of C and E using the KAN model
                pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_dose.to(device), t_span.to(device)).to(device)

                # Get the true trajectory of C and E from the training data
                true: torch.Tensor = CE_train[n][mask].to(device)

                # Separate the true and predicted trajectories into C and E components
                true_dCdt, true_dEdt = true[:, 0], true[:, 1]
                pred_dCdt, pred_dEdt = pred[:, 0], pred[:, 1]

                # Compute theweighted loss for C and E
                total_loss += (gamma_c/2)*torch.mean((true_dCdt - pred_dCdt)**2) + (gamma_e/2)*torch.mean((true_dEdt - pred_dEdt)**2)

                # Update the previous time and dose for the next iteration
                t_prev: float = bolus_time
                t_prev_dose: float = CE_train[n, torch.searchsorted(t_train[n], bolus_time, right=False)]

            # SAME AS ABOVE BUT FOR THE TIME SPAN AFTER THE LAST BOLUS UNTIL THE END OF THE SIMULATION
            mask: torch.Tensor = (t_train[n] > t_prev) & (t_train[n] <= simulation_tools.TSOLVE_END)
            t_span: torch.Tensor = t_train[n][mask] - t_prev

            pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_dose.to(device), t_span.to(device)).to(device)
            true: torch.Tensor = CE_train[n][mask].to(device)

            true_dCdt, true_dEdt = true[:, 0], true[:, 1]
            pred_dCdt, pred_dEdt = pred[:, 0], pred[:, 1]

            total_loss += (gamma_c/2)*torch.mean((true_dCdt - pred_dCdt)**2) + (gamma_e/2)*torch.mean((true_dEdt - pred_dEdt)**2)
        
        total_loss /= CE_train.shape[0]

        total_loss.backward() 
        optimizer.step() 
        
        # SAME AS ABOVE BUT FOR THE TEST
        if (epoch + 1) % hyperparams["pace_eval"] == 0:
            
            KAN_model.eval()

            test_loss: torch.Tensor = torch.zeros(1, dtype=torch.float32).to(device)

            for n in range(CE_test.shape[0]):
                
                t_prev: float = simulation_tools.TSOLVE_START
                t_prev_dose: float = CE_test[n, 0]
                
                for bolus_time in PATIENTS_Ke[CE_train.shape[0] + n].dose_events:

                    mask: torch.Tensor = (t_test[n] > t_prev) & (t_test[n] < bolus_time)
                    t_span: torch.Tensor = t_test[n][mask] - t_prev

                    pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_dose.to(device), t_span.to(device)).to(device)
                    true: torch.Tensor = CE_test[n][mask].to(device)

                    true_dCdt, true_dEdt = true[:, 0], true[:, 1]
                    pred_dCdt, pred_dEdt = pred[:, 0], pred[:, 1]

                    test_loss += (gamma_c/2)*torch.mean((true_dCdt - pred_dCdt)**2) + (gamma_e/2)*torch.mean((true_dEdt - pred_dEdt)**2)

                    t_prev: float = bolus_time
                    t_prev_dose: float = CE_test[n, torch.searchsorted(t_test[n], bolus_time, right=False)]

                mask: torch.Tensor = (t_test[n] > t_prev) & (t_test[n] <= simulation_tools.TSOLVE_END)
                t_span: torch.Tensor = t_test[n][mask] - t_prev

                pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_dose.to(device), t_span.to(device)).to(device)
                true: torch.Tensor = CE_test[n][mask].to(device)

                true_dCdt, true_dEdt = true[:, 0], true[:, 1]
                pred_dCdt, pred_dEdt = pred[:, 0], pred[:, 1]

                test_loss += (gamma_c/2)*torch.mean((true_dCdt - pred_dCdt)**2) + (gamma_e/2)*torch.mean((true_dEdt - pred_dEdt)**2)
                test_loss /= CE_test.shape[0]

            print(f"Epoch {epoch + 1}: Train Loss = {round(total_loss.item(), 4)}, Test Loss = {round(test_loss.item(), 4)}")

            # Keep the best model based on the test loss
            if best_[0] > test_loss.item(): best_ = (test_loss.item(), copy.deepcopy(KAN_model))

            if plot:
                loss_train.append(total_loss.item())
                loss_test.append(test_loss.item())
                EPOCHS.append(epoch)

    if plot:
        plt.plot(EPOCHS, loss_train, label="train", c="blue")
        plt.plot(EPOCHS, loss_test, label="test", c="red")
        plt.xlabel("EPOCHS")
        plt.ylabel("LOSS")
        plt.legend()
        plt.savefig(f"plots/loss/kt_{n}_gs_{hyperparams["grid_size"]}_hl_{hyperparams["layers_hidden"][1]}.svg")
        plt.close()

    return best_[1]


def pkpd_plot_evaluate_model(model: KAN, patient: simulation_tools.pk_SyntheticPatientData, CE_test: torch.Tensor, t_test: torch.Tensor, gs: int = None, hl: int = None, n: int = None, save: bool = False) -> None:
    """
    Plot the predicted trajectory of C and E against the true trajectory and observed data for a given patient using a KAN model.
    """
    # SAME AS train_KAN_for_pkpd BUT WITHOUT THE TRAINING LOOP, JUST THE PREDICTION AND PLOTTING

    dCdt_dEdt : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda t, state: model(state)
    model.eval()

    true_dCdt_dEdt: torch.Tensor = torch.Tensor(simulation_tools.simulate_true_pkpd_python(patient))
    true_dCdt_dEdt: torch.Tensor = torch.concat([true_dCdt_dEdt[0, :][..., None], true_dCdt_dEdt[1, :][..., None]], dim=-1)
    true_dCdt_dEdt: torch.Tensor = true_dCdt_dEdt[-CE_test.shape[0]:]

    t_prev: float = simulation_tools.TSOLVE_START
    t_prev_state: float = true_dCdt_dEdt[0]

    pred_trajectory_C: list[float] = []
    true_trajectory_C: list[float] = []
    obs_trajectory_C: list[float] = []

    pred_trajectory_E: list[float] = []
    true_trajectory_E: list[float] = []
    obs_trajectory_E: list[float] = []

    t_trajectory: list[float] = []
    
    for bolus_time in patient.dose_events:

        mask: torch.Tensor = (t_test > t_prev) & (t_test < bolus_time)
        t_span: torch.Tensor = t_test[mask] - t_prev

        pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_state, t_span)
        pred_C, pred_E = pred[:, 0], pred[:, 1]

        pred_trajectory_C += pred_C.tolist()
        pred_trajectory_E += pred_E.tolist()

        true: torch.Tensor = true_dCdt_dEdt[mask]
        true_C, true_E = true[:, 0], true[:, 1]
        
        true_trajectory_C += true_C.tolist()
        true_trajectory_E += true_E.tolist()

        obs: torch.Tensor = CE_test[mask]
        obs_C, obs_E = obs[:, 0], obs[:, 1]

        obs_trajectory_C += obs_C.tolist()
        obs_trajectory_E += obs_E.tolist()

        t_trajectory += t_test[mask].tolist()

        t_prev: float = bolus_time
        t_prev_state: float = true_dCdt_dEdt[torch.searchsorted(t_test, bolus_time, right=False)]


    mask: torch.Tensor = (t_test > t_prev) & (t_test <= simulation_tools.TSOLVE_END)
    t_span: torch.Tensor = t_test[mask] - t_prev

    pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_state, t_span)
    pred_C, pred_E = pred[:, 0], pred[:, 1]

    pred_trajectory_C += pred_C.tolist()
    pred_trajectory_E += pred_E.tolist()

    true: torch.Tensor = true_dCdt_dEdt[mask]
    true_C, true_E = true[:, 0], true[:, 1]
    
    true_trajectory_C += true_C.tolist()
    true_trajectory_E += true_E.tolist()

    obs: torch.Tensor = CE_test[mask]
    obs_C, obs_E = obs[:, 0], obs[:, 1]

    obs_trajectory_C += obs_C.tolist()
    obs_trajectory_E += obs_E.tolist()

    t_trajectory += t_test[mask].tolist()

    plt.plot(t_trajectory, pred_trajectory_C, label="KAN(true)", c="black")
    plt.plot(t_trajectory, true_trajectory_C, linestyle="--", label="true", c="red")
    plt.plot(t_trajectory, obs_trajectory_C, linestyle=":", label="observed", c="green")
    plt.title("Trajectory of C")
    plt.legend()
    if save:
        plt.savefig(f"plots/trajectory/C_kt_{n}_gs_{gs}_hl_{hl}.svg")
    if not save:
        plt.show()
    plt.close()

    plt.plot(t_trajectory, pred_trajectory_E, label="KAN(true)", c="black")
    plt.plot(t_trajectory, true_trajectory_E, linestyle="--", label="true", c="red")
    plt.plot(t_trajectory, obs_trajectory_E, linestyle=":", label="observed", c="green")
    plt.title("Trajectory of E")
    plt.legend()
    if save:
        plt.savefig(f"plots/trajectory/E_kt_{n}_gs_{gs}_hl_{hl}.svg")
    if not save:
        plt.show()
    plt.close()

    return None


def pkpd_pred_true_trajectory(model: KAN, PATIENTS_Ke: np.ndarray[simulation_tools.pk_SyntheticPatientData], CE_test: torch.Tensor, t_test: torch.Tensor) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """
    Compute the predicted and true trajectories of C and E for each patient in the test set using a KAN model.
    """
    # SAME AS train_KAN_for_pkpd BUT WITHOUT THE TRAINING LOOP JUST THE PREDICTION AND RETURNING THE TRAJECTORIES

    dCdt_dEdt : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda t, state: model(state)
    model.eval()

    true_dCdt_dEdt: torch.Tensor = torch.Tensor([simulation_tools.simulate_true_pkpd_python(PATIENTS_Ke[i]) for i in range(len(PATIENTS_Ke))])
    true_dCdt_dEdt = torch.concat([true_dCdt_dEdt[:, 0, :][..., None], true_dCdt_dEdt[:, 1, :][..., None]], dim=-1)
    true_dCdt_dEdt = true_dCdt_dEdt[-CE_test.shape[0]:]

    all_pred_trajectory_C: list[list[float]] = []
    all_pred_trajectory_E: list[list[float]] = []
    all_true_trajectory_C: list[list[float]] = []
    all_true_trajectory_E: list[list[float]] = []
    all_t_trajectory: list[list[float]] = []

    for n, patient in enumerate(PATIENTS_Ke[-CE_test.shape[0]:]):

        t_prev: float = simulation_tools.TSOLVE_START
        t_prev_state: float = true_dCdt_dEdt[n, 0]

        pred_trajectory_C: list[float] = []
        true_trajectory_C: list[float] = []

        pred_trajectory_E: list[float] = []
        true_trajectory_E: list[float] = []

        t_trajectory: list[float] = []
        
        for bolus_time in patient.dose_events:

            mask: torch.Tensor = (t_test[n] > t_prev) & (t_test[n] < bolus_time)
            t_span: torch.Tensor = t_test[n][mask] - t_prev

            pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_state, t_span)
            pred_C, pred_E = pred[:, 0], pred[:, 1]

            pred_trajectory_C += pred_C.tolist()
            pred_trajectory_E += pred_E.tolist()
 
            true: torch.Tensor = true_dCdt_dEdt[n][mask]
            true_C, true_E = true[:, 0], true[:, 1]
            
            true_trajectory_C += true_C.tolist()
            true_trajectory_E += true_E.tolist()

            t_trajectory += t_test[n][mask].tolist()

            t_prev: float = bolus_time
            t_prev_state: float = true_dCdt_dEdt[n, torch.searchsorted(t_test[n], bolus_time, right=False)]


        mask: torch.Tensor = (t_test[n] > t_prev) & (t_test[n] <= simulation_tools.TSOLVE_END)
        t_span: torch.Tensor = t_test[n][mask] - t_prev

        pred: torch.Tensor = torchodeint(dCdt_dEdt, t_prev_state, t_span)
        pred_C, pred_E = pred[:, 0], pred[:, 1]

        pred_trajectory_C += pred_C.tolist()
        pred_trajectory_E += pred_E.tolist()

        true: torch.Tensor = true_dCdt_dEdt[n][mask]
        true_C, true_E = true[:, 0], true[:, 1]
        
        true_trajectory_C += true_C.tolist()
        true_trajectory_E += true_E.tolist()

        t_trajectory += t_test[n][mask].tolist()

        all_pred_trajectory_C.append(pred_trajectory_C)
        all_pred_trajectory_E.append(pred_trajectory_E)
        
        all_t_trajectory.append(t_trajectory)

        all_true_trajectory_C.append(true_trajectory_C)
        all_true_trajectory_E.append(true_trajectory_E)

    return all_pred_trajectory_C, all_pred_trajectory_E, all_true_trajectory_C, all_true_trajectory_E, all_t_trajectory


def plot_splines(layer: KANLinear, in_feature: int = 0, out_feature: int = 0, num_points: int = 100) -> None:
    """
    Plot the B-spline basis functions for a given KANLinear layer and specified input and output features.
    """
    device: torch.device = next(layer.parameters()).device

    # Get the grid range
    grid_min: float = layer.grid[in_feature, layer.spline_order].item()
    grid_max: float = layer.grid[in_feature, layer.grid_size + layer.spline_order].item()

    # Create a range of input
    x: torch.Tensor = torch.zeros(num_points, layer.in_features, device=device)
    x[:, in_feature] = torch.linspace(grid_min, grid_max, num_points, device=device)

    # Compute the spline values
    spline_values = layer.b_splines(x)[:, in_feature, :].detach().cpu() @ layer.scaled_spline_weight.detach().cpu()[out_feature, in_feature, :]

    plt.plot(x[:, in_feature].cpu(), spline_values)
    plt.title(f'Spline for in_feature={in_feature}, out_feature={out_feature}')
    plt.xlabel('Input feature value')
    plt.ylabel('Spline output')
    plt.grid(True)
    plt.show()
    return None