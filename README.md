# PK/PD Modeling with Kolmogorov–Arnold Networks (KAN)

Short project aiming to use **Kolmogorov–Arnold Networks (KAN)** to learn **patient PK/PD dynamics**.

---

## Usage

Training scripts are available at the end of the notebook `display.ipynb`.  
Training was executed on a **cluster via SSH** so the appropriate directories must be created beforehand to store artifacts generated during the runs (plots, metrics, checkpoints, etc)

---

## Project Structure

- `simulation_tools.py`  
  Contains all the scripts required for **PK and PK/PD simulation**, including:
  - trajectory generation,
  - computation of the **exact patient trajectory** (not the observed).

- `utils.py`  
  Contains all remaining utilities, including:
  - model training,
  - visualization,
  - evaluation tools.

---

## Results

KAN appear to be **very effective models for approximating dynamical systems**.  
The **DualKAN** architecture seems particularly promising, with each head specialized on a specific ODE.

By extending this approach, it may be possible to model **much more complex and deeply coupled systems**.

---

## Notes & Limitations

Only a **simple system** was explored in this project.  
Future work should focus on **larger and more complex systems** to further assess the scalability and robustness of the approach.

---

## References

- **Ziming Liu et al. (2024).**  
  *KAN: Kolmogorov–Arnold Networks.*  
  arXiv:2404.19756.  
  Introduces the KAN framework and its theoretical foundations.

- **Benjamin König et al. (2024).**  
  *KAN-ODEs: Learning Ordinary Differential Equations via Kolmogorov–Arnold Networks.*  
  GitHub: `DENG-MIT/KAN-ODEs`

- **M. Petrizzelli et al.**  
  *KANODE_PKPD.*  
  GitHub: `SanofiMPetrizzelli/KANODE_PKP_
