# compare_plot.py
# Compare EKF estimated bus voltages vs reference (e.g., loadflow solution)
# Plots magnitude and angle comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Filenames ---
EKF_FILE = "ekf_state_estimates.csv"   # output from ekf_state_est.py
REF_FILE = "IEEE_14.xlsx"              # reference load flow file (busdata sheet)

# --- Load EKF results ---
df_ekf = pd.read_csv(EKF_FILE)

# --- Load reference values from loadflow busdata sheet ---
# busdata format: [bus, Vm, Va(rad), ...]
ref_busdata = pd.read_excel(REF_FILE, sheet_name="busdata", header=None).values
ref_vm = ref_busdata[:,1]   # Vm column
ref_va = ref_busdata[:,2]   # Va (rad) column

# --- Align lengths ---
buses = df_ekf["bus"].values
ekf_vm = df_ekf["V_abs"].values
ekf_va = df_ekf["V_ang_rad"].values

# --- Compute errors ---
vm_err = ekf_vm - ref_vm
va_err = np.degrees(ekf_va - ref_va)   # in degrees

# --- Plot magnitude comparison ---
plt.figure(figsize=(10,5))
plt.plot(buses, ref_vm, 'o-', label="Reference (Loadflow)")
plt.plot(buses, ekf_vm, 's-', label="EKF Estimate")
plt.xlabel("Bus")
plt.ylabel("|V| (p.u.)")
plt.title("Bus Voltage Magnitude Comparison")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot angle comparison ---
plt.figure(figsize=(10,5))
plt.plot(buses, np.degrees(ref_va), 'o-', label="Reference (deg)")
plt.plot(buses, np.degrees(ekf_va), 's-', label="EKF Estimate (deg)")
plt.xlabel("Bus")
plt.ylabel("Voltage Angle (deg)")
plt.title("Bus Voltage Angle Comparison")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot errors ---
plt.figure(figsize=(10,5))
plt.bar(buses-0.2, vm_err, width=0.4, label="Magnitude Error (pu)")
# plt.bar(buses+0.2, va_err, width=0.4, label="Angle Error (deg)")
plt.xlabel("Bus")
plt.ylabel("Error")
plt.title("Estimation Errors (EKF - Reference)")
plt.legend()
plt.grid(True)
plt.show()
