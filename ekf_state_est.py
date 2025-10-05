# ekf_state_est.py
# Extended Kalman Filter (EKF) for static power system state estimation
# Supports PMU (phasors) + SCADA (P/Q/|V|/flow) measurements
#
# Usage: put your Excel files in the same folder and run:
#   python ekf_state_est.py
#
# Requires: numpy, pandas, scipy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import inv
import math
import sys

# -------------------------
# USER CONFIG / Filenames
# -------------------------
# If using simulation mode (like MATLAB), provide loadflow file (busdata + linedata)
LOADFLOW_FILE = "IEEE_14.xlsx"     # for mode 1 (if you want to initialize from loadflow)
MEAS_FILE = "meas_data.xlsx"       # for mode 0 (direct measurement)
USE_LOADFLOW_INIT = True           # If True, try to read LOADFLOW_FILE to create initial state
SIMULATION_MODE = None             # None => attempt to detect: if meas_data.xlsx exists use mode 0
MAX_ITER = 30
TOL = 1e-6

# -------------------------
# Utility functions
# -------------------------
def read_excel_sheet_safe(fn, sheet):
    try:
        return pd.read_excel(fn, sheet_name=sheet, header=None).values
    except Exception as e:
        print(f"[WARN] Could not read {sheet} from {fn}: {e}")
        return None

def complex_from_state(x, n):
    Vr = x[:n]
    Vi = x[n:]
    return Vr + 1j*Vi

def state_from_complex(V):
    Vr = V.real
    Vi = V.imag
    return np.concatenate([Vr, Vi])

def polar_from_complex(V):
    mag = np.abs(V)
    ang = np.angle(V)
    return mag, ang

# -------------------------
# Build YBUS / line pi params
# -------------------------
def build_ybus(line_data, nob):
    # line_data rows: start end r x b_shunt tap phase
    # MATLAB format had: Column1 start bus, 2 end bus, 3 r, 4 x, 5 b_shunt, 6 tap, 7 phase
    Y = np.zeros((nob, nob), dtype=complex)
    for row in line_data:
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        r = float(row[2])
        x = float(row[3])
        b_sh = float(row[4]) if not np.isnan(row[4]) else 0.0
        tap = float(row[5]) if not np.isnan(row[5]) and row[5]!=0 else 1.0
        phase_deg = float(row[6]) if not np.isnan(row[6]) else 0.0
        z = r + 1j*x
        y = 1.0 / z
        tap_complex = tap * np.exp(1j * math.radians(phase_deg))
        # off-diagonals
        Y[i, j] -= y / np.conjugate(tap_complex)
        Y[j, i] -= y / tap_complex
        # diag
        Y[i, i] += y / (tap_complex * np.conjugate(tap_complex)) + 1j * b_sh / 2.0
        Y[j, j] += y + 1j * b_sh / 2.0
    return Y

def branch_current(V, line_row):
    # For branch with orientation start->end (as in file)
    # returns current from start bus to end bus using simple pi-model:
    # I_km = (V_k - V_m) / (r + jx) + j*b_sh/2 * V_k     (ignoring transformer tap-phase for simplicity)
    r = float(line_row[2]); x = float(line_row[3])
    b_sh = float(line_row[4]) if not np.isnan(line_row[4]) else 0.0
    tap = float(line_row[5]) if not np.isnan(line_row[5]) and line_row[5]!=0 else 1.0
    phase_deg = float(line_row[6]) if not np.isnan(line_row[6]) else 0.0
    z = r + 1j*x
    y = 1.0 / z
    # incorporate tap/phase in approximate way: V_k_effective = V_k / tap_complex
    tap_complex = tap * np.exp(1j * math.radians(phase_deg))
    def I_from_VkVm(Vk, Vm):
        Vk_eff = Vk / tap_complex
        I = y * (Vk_eff - Vm) + 1j * b_sh / 2.0 * Vk_eff
        return I
    return I_from_VkVm

# -------------------------
# Read measurement file and create measurement vector (z), measurement function mapping
# -------------------------
def parse_measurements(meas_file, line_data, nob, nol, err_spec, flag_error, slack_bus_index, flag_eq):
    # Reads sheets from meas_file and returns:
    # meas_list: list of dicts { 'type':str, 'value':float, 'loc':..., 'var':float }
    # type can be: 'Pinj','Qinj','Pflow','Qflow','Vmag','Vang','VMA_phasor','Imag','Iang','Ireal','Iimag','I_IMA','I_IRI'
    # But to keep consistent: we'll produce items with 'code' (int) same as MATLAB:
    # bus meas types: 1 Pinj,2 Qinj,3 VM,4 VA
    # line meas types: 1 Pflow,2 Qflow,3 Iabs,4 Iang,5 Ireal,6 Iimag
    # The meas sheets in MATLAB: meas_bus cols -> [meas_value, meas_type, bus_no]
    # meas_line cols -> [meas_value, meas_type, line_no, from_bus, to_bus]
    meas_bus = read_excel_sheet_safe(meas_file, "bus_meas")
    meas_line = read_excel_sheet_safe(meas_file, "line_meas")
    z_list = []  # store dicts
    # If not present, return empty list
    if meas_bus is None:
        meas_bus = np.empty((0,3))
    if meas_line is None:
        meas_line = np.empty((0,5))
    # bus measurements
    for i in range(meas_bus.shape[0]):
        val = float(meas_bus[i,0])
        mtype = int(meas_bus[i,1])
        bus = int(meas_bus[i,2]) - 1
        if mtype == 1:   # Pinj
            var = ( (val*err_spec[0]/100.0)**2 / 3.0 ) if flag_error==2 else err_spec[0]**2
            z_list.append({'code':1,'value':val,'bus':bus,'var':var})
        elif mtype == 2: # Qinj
            var = ( (val*err_spec[0]/100.0)**2 / 3.0 ) if flag_error==2 else err_spec[0]**2
            z_list.append({'code':2,'value':val,'bus':bus,'var':var})
        elif mtype == 3: # VM
            var = ( (val*err_spec[2]/100.0)**2 / 3.0 ) if flag_error==2 else err_spec[2]**2
            z_list.append({'code':3,'value':val,'bus':bus,'var':var})
        elif mtype == 4: # VA
            if flag_error==2:
                var = ( math.pi*err_spec[4]/180.0 )**2 / 3.0
            else:
                var = err_spec[3]**2
            # value is angle in radian
            # shift by slack later in measurement model by subtracting slack angle reference
            z_list.append({'code':4,'value':val,'bus':bus,'var':var})
    # line measurements
    for i in range(meas_line.shape[0]):
        val = float(meas_line[i,0])
        mtype = int(meas_line[i,1])
        line_no = int(meas_line[i,2]) - 1
        from_bus = int(meas_line[i,3]) - 1
        to_bus = int(meas_line[i,4]) - 1
        # determine orientation index similar to MATLAB: if from==line(start) and to==line(end) => idx=line_no else idx=line_no+nol
        start = int(line_data[line_no][0]) - 1
        end = int(line_data[line_no][1]) - 1
        if from_bus==start and to_bus==end:
            temp_idx = line_no
            orientation = 0
        elif from_bus==end and to_bus==start:
            temp_idx = line_no + nol
            orientation = 1
        else:
            temp_idx = None
            orientation = None
        if temp_idx is None:
            continue
        if mtype == 1:  # Pflow
            var = ((val*err_spec[1]/100.0)**2/3.0) if flag_error==2 else err_spec[1]**2
            z_list.append({'code':101,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
        elif mtype == 2: # Qflow
            var = ((val*err_spec[1]/100.0)**2/3.0) if flag_error==2 else err_spec[1]**2
            z_list.append({'code':102,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
        elif mtype == 3: # I magnitude
            var = ((val*err_spec[3]/100.0)**2/3.0) if flag_error==2 else err_spec[2]**2
            z_list.append({'code':103,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
        elif mtype == 4: # I angle
            if flag_error==2:
                var = (math.pi*err_spec[4]/180.0)**2/3.0
            else:
                var = err_spec[3]**2
            z_list.append({'code':104,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
        elif mtype == 5: # Ireal
            var = ((val*err_spec[3]/100.0)**2/3.0) if flag_error==2 else err_spec[2]**2
            z_list.append({'code':105,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
        elif mtype == 6: # Iimag
            var = ((val*err_spec[3]/100.0)**2/3.0) if flag_error==2 else err_spec[2]**2
            z_list.append({'code':106,'value':val,'line_idx':line_no,'orientation':orientation,'var':var})
    # Add zero injection/flow constraints if flag_eq==0 (same as MATLAB)
    # zero_inj and zero_flow must be read from sheets
    zero_inj = read_excel_sheet_safe(meas_file, "zero_inj")
    zero_flow = read_excel_sheet_safe(meas_file, "zero_flow")
    if flag_eq==0 and zero_inj is not None and zero_inj.size>0:
        # MATLAB uses z_Pinj and z_Qinj etc - they are probably loaded already. For simplicity, assume zero_inj rows [bus, type]
        for row in zero_inj:
            if np.isnan(row[0]):
                continue
            bus = int(row[0]) - 1
            ztype = int(row[1])
            if ztype == 1: # Pinj zero
                z_list.append({'code':1,'value':0.0,'bus':bus,'var':1e-8})
            elif ztype == 2: # Qinj zero
                z_list.append({'code':2,'value':0.0,'bus':bus,'var':1e-8})
    if flag_eq==0 and zero_flow is not None and zero_flow.size>0:
        # zero_flow has [type, branch_no, start, end]
        for row in zero_flow:
            if np.isnan(row[0]):
                continue
            ztype = int(row[0])
            brno = int(row[1]) - 1
            if ztype == 1:
                z_list.append({'code':101,'value':0.0,'line_idx':brno,'orientation':0,'var':1e-8})
            elif ztype == 2:
                z_list.append({'code':102,'value':0.0,'line_idx':brno,'orientation':0,'var':1e-8})
    return z_list

# -------------------------
# Measurement prediction function h(x) and numerical Jacobian
# -------------------------
def build_measurement_func_and_jac(z_list, Ybus, line_data, nob, nol, slack_idx):
    # returns two functions: h(x) -> predicted measurement vector, H(x) -> Jacobian matrix (len(z)x(2n))
    n = nob
    def h_of_x(x):
        V = complex_from_state(x, n)
        hvals = []
        # precompute currents for each branch orientation
        for m in z_list:
            code = m['code']
            if code == 1: # Pinj at bus
                bus = m['bus']
                Iinj = Ybus[bus,:].dot(V)
                S = V[bus] * np.conjugate(Iinj)
                P = S.real
                hvals.append(P)
            elif code == 2: # Qinj
                bus = m['bus']
                Iinj = Ybus[bus,:].dot(V)
                S = V[bus] * np.conjugate(Iinj)
                Q = S.imag
                hvals.append(Q)
            elif code == 3: # VM
                bus = m['bus']
                hvals.append(np.abs(V[bus]))
            elif code == 4: # VA
                bus = m['bus']
                # values in file are absolute angles; MATLAB subtracts slack angle; we predict (angle - slack_angle)
                angle_rel = np.angle(V[bus]) - np.angle(V[slack_idx])
                hvals.append(angle_rel)
            elif code in (101,102,103,104,105,106):
                # line measurements: need to compute current from near bus to far bus depending on orientation.
                line_idx = m['line_idx']
                orientation = m['orientation']
                row = line_data[line_idx]
                start = int(row[0]) - 1
                end = int(row[1]) - 1
                # if orientation == 0 means measurement is from start->end, so near bus = start, far = end
                if orientation == 0:
                    near = start; far = end
                else:
                    near = end; far = start
                # compute I_near_to_far using branch_current routine
                Ifun = branch_current(V, row)
                I_km = Ifun(V[near], V[far])
                if code == 101:
                    # Pflow = real(S_km) where S_km = V_k * conj(I_km)
                    S = V[near] * np.conjugate(I_km)
                    hvals.append(S.real)
                elif code == 102:
                    S = V[near] * np.conjugate(I_km)
                    hvals.append(S.imag)
                elif code == 103: # I magnitude
                    hvals.append(np.abs(I_km))
                elif code == 104: # I angle (relative to slack)
                    angle_rel = np.angle(I_km) - np.angle(V[slack_idx])
                    hvals.append(angle_rel)
                elif code == 105: # I real
                    hvals.append(I_km.real)
                elif code == 106: # I imag
                    hvals.append(I_km.imag)
            else:
                # unknown code
                hvals.append(0.0)
        return np.array(hvals, dtype=float)

    def jacobian_numeric(x, eps=1e-6):
        # central difference numeric Jacobian: for speed use forward diff (simple) but central is more accurate
        mlen = len(z_list)
        nstate = len(x)
        H = np.zeros((mlen, nstate))
        hx = h_of_x(x)
        for k in range(nstate):
            dx = np.zeros(nstate)
            dx[k] = eps
            h1 = h_of_x(x + dx)
            h0 = h_of_x(x - dx)
            H[:, k] = (h1 - h0) / (2*eps)
        return H
    return h_of_x, jacobian_numeric

# -------------------------
# EKF Implementation
# -------------------------
def ekf_run(x0, P0, Q_proc, z_vec, R_mat, hfunc, Hfunc, max_iter=30, tol=1e-6):
    x = x0.copy()
    P = P0.copy()
    nstate = len(x)
    m = len(z_vec)
    for ite in range(max_iter):
        # Prediction: static model => x_pred = x; P_pred = P + Q_proc
        x_pred = x.copy()
        P_pred = P + Q_proc
        # Measurement prediction and Jacobian
        hx = hfunc(x_pred)
        H = Hfunc(x_pred)
        # Kalman gain
        S = H.dot(P_pred).dot(H.T) + R_mat
        try:
            K = P_pred.dot(H.T).dot(inv(S))
        except Exception as e:
            K = P_pred.dot(H.T).dot(np.linalg.pinv(S))
        # Update
        y = z_vec - hx
        x_new = x_pred + K.dot(y)
        P_new = (np.eye(nstate) - K.dot(H)).dot(P_pred)
        err = np.linalg.norm(x_new - x)
        print(f"[EKF] iter {ite+1}, ||dx|| = {err:.3e}")
        x = x_new
        P = P_new
        if err < tol:
            break
    return x, P

# -------------------------
# Main routine
# -------------------------
def main():
    # Try to detect measurement file mode
    global SIMULATION_MODE
    import os
    if os.path.exists(MEAS_FILE):
        SIMULATION_MODE = 0
    else:
        SIMULATION_MODE = 1
    print(f"[INFO] Detected SIMULATION_MODE = {SIMULATION_MODE} (0 => use meas_data.xlsx, 1 => simulate from loadflow)")

    # Read either meas_data.xlsx or loadflow files
    if SIMULATION_MODE == 0:
        msys = read_excel_sheet_safe(MEAS_FILE, "sys_param")
        if msys is None:
            print("[ERROR] meas_data.xlsx: sys_param sheet missing.")
            return
        nob = int(msys[0,0])
        nol = int(msys[1,0])
        slack = int(msys[2,0]) - 1
        line_data = read_excel_sheet_safe(MEAS_FILE, "linedata")
        bus_shunt = read_excel_sheet_safe(MEAS_FILE, "bus_shunt")
        # measurement spec: err_spec and flag_error likely defined in input_data in MATLAB.
        # We'll choose a default err_spec similar to MATLAB indexing: [Perr, Qerr, Verr, Ierr, ang_err_deg]
        err_spec = [0.01, 0.01, 0.005, 0.005, 0.1]  # these are placeholders (pu, pu, pu, pu, degrees) - user can adjust
        flag_error = 1  # Gaussian default
        flag_eq = 0  # assume zero injection constraints were loaded
        # parse measurements into list
        z_list = parse_measurements(MEAS_FILE, line_data, nob, nol, err_spec, flag_error, slack, flag_eq)
    else:
        # SIM mode - read loadflow file for busdata/linedata and optionally meas generation (not implemented)
        lf = read_excel_sheet_safe(LOADFLOW_FILE, "busdata")
        line_data = read_excel_sheet_safe(LOADFLOW_FILE, "linedata")
        if lf is None or line_data is None:
            print("[ERROR] Could not read loadflow data.")
            return
        nob = lf.shape[0]
        nol = line_data.shape[0]
        slack = int(np.where(lf[:,9]==1)[0][0]) if np.any(lf[:,9]==1) else 0
        # No measurement file: we will still look for meas_data.xlsx
        if os.path.exists(MEAS_FILE):
            msys = read_excel_sheet_safe(MEAS_FILE, "sys_param")
            if msys is not None:
                z_list = parse_measurements(MEAS_FILE, line_data, nob, nol, [0.01,0.01,0.005,0.005,0.1], 1, slack, 0)
            else:
                z_list = [] # empty
        else:
            z_list = []
        # set some defaults
        err_spec = [0.01, 0.01, 0.005, 0.005, 0.1]
        flag_error = 1
        flag_eq = 0

    # Build Ybus
    Ybus = build_ybus(line_data, nob)
    print("[INFO] Ybus built.")

    # Prepare measurement vector z and R
    if len(z_list) == 0:
        print("[ERROR] No measurements found. Place meas_data.xlsx with bus_meas/line_meas sheets and run again.")
        return
    z_vec = np.array([m['value'] for m in z_list], dtype=float)
    R = np.diag([m['var'] for m in z_list])

    # State initialization
    if USE_LOADFLOW_INIT and os.path.exists(LOADFLOW_FILE):
        busdata = read_excel_sheet_safe(LOADFLOW_FILE, "busdata")
        if busdata is not None:
            # busdata columns as earlier: Column2 Voltage magnitude Column3 angle radian
            V_init = np.ones(nob, dtype=complex)
            for i in range(nob):
                Vm = busdata[i,1] if not np.isnan(busdata[i,1]) else 1.0
                Va = busdata[i,2] if not np.isnan(busdata[i,2]) else 0.0
                V_init[i] = Vm * np.exp(1j * Va)
            x0 = state_from_complex(V_init)
            print("[INFO] Initialized state from loadflow file.")
        else:
            V_init = np.ones(nob, dtype=complex)
            x0 = state_from_complex(V_init)
            print("[INFO] Loadflow file missing, used flat start.")
    else:
        V_init = np.ones(nob, dtype=complex)
        x0 = state_from_complex(V_init)
        print("[INFO] Flat start initialization.")

    nstate = 2 * nob
    P0 = np.eye(nstate) * 1e-3   # initial covariance (small)
    Q_proc = np.eye(nstate) * 1e-6  # process noise (small for static model)

    # Build measurement function and Jacobian
    hfunc, Hfunc = build_measurement_func_and_jac(z_list, Ybus, line_data, nob, nol, slack)

    # Run EKF
    x_est, P_est = ekf_run(x0, P0, Q_proc, z_vec, R, hfunc, Hfunc, max_iter=MAX_ITER, tol=TOL)

    # Postprocess: complex voltages, magnitude, angle
    V_est = complex_from_state(x_est, nob)
    Vm, Va = polar_from_complex(V_est)
    print("\nEstimated bus voltages:")
    for i in range(nob):
        print(f"Bus {i+1:3d}: |V| = {Vm[i]:.6f}  angle(deg) = {math.degrees(Va[i]):.6f}")

    # Save results
    df = pd.DataFrame({
        'bus': np.arange(1, nob+1),
        'V_abs': Vm,
        'V_ang_rad': Va,
        'V_ang_deg': np.degrees(Va)
    })
    outfn = "ekf_state_estimates.csv"
    df.to_csv(outfn, index=False)
    print(f"[INFO] Results written to {outfn}")

if __name__ == "__main__":
    main()
