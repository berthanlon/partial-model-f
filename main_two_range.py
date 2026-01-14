# main_two_range.py
# Neural EKF (train-or-load) + UKF (known f,h,Q,R) + KNet overlay; y>=0 evaluation

from __future__ import annotations
import os, importlib, numpy as np, torch, glob
import matplotlib.pyplot as plt
import parameters as params
importlib.reload(params)
import EKF_helpers as new_sim
from nn_train import train_unsupervised_nonlinear_batched, load_kf_from_checkpoint
from plot_gen import (
    plot_state_timeseries, plot_xy_trajectory, save_states_csv,
    plot_rmse_vs_time, plot_total_rmse_vs_time_xy, plot_est_vs_gt_seq,
)
from metrics import (
    batch_rmse_states_xy, nll_decomposition, nis_fraction_below,
    rmse_time_across_sequences_xy, y_sign_from_gt, apply_y_sign, _to_B_T_nx_lenient, _finite_or_none, _to_B_T_nx
)
import UKF3
from UKF3 import run_ukf
from parameters import mean_ini, P_ini, Q, R, F, to_batched_time_major, h_torch, H_jac_torch, h, make_R_full
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", "CUDA" if torch.cuda.is_available() else "CPU")

TRAIN_MODEL = True # True=train & save; False=load checkpoint if present
EPOCHS, LR = 600, 1e-3
SEQ_FOR_PLOTS = 0
print("n_steps,", params.n_steps)

def _fmt(x: float) -> str:
    try:
        return ("{:.6g}".format(float(x))).rstrip("0").rstrip(".")
    except Exception:
        return str(x)

root_dir = os.getcwd()
run_dir = os.path.join(root_dir, f"KNetFiles_{params.n_steps}")
subdirectory_path = os.path.join(
    run_dir, f"q{_fmt(params.sigma_u)}_r{_fmt(params.sigma_r)}_T{params.n_steps}"
)
os.makedirs(subdirectory_path, exist_ok=True)
print("Run folder:", subdirectory_path)

# point to your pre-generated dataset
data_root = rf"C:\Users\betti\Desktop\Unsupervised_BettiNet\two_radar\Data\KNetFiles_{params.n_steps}\q{_fmt(params.sigma_u)}_r{_fmt(params.sigma_r)}_T{params.n_steps}"
data_path = os.path.join(data_root, f"MCSim_test_data_{params.n_steps}.pt")
print(f"[data] Loading existing dataset from: {data_path}")

(training_measurements, training_gt,
 cv_measurements, cv_gt,
 test_measurements, test_gt) = torch.load(data_path)

# to (B,T,nz)
z_train = to_batched_time_major(training_measurements)
z_cv = to_batched_time_major(cv_measurements)
z_test = to_batched_time_major(test_measurements)
nz = z_train.shape[-1]
nx = params.F.shape[0]
R_full = make_R_full(params.R, nz)

# ---------------- neural KF ----------------
h_torch = h_torch
ckpt_name = f"ekf_batched_q{_fmt(params.sigma_u)}_r{_fmt(params.sigma_r)}_T{params.n_steps}.pth"
ckpt_path = os.path.join(subdirectory_path, ckpt_name)

# Prepare initial conditions
x0 = torch.tensor(params.mean_ini, dtype=torch.float32, device=device)
P0 = torch.tensor(params.P_ini, dtype=torch.float32, device=device)

if TRAIN_MODEL or (not os.path.isfile(ckpt_path)):
    if not TRAIN_MODEL:
        print(f"[model] Checkpoint not found, training instead: {ckpt_path}")
    
    print(f"[model] Training neural KF with Cholesky-based covariance learning...")
    kf, history = train_unsupervised_nonlinear_batched(
        z_seq=z_train, 
        nx=nx, nz=nz, R=R_full,
        h_fn=h_torch,
        x0=x0,
        P0=P0,
        epochs=EPOCHS, 
        lr=LR,
        device=str(device),
        checkpoint_dir=subdirectory_path, 
        checkpoint_name=ckpt_name,
    )
    
    # Plot training loss if history is available
    if history is not None and 'loss' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (NLL)')
        plt.title('Training Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(subdirectory_path, 'training_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"[plot] Saved training loss: {loss_plot_path}")
else:
    print(f"[model] Loading checkpoint: {ckpt_path}")
    kf = load_kf_from_checkpoint(nx, nz, R_full, ckpt_path, device)

# ---------------- neural TEST --------------
P0_OVERRIDE = {0: 0.5, 2: 0.5, 1: 10.0, 3: 10.0}
x_test, P_test, zhat_test, S_test, y_test = new_sim.run_filter_batched(
    kf, z_test,
    init_mode="triangulate",
    P_diag_override=P0_OVERRIDE,
    estimate_vel_from_second=False,
)

# ------------- GT to (B,T,nx) -------------
gt_bt_raw = np.transpose(
    test_gt.detach().cpu().numpy() if hasattr(test_gt, "detach") else np.asarray(test_gt),
    (0, 2, 1)
)
s_y = y_sign_from_gt(gt_bt_raw)
gt_bt = apply_y_sign(gt_bt_raw, s_y)
x_test_pos = apply_y_sign(x_test, s_y)

# === SAVE ALL TEST TRAJECTORIES ===
np.save(os.path.join(subdirectory_path, "test_ground_truth.npy"), gt_bt)
np.save(os.path.join(subdirectory_path, "test_neural_estimates.npy"), x_test_pos)
print("[save] Saved ground truth and neural estimates as .npy")

# ---------------- metrics (neural quick print) ----------
rmse_xy = batch_rmse_states_xy(x_test_pos, gt_bt, reflect_y=False)
rmseT_xy, overall_xy = rmse_time_across_sequences_xy(x_test_pos, gt_bt, reflect_y=False)
print("[Neural y>=0] per-dim RMSE(x,y):", rmse_xy.tolist(), "| total:", overall_xy)
mean_logdet, mean_nis, nis_vals = nll_decomposition(S_test, y_test)
print("mean log|S|:", mean_logdet, "| mean NIS:", mean_nis,
      "| frac<95%:", nis_fraction_below(nis_vals, nz=y_test.shape[-1], alpha=0.95))

# ---------------- UKF (known model) ----------------
print("=== Running UKF3 with the true measurement function (known f,h,Q,R) ===")
ukf_known = UKF3.UnscentedKalmanFilter(
    mean_ini, P_ini,
    Q, R,
    F,
    new_sim.h
)
state_histories_ukf3_known_fhQR = run_ukf(
    ukf_known,
    test_measurements,
    test_gt,
    mean_ini, P_ini,
    device
)
ukf_bt_raw = _to_B_T_nx(state_histories_ukf3_known_fhQR, params.n_steps, nx)
ukf_bt_pos = apply_y_sign(ukf_bt_raw, s_y)

# === SAVE UKF ESTIMATES ===
np.save(os.path.join(subdirectory_path, "test_ukf_estimates.npy"), ukf_bt_pos)
print("[save] Saved UKF estimates as .npy")

# ---------------- Plotting ----------------
# Per-sequence plots
seq = SEQ_FOR_PLOTS
path_ts = os.path.join(subdirectory_path, f"state_timeseries_seq{seq}_T{params.n_steps}.png")
plot_state_timeseries(x_test_pos[seq], P_test[seq], gt_bt[seq], path_ts)
print("[plot] Saved:", path_ts)

path_trj = os.path.join(subdirectory_path, f"xy_trajectory_seq{seq}_T{params.n_steps}.png")
plot_xy_trajectory(x_test_pos[seq], gt_bt[seq], path_trj,
                   radars=((0, 0), (new_sim.RADAR_BASELINE, 0)))
print("[plot] Saved:", path_trj)

csv_path = os.path.join(subdirectory_path, f"states_vs_gt_seq{seq}_T{params.n_steps}.csv")
save_states_csv(x_test_pos[seq], gt_bt[seq], csv_path)
print("[save] Saved CSV:", csv_path)

# All sequences XY trajectories
all_xy_dir = os.path.join(subdirectory_path, "all_xy_trajectories")
os.makedirs(all_xy_dir, exist_ok=True)
B = x_test_pos.shape[0]
for seq in range(B):
    path_trj_seq = os.path.join(all_xy_dir, f"xy_trajectory_seq{seq}_T{params.n_steps}.png")
    plot_xy_trajectory(x_test_pos[seq], gt_bt[seq], path_trj_seq,
                       radars=((0, 0), (new_sim.RADAR_BASELINE, 0)))
print(f"[plot] Saved XY trajectories for all {B} sequences in:", all_xy_dir)

# RMSE vs time plots
p_panels = os.path.join(subdirectory_path, f"rmse_vs_time_T{params.n_steps}.png")
plot_rmse_vs_time(x_test_pos, gt_bt, ukf_bt_pos, p_panels)
print("[plot] Saved:", p_panels)

p_total = os.path.join(subdirectory_path, f"rmse_TOTAL_vs_time_XY_T{params.n_steps}.png")
plot_total_rmse_vs_time_xy(x_test_pos, gt_bt, ukf_bt_pos, p_total)
print("[plot] Saved:", p_total)

# Est vs GT
p_estgt = os.path.join(subdirectory_path, f"est_vs_gt_seq{SEQ_FOR_PLOTS}_T{params.n_steps}.png")
plot_est_vs_gt_seq(x_test_pos, gt_bt, SEQ_FOR_PLOTS, p_estgt,
                   keep_idx=(0, 2), names=("x", "y"))
print("[plot] Saved:", p_estgt)

print("All done – figures and .npy files saved in:", subdirectory_path)