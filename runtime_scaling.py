"""
Runtime and scalability study (Reviewer 1, Comment 5).

Measures the wall-clock runtime and detailed iteration statistics of the
proposed BCD / penalty-SCA / power-allocation algorithm as the number of IRS
elements L and the number of users M increase.

The algorithm is a self-contained copy of the AR_IRS class from 1.py
(BCD over {IRS phase via penalty-SDP, UAV position via SCA, power via
convex optimization}). It is instrumented to record, per block, the solver
calls, inner SCA iterations, and penalty updates, plus the wall-clock time.

The channel model (vector, N antennas, L IRS elements) and global parameters
match the paper's main simulation. Runtime excludes data generation and
Monte Carlo evaluation.
"""

import time
import os
import json
import numpy as np
import cvxpy as cp
import mosek
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Global system parameters (identical to 1.py)
# ----------------------------------------------------------------------
N = 2                   # HAP antennas
rho0 = 0.001
K1 = 0.0                # direct HAP-GMD link blocked -> pure NLoS (kappa1 = 0)
K2 = 10.0               # Rician factor of the AIRS-GMD LoS link (10 dB = 10 linear)
sigmma = 10.0 ** (-11)   # noise power -80 dBm = 1e-11 W
d = 0.05                # element spacing (lambda/2)
wavelength = 0.1        # carrier wavelength
L_k = 5
Z_min = 30.0            # minimum UAV altitude
Z_max = 60.0            # maximum UAV altitude

np.random.seed(0)
s_ki = np.exp(1j * 2 * np.pi * d / wavelength * np.random.rand(N))
s_ki = s_ki / np.linalg.norm(s_ki)

# GMDs distributed over a ~350 m^2 region ~100 m from the HAP (direct link blocked).
positions = np.array([
    [100.0,   0.00, 0.0],
    [106.0,   1.00, 0.0],
    [104.0,  -1.25, 0.0],
    [ 97.0,   1.50, 0.0],
    [109.0,  -0.50, 0.0],
    [ 93.0,   1.25, 0.0],
    [101.0,  -2.00, 0.0],
    [107.0,  -1.50, 0.0],
    [ 95.0,  -1.00, 0.0],
    [103.0,   2.00, 0.0],
    [ 99.0,   0.50, 0.0],
    [108.0,   0.75, 0.0],
    [ 92.0,  -1.75, 0.0],
    [105.0,   1.75, 0.0],
])

W_s = np.array([0.0, 0.0, 20.0])   # HAP position
kappa1 = K1 * rho0 / (K1 + 1.0)
kappa2 = K2 * rho0 / (K2 + 1.0)

# Numerical rescaling: multiply every channel gain and the noise by a common
# factor so the SDP/SCA solvers see O(1) numbers. SINR = P*g/sigma^2 and hence the
# achievable rate tau (nats/s) are unchanged; only the conditioning is improved.
GAIN_SCALE = 1e11
sigmma = sigmma * GAIN_SCALE  # 1e-11 W -> 1.0


class AR_IRS:
    def __init__(self):
        self.solver_calls = 0
        self.inner_irs = []          # inner SCA iterations per IRS block call
        self.inner_q = []            # inner SCA iterations per UAV block call
        self.penalty_updates = 0     # number of penalty-coefficient increases
        self.irs_time = 0.0
        self.q_time = 0.0
        self.p_time = 0.0
        self.bcd_iters = 0

    def initialize_Theta(self, L, rng=None):
        if rng is None:
            psi = np.random.uniform(0, 2 * np.pi, L)
        else:
            psi = rng.uniform(0, 2 * np.pi, L)
        return np.diag(np.exp(1j * psi))

    def theta_to_Phi(self, Theta, L):
        theta = np.diagonal(Theta)
        breve_theta = np.append(theta, 1.0 + 0j)
        return np.outer(breve_theta, np.conj(breve_theta))

    def recover_Theta_from_Phi(self, Phi, L):
        Phi = 0.5 * (Phi + Phi.conj().T)
        eigvals, eigvecs = np.linalg.eigh(Phi)
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]
        breve_theta = v if abs(v[-1]) < 1e-12 else v / v[-1]
        theta = np.exp(1j * np.angle(breve_theta[:L]))
        return np.diag(theta)

    def rank_one_metrics(self, Phi):
        Phi = 0.5 * (Phi + Phi.conj().T)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(Phi)))[::-1]
        eigvals = np.maximum(eigvals, 0)
        eig_sum = np.sum(eigvals) + 1e-12
        return {
            "lambda1_ratio": eigvals[0] / eig_sum,
            "rank_error": (eig_sum - eigvals[0]) / eig_sum,
        }

    def bcd_optimization(self, max_iter=10, L=20, W=1e6, M=5, P_HAP=30,
                         beta1=3.5, beta2=2.0, seed=None, inner_max=30,
                         opt_theta=True, opt_Q=True, opt_P=True, Q_init=None,
                         P_init=None):
        rng = np.random.default_rng(seed)
        if Q_init is not None:
            Q = np.array(Q_init, dtype=float)
        else:
            Q = np.array([np.mean(positions[:M, 0]), np.mean(positions[:M, 1]),
                          (Z_min + Z_max) / 2.0])
        Theta = self.initialize_Theta(L, rng=rng)

        if P_init is not None:
            P = np.array(P_init, dtype=float)
        else:
            first = 0.9 * (10 ** (P_HAP / 10 - 3))
            rest = 0.1 * (10 ** (P_HAP / 10 - 3)) / (M - 1)
            P = np.append([first], [rest] * (M - 1))

        X_prev = {
            "u": np.full(M, 55.0), "l": np.full(M, 30.0),
            "uu": 120.0, "ll": 100.0,
            "Omega_underline": np.full(M, 0.0001),
            "Omega_bar": np.full(M, 0.0001),
        }

        tau_list = []
        for iteration in range(max_iter):
            if opt_theta:
                t0 = time.perf_counter()
                Theta, tau, Phi_final, rank_info = self.optimize_pi_Theta(
                    Q, Theta, P, L, W, M, beta1, beta2, max_iter=inner_max)
                self.irs_time += time.perf_counter() - t0

            if opt_Q:
                t0 = time.perf_counter()
                Q, X_prev, tau = self.optimize_Q(
                    X_prev, Q, P, Theta, L, W, M, beta1, beta2, max_iter=inner_max)
                self.q_time += time.perf_counter() - t0

            if opt_P:
                t0 = time.perf_counter()
                P, tau = self.optimize_P(Q, P, Theta, L, P_HAP, W, M, beta1, beta2)
                self.p_time += time.perf_counter() - t0

            tau_list.append(float(tau))
            self.bcd_iters += 1

        self.Q = np.array(Q, dtype=float)
        self.Theta = np.array(Theta, dtype=complex)
        self.P = np.array(P, dtype=float)
        return tau_list

    def optimize_pi_Theta(self, Q, Theta, P, L, W, M, beta1, beta2,
                          max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        Delta = 1e-3
        alpha_Delta = 1.5
        Delta_max = 1e3
        tau_k_i = L * rho0 * (rho0 - kappa2)
        W_k = positions[:M]
        Phi_final = self.theta_to_Phi(Theta, L)

        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)

        while t < max_iter:
            Phi = self.theta_to_Phi(Theta, L) if t == 0 else Phi_final.copy()
            Phi = 0.5 * (Phi + Phi.conj().T)

            tau_var = cp.Variable()
            Phi_real_var = cp.Variable((L + 1, L + 1))
            Phi_image_var = cp.Variable((L + 1, L + 1))
            Phi_var = Phi_real_var + 1j * Phi_image_var

            constraints = [Phi_real_var == Phi_real_var.T,
                           Phi_image_var == -Phi_image_var.T,
                           Phi_var >> 0]

            for i in range(M):
                gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
                A_i = (gamma_k_i + tau_k_i / (
                    (np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - W_s) ** 2))) * GAIN_SCALE

                gain = self.compute_channel_gain(Q, Theta, i, L, W_k, beta1, beta2)
                interference = 0.0
                for j in range(i + 1, M):
                    interference += P[j] * gain
                interference += sigmma

                SINR_tilde = (gain * P[i]) / interference
                epsilon_i = SINR_tilde / (1 + SINR_tilde)
                eta_i = np.log(1 + SINR_tilde) - epsilon_i * np.log(SINR_tilde)

                phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
                hat_s_i = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
                g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
                g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
                hat_h_i = np.sqrt(kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
                    -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
                G = np.sqrt(rho0 / np.linalg.norm(Q - W_s) ** 2) * g_bar_k * g_title_k

                diag_h_i = np.diagflat(np.conj(hat_h_i.T))
                X_i = np.vstack([diag_h_i @ G, np.conj(hat_s_i.T)]) * np.sqrt(GAIN_SCALE)

                p_i = np.sqrt(P[i]) * np.sqrt(np.conj(P[i]))
                Y_i = np.conj(X_i.T) @ Phi_var @ X_i

                f_phi = cp.log(cp.real(cp.trace(p_i * Y_i)) + A_i * P[i])
                g_Phi = cp.log(sum(cp.real(
                    cp.trace(np.sqrt(P[j]) * np.sqrt(np.conj(P[j].T)) * np.conj(X_i.T) @ Phi @ X_i) + A_i * P[j])
                    for j in range(i + 1, M)) + sigmma)

                numerator = sum(
                    np.sqrt(P[j]) * np.sqrt(np.conj(P[j])) * X_i @ np.conj(X_i.T)
                    for j in range(i + 1, M))
                denominator = sum(
                    np.trace(np.sqrt(P[j]) * np.sqrt(np.conj(P[j])) * np.conj(X_i.T) @ Phi @ X_i + A_i * P[j])
                    for j in range(i + 1, M)) + sigmma
                grad_g = numerator / denominator
                if i < M - 1:
                    tilde_g_phi = cp.real(cp.trace(grad_g @ (Phi_var - Phi))) + g_Phi
                else:
                    tilde_g_phi = cp.real(cp.trace(grad_g * (Phi_var - Phi))) + g_Phi

                constraints.append(epsilon_i * (f_phi - tilde_g_phi) + eta_i >= tau_var / W)

            constraints.append(cp.diag(Phi_var) == 1)

            eigvals, eigvecs = np.linalg.eigh(Phi)
            idx_max = np.argmax(eigvals)
            lambda_max = np.real(eigvals[idx_max])
            u_max = eigvecs[:, idx_max].reshape(-1, 1)
            grad_B = u_max @ u_max.conj().T
            tilde_B_phi = lambda_max + cp.real(cp.trace(grad_B.conj().T @ (Phi_var - Phi)))
            # 核范数 = trace = L+1（diag(Phi)==1 且 Phi >> 0），用常数替换以去掉冗余 PSD 锥
            rank_penalty = (L + 1) - tilde_B_phi

            objective = cp.Minimize(Delta * rank_penalty - tau_var)
            problem = cp.Problem(objective, constraints)
            self.solver_calls += 1
            problem.solve(solver=cp.MOSEK, mosek_params={
                mosek.dparam.intpnt_co_tol_rel_gap: 1e-4})

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                break

            Phi_final = 0.5 * (Phi_var.value + Phi_var.value.conj().T)
            Theta = self.recover_Theta_from_Phi(Phi_final, L)

            if Delta < Delta_max:
                Delta = min(alpha_Delta * Delta, Delta_max)
                self.penalty_updates += 1

            if t != 0 and abs(tau_var.value - tau) < tol:
                tau = tau_var.value
                t += 1
                break
            tau = tau_var.value
            t += 1

        self.inner_irs.append(t)
        return Theta, tau, Phi_final, self.rank_one_metrics(Phi_final)

    def compute_channel_gain(self, Q, Theta, i, L, W_k, beta1, beta2):
        gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
        tau_k_i = L * rho0 * (rho0 - kappa2)

        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)
        phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
        hat_s_ki = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
        g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
        g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
        hat_h_k_i = np.sqrt(kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
            -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
        G_k = np.sqrt(rho0 / np.linalg.norm(Q - W_s) ** 2) * g_bar_k * g_title_k

        gain = np.conj(hat_s_ki.T) + np.conj(hat_h_k_i.T) @ Theta @ G_k
        gain = np.linalg.norm(gain) ** 2
        effective_gain = (gain + gamma_k_i + tau_k_i / (
            (np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - W_s) ** 2)))
        return effective_gain * GAIN_SCALE

    def compute_ABC(self, Theta, Q, i, L, W_k, beta1, beta2):
        q_k = Q
        w_ki = W_k[i]
        phi_k = (q_k[1] - W_s[1]) / np.linalg.norm(q_k - W_s)
        psi_k = (q_k[2] - W_s[2]) / np.linalg.norm(q_k - W_s)
        phi_ki = (w_ki[1] - q_k[1]) / np.linalg.norm(q_k - w_ki)
        dist_ki = np.linalg.norm(W_s - w_ki)

        gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - w_ki) ** beta1)
        tau_ki = L * rho0 * (rho0 - kappa2)
        A = kappa1 / (dist_ki ** beta1)

        s_bar_ki = s_ki.reshape(-1, 1)
        h_bar_ki = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
        g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
        g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)

        gain = np.conj(h_bar_ki.T) @ Theta @ g_bar_k @ g_title_k
        gain = np.linalg.norm(gain, ord=2) ** 2
        B = rho0 * kappa2 * gain + tau_ki
        C = 2 * np.real(np.sqrt(rho0 * kappa2 * A) * np.conj(h_bar_ki.T) @ Theta @ g_bar_k @ g_title_k @ s_bar_ki)
        A += gamma_k_i
        return A * GAIN_SCALE, B * GAIN_SCALE, C * GAIN_SCALE

    def optimize_Q(self, X_prev, Q, P, Theta, L, W, M, beta1, beta2,
                   max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        W_k = positions[:M]

        while t < max_iter:
            Q_prev = np.asarray(Q, dtype=float)
            u_prev = np.maximum(np.asarray(X_prev["u"], dtype=float), 1e-3)
            uu_prev = max(float(X_prev["uu"]), 1e-3)
            Omega_prev = np.maximum(np.asarray(X_prev["Omega_underline"], dtype=float), 1e-12)

            Q_var = cp.Variable(3)
            u_var = cp.Variable(M)
            uu_var = cp.Variable()
            Omega_underline_var = cp.Variable(M)
            tau_var = cp.Variable()

            objective = cp.Maximize(tau_var)
            constraints = [u_var >= 1e-3, uu_var >= 1e-3, Omega_underline_var >= 1e-12,
                           Q_var[2] >= Z_min, Q_var[2] <= Z_max,
                           Q_var[0] >= 85, Q_var[0] <= 115,
                           Q_var[1] >= -15, Q_var[1] <= 15]

            for i in range(M):
                inter_power = sum(P[i + 1:M])
                total_power = sum(P[i:M])
                title_i_n = np.log(Omega_prev[i] * inter_power + sigmma) + (
                    inter_power * (Omega_underline_var[i] - Omega_prev[i]) /
                    (Omega_prev[i] * inter_power + sigmma))
                constraints.append(
                    cp.log(Omega_underline_var[i] * total_power + sigmma)
                    >= (tau_var / W) + title_i_n)

            for i in range(M):
                constraints.append(
                    u_prev[i] ** 2 + 2 * u_prev[i] * (u_var[i] - u_prev[i])
                    >= cp.norm(Q_var - W_k[i]) ** 2)
            constraints.append(
                uu_prev ** 2 + 2 * uu_prev * (uu_var - uu_prev)
                >= cp.norm(Q_var - W_s) ** 2)

            for i in range(M):
                f_ki_lb = (uu_prev ** (-2) * u_prev[i] ** (-beta2)
                           - 2 * uu_prev ** (-3) * u_prev[i] ** (-beta2) * (uu_var - uu_prev)
                           - beta2 * uu_prev ** (-2) * u_prev[i] ** ((-beta2) - 1) * (u_var[i] - u_prev[i]))
                g_ki_lb = (uu_prev ** (-1) * u_prev[i] ** (-beta2 / 2)
                           - uu_prev ** (-2) * u_prev[i] ** (-beta2 / 2) * (uu_var - uu_prev)
                           - (beta2 / 2) * uu_prev ** (-1) * u_prev[i] ** ((-beta2 / 2) - 1) * (u_var[i] - u_prev[i]))
                A, B, C = self.compute_ABC(Theta, Q_prev, i, L, W_k, beta1, beta2)
                g_ki = cp.exp(-cp.log(uu_var) - (beta2 / 2) * cp.log(u_var[i]))
                if C >= 0:
                    lb = A + B * f_ki_lb + abs(C) * g_ki_lb
                else:
                    lb = A + B * f_ki_lb - abs(C) * g_ki
                constraints.append(Omega_underline_var[i] <= lb)

            prob = cp.Problem(objective, constraints)
            self.solver_calls += 1
            try:
                prob.solve(solver=cp.MOSEK)
            except Exception as e:
                break

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                break

            if (u_var.value is None or uu_var.value is None
                    or Omega_underline_var.value is None or Q_var.value is None
                    or tau_var.value is None):
                break

            new_u = np.asarray(u_var.value, dtype=float)
            new_uu = float(uu_var.value)
            new_omega = np.asarray(Omega_underline_var.value, dtype=float)
            new_Q = np.asarray(Q_var.value, dtype=float)
            new_tau = float(tau_var.value)

            if (not np.all(np.isfinite(new_u)) or not np.isfinite(new_uu)
                    or not np.all(np.isfinite(new_omega))
                    or not np.all(np.isfinite(new_Q)) or not np.isfinite(new_tau)):
                break

            X_prev["u"] = np.maximum(new_u, 1e-3)
            X_prev["uu"] = max(new_uu, 1e-3)
            X_prev["Omega_underline"] = np.maximum(new_omega, 1e-12)
            Q = new_Q

            if t != 0 and abs(new_tau - tau) < tol:
                tau = new_tau
                t += 1
                break
            tau = new_tau
            t += 1

        self.inner_q.append(t)
        return Q, X_prev, tau

    def optimize_P(self, Q, P, Theta, L, P_HAP, W, M, beta1, beta2):
        P_H = 10 ** (P_HAP / 10 - 3)
        W_k = positions[:M]
        p = cp.Variable(M)
        tau_var = cp.Variable()
        constraints = []

        for i in range(M):
            gain = self.compute_channel_gain(Q, Theta, i, L, W_k, beta1, beta2)
            interference = 0.0
            for j in range(i + 1, M):
                interference += P[j] * gain
            interference += sigmma

            SINR_tilde = (gain * P[i]) / interference
            epsilon = SINR_tilde / (1 + SINR_tilde)
            eta = np.log(1 + SINR_tilde) - epsilon * np.log(SINR_tilde)
            constant = cp.log(sigmma / gain)
            a = cp.hstack([p[i + 1:M], constant])
            if i <= M - 2:
                SINR_term = cp.log(gain) + p[i] - cp.log(gain) - cp.log_sum_exp(a)
            else:
                SINR_term = cp.log(gain) + p[i] - cp.log(sigmma)
            constraints.append(epsilon * SINR_term + eta >= tau_var / W)

        constraints.append(sum(cp.exp(p[i]) for i in range(M)) <= P_H)
        objective = cp.Maximize(tau_var)
        prob = cp.Problem(objective, constraints)
        self.solver_calls += 1
        prob.solve(cp.CLARABEL)

        P = np.array([np.exp(p[i].value) for i in range(M)])
        return P, float(tau_var.value)


def n_vars_irs(L):
    return 2 * (L + 1) ** 2 + 1


def n_vars_uav(M):
    return 2 * M + 5


def n_vars_power(M):
    return M + 1


# ----------------------------------------------------------------------
# Run the sweep
# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUTER = 5
    INNER = 30
    REPS = 3
    L_LIST = [8, 16, 24, 32]
    M_LIST = [2, 3, 4, 5, 6]

    results = {"L": {}, "M": {}}

    def run_once(L, M):
        irs = AR_IRS()
        t0 = time.perf_counter()
        tau_list = irs.bcd_optimization(max_iter=OUTER, L=L, W=1e6, M=M, P_HAP=30,
                                        seed=0, inner_max=INNER)
        wall = time.perf_counter() - t0
        return {
            "wall_s": wall,
            "irs_time_s": irs.irs_time,
            "q_time_s": irs.q_time,
            "p_time_s": irs.p_time,
            "bcd_iters": irs.bcd_iters,
            "solver_calls": irs.solver_calls,
            "penalty_updates": irs.penalty_updates,
            "irs_inner_iters": irs.inner_irs,
            "q_inner_iters": irs.inner_q,
            "n_vars_irs": n_vars_irs(L),
            "n_vars_uav": n_vars_uav(M),
            "n_vars_power": n_vars_power(M),
            "final_tau": tau_list[-1] if tau_list else None,
        }

    # warm-up
    run_once(L_LIST[0], M_LIST[0])

    for L in L_LIST:
        M = 4
        recs = [run_once(L, M) for _ in range(REPS)]
        wall = [r["wall_s"] for r in recs]
        results["L"][str(L)] = {
            "M": M,
            "mean_wall_s": float(np.mean(wall)),
            "std_wall_s": float(np.std(wall)),
            "median_wall_s": float(np.median(wall)),
            "mean_irs_time_s": float(np.mean([r["irs_time_s"] for r in recs])),
            "mean_q_time_s": float(np.mean([r["q_time_s"] for r in recs])),
            "mean_p_time_s": float(np.mean([r["p_time_s"] for r in recs])),
            "solver_calls": recs[0]["solver_calls"],
            "penalty_updates": recs[0]["penalty_updates"],
            "mean_irs_inner_iters": float(np.mean([sum(r["irs_inner_iters"]) for r in recs])),
            "mean_q_inner_iters": float(np.mean([sum(r["q_inner_iters"]) for r in recs])),
            "n_vars_irs": recs[0]["n_vars_irs"],
            "n_vars_uav": recs[0]["n_vars_uav"],
            "n_vars_power": recs[0]["n_vars_power"],
        }
        print(f"[L={L}] wall={np.mean(wall):.3f}s irs={results['L'][str(L)]['mean_irs_time_s']:.3f} "
              f"q={results['L'][str(L)]['mean_q_time_s']:.3f} p={results['L'][str(L)]['mean_p_time_s']:.3f} "
              f"solver_calls={recs[0]['solver_calls']}")

    for M in M_LIST:
        L = 16
        recs = [run_once(L, M) for _ in range(REPS)]
        wall = [r["wall_s"] for r in recs]
        results["M"][str(M)] = {
            "L": L,
            "mean_wall_s": float(np.mean(wall)),
            "std_wall_s": float(np.std(wall)),
            "median_wall_s": float(np.median(wall)),
            "mean_irs_time_s": float(np.mean([r["irs_time_s"] for r in recs])),
            "mean_q_time_s": float(np.mean([r["q_time_s"] for r in recs])),
            "mean_p_time_s": float(np.mean([r["p_time_s"] for r in recs])),
            "solver_calls": recs[0]["solver_calls"],
            "penalty_updates": recs[0]["penalty_updates"],
            "mean_irs_inner_iters": float(np.mean([sum(r["irs_inner_iters"]) for r in recs])),
            "mean_q_inner_iters": float(np.mean([sum(r["q_inner_iters"]) for r in recs])),
            "n_vars_irs": recs[0]["n_vars_irs"],
            "n_vars_uav": recs[0]["n_vars_uav"],
            "n_vars_power": recs[0]["n_vars_power"],
        }
        print(f"[M={M}] wall={np.mean(wall):.3f}s solver_calls={recs[0]['solver_calls']}")

    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runtime_scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))

    Ls = [int(k) for k in results["L"].keys()]
    axes[0].plot(Ls, [results["L"][str(l)]["mean_wall_s"] for l in Ls], "o-", color="#1f77b4")
    axes[0].set_xlabel("Number of IRS elements $L$")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("(a) Scaling with $L$ ($M=4$)")
    axes[0].grid(linestyle="--", alpha=0.4)

    Ms = [int(k) for k in results["M"].keys()]
    axes[1].plot(Ms, [results["M"][str(m)]["mean_wall_s"] for m in Ms], "o-", color="#d62728")
    axes[1].set_xlabel("Number of users $M$")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("(b) Scaling with $M$ ($L=16$)")
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig("fig_runtime.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_runtime.pdf")
    print("Done.")
