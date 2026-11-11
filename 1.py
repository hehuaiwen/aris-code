import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
import mosek
import json

from new_func.myfunc import *


class AR_IRS:
    def initialize_Theta(self, L, rng=None):
        if rng is None:
            psi = np.random.uniform(0, 2 * np.pi, L)
        else:
            psi = rng.uniform(0, 2 * np.pi, L)

        theta = np.exp(1j * psi)
        Theta = np.diag(theta)
        return Theta

    def theta_to_Phi(self, Theta, L):
        theta = np.diagonal(Theta)
        breve_theta = np.append(theta, 1.0 + 0j)
        Phi = np.outer(breve_theta, np.conj(breve_theta))
        return Phi

    def recover_Theta_from_Phi(self, Phi, L):
        # 保证 Hermitian
        Phi = 0.5 * (Phi + Phi.conj().T)

        eigvals, eigvecs = np.linalg.eigh(Phi)
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]

        # 用最后一个元素归一化，恢复 [theta; 1]
        if abs(v[-1]) < 1e-12:
            breve_theta = v
        else:
            breve_theta = v / v[-1]

        theta = breve_theta[:L]
        theta = np.exp(1j * np.angle(theta))  # 强制单位模

        Theta = np.diag(theta)
        return Theta

    def rank_one_metrics(self, Phi):
        Phi = 0.5 * (Phi + Phi.conj().T)
        eigvals = np.linalg.eigvalsh(Phi)
        eigvals = np.sort(np.real(eigvals))[::-1]
        eigvals = np.maximum(eigvals, 0)

        eig_sum = np.sum(eigvals) + 1e-12
        lambda_ratio = eigvals[0] / eig_sum

        rank_error = (eig_sum - eigvals[0]) / eig_sum

        return {
            "eigvals": eigvals,
            "lambda1_ratio": lambda_ratio,
            "rank_error": rank_error,
        }

    # BCD 迭代优化框架
    def bcd_optimization(self, max_iter=10, L=20, W=1e6, M=5, P_HAP=30, beta1=3.5, beta2=2.0, seed=None):
        Q_list = []
        P_list = []
        tau_list = []
        Phi_list = []
        rank_info_list = []
        rng = np.random.default_rng(seed)

        # 初始化变量 (UAV starts at the GMD cluster centroid)
        Q = np.array([np.mean(positions[:M, 0]), np.mean(positions[:M, 1]),
                      (Z_min + Z_max) / 2])
        Theta = AR_IRS.initialize_Theta(self, L, rng=rng)

        # 构造 P 数组
        first_element = 0.9 * (10 ** (P_HAP / 10 - 3))
        remaining_element = 0.1 * (10 ** (P_HAP / 10 - 3)) / (M - 1)
        P = np.append([first_element], [remaining_element] * (M - 1))

        X_prev = {
            "u": np.full(M, 55),
            "l": np.full(M, 30),
            "uu": 120,
            "ll": 100,
            "Omega_underline": np.full(M, 0.0001),
            "Omega_bar": np.full(M, 0.0001),
        }

        for iteration in range(max_iter):
            # Step 1: 优化 pi 和 Theta
            Theta, tau,  Phi_final, rank_info = AR_IRS.optimize_pi_Theta(self, Q, Theta, P, L, W, M, beta1, beta2)

            # Step 2: 固定 A 和 P，优化 Q
            Q, X_prev, tau = AR_IRS.optimize_Q(self, X_prev, Q, P, Theta, L, W, M, beta1, beta2)

            # Step 3: 固定 Q，优化 A 和 P
            P, tau = AR_IRS.optimize_P(self, Q, P, Theta, L, P_HAP, W, M, beta1, beta2)
            print("t", tau)

            Q_list.append(Q)
            P_list.append(P)
            tau_list.append(tau)
            Phi_list.append(Phi_final)
            rank_info_list.append(rank_info)

        return P_list, tau_list, Phi_list, rank_info_list

    def optimize_pi_Theta(self, Q, Theta, P, L, W, M, beta1, beta2, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        # 设定参数
        Delta = 1e-3
        alpha_Delta = 1.5
        Delta_max = 1e3  # 惩罚系数
        tau_k_i = L * rho0 * (rho0 - kappa2)
        W_k = positions[:M]
        Phi_final = self.theta_to_Phi(Theta, L)
        rank_info = self.rank_one_metrics(Phi_final)

        # 初始化功率分配变量

        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoA
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoD

        while t < max_iter:
            if t == 0:
                Phi = self.theta_to_Phi(Theta, L)
            else:
                Phi = Phi_final.copy()

            Phi = 0.5 * (Phi + Phi.conj().T)

            # 定义变量
            tau_var = cp.Variable()  # 目标变量 tau
            Phi_real_var = cp.Variable((L + 1, L + 1))
            Phi_image_var = cp.Variable((L + 1, L + 1))
            Phi_var = Phi_real_var + 1j * Phi_image_var  # 反射矩阵，要求半正定

            # 定义约束
            constraints = []

            constraints.append(Phi_real_var == Phi_real_var.T)  # 实部对称
            constraints.append(Phi_image_var == -Phi_image_var.T)  # 虚部反对称

            constraints.append(Phi_var >> 0)
            a = []
            # 速率约束
            for i in range(M):
                gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
                A_i = gamma_k_i + tau_k_i / (
                        (np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - W_s) ** 2))

                gain = AR_IRS.compute_channel_gain(self, Q, Theta, i, L, W_k, beta1, beta2)
                a.append(gain)
                # 计算干扰
                interference = 0
                for j in range(i + 1, M):
                    interference += P[j] * gain
                interference += sigmma  # 加上干扰项

                SINR_tilde = (gain * P[i]) / interference
                epsilon_i = SINR_tilde / (1 + SINR_tilde)
                eta_i = np.log(1 + SINR_tilde) - epsilon_i * np.log(SINR_tilde)

                # 计算X_i
                phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
                hat_s_i = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
                g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
                g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
                hat_h_i = np.sqrt(kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
                    -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
                G = np.sqrt(rho0 / np.linalg.norm(Q - W_s) ** 2) * g_bar_k * g_title_k

                diag_h_i = np.diagflat(np.conj(hat_h_i.T))
                first_part = diag_h_i @ G
                X_i = np.vstack([first_part, np.conj(hat_s_i.T)])


                # 计算P_i,Y_i,Phi
                p_i = np.sqrt(P[i]) * np.sqrt(np.conj(P[i]))
                Y_i = np.conj(X_i.T) @ Phi_var @ X_i

                # 计算f(Phi),g(Phi)
                # 定义 g(Phi) 和 近似的 \widetilde{g}(Phi)
                f_phi = cp.log((cp.real(cp.trace(p_i * Y_i)) + A_i * P[i]))
                g_Phi = cp.log(sum(cp.real(
                    cp.trace(np.sqrt(P[j]) * np.sqrt(np.conj(P[j].T)) * np.conj(X_i.T) @ Phi @ X_i) + A_i * P[j]) for j
                                   in
                                   range(i + 1, M)) + sigmma)

                # 计算梯度
                numerator = sum(
                    np.sqrt(P[j]) * np.sqrt(np.conj(P[j])) * X_i @ np.conj(X_i.T) for j in
                    range(i + 1, M))  # 用户的贡献，P_j 和 X_i 是已知的
                denominator = sum(
                    np.trace(np.sqrt(P[j]) * np.sqrt(np.conj(P[j])) * np.conj(X_i.T) @ Phi @ X_i + A_i * P[j]) for j in
                    range(i + 1, M)) + sigmma
                grad_g =  numerator / denominator
                if i < M - 1:
                    tilde_g_phi = cp.real(cp.trace(grad_g @ (Phi_var - Phi))) + g_Phi
                else:
                    tilde_g_phi = cp.real(cp.trace(grad_g * (Phi_var - Phi))) + g_Phi

                constraints.append(epsilon_i * (f_phi - tilde_g_phi) + eta_i >= tau_var / W)

            constraints.append(cp.diag(Phi_var) == 1)

            # 定义目标函数
            eigvals, eigvecs = np.linalg.eigh(Phi)
            idx_max = np.argmax(eigvals)
            lambda_max = np.real(eigvals[idx_max])
            u_max = eigvecs[:, idx_max].reshape(-1, 1)

            grad_B = u_max @ u_max.conj().T

            tilde_B_phi = lambda_max + cp.real(
                cp.trace(grad_B.conj().T @ (Phi_var - Phi))
            )

            # 核范数 = trace = L+1（diag(Phi)==1 且 Phi >> 0），用常数替换以去掉冗余 PSD 锥
            rank_penalty = (L + 1) - tilde_B_phi

            objective = cp.Minimize(Delta * rank_penalty - tau_var)

            # 定义问题
            problem = cp.Problem(objective, constraints)
            mosek_params = {
                mosek.dparam.intpnt_co_tol_rel_gap: 1e-4  # 设置对偶间隙容忍度为0.0001
            }
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print("[WARN] optimize_pi_Theta solve status:", problem.status)
                break

            Phi_final = Phi_var.value
            Phi_final = 0.5 * (Phi_final + Phi_final.conj().T)

            Theta = self.recover_Theta_from_Phi(Phi_final, L)
            rank_info = self.rank_one_metrics(Phi_final)

            print(
                f"[IRS_RANK] iter={t}, "
                f"Delta={Delta:.3e}, "
                f"tau={tau_var.value:.6f}, "
                f"lambda1_ratio={rank_info['lambda1_ratio']:.6f}, "
                f"rank_error={rank_info['rank_error']:.3e}"
            )

            Delta = min(alpha_Delta * Delta, Delta_max)

            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        return Theta, tau,  Phi_final, rank_info

    def compute_channel_gain(self, Q, Theta, i, L, W_k, beta1, beta2):
        # 计算 γ_{k,i} 和 τ_{k,i}
        gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
        tau_k_i = L * rho0 * (rho0 - kappa2)

        # 计算 LoS 分量
        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoA
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoD
        phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
        hat_s_ki = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
        g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
        g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
        hat_h_k_i = np.sqrt(kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
            -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
        G_k = np.sqrt(rho0 / np.linalg.norm(Q - W_s) ** 2) * g_bar_k * g_title_k

        gain = np.conj(hat_s_ki.T) + np.conj(hat_h_k_i.T) @ Theta @ G_k
        gain = np.linalg.norm(gain) ** 2

        # 计算信道增益
        effective_gain = (gain + gamma_k_i
                          + tau_k_i / ((np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - W_s) ** 2))
                          )
        return effective_gain

    def compute_ABC(self, Theta, Q, i, L, W_k, beta1, beta2):
        # 提取位置
        q_k = Q
        w_ki = W_k[i]

        # 计算 AoA 和 AoD
        phi_k = (q_k[1] - W_s[1]) / np.linalg.norm(q_k - W_s)  # UAV 到 HAP 的 AoA
        psi_k = (q_k[2] - W_s[2]) / np.linalg.norm(q_k - W_s)  # UAV 到 HAP 的 AoD
        phi_ki = (w_ki[1] - q_k[1]) / np.linalg.norm(q_k - w_ki)  # IRS 到用户的 AoD

        # 通道相关变量
        dist_ki = np.linalg.norm(W_s - w_ki)

        # 计算 γ_{k,i} 和 τ_{k,i}
        gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - w_ki) ** beta1)
        tau_ki = L * rho0 * (rho0 - kappa2)
        # 计算 A_{k,i}
        A = kappa1 / (dist_ki ** beta1)

        # IRS 相关矩阵
        s_bar_ki = s_ki.reshape(-1, 1)  # 确保模为1
        h_bar_ki = np.exp(
            -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
        g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
        g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
        # 计算 B_{k,i}
        gain = np.conj(h_bar_ki.T) @ Theta @ g_bar_k @ g_title_k
        gain = np.linalg.norm(gain, ord=2) ** 2
        B = rho0 * kappa2 * gain + tau_ki

        # 计算 C_{k,i}
        C = 2 * np.real(
            np.sqrt(rho0 * kappa2 * A) * np.conj(h_bar_ki.T) @ Theta @ g_bar_k @ g_title_k @ s_bar_ki)
        A += gamma_k_i
        return A, B, C

    def optimize_Q(self, X_prev, Q, P, Theta, L, W, M, beta1, beta2, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        W_k = positions[:M]

        while t < max_iter:
            Q_prev = np.asarray(Q, dtype=float)

            # ================= 安全上一轮 SCA 点 =================
            u_prev = np.maximum(np.asarray(X_prev["u"], dtype=float), 1e-3)
            uu_prev = max(float(X_prev["uu"]), 1e-3)
            Omega_prev = np.maximum(np.asarray(X_prev["Omega_underline"], dtype=float), 1e-12)

            # ================= 定义变量 =================
            Q_var = cp.Variable(3)
            u_var = cp.Variable(M)
            uu_var = cp.Variable()
            Omega_underline_var = cp.Variable(M)
            tau_var = cp.Variable()

            objective = cp.Maximize(tau_var)
            constraints = []

            # ================= 基本数值下界 =================
            constraints += [
                u_var >= 1e-3,
                uu_var >= 1e-3,
                Omega_underline_var >= 1e-12,
            ]

            # 高度约束
            constraints += [
                Q_var[2] >= Z_min,
                Q_var[2] <= Z_max,
            ]

            # UAV horizontal service region covers the GMD cluster (~100 m from HAP)
            constraints += [
                Q_var[0] >= 85,
                Q_var[0] <= 115,
                Q_var[1] >= -15,
                Q_var[1] <= 15,
            ]

            # ================= 2. 最小速率约束 =================
            for i in range(M):
                inter_power = sum(P[i + 1:M])
                total_power = sum(P[i:M])

                title_i_n = np.log(Omega_prev[i] * inter_power + sigmma) + (
                        inter_power * (Omega_underline_var[i] - Omega_prev[i])
                        / (Omega_prev[i] * inter_power + sigmma)
                )

                constraints.append(
                    cp.log(Omega_underline_var[i] * total_power + sigmma)
                    >= (tau_var / W) + title_i_n
                )

            # ================= 3. 用户到 UAV 的距离上界约束 =================
            for i in range(M):
                constraints.append(
                    u_prev[i] ** 2
                    + 2 * u_prev[i] * (u_var[i] - u_prev[i])
                    >= cp.norm(Q_var - W_k[i]) ** 2
                )

            # ================= 5. UAV 到反射器的距离上界约束 =================
            constraints.append(
                uu_prev ** 2
                + 2 * uu_prev * (uu_var - uu_prev)
                >= cp.norm(Q_var - W_s) ** 2
            )

            # ================= 9. 全局信道下界约束 =================
            for i in range(M):
                f_ki_lb = (
                        uu_prev ** (-2) * u_prev[i] ** (-beta2)
                        - 2 * uu_prev ** (-3) * u_prev[i] ** (-beta2) * (uu_var - uu_prev)
                        - beta2 * uu_prev ** (-2) * u_prev[i] ** ((-beta2) - 1)
                        * (u_var[i] - u_prev[i])
                )

                g_ki_lb = (
                        uu_prev ** (-1) * u_prev[i] ** (-beta2 / 2)
                        - uu_prev ** (-2) * u_prev[i] ** (-beta2 / 2) * (uu_var - uu_prev)
                        - (beta2 / 2) * uu_prev ** (-1) * u_prev[i] ** ((-beta2 / 2) - 1)
                        * (u_var[i] - u_prev[i])
                )

                A, B, C = AR_IRS.compute_ABC(
                    self, Theta, Q_prev, i, L, W_k, beta1, beta2
                )

                g_ki = cp.exp(-cp.log(uu_var) - (beta2 / 2) * cp.log(u_var[i]))

                if C >= 0:
                    lb = A + B * f_ki_lb + abs(C) * g_ki_lb
                else:
                    lb = A + B * f_ki_lb - abs(C) * g_ki

                constraints.append(Omega_underline_var[i] <= lb)

            # ================= 求解 =================
            prob = cp.Problem(objective, constraints)

            try:
                prob.solve(solver=cp.MOSEK)
            except Exception as e:
                print("[WARN] optimize_Q solver failed:", e)
                break

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print("[WARN] optimize_Q solve status:", prob.status)
                break

            # ================= 取值前检查 =================
            if (
                    u_var.value is None
                    or uu_var.value is None
                    or Omega_underline_var.value is None
                    or Q_var.value is None
                    or tau_var.value is None
            ):
                print("[WARN] optimize_Q got None value, break")
                break

            new_u = np.asarray(u_var.value, dtype=float)
            new_uu = float(uu_var.value)
            new_omega = np.asarray(Omega_underline_var.value, dtype=float)
            new_Q = np.asarray(Q_var.value, dtype=float)
            new_tau = float(tau_var.value)

            if (
                    not np.all(np.isfinite(new_u))
                    or not np.isfinite(new_uu)
                    or not np.all(np.isfinite(new_omega))
                    or not np.all(np.isfinite(new_Q))
                    or not np.isfinite(new_tau)
            ):
                print("[WARN] optimize_Q got non-finite value, break")
                break

            # ================= 更新上一轮 SCA 点，带 clipping =================
            X_prev["u"] = np.maximum(new_u, 1e-3)
            X_prev["uu"] = max(new_uu, 1e-3)
            X_prev["Omega_underline"] = np.maximum(new_omega, 1e-12)

            # 当前没启用 l / ll / Omega_bar，保持即可
            X_prev["l"] = np.asarray(X_prev["l"], dtype=float)

            Q = new_Q

            # ================= 收敛判断 =================
            if t != 0 and abs(new_tau - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {new_tau}")
                tau = new_tau
                break

            tau = new_tau
            t += 1

        print("优化后的 Q:", Q[0], Q[1], Q[2])
        return Q, X_prev, tau

    def optimize_P(self, Q, P, Theta, L, P_HAP, W, M, beta1, beta2):
        P_H = 10 ** (P_HAP / 10 - 3)
        W_k = positions[:M]

        p = cp.Variable(M)
        tau_var = cp.Variable()

        constraints = []

        # 约束条件
        for i in range(M):
            gain = AR_IRS.compute_channel_gain(self, Q, Theta, i, L, W_k, beta1, beta2)

            # 计算干扰
            interference = 0
            for j in range(i + 1, M):
                interference += P[j] * gain
            interference += sigmma  # 加上干扰项

            SINR_tilde = (gain * P[i]) / interference
            epsilon = SINR_tilde / (1 + SINR_tilde)
            eta = np.log(1 + SINR_tilde) - epsilon * np.log(SINR_tilde)

            constant = cp.log(sigmma / gain)
            a = cp.hstack([p[i + 1:M], constant])

            # 计算凸近似的 SINR_term
            if i <= M - 2:
                SINR_term = cp.log(gain) + p[i] \
                            - cp.log(gain) - cp.log_sum_exp(a)
            else:
                SINR_term = cp.log(gain) + p[i] - cp.log(sigmma)

            # SINR 最小值约束
            constraints.append(epsilon * SINR_term + eta >= tau_var / W)

        # 总功率约束
        constraints.append(sum(cp.exp(p[i]) for i in range(M)) <= P_H)

        # 目标函数
        objective = cp.Maximize(tau_var)

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve(cp.CLARABEL)

        A = []
        for i in range(M):
            gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
            hat_s_ki = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
            gain = np.linalg.norm(hat_s_ki) ** 2 + gamma_k_i

            # 计算干扰
            interference = 0
            for j in range(i + 1, M):
                interference += P[j] * gain
            interference += sigmma  # 加上干扰项

            SINR_tilde = (gain * P[i]) / interference
            epsilon = SINR_tilde / (1 + SINR_tilde)
            eta = np.log(1 + SINR_tilde) - epsilon * np.log(SINR_tilde)

            # 计算凸近似的 SINR_term
            if i <= M - 2:
                SINR_term = np.log(gain) + p[i].value \
                            - np.log(gain) - np.log(sum(np.exp(p[j].value) for j in range(i + 1, M)) + sigmma / gain)
            else:
                SINR_term = np.log(gain) + p[i].value - np.log(sigmma)

            # SINR 最小值约束
            A.append((epsilon * SINR_term + eta) * W)

        P = [np.exp(p[i].value) for i in range(M)]
        P = np.array(P)
        tau = tau_var.value

        return P, tau

# 参数设置
N = 2  # 天线数量
rho0 = 0.001
K1 = 0   # direct HAP-GMD link is blocked -> pure NLoS (kappa1 = 0)
K2 = 10  # Rician factor of the AIRS-GMD LoS link (10 dB = 10 linear)
sigmma = 10 ** (-11)  # noise power -80 dBm = 1e-11 W
d = 0.05  # 元件距离 (lambda/2)
wavelength = 0.1  # 载波长
L_k = 5
Z_min = 30  # 飞行最低高度
Z_max = 60  # 飞行最高高度
# 初始化变量
np.random.seed(1)
s_ki = np.exp(1j * 2 * np.pi * d / wavelength * np.random.rand(N))
s_ki = s_ki / np.linalg.norm(s_ki)

random_numbers = np.random.uniform(1, 2, size=2)

np.random.seed(0)
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

W_s = np.array([0, 0, 20])  # HAP 的位置
# 计算常量值
kappa1 = K1 * rho0 / (K1 + 1)
kappa2 = K2 * rho0 / (K2 + 1)

# 执行 BCD 迭代优化
irs = AR_IRS()

# ================= 多随机初始化 + IRS秩一指标保存 =================
num_seeds = 20          # 调试先用5；论文正式可以改成10或20
rank_threshold = 1e-3
out_dir = "irs_rank_results"
os.makedirs(out_dir, exist_ok=True)

results = []

for seed in range(num_seeds):
    print(f"\n========== RANDOM INIT seed={seed} ==========")

    P_list, tau_list, Phi_list, rank_info_list = irs.bcd_optimization(
        max_iter=10,
        L=20,
        W=1e6,
        M=5,
        P_HAP=30,
        beta1=3.5,
        beta2=2.0,
        seed=seed
    )

    final_tau = float(tau_list[-1])
    final_rank = rank_info_list[-1]

    eigvals = np.real(final_rank["eigvals"])
    eig_sum = np.sum(eigvals) + 1e-12
    eigvals_norm = eigvals / eig_sum

    one_result = {
        "seed": int(seed),
        "final_tau": final_tau,
        "lambda1_ratio": float(final_rank["lambda1_ratio"]),
        "rank_error": float(final_rank["rank_error"]),
        "success": bool(final_rank["rank_error"] <= rank_threshold),

        # 保存前几个归一化特征值，论文画图/写表够用
        "eigvals_all": [float(x) for x in eigvals],
        "eigvals_norm_all": [float(x) for x in eigvals_norm],
        "eigvals_top10": [float(x) for x in eigvals[:10]],
        "eigvals_norm_top10": [float(x) for x in eigvals_norm[:10]],

        # 可选：保存每轮BCD的秩一误差变化
        "rank_error_curve": [
            float(info["rank_error"]) for info in rank_info_list
        ],
        "lambda1_ratio_curve": [
            float(info["lambda1_ratio"]) for info in rank_info_list
        ],
        "tau_curve": [
            float(x) for x in tau_list
        ],
    }

    results.append(one_result)

success_rate = float(np.mean([r["success"] for r in results]))
avg_rank_error = float(np.mean([r["rank_error"] for r in results]))
max_rank_error = float(np.max([r["rank_error"] for r in results]))
avg_lambda1_ratio = float(np.mean([r["lambda1_ratio"] for r in results]))
avg_tau = float(np.mean([r["final_tau"] for r in results]))

summary = {
    "num_seeds": int(num_seeds),
    "rank_threshold": float(rank_threshold),
    "success_rate": success_rate,
    "avg_rank_error": avg_rank_error,
    "max_rank_error": max_rank_error,
    "avg_lambda1_ratio": avg_lambda1_ratio,
    "avg_tau": avg_tau,
}

save_data = {
    "summary": summary,
    "results": results,
}

json_path = os.path.join(out_dir, "irs_rank_random_init_results.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)