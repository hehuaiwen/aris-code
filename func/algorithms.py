import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
import mosek

from new_func.myfunc import *


class AR_IRS:
    def initialize_Theta(self, L):
        psi = np.random.uniform(0, 2 * np.pi, L)
        diagonal_elements = np.exp(1j * psi)
        Theta = np.diag(diagonal_elements)
        return Theta

    # BCD 迭代优化框架
    def bcd_optimization(self, max_iter=20, L=20, W=1, M=5, P_HAP=30):
        Q_list = []
        P_list = []
        tau_list = []
        np.random.seed(1)
        W_k = positions[:M]

        # 初始化变量
        Q = np.random.rand(3)
        Q[2] = (Z_min + Z_max) / 2  # 无人机的 3D 位置
        Theta = AR_IRS.initialize_Theta(self, L)

        # 构造 P 数组
        first_element = 0.9 * (10 ** (P_HAP / 10))
        remaining_element = 0.1 * (10 ** (P_HAP / 10)) / (M - 1)
        P = np.append([first_element], [remaining_element] * (M - 1))

        X_prev = {
            "u": np.full(M, 55),
            "l": np.full(M, 30),
            "uu": 55,
            "ll": 30,
            "Omega_underline": np.full(M, 0.0001),
            "Omega_bar": np.full(M, 0.0001),
        }

        for iteration in range(max_iter):
            # Step 1: 优化 pi 和 Theta
            Theta, tau = AR_IRS.optimize_pi_Theta(self, Q, Theta, P, L, W, M)

            # Step 2: 固定 A 和 P，优化 Q
            Q, X_prev, tau = AR_IRS.optimize_Q(self, X_prev, Q, P, Theta, L, W, M)

            # Step 3: 固定 Q，优化 A 和 P
            P, tau = AR_IRS.optimize_P(self, Q, P, Theta, L, P_HAP, W, M)
            print("t", tau)

            Q_list.append(Q)
            P_list.append(P)
            tau_list.append(tau)

        return Q_list, W_k

    def optimize_pi_Theta(self, Q, Theta, P, L, W, M, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        # 设定参数
        Delta = 10 ** (-5)  # 惩罚系数
        tau_k_i = (rho0 - kappa1) * L * rho0
        W_k = positions[:M]

        # 初始化功率分配变量

        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoA
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoD

        while t < max_iter:
            theta_values = np.diagonal(Theta)
            breve_theta = np.append(theta_values, 1)
            Phi = np.outer(breve_theta, np.conj(breve_theta.T))

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
                gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
                A_i = gamma_k_i + tau_k_i * 10 ** (9) / (
                        (np.linalg.norm(Q - W_k[i]) ** beta1) * (np.linalg.norm(Q - W_s) ** 2))

                gain = AR_IRS.compute_channel_gain(self, Q, Theta, i, L, W_k)
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

                # gain_1 = np.conj(hat_s_i.T) + np.conj(hat_h_i.T) @ Theta @ G
                # gain_1=np.linalg.norm(gain_1) ** 2
                # Y_i_1 = np.trace(np.conj(X_i.T) @ Phi @ X_i)

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
                grad_g = 10 * numerator / denominator
                if i < M - 1:
                    tilde_g_phi = cp.real(cp.trace(grad_g @ (Phi_var - Phi))) + g_Phi
                else:
                    tilde_g_phi = cp.real(cp.trace(grad_g * (Phi_var - Phi))) + g_Phi

                constraints.append(epsilon_i * (f_phi - tilde_g_phi) + eta_i >= tau_var / W)

            constraints.append(cp.diag(Phi_var) == 1)

            # 定义目标函数
            grad_Phi = Phi.T / cp.norm(Phi, 2)
            tilde_B_phi = cp.norm(Phi, 2) + cp.trace((grad_Phi) @ (Phi_var - Phi))
            objective = cp.Minimize(Delta * cp.real((cp.norm(Phi_var, "nuc", axis=None) - tilde_B_phi)) - tau_var)

            # 定义问题
            problem = cp.Problem(objective, constraints)
            mosek_params = {
                mosek.dparam.intpnt_co_tol_rel_gap: 1e-4  # 设置对偶间隙容忍度为0.0001
            }
            problem.solve(mosek_params=mosek_params)

            phi_diagonal_elements = np.diagonal(Phi_var.value)

            # 计算平方根，得到 breve_theta 的元素（不包括额外的1）
            breve_theta_elements = np.sqrt(phi_diagonal_elements[:-1])

            # 构造 Theta 矩阵，这里假设 Theta 是一个对角矩阵
            Theta = np.diag(breve_theta_elements)
            b = []
            # 速率约束
            for i in range(M):
                gain_1 = AR_IRS.compute_channel_gain(self, Q, Theta, i, L, W_k)
                b.append(gain_1)

            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        return Theta, tau

    def compute_channel_gain(self, Q, Theta, i, L, W_k):
        # 计算 γ_{k,i} 和 τ_{k,i}
        gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
        tau_k_i = (rho0 - kappa1) * L * rho0 * 10 ** (9)

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
                          + tau_k_i / ((np.linalg.norm(Q - W_k[i]) ** beta1) * (np.linalg.norm(Q - W_s) ** 2))
                          )
        return effective_gain

    def compute_ABC(self, Theta, Q, i, L, W_k):
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
        gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - w_ki) ** beta2)
        tau_ki = L * rho0 * (rho0 - kappa2) * 10 ** (9)
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

    def optimize_Q(self, X_prev, Q, P, Theta, L, W, M, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        W_k = positions[:M]

        Q_var = cp.Variable(3)
        u_var = cp.Variable(M)  # u_{k,i}^k: 第k个无人机到第k个小区第i个用户的距离上限
        l_var = cp.Variable(M)  # l_{k,i}^b: 第b个无人机到第k个小区第i个用户的距离下限
        uu_var = cp.Variable()  # uu_k: 第k个无人机到基站的距离上限
        ll_var = cp.Variable()  # ll_k: 第b个无人机到基站的距离下限
        Omega_underline_var = cp.Variable(M)  # \underline{\Omega_{k,i}^k}: 第k个无人机到第k个小区第i个用户的信道下限
        # Omega_bar_var = cp.Variable(M)  # \bar{\Omge(K)}ega_{k,i}^b}: 第b个无人机到第k个小区第i个用户的信道上限
        tau_var = cp.Variable()

        while t < max_iter:
            Q_prev = Q
            # 目标函数
            objective = cp.Maximize(tau_var)

            # 约束条件
            constraints = []

            # 1. 高度约束
            constraints.append(Q_var[2] >= Z_min)
            constraints.append(Q_var[2] <= Z_max)

            # 2. 最小速率约束
            for i in range(M):
                # title_i_n = np.log(X_prev["Omega_bar"][i] * sum(P[i + 1:M]) + sigmma) + (
                #         sum(P[i + 1:M]) * (Omega_bar_var[i] - X_prev["Omega_bar"][i]) / (
                #         X_prev["Omega_bar"][i] * sum(P[i + 1:M]) + sigmma))
                title_i_n = np.log(X_prev["Omega_underline"][i] * sum(P[i + 1:M]) + sigmma) + (
                        sum(P[i + 1:M]) * (Omega_underline_var[i] - X_prev["Omega_underline"][i]) / (
                        X_prev["Omega_underline"][i] * sum(P[i + 1:M]) + sigmma))
                constraints.append(
                    cp.log(
                        Omega_underline_var[i] * sum(
                            P[i:M]) + sigmma) >= (tau_var / W) + title_i_n
                )

            # 3. 用户到 UAV 的距离约束
            for i in range(M):
                constraints.append(
                    X_prev["u"][i] ** 2 + 2 * X_prev["u"][i] * (u_var[i] - X_prev["u"][i]) >=
                    cp.norm(Q_var - W_k[i]) ** 2
                )

            # 4. UAV 到其他用户的距离约束
            # for i in range(M):
            #     constraints.append(
            #         cp.norm(Q_prev - W_k[i]) ** 2 +
            #         2 * (Q_prev - W_k[i]).T @ (Q_var - Q_prev) >= l_var[i] ** 2
            #     )

            # 5. UAV 到反射器的距离约束
            constraints.append(
                X_prev["uu"] ** 2 + 2 * X_prev["uu"] * (uu_var - X_prev["uu"]) >=
                cp.norm(Q_var - W_s) ** 2
            )

            # # 6. UAV 到反射器的约束（另一种形式）
            # constraints.append(
            #     cp.norm(Q_prev - W_s) ** 2 +
            #     2 * (Q_prev - W_s).T @ (Q_var - Q_prev) >= ll_var ** 2
            # )

            # 8. 全局上界约束
            # for i in range(M):
            #     f_bi = -2 * cp.log(ll_var) - beta2 * cp.log(l_var[i])
            #     f_bi = cp.exp(f_bi)
            #
            #     g_bi = -1 * cp.log(ll_var) - (beta2 / 2) * cp.log(l_var[i])
            #     g_bi = cp.exp(g_bi)
            #     A, B, C = AR_IRS.compute_ABC(self, Theta, Q_prev, i, L)
            #
            #     # 计算 [tilde{g_{b,i}}]^{lb}
            #     g_bi_lb = (X_prev["ll"] ** -1) * (X_prev["l"][i] ** (-beta2 / 2)) \
            #               - (X_prev["ll"] ** -2) * (X_prev["l"][i] ** (-beta2 / 2)) * (ll_var - X_prev["ll"]) \
            #               - (beta2 / 2) * (X_prev["ll"] ** -1) * (X_prev["l"][i] ** ((-beta2 / 2) - 1)) * (
            #                       l_var[i] - X_prev["l"][i])
            #
            #     # 计算 [tilde{f_{b,i}} + E_{b,i} * tilde{g_{b,i}}]^{ub}
            #     if C >= 0:
            #         ub = A + B * f_bi + abs(C) * g_bi
            #     else:
            #         ub = A + B * f_bi - abs(C) * g_bi_lb
            #
            #     constraints.append(Omega_bar_var[i] >= ub)

            # 9. 全局下界约束
            for i in range(M):
                f_ki_lb = (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** (-beta2)) \
                          - 2 * (X_prev["uu"] ** (-3)) * (X_prev["u"][i] ** (-beta2)) * (uu_var - X_prev["uu"]) \
                          - (beta2) * (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** ((-beta2) - 1)) * (
                                  u_var[i] - X_prev["u"][i])

                g_ki_lb = (X_prev["uu"] ** (-1)) * (X_prev["u"][i] ** (-beta2 / 2)) \
                          - (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** (-beta2 / 2)) * (uu_var - X_prev["uu"]) \
                          - (beta2 / 2) * (X_prev["uu"] ** (-1)) * (X_prev["u"][i] ** ((-beta2 / 2) - 1)) * (
                                  u_var[i] - X_prev["u"][i])

                A, B, C = AR_IRS.compute_ABC(self, Theta, Q_prev, i, L, W_k)
                # 计算 [tilde{f_{b,i}}]^{ub}
                g_ki = -1 * cp.log(uu_var) - (beta2 / 2) * cp.log(u_var[i])
                g_ki = cp.exp(g_ki)

                if C >= 0:
                    lb = A + B * f_ki_lb + abs(C) * g_ki_lb
                else:
                    lb = A + B * f_ki_lb - abs(C) * g_ki
                constraints.append(Omega_underline_var[i] <= lb)

            # constraints.append(Omega_bar_var[4] <= Omega_underline_var[4])

            # 优化问题
            prob = cp.Problem(objective, constraints)
            prob.solve(cp.MOSEK)

            # 更新上一轮的值
            X_prev["u"] = [u_var[i].value for i in range(M)]
            # X_prev["l"] = [l_var[i].value for i in range(M)]
            X_prev["uu"] = uu_var.value
            # X_prev["ll"] = ll_var.value
            X_prev["Omega_underline"] = [Omega_underline_var[i].value for i in range(M)]
            # X_prev["Omega_bar"] = [Omega_bar_var[i].value for i in range(M)]

            X_prev["u"] = np.array(X_prev["u"])
            X_prev["l"] = np.array(X_prev["l"])
            X_prev["Omega_underline"] = np.array(X_prev["Omega_underline"])
            # X_prev["Omega_bar"] = np.array(X_prev["Omega_bar"])
            Q = [Q_var[i].value for i in range(3)]

            # 检查收敛
            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        # 输出优化结果
        print("优化后的 Q:", Q[0], Q[1], Q[2])
        return Q, X_prev, tau

    def optimize_P(self, Q, P, Theta, L, P_HAP, W, M):
        P_H = 10 ** (P_HAP / 10)
        W_k = positions[:M]

        p = cp.Variable(M)
        tau_var = cp.Variable()

        constraints = []

        # 约束条件
        for i in range(M):
            gain = AR_IRS.compute_channel_gain(self, Q, Theta, i, L, W_k)

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
            gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
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

        print("优化后的 P:", P[0], P[1], P[2], P[3], P[4])
        return P, tau


class AR_IRS_IF:
    def initialize_Theta(self, L):
        psi = np.random.uniform(0, 2 * np.pi, L)
        diagonal_elements = np.exp(1j * psi)
        Theta = np.diag(diagonal_elements)
        return Theta

    # BCD 迭代优化框架
    def bcd_optimization(self, max_iter=20, L=20, W=1, M=5, P_HAP=30):
        Q_list = []
        P_list = []
        tau_list = []
        np.random.seed(1)
        W_k = positions[:M]


        # 初始化变量
        Q = np.random.rand(3)
        Q[2] = (Z_min + Z_max) / 2  # 无人机的 3D 位置
        Theta = AR_IRS_IF.initialize_Theta(self, L)
        # 构造 P 数组
        first_element = 0.9 * (10 ** (P_HAP / 10))
        remaining_element = 0.1 * (10 ** (P_HAP / 10)) / (M - 1)
        P = np.append([first_element], [remaining_element] * (M - 1))

        X_prev = {
            "u": np.full(M, 55),
            # "l": np.full(M, 30),
            "uu": 55,
            # "ll": 30,
            "Omega_underline": np.full(M, 0.00001),
            # "Omega_bar": np.full(M, 0.00001),
        }

        for iteration in range(max_iter):
            # Step 1: 优化 pi 和 Theta
            Theta, tau = AR_IRS_IF.optimize_pi_Theta(self, Q, Theta, P, L, W, M)

            # Step 2: 固定 A 和 P，优化 Q
            Q, X_prev, tau = AR_IRS_IF.optimize_Q(self, X_prev, Q, P, Theta, L, W, M)

            # Step 3: 固定 Q，优化 A 和 P
            P, tau = AR_IRS_IF.optimize_P(self, Q, P, Theta, L, P_HAP, W, M)
            print("t", tau)

            Q_list.append(Q)
            P_list.append(P)
            tau_list.append(tau)

        return Q_list, W_k

    # 优化函数（具体实现需根据问题定义）
    def optimize_pi_Theta(self, Q, Theta, P, L, W, M, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        # 设定参数
        Delta = 10 ** (-5)  # 惩罚系数
        tau_k_i = (rho0 - kappa1) * L * rho0
        W_k = positions[:M]  # 将 positions 赋值给 W_k

        # 初始化功率分配变量

        phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoA
        psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)  # UAV 到 HAP 的 AoD

        while t < max_iter:
            theta_values = np.diagonal(Theta)
            breve_theta = np.append(theta_values, 1)
            Phi = np.outer(breve_theta, np.conj(breve_theta.T))

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
                gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
                A_i = gamma_k_i + tau_k_i * 10 ** (10) / (
                        (np.linalg.norm(Q - W_k[i]) ** beta1) * (np.linalg.norm(Q - W_s) ** 2))

                gain = AR_IRS_IF.compute_channel_gain(self, Q, Theta, i, L, W_k)
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

                # gain_1 = np.conj(hat_s_i.T) + np.conj(hat_h_i.T) @ Theta @ G
                # gain_1=np.linalg.norm(gain_1) ** 2
                # Y_i_1 = np.trace(np.conj(X_i.T) @ Phi @ X_i)

                # 计算P_i,Y_i,Phi
                p_i = np.sqrt(P[i]) * np.sqrt(np.conj(P[i]))
                Y_i = np.conj(X_i.T) @ Phi_var @ X_i

                # 计算f(Phi),g(Phi)
                f_phi = cp.log(cp.real(cp.trace(p_i * Y_i)) + A_i * P[i])
                g_phi = np.log(sigmma)

                constraints.append(epsilon_i * (f_phi - g_phi) + eta_i >= tau_var * M / W)

            constraints.append(cp.diag(Phi_var) == 1)

            # 定义目标函数
            grad_Phi = Phi.T / cp.norm(Phi, 2)
            tilde_B_phi = cp.norm(Phi, 2) + cp.trace((grad_Phi) @ (Phi_var - Phi))
            objective = cp.Minimize(Delta * cp.real((cp.norm(Phi_var, "nuc", axis=None) - tilde_B_phi)) - tau_var)

            # 定义问题
            problem = cp.Problem(objective, constraints)
            mosek_params = {
                mosek.dparam.intpnt_co_tol_rel_gap: 1e-4  # 设置对偶间隙容忍度为0.0001
            }
            problem.solve(mosek_params=mosek_params)

            phi_diagonal_elements = np.diagonal(Phi_var.value)

            # 计算平方根，得到 breve_theta 的元素（不包括额外的1）
            breve_theta_elements = np.sqrt(phi_diagonal_elements[:-1])

            # 构造 Theta 矩阵，这里假设 Theta 是一个对角矩阵
            Theta = np.diag(breve_theta_elements)

            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        return Theta, tau

    def compute_channel_gain(self, Q, Theta, i, L, W_k):
        # 计算 γ_{k,i} 和 τ_{k,i}
        gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
        tau_k_i = (rho0 - kappa1) * L * rho0

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
                          + tau_k_i / ((np.linalg.norm(Q - W_k[i]) ** beta1) * (np.linalg.norm(Q - W_s) ** 2))
                          )
        return effective_gain

    def compute_ABC(self, Theta, Q, i, L, W_k):
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
        gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - w_ki) ** beta2)
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

    def optimize_Q(self, X_prev, Q, P, Theta, L, W, M, max_iter=100, tol=1e-6):
        tau = 0
        t = 0
        W_k = positions[:M]

        Q_var = cp.Variable(3)
        u_var = cp.Variable(M)  # u_{k,i}^k: 第k个无人机到第k个小区第i个用户的距离上限
        # l_var = cp.Variable(M)  # l_{k,i}^b: 第b个无人机到第k个小区第i个用户的距离下限
        uu_var = cp.Variable()  # uu_k: 第k个无人机到基站的距离上限
        # ll_var = cp.Variable()  # ll_k: 第b个无人机到基站的距离下限
        Omega_underline_var = cp.Variable(M)  # \underline{\Omega_{k,i}^k}: 第k个无人机到第k个小区第i个用户的信道下限
        # Omega_bar_var = cp.Variable(M)  # \bar{\Omge(K)}ega_{k,i}^b}: 第b个无人机到第k个小区第i个用户的信道上限
        tau_var = cp.Variable()

        while t < max_iter:
            Q_prev = Q
            # 目标函数
            objective = cp.Maximize(tau_var)

            # 约束条件
            constraints = []

            # 1. 高度约束
            constraints.append(Q_var[2] >= Z_min)
            constraints.append(Q_var[2] <= Z_max)

            # 2. 最小速率约束
            for i in range(M):
                constraints.append(
                    cp.log(Omega_underline_var[i] * P[i] + sigmma) >= tau_var * M / W + np.log(sigmma)
                )

            # 3. 用户到 UAV 的距离约束
            for i in range(M):
                constraints.append(
                    X_prev["u"][i] ** 2 + 2 * X_prev["u"][i] * (u_var[i] - X_prev["u"][i]) >=
                    cp.norm(Q_var - W_k[i]) ** 2
                )

            # # 4. UAV 到其他用户的距离约束
            # for i in range(M):
            #     constraints.append(
            #         cp.norm(Q_prev - W_k[i]) ** 2 +
            #         2 * (Q_prev - W_k[i]).T @ (Q_var - Q_prev) >= l_var[i] ** 2
            #     )

            # 5. UAV 到反射器的距离约束
            constraints.append(
                X_prev["uu"] ** 2 + 2 * X_prev["uu"] * (uu_var - X_prev["uu"]) >=
                cp.norm(Q_var - W_s) ** 2
            )

            # # 6. UAV 到反射器的约束（另一种形式）
            # constraints.append(
            #     cp.norm(Q_prev - W_s) ** 2 +
            #     2 * (Q_prev - W_s).T @ (Q_var - Q_prev) >= ll_var ** 2
            # )

            # 8. 全局上界约束
            # for i in range(M):
            #     f_bi = -2 * cp.log(ll_var) - beta2 * cp.log(l_var[i])
            #     f_bi = cp.exp(f_bi)
            #
            #     g_bi = -1 * cp.log(ll_var) - (beta2 / 2) * cp.log(l_var[i])
            #     g_bi = cp.exp(g_bi)
            #     A, B, C = AR_IRS_IF.compute_ABC(self, Theta, Q_prev, i, L)
            #
            #     # 计算 [tilde{g_{b,i}}]^{lb}
            #     g_bi_lb = (X_prev["ll"] ** -1) * (X_prev["l"][i] ** (-beta2 / 2)) \
            #               - (X_prev["ll"] ** -2) * (X_prev["l"][i] ** (-beta2 / 2)) * (ll_var - X_prev["ll"]) \
            #               - (beta2 / 2) * (X_prev["ll"] ** -1) * (X_prev["l"][i] ** ((-beta2 / 2) - 1)) * (
            #                       l_var[i] - X_prev["l"][i])
            #
            #     # 计算 [tilde{f_{b,i}} + E_{b,i} * tilde{g_{b,i}}]^{ub}
            #     if C >= 0:
            #         ub = A + B * f_bi + abs(C) * g_bi
            #     else:
            #         ub = A + B * f_bi - abs(C) * g_bi_lb
            #
            #     constraints.append(Omega_bar_var[i] >= ub)
            #     constraints.append(Omega_bar_var[i] >= ub)

            # 9. 全局下界约束
            for i in range(M):
                f_ki_lb = (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** (-beta2)) \
                          - 2 * (X_prev["uu"] ** (-3)) * (X_prev["u"][i] ** (-beta2)) * (uu_var - X_prev["uu"]) \
                          - (beta2) * (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** ((-beta2) - 1)) * (
                                  u_var[i] - X_prev["u"][i])

                g_ki_lb = (X_prev["uu"] ** (-1)) * (X_prev["u"][i] ** (-beta2 / 2)) \
                          - (X_prev["uu"] ** (-2)) * (X_prev["u"][i] ** (-beta2 / 2)) * (uu_var - X_prev["uu"]) \
                          - (beta2 / 2) * (X_prev["uu"] ** (-1)) * (X_prev["u"][i] ** ((-beta2 / 2) - 1)) * (
                                  u_var[i] - X_prev["u"][i])

                A, B, C = AR_IRS_IF.compute_ABC(self, Theta, Q_prev, i, L, W_k)
                # 计算 [tilde{f_{b,i}}]^{ub}
                g_ki = -1 * cp.log(uu_var) - (beta2 / 2) * cp.log(u_var[i])
                g_ki = cp.exp(g_ki)

                if C >= 0:
                    lb = A + B * f_ki_lb + abs(C) * g_ki_lb
                else:
                    lb = A + B * f_ki_lb - abs(C) * g_ki
                constraints.append(Omega_underline_var[i] <= lb)

            # constraints.append(Omega_bar_var[4] <= Omega_underline_var[4])

            # 优化问题
            prob = cp.Problem(objective, constraints)
            prob.solve(cp.ECOS)

            # 更新上一轮的值
            X_prev["u"] = [u_var[i].value for i in range(M)]
            # X_prev["l"] = [l_var[i].value for i in range(M)]
            X_prev["uu"] = uu_var.value
            # X_prev["ll"] = ll_var.value
            X_prev["Omega_underline"] = [Omega_underline_var[i].value for i in range(M)]
            # X_prev["Omega_bar"] = [Omega_bar_var[i].value for i in range(M)]

            X_prev["u"] = np.array(X_prev["u"])
            # X_prev["l"] = np.array(X_prev["l"])
            X_prev["Omega_underline"] = np.array(X_prev["Omega_underline"])
            # X_prev["Omega_bar"] = np.array(X_prev["Omega_bar"])
            Q = [Q_var[i].value for i in range(3)]

            # 检查收敛
            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        # 输出优化结果
        print("优化后的 Q:", Q[0], Q[1], Q[2])
        return Q, X_prev, tau

    def optimize_P(self, Q, P, Theta, L, P_HAP, W, M, max_iter=100, tol=1e-6):
        t = 0
        tau = 0
        P_H = 10 ** (P_HAP / 10)
        W_k = positions[:M]

        p = cp.Variable(M)
        tau_var = cp.Variable()

        while t < max_iter:
            constraints = []

            # 约束条件
            for i in range(M):
                gain = AR_IRS_IF.compute_channel_gain(self, Q, Theta, i, L, W_k)

                SINR_tilde = (gain * P[i]) / sigmma
                epsilon = SINR_tilde / (1 + SINR_tilde)
                eta = np.log(1 + SINR_tilde) - epsilon * np.log(SINR_tilde)

                # 计算凸近似的 SINR_term
                SINR_term = np.log(gain) + p[i] - np.log(sigmma)

                # SINR 最小值约束
                constraints.append(epsilon * SINR_term + eta >= tau_var * M / W)

                # 总功率约束
                constraints.append(sum(cp.exp(p[i]) for i in range(M)) <= P_H)

                # 目标函数
            objective = cp.Maximize(tau_var)

            # 求解
            prob = cp.Problem(objective, constraints)
            prob.solve(cp.CLARABEL)

            P = [np.exp(p[i].value) for i in range(M)]
            P = np.array(P)

            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1
        print("优化后的 P:", P[0], P[1], P[2], P[3], P[4])
        return P, tau


class NO_IRS_NOMA:
    def bcd_optimization(self, max_iter=20, L=5, W=1, M=5, P_HAP=30):
        P_list = []
        tau_list = []
        np.random.seed(1)

        # 构造 P 数组
        first_element = 0.9 * (10 ** (P_HAP / 10))
        remaining_element = 0.1 * (10 ** (P_HAP / 10)) / (M - 1)
        P = np.append([first_element], [remaining_element] * (M - 1))

        for iteration in range(max_iter):
            P, tau = NO_IRS_NOMA.optimize_P(self, P, P_HAP,L, W, M)
            print("t", tau)
            P_list.append(P)
            tau_list.append(tau)

        return P_list, P_list, tau_list

    def optimize_P(self, P, P_HAP,L=2,  W=1, M=5, max_iter=20, tol=1e-6):
        tau = 0
        t = 0
        W_k = positions[:M]  # 将 positions 赋值给 W_k

        P_H = 10 ** (P_HAP / 10)
        p = cp.Variable(M)
        tau_var = cp.Variable()

        constraints = []
        while t < max_iter:
            # 约束条件
            for i in range(M):
                gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
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

                constant = np.log(sigmma / gain)
                a = cp.hstack([p[i + 1:M], constant])

                # 计算凸近似的 SINR_term
                if i <= M - 2:
                    SINR_term = cp.log(gain) + p[i] \
                                - cp.log(gain) - cp.log_sum_exp(a)
                else:
                    SINR_term = cp.log(gain) + p[i] - cp.log(sigmma)

                # SINR 最小值约束
                constraints.append(epsilon * SINR_term + eta >= tau_var)

            # 总功率约束
            constraints.append(sum(cp.exp(p[i]) for i in range(M)) <= P_H)

            # 目标函数
            objective = cp.Maximize(tau_var)

            # 求解
            prob = cp.Problem(objective, constraints)
            prob.solve(cp.MOSEK)

            A = []
            for i in range(M):
                gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
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
                                - np.log(gain) - np.log(
                        sum(np.exp(p[j].value) for j in range(i + 1, M)) + sigmma / gain)
                else:
                    SINR_term = np.log(gain) + p[i].value - np.log(sigmma)

                # SINR 最小值约束
                A.append((epsilon * SINR_term + eta) * W)

            P = [np.exp(p[i].value) for i in range(M)]
            P = np.array(P)
            tau_1 = tau_var.value

            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        print("优化后的 P:", P[0], P[1], P[2], P[3], P[4])
        print("tau=", tau_1)
        return P, tau_var.value


class NO_IRS_IF:
    def bcd_optimization(self, max_iter=20, L=5, W=1, M=5, P_HAP=30):
        P_list = []
        tau_list = []
        np.random.seed(1)

        # 构造 P 数组
        first_element = 0.9 * (10 ** (P_HAP / 10))
        remaining_element = 0.1 * (10 ** (P_HAP / 10)) / (M - 1)
        P = np.append([first_element], [remaining_element] * (M - 1))

        for iteration in range(max_iter):
            P, tau = NO_IRS_IF.optimize_P(self, P, P_HAP,L, W, M)
            print("t", tau)
            P_list.append(P)
            tau_list.append(tau)

        return P_list, P_list, tau_list

    def optimize_P(self, P, P_HAP, L=2, W=1, M=5, max_iter=20, tol=1e-6):
        tau = 0
        t = 0

        W_k = positions[:M]  # 将 positions 赋值给 W_k

        P_H = 10 ** (P_HAP / 10)
        p = cp.Variable(M)
        tau_var = cp.Variable()

        while t < max_iter:
            constraints = []

            # 约束条件
            for i in range(M):
                gamma_k_i = (rho0 - kappa2) / (np.linalg.norm(W_s - W_k[i]) ** beta2)
                hat_s_ki = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
                gain = np.linalg.norm(hat_s_ki) ** 2 + gamma_k_i

                SINR_tilde = (gain * P[i]) / sigmma
                epsilon = SINR_tilde / (1 + SINR_tilde)
                eta = np.log(1 + SINR_tilde) - epsilon * np.log(SINR_tilde)

                # 计算凸近似的 SINR_term
                SINR_term = np.log(gain) + p[i] - np.log(sigmma)

                # SINR 最小值约束
                constraints.append(epsilon * SINR_term + eta >= tau_var * M / W)

            # 总功率约束
            constraints.append(sum(cp.exp(p[i]) for i in range(M)) <= P_H)

            # 目标函数
            objective = cp.Maximize(tau_var)

            # 求解
            prob = cp.Problem(objective, constraints)
            prob.solve(cp.MOSEK)

            P = [np.exp(p[i].value) for i in range(M)]
            P = np.array(P)
            tau_1 = tau_var.value
            if t != 0 and abs(tau_var.value - tau) < tol:
                print(f"迭代收敛在第 {t} 步，总速率: {tau_var.value}")
                break

            tau = tau_var.value
            t += 1

        print("优化后的 P:", P[0], P[1], P[2], P[3], P[4])
        print("tau=", tau_1)
        return  P, tau_var.value


# 参数设置
N = 2  # 天线数量
beta1 = 2.2
beta2 = 2.2
rho0 = 0.001
K1 = 10
K2 = 10
sigmma = 10 ** (-8)
d = 0.5  # 元件距离
wavelength = 0.1  # 载波长
L_k = 5
Z_min = 30  # 飞行最低高度
Z_max = 50  # 飞行最高高度
# 初始化变量
np.random.seed(1)
s_ki = np.exp(-1j * 2 * np.pi * d / wavelength * np.random.rand(N))
s_ki = s_ki / np.linalg.norm(s_ki)

np.random.seed(0)
positions = np.array([
    [30.0, 30.0, 0.0],  # 第一个位置的坐标
    # [3.0, 2.0, 0.0],  # 第二个位置的坐标
    # [2.0, 3.0, 0.0],  # 第三个位置的坐标
    [20.0, 20.0, 0.0],  # 第四个位置的坐标
    [20.0, 10.0, 0.0],  # 第五个位置的坐标
    [10.0, 30.0, 0.0],  # 第六个位置的坐标
    # [1.0, 2.0, 0.0],  # 第七个位置的坐标
    [10.0, 10.0, 0.0],  # 第八个位置的坐标
    np.random.rand(3),  # 第九个位置的坐标
    np.random.rand(3),
    np.random.rand(3),
    np.random.rand(3),
])

W_s = np.array([0, 0, 20])  # HAP 的位置

# 计算常量值
kappa1 = K1 * rho0 / (K1 + 1)
kappa2 = K2 * rho0 / (K2 + 1)

# 执行 BCD 迭代优化
irs = AR_IRS()
irs_if = AR_IRS_IF()
no_irs_na = NO_IRS_NOMA()
no_irs_if = NO_IRS_IF()

# Q, P, tau = irs.bcd_optimization()
