import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
from new_func.myfunc import *
from new_func.algorithms import *

def do_ris_Q(max_iter=20, M=5):
    irs.bcd_optimization(max_iter=max_iter, M=M)
    L_results = [irs.Q, positions[:M]]
    savedata(L_results, 'ris_Q')

def do_ris_if_Q(max_iter=20, M=5):
    irs_if.bcd_optimization(max_iter=max_iter, M=M)
    L_results = [irs_if.Q, positions[:M]]
    savedata(L_results, 'ris_if_Q')


def do_ris_difL(max_iter=20):
    L_results = []
    for L in np.arange(8, 33, 4):
        P, tau = irs.bcd_optimization(max_iter=max_iter, L=L)
        L_results.append(P)
        L_results.append(tau)
    L_results.append(np.arange(max_iter) + 1)
    savedata(L_results, 'difL_ris_ite')


def do_ris_difPHAP(max_iter=20, L=32):
    L_results = []
    for P_HAP in np.arange(10, 40, 10):
        P, tau = irs.bcd_optimization(max_iter=max_iter, L=L, P_HAP=P_HAP)
        L_results.append(P)
        L_results.append(tau)
    L_results.append(np.arange(max_iter) + 1)
    savedata(L_results, 'difPHAP_ris_ite')

def do_ris_difPHAP_L10(max_iter=20, L=10):
    L_results = []
    for P_HAP in np.arange(10, 40, 10):
        P, tau = irs.bcd_optimization(max_iter=max_iter, L=L, P_HAP=P_HAP)
        L_results.append(P)
        L_results.append(tau)
    L_results.append(np.arange(max_iter) + 1)
    savedata(L_results, 'difPHAP_ris_iteL10')


class difM:
    def do_irs_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = irs.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_irs')

    def do_irs_if_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = irs_if.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_irs_if')

    def do_no_irs_na_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = no_irs_na.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_no_irs_na')

    def do_no_irs_if_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = no_irs_if.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_no_irs_if')

    def do_fix_irs_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = fix_irs.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_fix_irs')

    def do_fix_irs_if_difM(self):
        tau = []
        for M in np.arange(2, 15, 2):
            data = fix_irs_if.bcd_optimization(M=M)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 15, 2)]
        savedata(data, 'difM_fix_irs_if')


class difW:
    def do_irs_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = irs.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_irs')

    def do_irs_if_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = irs_if.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_irs_if')

    def do_no_irs_na_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = no_irs_na.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_no_irs_na')

    def do_no_irs_if_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = no_irs_if.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_no_irs_if')

    def do_fix_irs_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = fix_irs.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_fix_irs')

    def do_fix_irs_if_difW(self):
        tau = []
        for W in np.arange(0.5, 3.5, 0.5):
            data = fix_irs_if.bcd_optimization(W=W * 1e6)
            tau.append(data[1][-1])
        data = [tau, np.arange(0.5, 3.5, 0.5)]
        savedata(data, 'difW_fix_irs_if')


class difL:
    def do_irs_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = irs.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_irs')

    def do_irs_if_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = irs_if.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_irs_if')

    def do_no_irs_na_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = no_irs_na.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_no_irs_na')

    def do_no_irs_if_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = no_irs_if.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_no_irs_if')

    def do_fix_irs_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = fix_irs.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_fix_irs')

    def do_fix_irs_if_difL(self):
        tau = []
        for L in np.arange(8, 33, 4):
            data = fix_irs_if.bcd_optimization(L=L)
            tau.append(data[1][-1])
        data = [tau, np.arange(8, 33, 4)]
        savedata(data, 'difL_fix_irs_if')


class difP:
    def do_irs_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = irs.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_irs')

    def do_irs_if_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = irs_if.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_irs_if')

    def do_no_irs_na_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = no_irs_na.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_no_irs_na')

    def do_no_irs_if_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = no_irs_if.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_no_irs_if')

    def do_fix_irs_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = fix_irs.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_fix_irs')

    def do_fix_irs_if_difP(self):
        tau = []
        for P_HAP in np.arange(10, 45, 5):
            data = fix_irs_if.bcd_optimization(P_HAP=P_HAP)
            tau.append(data[1][-1])
        data = [tau, np.arange(10, 45, 5)]
        savedata(data, 'difP_fix_irs_if')


class difBeta:
    def do_irs_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = irs.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_irs')

    def do_irs_if_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = irs_if.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_irs_if')

    def do_no_irs_na_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = no_irs_na.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_no_irs_na')

    def do_no_irs_if_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = no_irs_if.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_no_irs_if')

    def do_fix_irs_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = fix_irs.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_fix_irs')

    def do_fix_irs_if_difBeta(self):
        tau = []
        for Beta in np.arange(2, 3.2, 0.2):
            data = fix_irs_if.bcd_optimization(beta1=Beta,beta2=Beta)
            tau.append(data[1][-1])
        data = [tau, np.arange(2, 3.2, 0.2)]
        savedata(data, 'difBeta_fix_irs_if')




difM=difM()
difW=difW()
difL=difL()
difP=difP()
difBeta=difBeta()