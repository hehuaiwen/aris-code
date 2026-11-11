import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
from func.myfunc import *
from func.algorithms import *


def do_proposed(turn=10,miu=1.5,q1=420,q2=300):
    data = proposed(turn=turn,miu=miu,q1=q1,q2=q2)
    data.append(np.arange(turn) + 1)
    savedata(data, 'proposed')


def do_nocoop(turn=5000):
    data = nocoop(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'nocoop')


def do_noLy(turn=5000):
    data = noLy(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'noLy')


def do_xiezai(turn=5000):
    data = xiezai(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'xiezai')


# def do_suiji(turn=5000):
#     data = noLy(turn=turn)
#     data.append(np.arange(turn) + 1)
#     savedata(data, 'noLy')


def do_proposed_difV(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for v in np.arange(100, 1100, 100):
        data = proposed(turn=turn, v=v)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(100, 1000, 100)]
    savedata(data, 'difV_proposed')


def do_nocoop_difV(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for v in np.arange(100, 1100, 100):
        data = nocoop(turn=turn, v=v)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(100, 1000, 100)]
    savedata(data, 'difV_nocoop')


def do_proposed_difW(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    p1_list = []
    p2_list = []
    p3_list = []
    for w in np.arange(0.96, 1.05, 0.01):
        print(w)
        data = proposed(turn=turn, w=w)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])
        p1_list.append(data[5][0])
        p2_list.append(data[6][0])
        p3_list.append(data[7][0])
    p1_list = p1_list[::-1]
    p2_list = p2_list[::-1]
    p3_list = p3_list[::-1]

    data = [dmean, Qmean_s, Qmean_h, Ymean,p1_list,p2_list,p3_list, np.arange(0.96, 1.05, 0.01)]
    savedata(data, 'difW_proposed')


def do_nocoop_difW(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for w in np.arange(0.96, 1.05, 0.01):
        data = nocoop(turn=turn, w=w)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(0.96, 1.05, 0.01)]
    savedata(data, 'difW_nocoop')


def do_noLy_difW(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for w in np.arange(0.96, 1.05, 0.01):
        data = noLy(turn=turn, w=w)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(0.96, 1.05, 0.01)]
    savedata(data, 'difW_noLy')


def do_xiezai_difW(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for w in np.arange(0.96, 1.05, 0.01):
        data = xiezai(turn=turn, w=w)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(0.96, 1.05, 0.01)]
    savedata(data, 'difW_xiezai')


def do_proposed_difGama(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for g in np.arange(1.75, 2.3, 0.05):
        data = proposed(turn=turn, g=g)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(1.75, 2.3, 0.05)]
    savedata(data, 'difGama_proposed')


def do_proposed_difD(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for d in np.arange(120, 164, 4):
        data = proposed(turn=turn, d=d)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(120, 164, 4)]
    savedata(data, 'difD_proposed')


def do_nocoop_difD(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for d in np.arange(120, 164, 4):
        data = nocoop(turn=turn, d=d)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(120, 164, 4)]
    savedata(data, 'difD_nocoop')


def do_noLy_difD(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for d in np.arange(120, 164, 4):
        data = noLy(turn=turn, d=d)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(120, 164, 4)]
    savedata(data, 'difD_noLy')


def do_xiezai_difD(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for d in np.arange(120, 164, 4):
        data = xiezai(turn=turn, d=d)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(120, 164, 4)]
    savedata(data, 'difD_xiezai')


def do_proposed_difA(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for a_h in np.arange(1.9, 2.35, 0.05):
        data = proposed(turn=turn, a_h=a_h)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(1.9, 2.35, 0.05)]
    savedata(data, 'difA_proposed')

def do_nocoop_difA(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for a_h in np.arange(1.9, 2.35, 0.05):
        data = nocoop(turn=turn, a_h=a_h)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(1.9, 2.35, 0.05)]
    savedata(data, 'difA_nocoop')

def do_noLy_difA(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for a_h in np.arange(1.9, 2.35, 0.05):
        data = noLy(turn=turn, a_h=a_h)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(1.9, 2.35, 0.05)]
    savedata(data, 'difA_noLy')

def do_xiezai_difA(turn=5000):
    dmean = []
    Qmean_s = []
    Qmean_h = []
    Ymean = []
    for a_h in np.arange(1.9, 2.35, 0.05):
        data = xiezai(turn=turn, a_h=a_h)
        dmean.append(data[0][-1])
        Qmean_s.append(data[1][-1])
        Qmean_h.append(data[2][-1])
        Ymean.append(data[3][-1])

    data = [dmean, Qmean_s, Qmean_h, Ymean, np.arange(1.9, 2.35, 0.05)]
    savedata(data, 'difA_xiezai')


def do_proposed_difV_detail(turn=5000):
    dmean_detail = []
    Qmean_s_detail = []
    Qmean_h_detail = []
    Ymean_detail = []
    tao0_detail = []
    for v in np.arange(100, 1100, 100):
        data = proposed(turn=turn, v=v)
        dmean_detail.append(data[0])
        Qmean_s_detail.append(data[1])
        Qmean_h_detail.append(data[2])
        Ymean_detail.append(data[3])
        tao0_detail.append(data[4])
    dmean_detail.append(np.arange(turn) + 1)
    Qmean_s_detail.append(np.arange(turn) + 1)
    Qmean_h_detail.append(np.arange(turn) + 1)
    Ymean_detail.append(np.arange(turn) + 1)
    tao0_detail.append(np.full(turn, 2))
    tao0_detail.append(np.arange(turn) + 1)

    savedata(dmean_detail, 'difV_proposed_dmean')
    savedata(Qmean_s_detail, 'difV_proposed_Qmean_s')
    savedata(Qmean_h_detail, 'difV_proposed_Qmean_h')
    savedata(Ymean_detail, 'difV_proposed_Ymean')
    savedata(tao0_detail, "tao0_difV")


def do_proposed_difW_detail(turn=5000):
    dmean_detail = []
    Qmean_s_detail = []
    Qmean_h_detail = []
    Ymean_detail = []
    tao0_detail = []
    for w in np.arange(0.75, 1.3, 0.05):
        data = proposed(turn=turn, w=w, a_s=1.8)
        dmean_detail.append(data[0])
        Qmean_s_detail.append(data[1])
        Qmean_h_detail.append(data[2])
        Ymean_detail.append(data[3])
        tao0_detail.append(data[4])
    dmean_detail.append(np.arange(turn) + 1)
    Qmean_s_detail.append(np.arange(turn) + 1)
    Qmean_h_detail.append(np.arange(turn) + 1)
    Ymean_detail.append(np.arange(turn) + 1)
    tao0_detail.append(np.full(turn, 2))
    tao0_detail.append(np.arange(turn) + 1)

    savedata(dmean_detail, 'difW_proposed_dmean')
    savedata(Qmean_s_detail, 'difW_proposed_Qmean_s')
    savedata(Qmean_h_detail, 'difW_proposed_Qmean_h')
    savedata(Ymean_detail, 'difW_proposed_Ymean')
    savedata(tao0_detail, "tao0_difW")


def do_proposed_difGama_detail(turn=5000):
    dmean_detail = []
    Qmean_s_detail = []
    Qmean_h_detail = []
    Ymean_detail = []
    tao0_detail = []
    for g in np.arange(1.8, 2.24, 0.04):
        data = proposed(turn=turn, g=g)
        dmean_detail.append(data[0])
        Qmean_s_detail.append(data[1])
        Qmean_h_detail.append(data[2])
        Ymean_detail.append(data[3])
        tao0_detail.append(data[4])
    dmean_detail.append(np.arange(turn) + 1)
    Qmean_s_detail.append(np.arange(turn) + 1)
    Qmean_h_detail.append(np.arange(turn) + 1)
    Ymean_detail.append(np.arange(turn) + 1)
    tao0_detail.append(np.full(turn, 2))
    tao0_detail.append(np.arange(turn) + 1)

    savedata(dmean_detail, 'difGama_proposed_dmean')
    savedata(Qmean_s_detail, 'difGama_proposed_Qmean_s')
    savedata(Qmean_h_detail, 'difGama_proposed_Qmean_h')
    savedata(Ymean_detail, 'difGama_proposed_Ymean')
    savedata(tao0_detail, "tao0_difGama")
