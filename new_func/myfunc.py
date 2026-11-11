import ast

import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import sys

def produce_A(taskpertime, num_tasks=30000, time_frame_length=1.0):
    """
    使用指数分布，后保持均值不变向均值收缩0.5以保证数据稳定
    :param taskpertime: 一个时隙到达任务的均值
    :param num_tasks: 任务总数量
    :param time_frame_length: 时隙长度
    :return: 每个时隙到达的任务数量
    """

    # 设定参数
    lambda_param = taskpertime
    mean_interval = 1 / lambda_param
    numpy.random.seed(43)

    # 生成服从指数分布的任务到达时间间隔
    interarrival_times = np.random.exponential(scale=mean_interval, size=num_tasks)

    # 累积任务到达时间点
    arrival_times = np.cumsum(interarrival_times)

    # 计算每个时间帧内到达的任务数量
    max_time = arrival_times[-1]
    num_frames = int(np.ceil(max_time / time_frame_length))
    task_counts_per_frame = np.zeros(num_frames)

    for time in arrival_times:
        frame_index = int(time // time_frame_length)
        task_counts_per_frame[frame_index] += 1

    mean = sum(task_counts_per_frame) / len(task_counts_per_frame)

    # 计算每个元素与均值的差值
    differences = [x - mean for x in task_counts_per_frame]

    # 将差值缩小，使其向均值靠近
    compressed_differences = [0.6 * diff for diff in differences]  # 可调整0.9这个比例因子

    # 将新的差值与均值相加，得到压缩后的数据
    task_counts_per_frame = [(diff + mean) for diff in compressed_differences]

    return task_counts_per_frame


# C:/Users/javakaifa/PycharmProjects/lunwen2
def savedata(data, name):
    with open("./new_data/"+name+".csv", mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        # 写入数据
        csv_writer.writerows(data)

    return


def readdata(name):
    data = np.loadtxt("./new_data/"+name+".csv", delimiter=",")
    return np.array(data)

