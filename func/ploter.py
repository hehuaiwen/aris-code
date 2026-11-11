import os
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

from func.myfunc import *
from mpl_toolkits.axisartist import ParasiteAxes, HostAxes

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'


class Plotter:
    def __init__(self, data_save_dir='D:/HF/lunwen2/data',
                 pic_save_dir='D:/HF/lunwen2/pic', dpi=500, ):
        self.data_save_dir = data_save_dir
        self.pic_save_dir = pic_save_dir
        self.figsize = (12, 8)
        self.legendsize = 20

        self.colors = ['#0a7ac7','#de8a00','#bf00a8','#1aa600','#bf2c00','#bf9300','#8a00ab','#003496',]

        self.grid_color = '#ddd'
        self.grid_line = '--'
        self.grid_line_width = 2
        self.grid_alpha = 0.4

        self.line_width = 3
        self.line_width_list= []
        self.linestyle = ['-', '--','-','--','-','--']

        self.spine_line_width = 2

        self.lable_fontsize = 25
        self.offset_fontsize = 17

        self.markersize = 14
        self.markerstyles = ['>', 'p','*', 'o', '+', 'v','s']
        self.markerfacecolor = self.colors

        self.dpi = dpi
        self.line_width_small = 2

        self.tick_length = 8
        self.tick_width = 2
        self.tick_labelsize = 21
        self.tick_pad = 11

    def save_fig(self,plt,save_file):
        save_path = os.path.join(self.pic_save_dir, save_file)
        plt.savefig(save_path)
        print(save_file+"saved")
        return


    def plot(self, x, y, xlabel, ylabel, label, markevery, tickin, ylim,
             xoffset=None, xticklabels=None):
        if xoffset is None:
            xoffset = [0.0, 0.0]
        # 创建图表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_line_width)

        # 绘制第一个纵坐标
        line = ax.plot(x, y, marker=self.markerstyles[0], markevery=markevery, markerfacecolor=self.markerfacecolor[0],
                       markersize=self.markersize, markeredgewidth=self.line_width, linestyle=self.linestyle[0],
                       color=self.colors[0],
                       label=label, linewidth=self.line_width)

        # 坐标标签字体
        ax.set_xlabel(xlabel, fontsize=self.lable_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.lable_fontsize)

        # 坐标轴字体
        ax.xaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)

        # 坐标轴刻度
        ax.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        ax.set_ylim(ylim)

        # x10科学计数法
        ax_x_formatter = ticker.ScalarFormatter(useMathText=True)
        ax_y_formatter = ticker.ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(ax_x_formatter)
        ax.yaxis.set_major_formatter(ax_y_formatter)

        # 设置网格和刻度
        ax.grid(True, alpha=self.grid_alpha, color=self.grid_color, linestyle=self.grid_line,
                linewidth=self.grid_line_width)

        if xticklabels is not None:
            ax.set_xticks(x, labels=xticklabels)

        ax.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)
        ax.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tickin))


        return fig, ax, line

    def plotyy(self, x, y1, y2, save_file, xlabel, ylabel1, ylabel2, label1, label2, markevery, tickin, y1lim, y2lim,
               xoffset=None, bbox_to_anchor=(0.095, 0.95),ytickin2 = None):
        # 读取数据
        if xoffset is None:
            xoffset = [0.0, 0.0]

        # 创建图表
        fig, ax, line1 = self.plot(x, y1, xlabel, ylabel1, label1, markevery, tickin, y1lim,
                                   xoffset, xticklabels=None)
        ax.set_ylabel(ylabel1,color=self.colors[0])
        ax.tick_params(axis='y', which='major',colors=self.colors[0])
        ax.spines['left'].set_color(self.colors[0])  # 左侧坐标轴
        ax.spines['right'].set_color(self.colors[1])  # 右侧坐标轴

        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_line_width)

        # 上方横坐标
        new_x = ax.twiny()
        new_x.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        new_x.set_xticklabels([])
        new_x.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width)
        new_x.xaxis.set_major_locator(ticker.MultipleLocator(tickin))

        new_x.spines['left'].set_color(self.colors[0])  # 左侧坐标轴
        new_x.spines['right'].set_color(self.colors[1])  # 右侧坐标轴

        # 创建第二个纵坐标
        new_y = ax.twinx()
        new_y.set_ylabel(ylabel2, fontsize=self.lable_fontsize, color=self.colors[1])
        new_y.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        new_y_formatter = ticker.ScalarFormatter(useMathText=True)
        new_y.yaxis.set_major_formatter(new_y_formatter)
        if ytickin2 != None:
            new_y.yaxis.set_major_locator(ticker.MultipleLocator(ytickin2))

        new_y.spines['left'].set_color(self.colors[0])  # 左侧坐标轴
        new_y.spines['right'].set_color(self.colors[1])  # 右侧坐标轴

        new_y.set_ylim(y2lim)
        line2 = new_y.plot(x, y2, marker=self.markerstyles[1], markevery=markevery, markerfacecolor=self.markerfacecolor[1],
                           markersize=self.markersize, markeredgewidth=self.line_width, linestyle=self.linestyle[1],
                           color=self.colors[1],
                           label=label2, linewidth=self.line_width)
        new_y.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                          labelsize=self.tick_labelsize, pad=self.tick_pad, colors=self.colors[1])

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, prop={'size': self.legendsize}, loc='upper left', bbox_to_anchor=bbox_to_anchor)

        # 保存图表
        # save_path = os.path.join(self.pic_save_dir, save_file)
        # plt.savefig(save_path)
        self.save_fig(plt,save_file)
        plt.close()

    def plotdetail(self, x, y, save_file, xlabel, ylabel,
                   label, markevery, tickin, ylim, xoffset=None, bbox_to_anchor=(0.095, 0.95), xticklabels=None):
        if xoffset is None:
            xoffset = [0.0, 0.0]

        # 创建图表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # print(ax.spines.values)
        spines = ax.spines
        # spines[['top','right']].set_visible(False)
        for spine in spines.values():
            spine.set_linewidth(self.spine_line_width)

        lines = ax.plot(x, y[0], marker=self.markerstyles[0], markevery=markevery,
                         markerfacecolor=self.markerfacecolor[0],
                         markersize=self.markersize, markeredgewidth=self.line_width,
                         linestyle=self.linestyle[0],
                         color=self.colors[0], label=label[0], linewidth=self.line_width)
        print(len(y),len(y[0]),len(y[-1]))
        # 绘制第一个纵坐标
        for i in range(len(y)-1):
            lines += ax.plot(x, y[i+1], marker=self.markerstyles[i+1], markevery=markevery,
                             markerfacecolor=self.markerfacecolor[i+1],
                             markersize=self.markersize, markeredgewidth=self.line_width,
                             linestyle=self.linestyle[i+1],
                             color=self.colors[i+1], label=label[i+1], linewidth=self.line_width)

        # ax.spines['left'].set_color(self.colors[0])  # 左侧坐标轴

        # 坐标标签字体
        ax.set_xlabel(xlabel, fontsize=self.lable_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.lable_fontsize)

        # 坐标轴字体
        ax.xaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tickin))

        # 坐标轴刻度
        ax.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        ax.set_ylim(ylim)

        # x10科学计数法
        ax_x_formatter = ticker.ScalarFormatter(useMathText=True)
        ax_y_formatter = ticker.ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(ax_x_formatter)
        ax.yaxis.set_major_formatter(ax_y_formatter)

        # 设置网格和刻度
        ax.grid(True, alpha=self.grid_alpha, color=self.grid_color, linestyle=self.grid_line,
                linewidth=self.grid_line_width)

        if xticklabels is not None:
            ax.set_xticks(x, labels=xticklabels)

        ax.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)
        ax.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)


        # 上方横坐标
        new_x = ax.twiny()
        new_x.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        new_x.set_xticklabels([])
        new_x.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width)
        new_x.xaxis.set_major_locator(ticker.MultipleLocator(tickin))
        # 右边坐标
        new_y = ax.twinx()
        new_y.set_ylim(ylim)
        # new_y.yaxis.set_major_locator(ticker.MultipleLocator(tickin))
        new_y.set_yticklabels([])
        new_y.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width)

        # new_x.spines['left'].set_color(self.colors[0])  # 左侧坐标轴
        # new_x.spines['right'].set_color(self.colors[1])  # 右侧坐标轴

        # 图例
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, prop={'size': self.legendsize}, loc='upper left', bbox_to_anchor=bbox_to_anchor)

        # 保存图表
        # save_path = os.path.join(self.pic_save_dir, save_file)
        # plt.savefig(save_path)
        self.save_fig(plt,save_file)
        plt.close()

    def plottripley(self, x, y1, y2,y3, save_file, xlabel, ylabel1, ylabel2, ylabel3,label1, label2,label3, markevery, tickin, y1lim, y2lim,y3lim,
               xoffset=None, bbox_to_anchor=(0.095, 0.95)):
        # 读取数据
        if xoffset is None:
            xoffset = [0.0, 0.0]
        fig = plt.figure(figsize=(14,8), dpi=self.dpi)
        host = host_subplot(111, axes_class=axisartist.Axes,figure=fig)
        plt.subplots_adjust(right=0.8)

        par1 = host.twinx()
        par2 = host.twinx()
        # for spine in host.spines.values():
            # spine.set_linewidth(self.spine_line_width)
            # spine.set_linewidth(1000000)




        par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(100, 0))

        par1.axis["right"].toggle(all=True)
        par2.axis["right"].toggle(all=True)

        line1, = host.plot(x, y1, marker=self.markerstyles[0], markevery=markevery,
                             markerfacecolor=self.markerfacecolor[0],
                             markersize=self.markersize, markeredgewidth=self.line_width,
                             linestyle=self.linestyle[0],
                             color=self.colors[0], label=label1, linewidth=self.line_width)
        line2, = par1.plot(x, y2, marker=self.markerstyles[1], markevery=markevery,
                             markerfacecolor=self.markerfacecolor[1],
                             markersize=self.markersize, markeredgewidth=self.line_width,
                             linestyle=self.linestyle[0],
                             color=self.colors[1], label=label2, linewidth=self.line_width)
        line3, = par2.plot(x, y3, marker=self.markerstyles[2], markevery=markevery,
                             markerfacecolor=self.markerfacecolor[1],
                             markersize=self.markersize, markeredgewidth=self.line_width,
                             linestyle=self.linestyle[1],
                             color=self.colors[1], label=label3, linewidth=self.line_width)



        host.set(xlim=(min(x) - xoffset[0], max(x) + xoffset[1]), ylim=(y1lim))



        # x10科学计数法
        host_x_formatter = ticker.ScalarFormatter(useMathText=True)
        host_y_formatter = ticker.ScalarFormatter(useMathText=True)
        host.xaxis.set_major_formatter(host_x_formatter)
        host.yaxis.set_major_formatter(host_y_formatter)
        par1_y_formatter = ticker.ScalarFormatter(useMathText=True)
        par1.yaxis.set_major_formatter(par1_y_formatter)
        par2_y_formatter = ticker.ScalarFormatter(useMathText=True)
        par2.yaxis.set_major_formatter(par2_y_formatter)
        par2.axis['right'].offsetText.xyann= (100, 0)

        # 设置网格和刻度
        host.grid(True, alpha=self.grid_alpha, color=self.grid_color, linestyle=self.grid_line,
                linewidth=self.grid_line_width)

        host.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)
        host.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                         labelsize=self.tick_labelsize, pad=self.tick_pad)

        host.xaxis.set_major_locator(ticker.MultipleLocator(tickin))


        # 坐标标签字体
        host.set_xlabel(xlabel, fontsize=self.lable_fontsize)
        host.set_ylabel(ylabel1, fontsize=self.lable_fontsize,color=self.colors[0])


        # 坐标轴字体
        # host.xaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        # host.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)

        # ax.offsetText



        host.axis['left'].line.set_color(self.colors[0])
        host.axis['left'].major_ticks.set_color(self.colors[0])
        host.axis['left'].major_ticklabels.set_color(self.colors[0])
        host.axis['left'].label.set_color(self.colors[0])
        host.axis['left'].offsetText.set_color(self.colors[0])
        # host.axis['left'].offsetText.set_size(self.offset_fontsize)


        par1.axis['right'].line.set_color(self.colors[1])
        par1.axis['right'].major_ticks.set_color(self.colors[1])
        par1.axis['right'].major_ticklabels.set_color(self.colors[1])
        par1.axis['right'].label.set_color(self.colors[1])
        par1.axis['right'].offsetText.set_color(self.colors[1])


        par2.axis['right'].line.set_color(self.colors[1])
        par2.axis['right'].major_ticks.set_color(self.colors[1])
        par2.axis['right'].major_ticklabels.set_color(self.colors[1])
        par2.axis['right'].label.set_color(self.colors[1])
        par2.axis['right'].offsetText.set_color(self.colors[1])



        for ax in host.axis.values():
            ax.label.set_fontsize(self.lable_fontsize)
            ax.major_ticklabels.set_fontsize(self.tick_labelsize)
            ax.line.set_linewidth(self.tick_width)
            ax.major_ticklabels.set_pad(self.tick_pad)
            ax.label.set_pad(self.tick_pad)
            ax.major_ticks.set(linewidth=self.tick_width,ticksize=self.tick_length)
            ax.offsetText.set_size(self.offset_fontsize)

        par1.axis['right'].label.set_fontsize(self.lable_fontsize)
        par1.axis['right'].major_ticklabels.set_fontsize(self.tick_labelsize)
        par1.axis['right'].line.set_linewidth(self.tick_width)
        par1.axis['right'].major_ticklabels.set_pad(self.tick_pad)
        par1.axis['right'].label.set_pad(self.tick_pad)
        par1.axis['right'].major_ticks.set(linewidth=self.tick_width,ticksize=self.tick_length)
        par1.axis['right'].offsetText.set_size(self.offset_fontsize)

        par2.axis['right'].label.set_fontsize(self.lable_fontsize)
        par2.axis['right'].major_ticklabels.set_fontsize(self.tick_labelsize)
        par2.axis['right'].line.set_linewidth(self.tick_width)
        par2.axis['right'].major_ticklabels.set_pad(self.tick_pad)
        par2.axis['right'].label.set_pad(self.tick_pad)
        par2.axis['right'].major_ticks.set(linewidth=self.tick_width,ticksize=self.tick_length)
        par2.axis['right'].offsetText.set_size(self.offset_fontsize)


        par1.set(ylim=y2lim)
        par1.set_ylabel(ylabel2, fontsize=self.lable_fontsize,color=self.colors[1])
        par1.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        # par1_y_formatter = ticker.ScalarFormatter(useMathText=True)
        # par1.yaxis.set_major_formatter(par1_y_formatter)
        par1.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad,colors=self.colors[1])
        par1.spines['right'].set_color(self.colors[1])  # 左侧坐标轴
        par1.spines['left'].set_color(self.colors[0])  # 左侧坐标轴

        par2.set(ylim=y3lim)
        par2.set_ylabel(ylabel3, fontsize=self.lable_fontsize, color=self.colors[1])
        par2.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        # par2_y_formatter = ticker.ScalarFormatter(useMathText=True)
        # par2.yaxis.set_major_formatter(par2_y_formatter)
        par2.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                         labelsize=self.tick_labelsize, pad=self.tick_pad, colors=self.colors[1])
        par2.spines['right'].set_color(self.colors[1])  # 左侧坐标轴
        par2.spines['left'].set_color(self.colors[0])  # 左侧坐标轴





        # 合并图例
        lines = [line1,line2,line3]
        labels = [l.get_label() for l in lines]
        host.legend(lines, labels, prop={'size': self.legendsize}, loc='upper left', bbox_to_anchor=bbox_to_anchor)

        # 保存图表
        # save_path = os.path.join(self.pic_save_dir, save_file)
        # plt.savefig(save_path)
        self.save_fig(plt,save_file)
        plt.close()

    def plotduibi(self, x, y,save_file, xlabel, ylabel, label, markevery, tickin, ylim,
                  xoffset=None,bbox_to_anchor=(0.1,0.95),xticklabels=None):
        if xoffset is None:
            xoffset = [0.0, 0.0]

        lines = None
        print(len(x),len(y))
        # 创建图表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_line_width)

        lines = ax.plot(x, y[0], marker=self.markerstyles[0], markevery=markevery,
                         markerfacecolor=self.markerfacecolor[0],
                         markersize=self.markersize, markeredgewidth=self.line_width,
                         linestyle=self.linestyle[0],
                         color=self.colors[0], label=label[0], linewidth=self.line_width)

        # 绘制第一个纵坐标
        for i in range(len(y)-1):
            lines += ax.plot(x, y[i+1], marker=self.markerstyles[i+1], markevery=markevery,
                             markerfacecolor=self.markerfacecolor[i+1],
                             markersize=self.markersize, markeredgewidth=self.line_width,
                             linestyle=self.linestyle[i+1],
                             color=self.colors[i+1], label=label[i+1], linewidth=self.line_width)


        # 坐标标签字体
        ax.set_xlabel(xlabel, fontsize=self.lable_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.lable_fontsize)

        # 坐标轴字体
        ax.xaxis.get_offset_text().set_fontsize(self.offset_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(self.offset_fontsize)

        # 坐标轴刻度
        ax.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        ax.set_ylim(ylim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tickin))

        # x10科学计数法
        ax_x_formatter = ticker.ScalarFormatter(useMathText=True)
        ax_y_formatter = ticker.ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(ax_x_formatter)
        ax.yaxis.set_major_formatter(ax_y_formatter)

        # 设置网格和刻度
        ax.grid(True, alpha=self.grid_alpha, color=self.grid_color, linestyle=self.grid_line,
                linewidth=self.grid_line_width)

        if xticklabels is not None:
            ax.set_xticks(x, labels=xticklabels)

        ax.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)
        ax.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width,
                       labelsize=self.tick_labelsize, pad=self.tick_pad)

        # 上方横坐标
        new_x = ax.twiny()
        new_x.set_xlim(min(x) - xoffset[0], max(x) + xoffset[1])
        new_x.set_xticklabels([])
        new_x.tick_params(axis='x', which='major', direction='in', length=self.tick_length, width=self.tick_width)
        new_x.xaxis.set_major_locator(ticker.MultipleLocator(tickin))

        # 右边坐标
        new_y = ax.twinx()
        new_y.set_ylim(ylim)
        # new_y.yaxis.set_major_locator(ticker.MultipleLocator(tickin))
        new_y.set_yticklabels([])
        new_y.tick_params(axis='y', which='major', direction='in', length=self.tick_length, width=self.tick_width)

        # new_x.spines['left'].set_color(self.colors[0])  # 左侧坐标轴
        # new_x.spines['right'].set_color(self.colors[1])  # 右侧坐标轴

        # 图例
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, prop={'size': self.legendsize}, loc='upper left', bbox_to_anchor=bbox_to_anchor)

        # 保存图表
        # save_path = os.path.join(self.pic_save_dir, save_file)
        # plt.savefig(save_path)
        self.save_fig(plt,save_file)

        plt.close()

