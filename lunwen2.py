import numpy as np
import numpy.random
from new_func.alldata import *
from new_func.ploter import *
import time

time_begin = time.time()
#
# do_ris_Q()
# do_ris_if_Q()
#
#
# do_ris_difL()
# do_ris_difPHAP()
#
#
# difM.do_irs_difM()
# difM.do_irs_if_difM()
# difM.do_no_irs_na_difM()
# difM.do_no_irs_if_difM()
# difM.do_fix_irs_difM()
# difM.do_fix_irs_if_difM()
#
# difW.do_irs_difW()
# difW.do_irs_if_difW()
# difW.do_no_irs_na_difW()
# difW.do_no_irs_if_difW()
# difW.do_fix_irs_difW()
# difW.do_fix_irs_if_difW()

# difL.do_irs_difL()
# difL.do_irs_if_difL()
# difL.do_no_irs_na_difL()
# difL.do_no_irs_if_difL()
# difL.do_fix_irs_difL()
# difL.do_fix_irs_if_difL()

# difP.do_irs_difP()
# difP.do_irs_if_difP()
# difP.do_no_irs_na_difP()
# difP.do_no_irs_if_difP()
# difP.do_fix_irs_difP()
# difP.do_fix_irs_if_difP()

# difBeta.do_irs_difBeta()
# difBeta.do_irs_if_difBeta()
# difBeta.do_no_irs_na_difBeta()
# difBeta.do_no_irs_if_difBeta()
# difBeta.do_fix_irs_difBeta()
# difBeta.do_fix_irs_if_difBeta()


# 以下是迪修改的代码 花禁止触碰
# 一 花的演示实例
# irs_difM = readdata("difM_irs")
# irs_if_difM=readdata("difM_irs_if")
# no_irs_na_difM=readdata("difM_no_irs_na")
# no_irs_if_difM=readdata("difM_no_irs_if")
# fix_irs_difM=readdata("difM_fix_irs")
# fix_irs_if_difM=readdata("difM_fix_irs_if")

# 2 beta组
# irs_difBet = readdata("difBet_irs")
# irs_if_difBet=readdata("difBeta_irs_if")
# no_irs_na_difBet=readdata("difBeta_no_irs_na")
# no_irs_if_difBet=readdata("difBeta_no_irs_if")
# fix_irs_difBet=readdata("difBeta_fix_irs")
# fix_irs_if_difBet=readdata("difBeta_fix_irs_if")

# 3 L组
# irs_difL = readdata("difL_irs")
# irs_if_difL=readdata("difL_irs_if")
# no_irs_na_difL=readdata("difL_no_irs_na")
# no_irs_if_difL=readdata("difL_no_irs_if")
# fix_irs_difL=readdata("difL_fix_irs")
# fix_irs_if_difL=readdata("difL_fix_irs_if")

# 4 P组
# irs_difP = readdata("difP_irs")
# irs_if_difP=readdata("difP_irs_if")
# no_irs_na_difP=readdata("difP_no_irs_na")
# no_irs_if_difP=readdata("difP_no_irs_if")
# fix_irs_difP=readdata("difP_fix_irs")
# fix_irs_if_difP=readdata("difP_fix_irs_if")

# 5 W组
# irs_difW = readdata("difW_irs")
# irs_if_difW = readdata("difW_irs_if")
# no_irs_na_difW = readdata("difW_no_irs_na")
# no_irs_if_difW = readdata("difW_no_irs_if")
# fix_irs_difW = readdata("difW_fix_irs")
# fix_irs_if_difW = readdata("difW_fix_irs_if")

# ite组
# 读取数据
iteL10_irs_difPHAP = readdata("difPHAP_ris_iteL10")
difP_ris_ite = readdata("difPHAP_ris_ite")
difL_ris_ite=readdata("difL_ris_ite")

plotter2 = Plotter()
plotter2.line_width = 2
plotter2.markersize = 10
plotter2.figsize = (12, 8)
plotter2.linestyle = ['--', '-', '--', '-', '--', '-', '--', '-', '--', '-']
plotter2.markerstyles = ['^', 'o', 'p', 's', '*', '+', 'v']

# ite组
plotter2.plotduibiite(difP_ris_ite[-1], [iteL10_irs_difPHAP[1],iteL10_irs_difPHAP[5],difL_ris_ite[13]],
                   save_file='duibi_ite.pdf',
                   xlabel=r'Iteration idex', ylabel='Minimum rate (bit/s)',
                   label=['L=10,$P_{HAP}=10dBm$', 'L=10,$P_{HAP}=30dBm$', 'L=32,$P_{HAP}=30dBm$'],
                   tickin=4, ytickin=0.5, ylim=(1.5, 4.5), markevery=1,
                   bbox_to_anchor=(0.59, 0.30))

# beta组
# plotter2.plotduibi(irs_difBet[-1], [irs_difBet[0],irs_if_difBet[0],no_irs_na_difBet[0],no_irs_if_difBet[0],fix_irs_difBet[0],fix_irs_if_difBet[0]],
#                    save_file='duibi_difBeta_RATE.pdf',
#                    xlabel=r'Path loss exponents $\beta_1$ and $\beta_2$', ylabel='Minimum rate (bit/s)',
#                    label=['NOMA with AR-IRS', 'OMA with AR-IRS','NOMA without IRS','OMA without IRS','NOMA with GR-IRS','OMA with GR-IRS'],
#                    tickin=0.2, ytickin=0.5,ylim=(1.0,5.0), markevery=1,
#                    bbox_to_anchor=(0.20,0.99))

# L组
# plotter2.plotduibi(irs_difL[-1], [irs_difL[0],irs_if_difL[0],no_irs_na_difL[0],no_irs_if_difL[0],fix_irs_difL[0],fix_irs_if_difL[0]],
#                    save_file='duibi_difL_RATE.pdf',
#                    xlabel=r'Numbers of IRS reflecting elements', ylabel='Minimum rate (bit/s)',
#                    label=['NOMA with AR-IRS', 'OMA with AR-IRS','NOMA without IRS','OMA without IRS','NOMA with GR-IRS','OMA with GR-IRS'],
#                    tickin=10, ytickin=0.5,ylim=(1.0,5.0), markevery=1,
#                    bbox_to_anchor=(0.20,0.99))

# P组
# plotter2.plotduibi(irs_difP[-1], [irs_difP[0],irs_if_difP[0],no_irs_na_difP[0],no_irs_if_difP[0],fix_irs_difP[0],fix_irs_if_difP[0]],
#                    save_file='duibi_difP_RATE.pdf',
#                    xlabel=r'HAP maximum transmit power $P_{HAP}$(dBm)', ylabel='Minimum rate (bit/s)',
#                    label=['NOMA with AR-IRS', 'OMA with AR-IRS','NOMA without IRS','OMA without IRS','NOMA with GR-IRS','OMA with GR-IRS'],
#                    tickin=5, ytickin=0.5,ylim=(0.5,5.0), markevery=1,
#                    bbox_to_anchor=(0.05,0.99))

# W组
# plotter2.plotduibi(irs_difW[-1], [irs_difW[0],irs_if_difW[0],no_irs_na_difW[0],no_irs_if_difW[0],fix_irs_difW[0],fix_irs_if_difW[0]],
#                    save_file='duibi_difW_RATE.pdf',
#                    xlabel=r'Bandwidth $W$(Hz)', ylabel='Minimum rate (bit/s)',
#                    label=['NOMA with AR-IRS', 'OMA with AR-IRS','NOMA without IRS','OMA without IRS','NOMA with GR-IRS','OMA with GR-IRS'],
#                    tickin=0.5, ytickin=1.0,ylim=(1.0,12.0), markevery=1,
#                    bbox_to_anchor=(0.05,0.99))




# plotter2.plotduibi(difga_proposed[-1], [difga_proposed[1]*10**6,difga_UC_BAC[1]*10**6,difga_UC_AC[1]*10**6,difga_nocoop[1]*10**6],
#                    save_file='duibi_difga_EE.pdf',tickin=1, markevery=1,
#                    xlabel=r'The performance gap (dB)', ylabel='Energy efficiency (EE) (bits/joule)',
#                    xoffset=[0,-1],label=['The Proposed Scheme','The UC with the BAC','The UC with the AC','Without UC with the integrated BAC and AC'],
#                     ytickin=0.5*10**6,ylim=(0*10**6,4*10**6),
#                    bbox_to_anchor=(0.02,0.98))


# plotter2.plotduibi(difW_proposed[-1]*10**6, [difW_proposed[1]*10**6,difW_UC_BAC[1]*10**6,difW_UC_AC[1]*10**6,difW_nocoop[1]*10**6],
#                    save_file='duibi_difW_EE.pdf',tickin=0.05*10**6, markevery=1,
#                    xlabel=r'Bandwidth $W$ (Hz)', ylabel='Energy efficiency (EE) (bits/joule)',
#                    xoffset=[0,0],label=['The Proposed Scheme','The UC with the BAC','The UC with the AC','Without UC with the integrated BAC and AC'],
#                     ytickin=0.5*10**6,ylim=(0*10**6,4*10**6),
#                    bbox_to_anchor=(0.01,0.98))
#
# plotter2.plotduibi(difmiu_proposed[-1], [difmiu_proposed[1]*10**6,difmiu_UC_BAC[1]*10**6,difmiu_UC_AC[1]*10**6,difmiu_nocoop[1]*10**6],
#                    save_file='duibi_difmiu_EE.pdf',tickin=0.5, markevery=1,
#                    xlabel=r'Wight $\omega$ ', ylabel='Energy efficiency (EE) (bits/joule)',
#                    xoffset=[0,0],label=['The Proposed Scheme','The UC with the BAC','The UC with the AC','Without UC with the integrated BAC and AC'],
#                     ytickin=0.2*10**6,ylim=(1*10**6,3*10**6),
#                    bbox_to_anchor=(0.30,0.98))
#
# plotter2.plotyy(difA_EE[-1]*10**6, difA_EE[1]*10**6,difA_EE[2]*10**6,
#                    save_file='duibi_difA_EE.pdf',tickin=0.1*10**6, markevery=1,
#                    xlabel=r'Average data arrival rate at $MD_2$ $\lambda_2$ (bits/s)', ylabel1='Energy efficiency EE (bits/joule)',ylabel2='Average queue length $\overline{Q} (bits)$',
#                    xoffset=[0,0],label1='Energy efficiency',label2='Task queue',
#                     ytickin1=1*10**5,y1lim=(13*10**5,20*10**5),ytickin2=4*10**7,y2lim=(4*10**7,28*10**7),
#                    bbox_to_anchor=(0.02,0.98))
# #
# plotter2.plotyy(difV_EE[-1], difV_EE[1]*10**6,difV_EE[2]*10**6,
#                    save_file='duibi_difV_EE.pdf',tickin=10, markevery=1,
#                    xlabel=r'$V$', ylabel1='Energy efficiency EE (bits/joule)',ylabel2='Average queue length $\overline{Q} (bits)$',
#                    xoffset=[0,0],label1='Energy efficiency',label2='Task queue',
#                     ytickin1=0.2*10**6,y1lim=(0.8*10**6,2.6*10**6),ytickin2=1*10**7,y2lim=(20*10**6,110*10**6),
#                    bbox_to_anchor=(0.02,0.98))
# # # #
# plotter2.plotduibi1(difD_EE[-1], difD_EE[1]*10**6,
#                    save_file='duibi_difD_EE.pdf',tickin=10, markevery=1,
#                    xlabel=r'Distance between the $MD_1$ and the $MD_2$ (m)', ylabel='Energy efficiency EE (bits/joule)',
#                    xoffset=[6,6],label='Energy efficiency',xticks=[80,90,100,110,120,130,140,150,160,170],xticklabels=['80','90','100','110','120','130','140','150','160','170'],
#                     ytickin=0.2*10**6,ylim=(1*10**6,3*10**6),
#                    bbox_to_anchor=(0.02,0.98))


time_end = time.time()
print("time", time_end - time_begin)
