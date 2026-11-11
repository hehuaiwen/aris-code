import numpy as np
import numpy.random
from new_func.alldata import *
from new_func.ploter import *
import time
time_begin = time.time()
#
# do_bcd_optimization()
# do_UC_AC()
# do_UC_BAC()
# do_nocoop()
#
do_ris_difPHAP_L10()
# do_ris_difL()
# do_ris_difPHAP()
#
# #
# difM.do_irs_difM()
# difM.do_irs_if_difM()
# difM.do_no_irs_na_difM()
# difM.do_no_irs_if_difM()
# difM.do_fix_irs_difM()
# difM.do_fix_irs_if_difM()

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
#
# difP.do_irs_difP()
# difP.do_irs_if_difP()
# difP.do_no_irs_na_difP()
# difP.do_no_irs_if_difP()
# difP.do_fix_irs_difP()
# difP.do_fix_irs_if_difP()





# proposed = readdata("proposed")
# UC_BAC=readdata("UC_BAC")
# UC_AC=readdata("UC_AC")
# nocoop=readdata("nocoop")
#
#
# difL_proposed=readdata("difL_proposed")
# dif_UC_BAC=readdata("difga_UC_BAC")
# difga_UC_AC=readdata("difga_UC_AC")
# difga_nocoop=readdata("difga_nocoop")
#
#
# difPH_proposed = readdata("difPH_proposed")
# difW_UC_BAC=readdata("difW_UC_BAC")
# difW_UC_AC=readdata("difW_UC_AC")
# difW_nocoop=readdata("difW_nocoop")
#
#
# difmiu_proposed = readdata("difmiu_proposed")
# difmiu_UC_BAC=readdata("difmiu_UC_BAC")
# difmiu_UC_AC=readdata("difmiu_UC_AC")
# difmiu_nocoop=readdata("difmiu_nocoop")
#
# difV_EE=readdata("difV_EE")
# difA_EE=readdata("difA_EE")
# difD_EE=readdata("difD_EE")


plotter2 = Plotter()
plotter2.line_width = 2
plotter2.markersize = 10
plotter2.figsize = (12,8)
plotter2.linestyle = ['--', '-', '--', '-', '--', '-', '--','-', '--','-']
plotter2.markerstyles =['^','o', 'p','s','*', '+', 'v']

# #
# plotter2.plotduibi(proposed[-1],[proposed[2]],save_file='main.pdf',tickin=5, markevery=1,
#                      xlabel='Time slots', ylabel='Energy efficiency (EE) (bits/joule)',bbox_to_anchor=(0.025,0.355),
#                  xoffset=[0,1],label=['The Proposed Scheme','The UC with the BAC','The UC with the AC','Without UC with the integrated BAC and AC'],ytickin=0.4*10**6,ylim=(0*10**6,2.4*10**6))
# # # #
# plotter2.plotduibi(difL_proposed[-1], [difL_proposed[2],difL_proposed[5]],
#                    save_file='duibi_difL_RATE.pdf',
#                    xlabel=r'iteration numbers', ylabel='Minimum rate',
#                    label=['$L=5$', '$L=10$'],
#                    tickin=1, ytickin=1,ylim=(0.0,5.0), markevery=1,
#                    bbox_to_anchor=(0.02,0.98))

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
print("time",time_end-time_begin)

