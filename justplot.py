from matplotlib import pyplot as plt
import numpy as np
import random

A = [15,17,13,16,18,20,16,15,15]
B = [19,21,24,17,21,20,25,21,28]
C = [25,26,31,32,32,35,20]

plt.rcParams.update({'font.size': 22})
plt.boxplot([A,B,C])
plt.xticks([1,2,3],['No.5','No.3','No.1'])
#plt.xlabel('nth leaves')
plt.ylabel('Age of clock (Days)')
#plt.boxplot(B)
#plt.boxplot(C)
plt.show()