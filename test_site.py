
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# READ DATA
# ============================================================

# Use whitespace for ALL files
cylinder = pd.read_csv('cylinder.txt', sep=r'\s+', header=None)
blade_data = pd.read_csv('bladedat.txt', sep=r'\s+', header=None)

file_names = ['FFA-W3-600.txt','FFA-W3-480.txt','FFA-W3-360.txt','FFA-W3-301.txt','FFA-W3-241.txt']
lists = [[] for _ in range(len(file_names))]
for num, file in enumerate(file_names):
    lists[num] = pd.read_csv(file, sep=r'\s+', header=None)


for i in range(len(lists)):
    print('NUMBER '+ str(i))
    print(lists[i])