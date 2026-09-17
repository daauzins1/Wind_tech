#reading txt

import pandas as pd 
import numpy as np

cylinder = pd.read_csv('cylinder.txt', sep= '\t',header =None, index_col=0,dtype={0: str})
blade600 = pd.read_csv('FFA-W3-600.txt', sep= '\t',header =None, index_col=0,dtype={0: float})
blade480 = pd.read_csv('FFA-W3-480.txt', sep= '\t',header =None, index_col=0,dtype={0: float})
blade360 = pd.read_csv('FFA-W3-360.txt', sep= '\t',header =None, index_col=0,dtype={0: float})
blade301 = pd.read_csv('FFA-W3-301.txt', sep= '\t',header =None, index_col=0,dtype={0: float})
blade241 = pd.read_csv('FFA-W3-241.txt', sep= '\t',header =None, index_col=0,dtype={0: float})

#print(cylinder.loc['180'])
#1 0.0 CL
#2 0.6 CD
#3 0.0 CM

alpha = np.linspace(-20,40,31)
alpha_HD = np.linspace(-20,40,601)
print(alpha_HD) #[-20. -15. -10.  -5.   0.   5.  10.  15.  20.  25.  30.  35.  40.]
cl_list = []
cd_list = []

for angle in alpha:
    cl = cylinder.loc[str(int(angle))][1]
    cd = cylinder.loc[str(int(angle))][2]
    cl_list.append(cl)
    cd_list.append(cd)

#print(cl_list)
cl_list_HD = np.interp(alpha_HD,alpha,cl_list)
cd_list_HD = np.interp(alpha_HD,alpha,cd_list)
#print(cl_list_HD)

df = pd.DataFrame({'alpha':alpha_HD,'Cl':cl_list_HD, 'Cd':cd_list_HD})
df.to_csv('Cylinder_HD',sep = ',', index=False)