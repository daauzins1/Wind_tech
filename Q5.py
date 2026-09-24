import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('power_data.txt',header=None,sep= " ")
power = df.iloc[1]
print(power)
#plt.plot(wind,power)
#plt.show()

#q5 weibull distribution
A = 9
k = 1.9

res = 50
V  = np.linspace(4,25,res)
V_interp = np.linspace(4,25,res*2)
h_list = [] #hours of windspeed per year
power_list = []
for V0 in V_interp:
    h = k/A * (V0/A)**(k-1)*np.exp(-(V0/A)**k) #weibull graph gives probability values 0-1
    h_list.append(h*8760) #actual hours per year of certain windspeed
    Power_at_windspeed = np.interp(V0,V,power)/1000/1000
    power_list.append(Power_at_windspeed)
V_desired = np.linspace(1,25,res)
KWh = np.array(power_list) * np.array(h_list)
total_power = np.trapezoid(KWh,V_interp)
print(np.shape(h_list))
print(np.shape(V))
print("MWh per year: ")
print(total_power)









#plt.plot(V,h_list)
#plt.show()