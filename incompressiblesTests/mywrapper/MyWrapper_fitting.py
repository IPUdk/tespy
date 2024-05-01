import numpy as np
from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper, CoolPropWrapper
#from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper
from tespy.tools.global_vars import gas_constants
from tespy.tools.fluid_properties.CustomWrapper import CustomWrapper as MyWrapper
import logging
#logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
#import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import linregress

# Define the Antoine equation function
def antoine_equation(P, A, B, C):
    return B / (np.log(P) - A) - C


myWrapper = MyWrapper("CUSTOM::WaterTwoPhase", 
                      Tref = 273.15, 
                      coefs = {
                        "CUSTOM::WaterTwoPhase": {
                            "name": "Custom water polynomial", 
                            "unit": "K", 
                            "cp": [7.79605665e+04,-1.12106166e+03,7.06771540e+00,-2.36638219e-02,4.43721794e-05,-4.41973243e-08,1.83159953e-11],
                            "d" : [1.35188573e+02,8.66049556e+00,-3.06549945e-02,4.62728683e-05,-2.80708081e-08],
                            "hfg"  : [3.73992983e+06, -8.02594391e+03, 1.80890144e+01, -1.93816772e-02],
                            "Tsat" : [23.22646886130465, -3842.204328212032, -44.75853983190677],
                            "dG"   : [8.62355442e+01,-1.05732250e+00,4.90467264e-03,-1.02406943e-05,8.15327490e-09],
                            "cpG"  : [ 4.70848101e+02,1.13556451e+01,-2.07921505e-02,-3.88616225e-05,1.18035083e-07],     
                            "VanDerWall" : [-1717.9874574726448, -0.02306278086667577]
                        }
                      }
                      )

cp = myWrapper.cp_pT(1e5, 273.15+15)
h = myWrapper.h_pT(1e5, 273.15+15)
T = myWrapper.T_ph(1e5, h)
print(T,h,cp)

CPW = CoolPropWrapper("Water")
h = CPW.h_pT(1e5, 273.15+15)
T = CPW.T_ph(1e5, h)
cp = CPW.cp_pT(1e5, 273.15+15)
print(T,h,cp)


N = 50
T = np.linspace(373.15,373.15+500,N)
G = [4.6e4,1.011249e3,8.3893e-1,-2.19989e-4,2.466190e-7,-9.7047e-11]
_cv = sum([G[i]*T**(i+1-2) for i in range(6)])
cv = np.array([CPW.cv_pT(1000, _T) for _T in T])

fit = np.polyfit(T, cv, 3)
cvfit = np.polyval(fit, T)
print(np.flip(fit))
slope, intercept, r_value, p_value, std_err = linregress(cv, cvfit)

T = np.linspace(273.2,373.15+500,N)
_cv = sum([G[i]*T**(i+1-2) for i in range(6)])
cv = np.array([CPW.cv_pT(1000, _T) for _T in T])

cvfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(cv,T)
ax.plot(_cv,T)
ax.plot(cvfit,T,'rx')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

plt.show(block=True)

N = 50
T = np.linspace(274.15,273.15+250,N)
#cp = np.zeros(N,N)
psat = np.zeros(N)

for i in range(N):
    psat[i] = CPW.p_sat(T[i])

# from scipy.stats import linregress
# fit = np.polyfit(psat, T, 4)
# Tfit = np.polyval(fit, psat)
# slope, intercept, r_value, p_value, std_err = linregress(T, Tfit)

# The Antoine-equation
A = 23.5771
B = -4042.90
C = -37.58
Tfit_orig = antoine_equation(psat,A,B,C) # B/(np.log(psat)-A)-C

# Fit the Antoine equation to the data
popt, pcov = curve_fit(antoine_equation, psat, T, p0=[23.5771, -4042.90, -37.58])
A, B, C = popt
Tfit_new = antoine_equation(psat,A,B,C) # B/(np.log(psat)-A)-C
print(A,B,C)
fig, ax = plt.subplots(1,1)
ax.scatter(psat,T)    
ax.plot(psat,Tfit_orig)    
ax.plot(psat,Tfit_new)    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
#plt.show(block=False)


# latent
hfg = np.array([CPW.h_QT(1.0,_T)-CPW.h_QT(0.0,_T) for _T in T])
fit = np.polyfit(T, hfg, 3)
print(fit)
hfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,hfg)    
ax.plot(T,hfit)    

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

DPDewDT = 1/((antoine_equation(psat+1e-6,A,B,C)-antoine_equation(psat,A,B,C))/1e-6)
Vgas = hfg/(DPDewDT*T)+0.001
# DPDewDT(TDew)*TDew*(VGAS-VL)

# test ideal gas law
R = 8314/18.02 
di = psat/(R*T)
d = np.array([CPW.d_QT(1.0,_T) for _T in T])

fit = np.polyfit(T,d,2)
print(np.flip(fit))
dfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,d)    
ax.plot(T,di)    
ax.plot(T,1/Vgas)    
ax.plot(T,dfit,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.show(block=True)

v = 1/d
fit = np.polyfit(T,v,5)
vfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,v)    
ax.plot(T,vfit,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.show(block=True)

def van_der_waals(x, a, b):
    T=x[1]
    R = 8314/18.02  # Gas constant in J/(mol*K)
    V=1/x[0]
    return R * T / (V - b) - a / V**2
   
popt, pcov = curve_fit(van_der_waals, (d,T), psat, p0=[0.0001, 0.0001])

a, b = popt
pfit_new = van_der_waals((d,T),a,b) # B/(np.log(psat)-A)-C

fit = np.polyfit(T,d,2)
print(np.flip(fit))

fig, ax = plt.subplots(1,1)
ax.scatter(T,psat)    
ax.plot(T,pfit_new,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

cp = np.array([CPW.cp_QT(1.0,_T) for _T in T])
fit = np.polyfit(T,cp,4)
print(np.flip(fit))
cpfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,cp)    
ax.plot(T,cpfit,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

plt.show(block=True)





_T, _p, cp = [],[],[]
for p in psat:
    for i in range(N):
        Tsat = CPW.T_sat(max(1000,p))
        if T[i] >= Tsat-0.01 and T[i] <= Tsat+0.01:
            continue
        else:
            print(T[i],p)
            _T.append(T[i])
            _p.append(p)
            cp.append(CPW.cp_pT(max(1000,p), T[i]))


_T = np.array(_T)
_p = np.array(_p)
cp = np.array(cp)



# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, cp)

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
#ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot Example')
# Show plot
plt.show(block=True)

# for i in range(N):
#     cp[i,N] = CPW.cp_QT(0.0, T[i])
#     psat[i] = CPW.p_sat(0.0, T[i])

T = np.linspace(274,274+250,200)
cpL = [CPW.cp_pT(max(1e5,CPW.p_sat(t)*1.0002), t) for t in T]
fit = np.polyfit(T,cpL,6)
print(np.flip(fit))
cpLfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,cpL)    
ax.plot(T,cpLfit,'rx')    
plt.show(block=True)

dL = [CPW.d_pT(max(1e5,CPW.p_sat(t))*1.0002, t) for t in T]
fit = np.polyfit(T,dL,4)
print(np.flip(fit))
dLfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,dL)    
ax.plot(T,dLfit,'rx')    
plt.show(block=True)



print("done")