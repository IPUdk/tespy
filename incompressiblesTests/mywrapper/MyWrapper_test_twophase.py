import numpy as np
from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper, CoolPropWrapper
#from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper
from tespy.tools.global_vars import gas_constants
#from tespy.tools.fluid_properties.CustomWrapper import CustomWrapper
from myWrapper import *
import logging
#logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
#import numpy as np

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
                            # "dG"   : [8.62355442e+01,-1.05732250e+00,4.90467264e-03,-1.02406943e-05,8.15327490e-09],
                            "cpG"  : [ 4.70848101e+02,1.13556451e+01,-2.07921505e-02,-3.88616225e-05,1.18035083e-07],     
                            # "VanDerWall" : [-1717.9874574726448, -0.02306278086667577],         
                            # "ci" : [1.34379777e+03,-6.04347700e-02,9.03468908e-04,-3.62413830e-07]
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
T = np.linspace(274.15,273.15+150,N)
psat = np.zeros(N)
psatfit = np.zeros(N)
for i in range(N):
    psat[i] = CPW.p_sat(T[i])
    psatfit[i] = myWrapper.p_sat(T[i])

fig, ax = plt.subplots(1,1)
ax.scatter(psat,T)    
ax.plot(psatfit,T)    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')






# latent
hfg = np.array([CPW.h_QT(1.0,_T)-CPW.h_QT(0.0,_T) for _T in T])
hfgfit = np.array([myWrapper.hfg_pT(1e5,T[i]) for i in range(N)])
fig, ax = plt.subplots(1,1)
ax.scatter(T,hfg)    
ax.plot(T,hfgfit)   
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

d = np.array([CPW.d_QT(1.0,_T) for _T in T])
dfit = np.array([myWrapper.d_pT(1e5,T[i]) for i in range(N)])
fig, ax = plt.subplots(1,1)
ax.scatter(T,d)    
ax.plot(T,dfit,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')


cp = np.array([CPW.cp_QT(1.0,_T) for _T in T])
cpfit = np.array([myWrapper.cp_pT(1e5,T[i]) for i in range(N)])
fig, ax = plt.subplots(1,1)
ax.scatter(T,cp)    
ax.plot(T,cpfit,'rx')    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

plt.show(block=True)


_T, _p, cp, h ,s ,d = [],[],[],[],[],[]
cp_, h_ ,s_ ,d_ = [],[],[],[]
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
            h.append(CPW.h_pT(max(1000,p), T[i]))
            s.append(CPW.s_pT(max(1000,p), T[i]))
            d.append(CPW.d_pT(max(1000,p), T[i]))
            cp_.append(myWrapper.cp_pT(max(1000,p), T[i]))
            h_.append(myWrapper.h_pT(max(1000,p), T[i]))
            s_.append(myWrapper.s_pT(max(1000,p), T[i]))
            d_.append(myWrapper.d_pT(max(1000,p), T[i]))

_T = np.array(_T)
_p = np.array(_p)
cp = np.array(cp)
h = np.array(h)
s = np.array(s)
d = np.array(d)

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, cp)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('cp')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, h)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('h')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, s)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('s')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, d)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d')
ax.set_zlim([0,10])

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, cp_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('cp')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, h_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('h')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, s_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('s')

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, d_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d')
ax.set_zlim([0,10])
plt.show(block=True)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, cp-cp_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d error')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, h-h_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d error')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, s-s_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d error')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(_T, _p, d-d_)
ax.set_xlabel('T')
ax.set_ylabel('p')
ax.set_zlabel('d error')
plt.show(block=True)

# for i in range(N):
#     cp[i,N] = CPW.cp_QT(0.0, T[i])
#     psat[i] = CPW.p_sat(0.0, T[i])

T = np.linspace(274,372,99)
cpL = [CPW.cp_pT(1e5, t) for t in T]
fit = np.polyfit(T,cpL,4)
print(np.flip(fit))
cpLfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,cpL)    
ax.plot(T,cpLfit,'rx')    
plt.show(block=True)

dL = [CPW.d_pT(1e5, t) for t in T]
fit = np.polyfit(T,dL,4)
print(np.flip(fit))
dLfit = np.polyval(fit, T)
fig, ax = plt.subplots(1,1)
ax.scatter(T,dL)    
ax.plot(T,dLfit,'rx')    
plt.show(block=True)






print("done")