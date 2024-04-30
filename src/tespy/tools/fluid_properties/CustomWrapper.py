import CoolProp.CoolProp as CP
from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper, CoolPropWrapper
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class CustomWrapper(FluidPropertyWrapper):
    def __init__(self, fluid, back_end=None, Tref = 293.15, coefs=[]) -> None:
        super().__init__(fluid, back_end)
        if self.fluid not in coefs:
            msg = "Fluid not available in database"
            raise KeyError(msg)

        # get coefs (converted to kelvin) and calculate reference
        self.T0 = Tref
        self.get_coefs(coefs)
        
        self._molar_mass = 1
        self._T_min = 273.15 - 50
        self._T_max = 273.15 + 1000
        self._p_min = 1000
        self._p_max = 10000000

    def get_coefs(self, coefs):

        self.C_c = coefs[self.fluid]["cp"]
        while self.C_c and self.C_c[-1] == 0.0:
            self.C_c.pop()
        self.n_c = len([c for c in coefs[self.fluid]["cp"] if c != 0.0])
        
        self.C_d = coefs[self.fluid]["d"]
        while self.C_d and self.C_d[-1] == 0.0:
            self.C_d.pop()            
        self.n_d = len([c for c in coefs[self.fluid]["d"] if c != 0.0])
        
        if coefs[self.fluid].get("hfg"):
            self.C_hfg = coefs[self.fluid]["hfg"]
            while self.C_hfg and self.C_hfg[-1] == 0.0:
                self.C_hfg.pop()            
            self.n_hfg = len([c for c in coefs[self.fluid]["hfg"] if c != 0.0])

        if coefs[self.fluid].get("Tsat"):
            self.C_Tsat = coefs[self.fluid]["Tsat"]
            while self.C_Tsat and self.C_Tsat[-1] == 0.0:
                self.C_Tsat.pop()            
            self.n_Tsat = len([c for c in coefs[self.fluid]["Tsat"] if c != 0.0])

        if coefs[self.fluid].get("cpG"):
            self.C_cG = coefs[self.fluid]["cpG"]
            while self.C_cG and self.C_cG[-1] == 0.0:
                self.C_cG.pop()            
            self.n_cG = len([c for c in coefs[self.fluid]["cpG"] if c != 0.0])

        if coefs[self.fluid].get("ci"):
            self.C_ci = coefs[self.fluid]["ci"]
            while self.C_ci and self.C_ci[-1] == 0.0:
                self.C_ci.pop()            
            self.n_ci = len([c for c in coefs[self.fluid]["ci"] if c != 0.0])

        if coefs[self.fluid].get("dG"):
            self.C_dG = coefs[self.fluid]["dG"]
            while self.C_dG and self.C_dG[-1] == 0.0:
                self.C_dG.pop()            
            self.n_dG = len([c for c in coefs[self.fluid]["dG"] if c != 0.0])                     

        if coefs[self.fluid].get("VanDerWall"):
            self.C_vdw = coefs[self.fluid]["VanDerWall"]
            while self.C_vdw and self.C_vdw[-1] == 0.0:
                self.C_vdw.pop()            
            self.n_vdw = len([c for c in coefs[self.fluid]["VanDerWall"] if c != 0.0])                     

        self.TwoPhaseMedium = False
        if coefs[self.fluid]["unit"] == "C" and (coefs[self.fluid].get("hfg") or coefs[self.fluid].get("Tsat")):
            raise ValueError("TwoPhase medium must have fits in terms of K")
        elif not coefs[self.fluid].get("hfg") and coefs[self.fluid].get("Tsat"):
            raise ValueError("TwoPhase medium must have fits in both hfg and Tsat")
        elif coefs[self.fluid].get("hfg") and not coefs[self.fluid].get("Tsat"):
            raise ValueError("TwoPhase medium must have fits in both hfg and Tsat")
        elif coefs[self.fluid].get("hfg") and coefs[self.fluid].get("Tsat"):
            self.TwoPhaseMedium = True

        if coefs[self.fluid]["unit"] == "C":
            # convert coefficients
            T_C = np.linspace(1,50)
            cp = self.cp_pT(None,T_C)
            d = self.d_pT(None,T_C)
            T_K = np.linspace(1+273.15,50+273.15)
            self.C_c = list(np.polyfit(T_K, cp, self.n_c-1))
            self.C_c = self.C_c[::-1]
            self.C_d = list(np.polyfit(T_K, d, self.n_d-1))
            self.C_d = self.C_d[::-1]
        elif coefs[self.fluid]["unit"] == "K":
            pass     
        else:
            raise ValueError("unit is not C or K")

    def get_state(self, state = None):
        if state:
            if state == 'g':
                return 1
            elif state == 'l':
                return 0
        return None

    def T_sat(self, p):
        # antoine_equation
        return self.C_Tsat[1] / (np.log(p) - self.C_Tsat[0]) - self.C_Tsat[2]

    def p_sat(self, T):
        # antoine_equation
        return np.exp(self.C_Tsat[0] + self.C_Tsat[1]/(T + self.C_Tsat[2]))

    def cp_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # assume saturated 
                return np.sum([self.C_cG[i] * Tsat**i for i in range(self.n_cG)], axis=0)
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                return np.sum([self.C_c[i] * T**i for i in range(self.n_c)], axis=0)
        return np.sum([self.C_c[i] * T**i for i in range(self.n_c)], axis=0)
   
    # def cpG_T(self, T):
    #     return np.sum([self.C_cG[i] * T**i for i in range(self.n_cG)], axis=0)

    def van_der_waals(self,d,T,p,R):
        V=1/d
        return (R * T / (V - self.C_vdw[1]) - self.C_vdw[0] / V**2 - p)/p
    
    def d_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # calculate d at saturation conditions and use ideal gas law
                #dsat = np.sum([self.C_dG[i] * Tsat**i for i in range(self.n_dG)], axis=0)
                #return dsat + p/(8314/18.02*T) - p/(8314/18.02*Tsat)
                #return p/(8314/18.02*T)
                #return np.sum([self.C_dG[i] * T**i for i in range(self.n_dG)], axis=0)
                # R=8314/18.02
                # res = root_scalar(self.van_der_waals, x0=p/(R*T), args=(T,p,R), bracket=[0.001,100])
                # return res.root
                R=8314/18.02
                return p/(R*T)
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                return np.sum([self.C_d[i] * T**i for i in range(self.n_d)], axis=0)
        return np.sum([self.C_d[i] * T**i for i in range(self.n_d)], axis=0)

   
    # def dG_T(self, T, **kwargs):
    #     return np.sum([self.C_dG[i] * T**i for i in range(self.n_dG)], axis=0)    

    def hfg_pT(self, p, T):
        return np.sum([self.C_hfg[i] * T**i for i in range(self.n_hfg)], axis=0)
    
    def u_pT(self, p, T):
        integral = 0
        for i in range(self.n_c):
            integral += (1 / (i + 1)) * self.C_c[i] * (T**(i + 1) - self.T0**(i + 1))
        return integral 

    def ui_pT(self, p, T):
        integral = 0
        for i in range(self.n_ci):
            integral += (1 / (i + 1)) * self.C_ci[i] * (T**(i + 1) - self.T0**(i + 1))
        return integral 

    def h_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # we do not want to use gas at T, but Tsat as the fit is saturated (keep the pressure)
                
                df  = self.d_pT(p, Tsat)
                hf  = self.u_pT(p, Tsat) - p/df
                hfg = self.hfg_pT(p, Tsat)
                dg  = self.d_pT(p, Tsat, force_state='g')
                d   = self.d_pT(p, T, force_state='g')
                # hg     = self.ui_pT(p,T) - d*self.C_vdw[0] - p/d - \
                #          (self.ui_pT(p,Tsat) - dg*self.C_vdw[0] - p/dg)
                hg = self.cp_pT(p, Tsat, force_state='g')*(T-Tsat) - p/d + p/dg
                h = hf + hfg + hg
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                d = self.d_pT(p, T, force_state='l')
                h = self.u_pT(p, T) - p/d
            return h
        else:
            u = self.u_pT(p, T)
            d = self.d_pT(p, T)
            return u - p/d

    def s_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                integral = self.C_c[0] * np.log(min(T,Tsat) / self.T0)
                for i in range(self.n_c - 1):
                    integral += (1 / (i + 1)) * self.C_c[i + 1] * (min(T,Tsat)**(i + 1) - self.T0**(i + 1))            
                # we do not want to use gas at T, but Tsat as the fit is saturated (keep the pressure)
                return integral + self.hfg_pT(p, Tsat)/Tsat + (self.cp_pT(p, Tsat, force_state='g')/Tsat)*(T-Tsat)
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                integral = self.C_c[0] * np.log(T / self.T0)
                for i in range(self.n_c - 1):
                    integral += (1 / (i + 1)) * self.C_c[i + 1] * (T**(i + 1) - self.T0**(i + 1))                            
                return integral
        else:
            integral = self.C_c[0] * np.log(T / self.T0)
            for i in range(self.n_c - 1):
                integral += (1 / (i + 1)) * self.C_c[i + 1] * (T**(i + 1) - self.T0**(i + 1))
            return integral 
    
    def Tx_ph(self, p, h):
        Tsat = self.T_sat(p)
        hL =  self.h_pT(p,Tsat)
        hG =  self.h_pT(p,Tsat,force_state='g')
        if h >= hL and h <= hG:
            return Tsat, (h-hL)/(hG-hL)
        elif h<hL:
            return self.newton(self.h_pT, self.cp_pT, h, p, Tmax=Tsat, T=Tsat-10), 0.0
        else:
            return self.newton(self.h_pT, self.cp_pT, h, p, Tmin=Tsat, T=Tsat+10, force_state='g'), 1.0
   
    def Q_ph(self, p, h):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ph(p, h)
            return x
        return False

    def T_ph(self, p, h):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ph(p, h)
            return T
        return self.newton(self.h_pT, self.cp_pT, h, p)   
        
    def d_ph(self, p, h):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ph(p, h)
            if x>=0.0 and x<=1.0:
                return self.d_pQ(p, x)
            else:
                return self.d_pT(p, T)
        T = self.T_ph(p, h)
        return self.d_pT(p, T)
    
    def d_pQ(self, p, Q):
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            dg = self.d_pT(p, Tsat, force_state='g')
            dl = self.d_pT(p, Tsat, force_state='l')
            return 1/(1/dl + Q*(1/dg - 1/dl))
        return False    

    def h_pQ(self, p, Q):
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            hL =  self.h_pT(p,Tsat)
            hG =  self.h_pT(p,Tsat,force_state='g')
            return hL + Q*(hG-hL)
        return False

    def Tx_ps(self, p, s):
        Tsat = self.T_sat(p)
        sL =  self.s_pT(p,Tsat)
        sG =  self.s_pT(p,Tsat,force_state='g')
        if s >= sL and s <= sG:
            return Tsat, (s-sL)/(sG-sL)
        elif s<sL:
            return self.newton(self.s_pT, self.dsdT, s, p, Tmax=Tsat, T=Tsat-10), 0.0
        else:
            return self.newton(self.s_pT, self.dsdT, s, p, Tmin=Tsat, T=Tsat+10, force_state='g'), 1.0

    def T_ps(self, p, s):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ps(p, s)
            return T
        return self.newton(self.s_pT, self.dsdT, s, p)
    def dsdT(self, p, T, **kwargs):
        return self.cp_pT(p, T, **kwargs)/T
    def h_ps(self, p, s):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ps(p, s)
            if x>=0.0 and x<=1.0:
                return self.h_pQ(p, x)
            else:
                return self.h_pT(p, T)
        T = self.T_ps(p, s)
        return self.h_pT(p, T)

    def s_ph(self, p, h):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ph(p, h)
            if x>=0.0 and x<=1.0:
                return self.h_pQ(p, x)
            else:
                return self.h_pT(p, T)
        T = self.T_ph(p, h)
        return self.s_pT(p, T)

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def newton(self, func, deriv, val, p, Tmin = -1000.0, Tmax = 3000.0, T = 300.0, **kwargs):
        # default valaues
        max_iter = 10
        tol_rel = 1e-6
        # start newton loop
        expr = True
        i = 0
        while expr:
            # calculate function residual and new value
            res = val - func(p, T, **kwargs)
            T += res / deriv(p, T, **kwargs)
            # check for value ranges
            if T < Tmin:
                T = Tmin
            if T > Tmax:
                T = Tmax
            i += 1
            if i > max_iter:
                break
            expr = abs(res / val) >= tol_rel
        return T    



if __name__ == "__main__":

    print("\n test protein started \n")

    # coefficients   a      b       c    d        
    COEF = {
    "protein": {
        "unit" : "C",
        "cp": [2008.2,     1.2089, -1.3129*1e-3,    0.0],
        "d" : [1329.9,    -0.5184,          0.0,    0.0],
        }
    }

    fluidwrap = CustomWrapper("protein",coefs=COEF) 

    T = 300
    p = 1e5 
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = 273.15
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = 373.15
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")
    T = fluidwrap.T_ph(p,h)
    s = fluidwrap.s_ph(p,h)
    print(f"recalc: T = {T}    s = {s}")
    T = fluidwrap.T_ps(p,s)
    h = fluidwrap.h_ps(p,s)
    print(f"recalc: T = {T}    h = {h}")

    CP_cp = []
    CP_k  = []
    CP_d  = []
    CP_h  = []
    CP_s  = []

    wrap_cp = []
    wrap_d = []
    wrap_h = []
    wrap_s = []

    p = 101325 * 5

    #Specific heat, kJ/(kg·K) 
    Tplt = np.linspace(273.15,373.15)
    for T in Tplt:
        CP_cp      += [CP.PropsSI('C','T',T,'P',p,'INCOMP::FoodProtein')]
        CP_k       += [CP.PropsSI('L','T',T,'P',p,'INCOMP::FoodProtein')]
        CP_d       += [CP.PropsSI('D','T',T,'P',p,'INCOMP::FoodProtein')]
        CP_h       += [CP.PropsSI('H','T',T,'P',p,'INCOMP::FoodProtein')]
        CP_s       += [CP.PropsSI('S','T',T,'P',p,'INCOMP::FoodProtein')]
        wrap_cp    += [fluidwrap.cp_pT(p, T)]
        wrap_d     += [fluidwrap.d_pT(p, T)]
        wrap_h    += [fluidwrap.h_pT(p, T)]
        wrap_s     += [fluidwrap.s_pT(p, T)]



    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax = ax.flatten()
    [a.grid() for a in ax]
    [a.set_xlabel('temperature, K') for a in ax]

    ax[0].plot(Tplt, wrap_cp)
    ax[1].plot(Tplt, wrap_d)
    ax[2].plot(Tplt, wrap_h)
    ax[3].plot(Tplt, wrap_s)

    ax[0].scatter(Tplt, CP_cp)
    ax[1].scatter(Tplt, CP_d)
    ax[2].scatter(Tplt, CP_h)
    ax[3].scatter(Tplt, CP_s)

    ax[0].set_ylabel('cp')
    ax[1].set_ylabel('d')
    ax[2].set_ylabel('h')
    ax[3].set_ylabel('s')

    plt.show()

    print("\n test protein finished \n")


    print("\n test custom two-phase water started \n")

    # coefficients   a      b       c    d        
    COEF = coefs = {
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

    fluidwrap = CustomWrapper("CUSTOM::WaterTwoPhase", 
                              Tref = 273.15, 
                              coefs=COEF) 

    T = 300
    p = 1e5 
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = 273.15
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = 373.15
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = fluidwrap.T_ph(p,h)
    s = fluidwrap.s_ph(p,h)
    print(f"recalc: T = {T}    s = {s}")
    T = fluidwrap.T_ps(p,s)
    h = fluidwrap.h_ps(p,s)
    print(f"recalc: T = {T}    h = {h}")    

    T = 393.15
    p = 1e5 
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = fluidwrap.T_ph(p,h)
    s = fluidwrap.s_ph(p,h)
    print(f"recalc: T = {T}    s = {s}")
    T = fluidwrap.T_ps(p,s)
    h = fluidwrap.h_ps(p,s)
    print(f"recalc: T = {T}    h = {h}")

    p = 1e5 
    T = fluidwrap.T_sat(p)
    u = fluidwrap.u_pT(p, T)
    d = fluidwrap.d_pT(p, T)
    h = fluidwrap.h_pT(p, T)
    s = fluidwrap.s_pT(p, T)
    print(f"u = {u}    d = {d}    h = {h}    s = {s}")

    T = fluidwrap.T_ph(p,h)
    s = fluidwrap.s_ph(p,h)
    print(f"recalc: T = {T}    s = {s}")
    T = fluidwrap.T_ps(p,s)
    h = fluidwrap.h_ps(p,s)
    print(f"recalc: T = {T}    h = {h}")    

    CP_cp = []
    CP_k  = []
    CP_d  = []
    CP_h  = []
    CP_s  = []

    wrap_cp = []
    wrap_d = []
    wrap_h = []
    wrap_s = []

    #p = 101325 * 5

    #Specific heat, kJ/(kg·K) 
    Tplt = np.linspace(274.15,473.15)
    for T in Tplt:
        CP_cp      += [CP.PropsSI('C','T',T,'P',p,'HEOS::Water')]
        CP_k       += [CP.PropsSI('L','T',T,'P',p,'HEOS::Water')]
        CP_d       += [CP.PropsSI('D','T',T,'P',p,'HEOS::Water')]
        CP_h       += [CP.PropsSI('H','T',T,'P',p,'HEOS::Water')]
        CP_s       += [CP.PropsSI('S','T',T,'P',p,'HEOS::Water')]
        wrap_cp    += [fluidwrap.cp_pT(p, T)]
        wrap_d     += [fluidwrap.d_pT(p, T)]
        wrap_h    += [fluidwrap.h_pT(p, T)]
        wrap_s     += [fluidwrap.s_pT(p, T)]



    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax = ax.flatten()
    [a.grid() for a in ax]
    [a.set_xlabel('temperature, K') for a in ax]

    ax[0].plot(Tplt, wrap_cp)
    ax[1].plot(Tplt, wrap_d)
    ax[2].plot(Tplt, wrap_h)
    ax[3].plot(Tplt, wrap_s)

    ax[0].scatter(Tplt, CP_cp)
    ax[1].scatter(Tplt, CP_d)
    ax[2].scatter(Tplt, CP_h)
    ax[3].scatter(Tplt, CP_s)

    ax[0].set_ylabel('cp')
    ax[1].set_ylabel('d')
    ax[2].set_ylabel('h')
    ax[3].set_ylabel('s')

    plt.show()

    print("\n test protein finished \n")                        