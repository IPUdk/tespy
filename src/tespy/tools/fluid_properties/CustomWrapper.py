import CoolProp.CoolProp as CP
from tespy.tools.fluid_properties.wrappers import FluidPropertyWrapper, CoolPropWrapper
import numpy as np
import matplotlib.pyplot as plt

class CustomWrapper(FluidPropertyWrapper):
    def __init__(self, fluid, back_end=None, Tref = 293.15, coefs=[]) -> None:
        super().__init__(fluid, back_end)
        if self.fluid not in coefs:
            msg = f"Fluid ({self.fluid}) not available in database"
            raise KeyError(msg)

        # get coefs (converted to kelvin) and calculate reference
        self.T0 = Tref
        self.get_coefs(coefs[self.fluid])
        
        self._molar_mass = coefs[self.fluid].get('molarmass',None)
        if self._molar_mass is None and self.TwoPhaseMedium:
            raise ValueError("Must have key (molarmass) for two-phase medium")

        self._T_min = 273.15 - 50
        self._T_max = 273.15 + 1000
        self._p_min = 1000
        self._p_max = 10000000
 

    def get_coefs(self, coefs):
        self.flddat = {}
        for k in ['cp','d','hfg','Tsat','cpG']:
            if coefs.get(k):
                if coefs[k].get("eqn") and coefs[k].get("coefs"):
                    self.flddat[k] = {}
                    self.flddat[k]['eqn'] = coefs[k]['eqn']
                    self.flddat[k]['coefs'] = coefs[k]['coefs']
                    self.flddat[k]['unit'] = coefs[k].get('unit',None)
                    # remove zero  coeffs for higher order terms, if any
                    [self.flddat[k]["coefs"][i].pop() for i in range(len(coefs[k]["coefs"]),0)]
                    self.flddat[k]["n"] = len(self.flddat[k]["coefs"])
                    if self.flddat[k]["unit"] == "C" and self.flddat[k]['eqn'] == "polynomial":
                        self.convert_coefs_to_Kelvin(k)
                    elif not self.flddat[k]["unit"] == "K":
                        raise ValueError("We only support Kelvin as T unit for saturation equation")
                else:
                    raise ValueError(f"Custom model ({coefs['name']}) for variable ({self.flddat[k]}) does not have (eqn) and/or (coefs) keys")

        if self.flddat.get("cp") and self.flddat.get("d"):
            self.TwoPhaseMedium = False
            if self.flddat.get("hfg") and self.flddat.get("Tsat") and self.flddat.get("cpG"):
                self.TwoPhaseMedium = True
            elif self.flddat.get("hfg") or self.flddat.get("Tsat") or self.flddat.get("cpG"):
                raise ValueError(f"Custom two-phase model ({coefs['name']}) need hfg, Tsat, and cpG")
        else:
            raise ValueError(f"Custom model ({self.fluid}) need cp, d for single phase fluid and hfg, Tsat, and cpG for two-phase fluid")

    def convert_coefs_to_Kelvin(self, k):
        # convert coefficients
        T_C = np.linspace(1,80)
        val = self.polyval(T_C, k)
        T_K = np.linspace(1+273.15,80+273.15)
        self.flddat[k]['coefs'] = list(np.polyfit(T_K, val, self.flddat[k]['n']-1))
        self.flddat[k]['coefs'] = self.flddat[k]['coefs'][::-1]
        valK = self.polyval(T_K, k)
        r_value = self.r_squared_adj(val, valK, self.flddat[k]['n'])
        if r_value < 0.999:
            raise ValueError("could not convert polynomial satisfactory")
            # I think this convertion should be done exact, TODO
            
    def polyval(self, T, key):
        if self.flddat[key]['eqn'] == "polynomial":
            return np.sum([self.flddat[key]['coefs'][i] * T**i for i in range(self.flddat[key]['n'])], axis=0)
        else:
            raise ValueError(f"Equation for ({key}) must be a (polynomial), use a single coefficient if you want a constant")

    def T_sat(self, p):
        c = self.flddat['Tsat']['coefs']
        if self.flddat['Tsat']['eqn'] == "antoine":
            return c[1] / (np.log(p) - c[0]) - c[2]
        elif self.flddat['Tsat']['eqn'] == "cstpair":
            return c[1]
        else:
            raise ValueError("Saturation equation must be (antoine) or (cstpair = [pressure,Temperature])")

    def p_sat(self, T):
        c = self.flddat['Tsat']['coefs']
        if self.flddat['Tsat']['eqn'] == "antoine":
            return np.exp(c[0] + c[1]/(T + c[2]))
        elif self.flddat['Tsat']['eqn'] == "cstpair":
            return c[0]               
        else:
            raise ValueError("Saturation equation must be (antoine) or (cstpair = [pressure,Temperature])")        

    def get_state(self, state = None):
        if state:
            if state == 'g':
                return 1
            elif state == 'l':
                return 0
        return None
    
    def cp_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # assume saturated gas at Tsat (no increase with pressure)
                return self.polyval(Tsat, "cpG")
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                return self.polyval(T, "cp")
        return self.polyval(T, "cp")
   
    def d_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # calculate d using ideal gas law
                R = 8314.46261815324 / self._molar_mass
                return p/(R*T)
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                return self.polyval(T, "d")
        return self.polyval(T, "d")

    def hfg_pT(self, p, T):
        return self.polyval(T, "hfg")
    
    def _u_T(self, T):
        if self.flddat['cp']['eqn'] == "polynomial":
            integral = 0
            for i in range(self.flddat['cp']['n']):
                integral += (1 / (i + 1)) * self.flddat['cp']['coefs'][i] * (T**(i + 1) - self.T0**(i + 1))
            return integral 
        else:
            raise ValueError(f"Equation for (cp) must be a (polynomial), use a single coefficient if you want a constant")
        
    def _s_T(self, T):
        if self.flddat['cp']['eqn'] == "polynomial":
            integral = self.flddat['cp']['coefs'][0] * np.log(T / self.T0)
            for i in range(self.flddat['cp']['n'] - 1):
                integral += (1 / (i + 1)) * self.flddat['cp']['coefs'][i + 1] * (T**(i + 1) - self.T0**(i + 1))            
            return integral 
        else:
            raise ValueError(f"Equation for (cp) must be a (polynomial), use a single coefficient if you want a constant")        

    def u_pT(self, p, T):
        return self._u_T(T)

    def h_pT(self, p, T, **kwargs):
        state = self.get_state(kwargs.get('force_state',None))
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            if (T > Tsat or state == 1) and not state == 0:
                # we do not want to use gas at T, but Tsat as the fit is saturated (keep the pressure)
                # and extrapolate cpG*(T-Tsat) instead 
                df  = self.d_pT(p, Tsat)
                hf  = self.u_pT(p, Tsat) - p/df
                hfg = self.hfg_pT(p, Tsat)
                dg  = self.d_pT(p, Tsat, force_state='g')
                d   = self.d_pT(p, T, force_state='g')
                hg = self.cp_pT(p, Tsat, force_state='g')*(T-Tsat) - (p/d - p/dg)
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
                # we do not want to use gas at T, but Tsat as the fit is saturated (keep the pressure)
                sf  = self._s_T(Tsat)
                shf = self.hfg_pT(p, Tsat)/Tsat
                sg  = (self.cp_pT(p, Tsat, force_state='g')/Tsat)*(T-Tsat)
                return sf + shf + sg
            else:
                # Liquid and forced liquid, i.e. (T>Tsat)
                return self._s_T(T)
        else:
            return self._s_T(T)
    
    def Tx_ph(self, p, h):
        Tsat = self.T_sat(p)
        hL =  self.h_pT(p,Tsat)
        hG =  self.h_pT(p,Tsat,force_state='g')
        x = (h-hL)/(hG-hL)
        if h >= hL and h <= hG:
            return Tsat, x
        elif h<hL:
            return self.newton(self.h_pT, self.cp_pT, h, p, Tmax=Tsat, T=Tsat-10), x
        else:
            return self.newton(self.h_pT, self.cp_pT, h, p, Tmin=Tsat, T=Tsat+10, force_state='g'), x
   
    def Q_ph(self, p, h):
        if self.TwoPhaseMedium:
            T, x = self.Tx_ph(p, h)
            return min(max(0.0,x),1.0)
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
    
    def s_pQ(self, p, Q):
        if self.TwoPhaseMedium:
            Tsat = self.T_sat(p)
            sL =  self.s_pT(p,Tsat)
            sG =  self.s_pT(p,Tsat,force_state='g')
            return sL + Q*(sG-sL)
        return False    

    def Tx_ps(self, p, s):
        Tsat = self.T_sat(p)
        sL =  self.s_pT(p,Tsat)
        sG =  self.s_pT(p,Tsat,force_state='g')
        x = (s-sL)/(sG-sL)
        if s >= sL and s <= sG:
            return Tsat, x
        elif s<sL:
            return self.newton(self.s_pT, self.dsdT, s, p, Tmax=Tsat, T=Tsat-10), x
        else:
            return self.newton(self.s_pT, self.dsdT, s, p, Tmin=Tsat, T=Tsat+10, force_state='g'), x

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
                return self.s_pQ(p, x)
            else:
                return self.s_pT(p, T)
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

    def r_squared_adj(self, x, y, k=1):
        """
        x is true values
        y is predicted
        k is no. independent variables
        """ 
        N = len(x)

        # Calculate R-squared without regression
        mean_y = np.mean(y)
        total_sum_of_squares = np.sum((y - mean_y)**2)
        residual_sum_of_squares = np.sum((y - x)**2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # Calculate adjusted R-squared
        return 1 - ((1 - r_squared) * (N - 1) / (N - k - 1))

if __name__ == "__main__":

    print("\n test protein started \n")

    # coefficients
    COEF = coefs = {
        'CUSTOM::Protein': {
            'cp': {'eqn': "polynomial", 'unit': "C", 'coefs': [2008.2,     1.2089, -0.0013129]},
            'd' : {'eqn': "polynomial", 'unit': "C", 'coefs': [1329.9,    -0.5184]},
        },
        'CUSTOM::WaterTwoPhase': {
            'name' : "Custom water model", 
            'molarmass': 18.01528, 
            'cp'   : {'eqn': "polynomial", 'unit': "K", 'coefs': [7.79605665e+04,-1.12106166e+03,7.06771540e+00,-2.36638219e-02,4.43721794e-05,-4.41973243e-08,1.83159953e-11]},
            'd'    : {'eqn': "polynomial", 'unit': "K", 'coefs': [1.35188573e+02,8.66049556e+00,-3.06549945e-02,4.62728683e-05,-2.80708081e-08]},
            'hfg'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [3.73992983e+06, -8.02594391e+03, 1.80890144e+01, -1.93816772e-02]},
            'Tsat' : {'eqn': "antoine"   , 'unit': "K", 'coefs': [23.22646886130465, -3842.204328212032, -44.75853983190677]},
            'cpG'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [4.70848101e+02,1.13556451e+01,-2.07921505e-02,-3.88616225e-05,1.18035083e-07]},     
        },
        'CUSTOM::WaterTwoPhaseSimple': {
            'name' : "Custom water simple", 
            'molarmass': 18.01528,             
            'cp'   : {'eqn': "polynomial", 'unit': "K", 'coefs': [4180]},
            'd'    : {'eqn': "polynomial", 'unit': "K", 'coefs': [1000]},
            'hfg'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [2250e3]},
            'Tsat' : {'eqn': "cstpair"   , 'unit': "K", 'coefs': [1e5, 373.15]},
            'cpG'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [2000]},
        }      
    }  


    fluidwrap = CustomWrapper("CUSTOM::Protein",coefs=COEF) 

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