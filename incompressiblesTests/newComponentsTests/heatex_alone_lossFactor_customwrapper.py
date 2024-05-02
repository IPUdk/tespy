# %%

import logging


from tespy.components import SimpleHeatExchanger, Source, Sink, Merge, Separator 
from tespy.tools import ComponentProperties
from tespy.connections import Connection
from tespy.networks import Network
import numpy as np

from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp

from tespy.components.newComponents import SimpleHeatExchangerDeltaPLossFactor,MergeDeltaP,SeparatorWithSpeciesSplits

logging.basicConfig(level=logging.DEBUG)


from tespy.tools.fluid_properties.CustomWrapper import CustomWrapper
from tespy.tools.fluid_properties.wrappers import CoolPropWrapper

# %%

# caution, must write "Water" (capital W) in INCOMP backend -> CoolProp bug? Intentional?

nw = Network(m_unit='kg / s', p_unit='bar', T_unit='C',h_unit='kJ / kg', h_range=[-1e2,4e3], iterinfo=True)

so = Source("Source")
#  Variant 2: Q is m (h_2 - h_1), Q_total is taking efficiency into account and represents the heat transfer over system
# boundary. For heat transfer into the system: Q = Q_total * eta, for heat transfer from the system: Q_total = Q * eta

he = SimpleHeatExchangerDeltaPLossFactor("Heater")
#he = SimpleHeatExchangerDeltaP("Heater")


si = Sink("Sink")

c1 = Connection(so, "out1", he, "in1", label="1")
c2 = Connection(he, "out1", si, "in1", label="4")

nw.add_conns(c1, c2)

# # set some generic data for starting values
# c1.set_attr(fluid={'HEOS::Water': 0.80,'INCOMP::PHE': 0.15,'INCOMP::S800': 0.05},
#             fluid_engines = {"HEOS::Water": CoolPropWrapper, "INCOMP::PHE" : CoolPropWrapper, "INCOMP::S800": CoolPropWrapper}, 
#             mixing_rule="incompressible")    

# set some generic data for starting values
c1.set_attr(fluid={'CUSTOM::WaterTwoPhase': 0.80,'INCOMP::PHE': 0.15,'INCOMP::S800': 0.05},
            fluid_engines = {"CUSTOM::WaterTwoPhase": CustomWrapper, "INCOMP::PHE" : CoolPropWrapper, "INCOMP::S800": CoolPropWrapper}, 
            fluid_coefs = {
                        'CUSTOM::Protein': {
                            'cp': {'eqn': "polynomial", 'unit': "C", 'coefs': [2008.2,     1.2089, -0.0013129]},
                            'd' : {'eqn': "polynomial", 'unit': "C", 'coefs': [1329.9,    -0.5184]},
                        },
                        'CUSTOM::WaterTwoPhase': {
                            'name' : "Custom water model", 
                            'unit' : "K", 
                            'molarmass': 18.01528, 
                            'cp'   : {'eqn': "polynomial", 'unit': "K", 'coefs': [7.79605665e+04,-1.12106166e+03,7.06771540e+00,-2.36638219e-02,4.43721794e-05,-4.41973243e-08,1.83159953e-11]},
                            'd'    : {'eqn': "polynomial", 'unit': "K", 'coefs': [1.35188573e+02,8.66049556e+00,-3.06549945e-02,4.62728683e-05,-2.80708081e-08]},
                            'hfg'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [3.73992983e+06, -8.02594391e+03, 1.80890144e+01, -1.93816772e-02]},
                            'Tsat' : {'eqn': "antoine"   , 'unit': "K", 'coefs': [23.22646886130465, -3842.204328212032, -44.75853983190677]},
                            'cpG'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [4.70848101e+02,1.13556451e+01,-2.07921505e-02,-3.88616225e-05,1.18035083e-07]},     
                        },
                        'CUSTOM::WaterTwoPhaseSimple': {
                            'name' : "Custom water simple", 
                            'unit' : "K", 
                            'molarmass': 18.01528,             
                            'cp'   : {'eqn': "polynomial", 'unit': "K", 'coefs': [4180]},
                            'd'    : {'eqn': "polynomial", 'unit': "K", 'coefs': [1000]},
                            'hfg'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [2250e3]},
                            'Tsat' : {'eqn': "cstpair"   , 'unit': "K", 'coefs': [1e5, 373.15]},
                            'cpG'  : {'eqn': "polynomial", 'unit': "K", 'coefs': [2000]},
                        }   
            },            
            mixing_rule="incompressible")    


c1.set_attr(m=1, p=1.0, T=30)

#c2.set_attr(h=1500)

# set pressure ratios of heater and merge
he.set_attr(deltaP=0.0)

he.set_attr(LF=0) 
he.set_attr(Q_total=1.16e+06) 
#he.set_attr(Q_loss=-7.42e+03)
nw.solve("design")
if not nw.converged:
    raise Exception("not converged")
nw.print_results()
print(nw.results['Connection'])

print("done")

# he.set_attr(LF=None)
# he.set_attr(Q_total=8.16e+04) 
# he.set_attr(Q_loss=-7.42e+03) 
# nw.solve("design")
# if not nw.converged:
#     raise Exception("not converged")
# nw.print_results()

# he.set_attr(LF=0.1)
# he.set_attr(Q_total=None) 
# he.set_attr(Q_loss=-7.42e+03) 
# nw.solve("design")
# if not nw.converged:
#     raise Exception("not converged")
# nw.print_results()





# print(nw.results['Connection'])
# he.Q.val
# he.Q_loss.val
# he.Q_total.val

print(he.LF.val)
print(he.Q_total.val)
#print(he.Q_loss.val)
