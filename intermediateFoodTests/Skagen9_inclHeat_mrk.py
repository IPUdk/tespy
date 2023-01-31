# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:00:18 2022

@author: mrk

This model includes 
- Boiler 
- Press
- Decanter (no steam is mixed into product in the presswater)
- Centrifuge
- Thickener 
- Merge
- Drier 

- Oil cleansing

- solve energy balances too

"""

from tespy.components import Sink, Source, HeatExchangerSimple, Splitter
from tespy.connections import Connection
from tespy.networks import Network
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tespy.components import Separator,Merge,CycleCloser,Valve
from tespy.components.newcomponents import DiabaticSimpleHeatExchanger,MergeWithPressureLoss,SeparatorWithSpeciesSplits,SplitWithFlowSplitter,SeparatorWithSpeciesSplitsAndDeltaT

import logging
logging.basicConfig(level=logging.DEBUG)

fluid_list = ['HEOS::Water','INCOMP::FoodProtein','INCOMP::FoodFat']
network = Network(fluids=fluid_list, m_unit='kg / s', p_unit='bar', T_unit='C',h_unit='kJ / kg', h_range=[-1e2,4e3], iterinfo=True)

# Objects
source               = Source('source')
boiler               = HeatExchangerSimple('boiler')
press                = SeparatorWithSpeciesSplitsAndDeltaT('press', num_out=2)
presswaterheater     = HeatExchangerSimple('presswaterheater')
#presswater          = Sink('presswater')
#presscake           = Sink('presscake')
decanter             = SeparatorWithSpeciesSplitsAndDeltaT('decanter', num_out=2)
#grax                = Sink('grax')
oil                  = Sink('oil')
centrifuge1          = SeparatorWithSpeciesSplitsAndDeltaT('centrifuge1',num_out=2)

evaporator        = SeparatorWithSpeciesSplitsAndDeltaT('evaporator',num_out=2)
evaporatedVapour  = Sink('evaporatedVapour')
solubleHeater      = HeatExchangerSimple('solubleHeater')
#solubles          = Sink('solubles')

liquidmerge      = MergeWithPressureLoss('liquidmerge', num_in = 3)
wetproduct       = Sink('wetproduct')
drier            = SeparatorWithSpeciesSplitsAndDeltaT('drier',num_out=2)
meal             = Sink('meal')
driedVapour      = Sink('driedVapour')

# Connections
c1 = Connection(source, 'out1', boiler, 'in1')
c2 = Connection(boiler, 'out1', press, 'in1')
c3 = Connection(press, 'out1', liquidmerge, 'in1')
c4a = Connection(press, 'out2', presswaterheater, 'in1')
c4b= Connection(presswaterheater, 'out1', decanter, 'in1')
c5 = Connection(decanter, 'out1', liquidmerge, 'in2')
c6 = Connection(decanter, 'out2', centrifuge1, 'in1')
c7 = Connection(centrifuge1, 'out1', evaporator, 'in1')
c8 = Connection(centrifuge1, 'out2', oil, 'in1')
c9a = Connection(evaporator, 'out1', solubleHeater, 'in1')
c9b = Connection(solubleHeater, 'out1', liquidmerge, 'in3')
c10 = Connection(evaporator, 'out2', evaporatedVapour, 'in1')
c11 = Connection(liquidmerge, 'out1', drier, 'in1')
c12 = Connection(drier, 'out1', meal, 'in1')
c13 = Connection(drier, 'out2', driedVapour, 'in1')

network.add_conns(c1,c2,c3,c4a,c4b,c5,c6,c7,c8,c9a,c9b,c10,c11,c12,c13)

# set global guess values 
m0 = 108.143    # transform unit at some point [this is kt/yr]
h0 = 1e2        # global guess value in kJ/kg
p0 = 10        # global guess value in bar

for c in network.conns['object']:
    n_fl = len(network.fluids)
    c.set_attr(m0=108.143,h0=h0,p0=p0,fluid0={'Water': 1/n_fl, 'FoodFat': 1/n_fl, 'FoodProtein': 1/n_fl})

# set conditions around boiler 
c1.set_attr(fluid={'Water': 0.81,'FoodProtein': 0.163,'FoodFat': 0.027}, m=108.143, T=5, p=p0)
c2.set_attr(T=95,p=p0)

# set conditions around press
press.set_attr(SFS={
    'val': 0.65, 'is_set': True, 
    'split_fluid' : 'FoodProtein', 'split_outlet' : "out1"})
c3.set_attr(fluid={'Water': 0.51, 'FoodFat': 0.04, 'FoodProtein': 0.45})
c3.set_attr(T=85)
c4a.set_attr(T=85)

# set conditions around presswater heater
c4b.set_attr(T=95)
#c4b.set_attr(p0=1) or the below
presswaterheater.set_attr(pr=1)

# set conditions around decanter
decanter.set_attr(SFS={
    'val': 0.35, 'is_set': True, 
    'split_fluid' : 'FoodProtein', 'split_outlet' : "out1"})
c5.set_attr(fluid={'Water': 0.648, 'FoodFat': 0.022, 'FoodProtein': 0.33})
# notice the decanter use steam that is not modelled by V&M
c5.set_attr(T=90)
c6.set_attr(T=90)

# set conditions around centrifuge
centrifuge1.set_attr(SFS={
    'val': 0.85, 'is_set': True, 
    'split_fluid' : 'FoodFat', 'split_outlet' : "out2"})
c8.set_attr(fluid={'Water': 0, 'FoodFat': 0.99, 'FoodProtein': 0.01})
# notice the oil split V&M use just upsteam fat value for splitting, while skagen use upstream of decanter 
c7.set_attr(T=80)
c8.set_attr(T=80)

# set conditions around evaporator
c10.set_attr(fluid={'Water': 1, 'FoodFat': 0, 'FoodProtein': 0})
c9a.set_attr(fluid={'FoodProtein': 0.30})
c10.set_attr(T=70)
c9a.set_attr(T=70)

# set conditions around soluable heater
c9b.set_attr(T=105,p=p0)

# set conditions around liquidMerge
c11.set_attr(p=1)

# set conditions around drier
c12.set_attr(fluid={'Water': 0.08})
c13.set_attr(fluid={'Water': 1, 'FoodFat': 0, 'FoodProtein': 0})
c12.set_attr(T=99)
c13.set_attr(T=110)
c12.set_attr(p0=1)
c13.set_attr(p0=1)

#c12.set_attr(state='l')

network.solve('design',init_only=True)

for c in network.conns['object']:
    print(c.p.val_SI)
for c in network.conns['object']:
    print(c.h.val_SI)
for c in network.conns['object']:
    print(c.T.val_SI)


# solve and print results
network.solve('design')

network.print_results()
print(network.results['Connection'])

oilmassflow = c8.m.val
print(f"oil mass flow is {oilmassflow}")
print(f"\n")

# MJ to kwh 
# 
for o in network.comps['object']:
    if isinstance(o,SeparatorWithSpeciesSplitsAndDeltaT):
        print(f"heat exchange for {o.label} = " + str(['{:.2f}'.format(Q/3.6) for Q in o.Q.val]))
print(f"\n")

for o in network.comps['object']:
    if isinstance(o,SeparatorWithSpeciesSplitsAndDeltaT):
        print(f"Total heat for {o.label} is {np.sum(o.Q.val)/3.6:.2f}")
print(f"\n")

print(f"Total heat for boiler is {boiler.Q.val/3.6:.2f}")
print(f"Total heat for presswater heater is {presswaterheater.Q.val/3.6:.2f}")


import sys
sys.exit()


c13.set_attr(T=None)
c13.set_attr(h=2700)
network.solve('design')
network.print_results()
print(network.results['Connection'])

for o in network.comps['object']:
    if isinstance(o,SeparatorWithSpeciesSplitsAndDeltaT):
        o.Q.val = []

c13.set_attr(h=None)
c13.set_attr(T=110)
network.solve('design')
network.print_results()
print(network.results['Connection'])

# MJ to kwh 
# 
for o in network.comps['object']:
    if isinstance(o,SeparatorWithSpeciesSplitsAndDeltaT):
        print(f"heat exchange for {o.label} = " + str(['{:.2f}'.format(Q/3.6) for Q in o.Q.val]))
print(f"\n")

for o in network.comps['object']:
    if isinstance(o,SeparatorWithSpeciesSplitsAndDeltaT):
        print(f"Total heat for {o.label} is {np.sum(o.Q.val)/3.6:.2f}")
print(f"\n")

print(f"Total heat for boiler is {boiler.Q.val/3.6:.2f}")
print(f"Total heat for presswater heater is {presswaterheater.Q.val/3.6:.2f}")
