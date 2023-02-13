import logging

from tespy.components import HeatExchangerSimple, Source, Sink, Merge, Separator 
from tespy.tools import ComponentProperties
from tespy.connections import Connection
from tespy.networks import Network
import numpy as np

from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp

from tespy.components.newcomponents import \
    DiabaticSimpleHeatExchanger,MergeWithPressureLoss,SeparatorWithSpeciesSplits, \
        SeparatorWithSpeciesSplitsAndDeltaT

# %%

# caution, must write "Water" (capital W) in INCOMP backend -> CoolProp bug? Intentional?
fluids = ["INCOMP::Water", "INCOMP::T66"]
nw = Network(fluids=fluids, p_unit="bar", T_unit="C")

so = Source("Source")
se = SeparatorWithSpeciesSplitsAndDeltaT("Separator") #,num_out=2)
si1 = Sink("Sink 1")
si2 = Sink("Sink 2")

c1 = Connection(so, "out1", se, "in1", label="1")
c2 = Connection(se, "out1", si1, "in1", label="2")
c3 = Connection(se, "out2", si2, "in1", label="3")

nw.add_conns(c1, c2, c3)

# set some generic data for starting values
c1.set_attr(m=1, p=1.2, T=30, fluid={"Water": 0.9, "T66": 0.1})
c2.set_attr(fluid={"Water": 0.85, "T66": 0.15})

se.set_attr(SFS={
    'val': 0.6, 'is_set': True, 
    'split_fluid' : 'T66', 'split_outlet' : "out1"})


# Now it is possible to set the temperatures out of the separator differently
c2.set_attr(T=20)
c3.set_attr(T=10)

# Or to use a deltaT array instead
#se.set_attr(deltaT=[-10,-20])
#se.set_attr(deltaT=[0,0])

# add some guess values
c2.set_attr(m0=0.5,p0=1.2,h0=1e5,T0=50,fluid0={"Water": 0.5, "T66": 0.5})
c3.set_attr(m0=0.5,p0=1.2,h0=1e5,T0=50,fluid0={"Water": 0.5, "T66": 0.5})

nw.solve("design")
nw.print_results()

print(nw.results['Connection'])

m_T66_c1 = c1.m.val * c1.fluid.val['T66']
m_T66_c2 = c2.m.val * c2.fluid.val['T66']


print(f"\n Species flow split is {m_T66_c2/m_T66_c1}")

print(f"\n heat flows are  {se.Q.val}")
print(f"\n")



