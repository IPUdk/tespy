# -*- coding: utf-8 -*-

from tespy import con, cmp, nwk
import math
import matplotlib.pyplot as plt

# %% components

# turbine part
vessel_turb = cmp.vessel(label='vessel_turb', mode='man')
turbine_hp = cmp.turbine(label='turbine_hp', eta_s=0.9)
split = cmp.splitter(label='splitter1')
turbine_lp = cmp.turbine(label='turbine_lp', eta_s=0.9)

# condenser and preheater
condenser = cmp.condenser(label='condenser', dp1=0.95, dp2=0.95, ttd_u=7)
preheater = cmp.condenser(label='preheater', dp1=0.95, dp2=0.99, ttd_u=7)
vessel = cmp.vessel(label='vessel1', mode='man')
merge = cmp.merge(label='merge1')

# feed water
pump = cmp.pump(label='pump', eta_s=0.8, mode='man')
steam_generator = cmp.heat_exchanger_simple(label='steam generator',
                                            dp=0.95, mode='man')

# sources and sinks
source = cmp.source(label='source')
sink = cmp.sink(label='sink')

# for cooling water
source_cw = cmp.source(label='source_cw')
sink_cw = cmp.sink(label='sink_cw')

# %% network

fluids = ['water']

nw = nwk.network(fluids=fluids, p='bar', T='C',
                 p_range=[0.02, 150], T_range=[20, 800])

# %% connections

# turbine part
fs_in = con.connection(source, 'out1', vessel_turb, 'in1', p=100, T=550)
fs = con.connection(vessel_turb, 'out1', turbine_hp, 'in1',
                    p=100, m=47, fluid={'water': 1})
ext = con.connection(turbine_hp, 'out1', split, 'in1', p=10)
ext_pre = con.connection(split, 'out1', preheater, 'in1', m0=10)
ext_turb = con.connection(split, 'out2', turbine_lp, 'in1', h0=3000000)
nw.add_conns(fs_in, fs, ext, ext_pre, ext_turb)

# preheater and condenser
ext_cond = con.connection(preheater, 'out1', vessel, 'in1', h0=400000)
cond_ws = con.connection(vessel, 'out1', merge, 'in2')
turb_ws = con.connection(turbine_lp, 'out1', merge, 'in1')
ws = con.connection(merge, 'out1', condenser, 'in1')
nw.add_conns(ext_cond, cond_ws, turb_ws, ws)

# feed water
cond = con.connection(condenser, 'out1', pump, 'in1')
fw_c = con.connection(pump, 'out1', preheater, 'in2', h0=300000)
fw_w = con.connection(preheater, 'out2', steam_generator, 'in1', h0=310000)
fs_out = con.connection(steam_generator, 'out1', sink, 'in1',
                        p=con.ref(fs_in, 1, 0), h=con.ref(fs_in, 1, 0))
nw.add_conns(cond, fw_c, fw_w, fs_out)

# cooling water
cw_in = con.connection(source_cw, 'out1', condenser, 'in2',
                       T=60, p=10, fluid={'water': 1})
cw_out = con.connection(condenser, 'out2', sink_cw, 'in1', T=110)
nw.add_conns(cw_in, cw_out)

# %% busses

# power bus
power_bus = con.bus('power')
power_bus.add_comps([turbine_hp, -1], [turbine_lp, -1], [pump, -1])

# heating bus
heat_bus = con.bus('heat')
heat_bus.add_comps([condenser, -1])

nw.add_busses(power_bus, heat_bus)


# %% solving

mode = 'design'

nw.solve(init_file=None, mode=mode)
nw.process_components(mode='post')
nw.save('CHP_'+mode)

file = 'CHP_'+mode+'_conn.csv'
mode = 'offdesign'

fs.set_attr(p=math.nan)
ext.set_attr(p=math.nan)

# representation of part loads
m = [50, 45, 40, 35, 30]

# temperatures for the heating system
t_vl = [80, 90, 100, 110, 120]

P = {}
Q = {}

# iterate over temperatures
for i in t_vl:
    cw_out.set_attr(T=i)
    P[i] = []
    Q[i] = []
    # iterate over mass flows
    for j in m:
        fs.set_attr(m=j)

        nw.solve(init_file=file, design_file=file, mode=mode)
        nw.process_components(mode='post')
        P[i] += [power_bus.P]
        Q[i] += [heat_bus.P]

# plotting
colors = ['#00395b', '#74adc1', '#bfbfbf', '#b54036', '#ec6707']

fig, ax = plt.subplots()
j = 0
for i in t_vl:
    plt.plot(Q[i], P[i], '.-', Color=colors[j],
             label='$T_{VL}$ = '+str(i)+' °C', markersize=15, linewidth=2)
    j += 1
ax.set_ylabel('$P$ in W')
ax.set_xlabel('$\dot{Q}$ in W')
plt.title('P-Q diagram for CHP with backpressure steam turbine')
plt.legend(loc='lower left')

ax.set_ylim([0, 1e8])
ax.set_xlim([0, 1e8])

plt.show()

fig.savefig('PQ_diagram.svg')
