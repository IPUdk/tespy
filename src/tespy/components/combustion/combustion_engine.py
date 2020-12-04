# -*- coding: utf-8

"""Module of class CombustionEngine.


This file is part of project TESPy (github.com/oemof/tespy). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location
tespy/components/combustion/combustion_engine.py

SPDX-License-Identifier: MIT
"""

import logging

import numpy as np

from tespy.components.combustion.combustion_chamber import CombustionChamber
from tespy.components.component import Component
from tespy.tools.data_containers import ComponentCharacteristics as dc_cc
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import DataContainerSimple as dc_simple
from tespy.tools.fluid_properties import h_mix_pQ
from tespy.tools.fluid_properties import h_mix_pT
from tespy.tools.fluid_properties import h_pT
from tespy.tools.fluid_properties import v_mix_ph
from tespy.tools.global_vars import err
from tespy.tools.global_vars import molar_masses
from tespy.tools.helpers import molar_mass_flow


class CombustionEngine(CombustionChamber):
    r"""
    An internal combustion engine supplies power and heat cogeneration.

    The combustion engine produces power and heat in cogeneration from fuel
    combustion. The combustion properties are identical to the combustion
    chamber. Thermal input and power output, heat output and heat losses are
    linked with an individual characteristic line for each property.

    Equations

        **mandatory equations**

        - :py:meth:`tespy.components.combustion.CombustionChamber.reaction_balance`
        - :py:meth:`tespy.components.combustion.CombustionEngine.fluid_func`
          (for cooling water)
        - :py:meth:`tespy.components.combustion.CombustionEngine.mass_flow_func`

        .. math::

            0 = p_{3,in} - p_{3,out}\\
            0 = p_{4,in} - p_{3,out}

        - :py:meth:`tespy.components.combustion.CombustionEngine.energy_balance`

        **optional equations**

        - :py:meth:`tespy.components.combustion.CombustionChamber.lambda_func`
        - :py:meth:`tespy.components.combustion.CombustionChamber.ti_func`
        - :py:meth:`tespy.components.combustion.CombustionEngine.Q1_func`
        - :py:meth:`tespy.components.combustion.CombustionEngine.Q2_func`

        .. math::

            0 = p_{1,in} \cdot pr1 - p_{1,out}\\
            0 = p_{2,in} \cdot pr2 - p_{2,out}

        - loop 1 :py:meth:`tespy.components.component.Component.zeta_func`
        - loop 2 :py:meth:`tespy.components.component.Component.zeta_func`

    Available fuels

        - methane, ethane, propane, butane, hydrogen

    Inlets/Outlets

        - in1, in2 (cooling water), in3, in4 (air and fuel)
        - out1, out2 (cooling water), out3 (flue gas)

    Image

        .. image:: _images/CombustionEngine.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    .. note::

        The fuel and the air components can be connected to either of the
        inlets.

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    design_path: str
        Path to the components design case.

    local_offdesign : boolean
        Treat this component in offdesign mode in a design calculation.

    local_design : boolean
        Treat this component in design mode in an offdesign calculation.

    char_warnings: boolean
        Ignore warnings on default characteristics usage for this component.

    printout: boolean
        Include this component in the network's results printout.

    lamb : float, tespy.tools.data_containers.dc_cp
        Air to stoichiometric air ratio, :math:`\lambda/1`.

    ti : float, tespy.tools.data_containers.dc_cp
        Thermal input, (:math:`{LHV \cdot \dot{m}_f}`),
        :math:`ti/\text{W}`.

    P : str, float, tespy.tools.data_containers.dc_cp
        Power output, :math:`P/\text{W}`.

    Q1 : float, tespy.tools.data_containers.dc_cp
        Heat output 1, :math:`\dot Q/\text{W}`.

    Q2 : float, tespy.tools.data_containers.dc_cp
        Heat output 2, :math:`\dot Q/\text{W}`.

    Qloss : str, float, tespy.tools.data_containers.dc_cp
        Heat loss, :math:`\dot Q_{loss}/\text{W}`.

    pr1 : float, tespy.tools.data_containers.dc_cp
        Pressure ratio heat outlet 1, :math:`pr/1`.

    pr2 : float, tespy.tools.data_containers.dc_cp
        Pressure ratio heat outlet 2, :math:`pr/1`.

    zeta1 : float, tespy.tools.data_containers.dc_cp
        Geometry independent friction coefficient heating loop 1,
        :math:`\zeta/\frac{1}{\text{m}^4}`.

    zeta2 : float, tespy.tools.data_containers.dc_cp
        Geometry independent friction coefficient heating loop 2,
        :math:`\zeta/\frac{1}{\text{m}^4}`.

    tiP_char : tespy.tools.charactersitics.char_line, tespy.tools.data_containers.dc_cc
        Characteristic line linking fuel input to power output.

    Q1_char : tespy.tools.charactersitics.char_line, tespy.tools.data_containers.dc_cc
        Characteristic line linking heat output 1 to power output.

    Q2_char : tespy.tools.charactersitics.char_line, tespy.tools.data_containers.dc_cc
        Characteristic line linking heat output 2 to power output.

    Qloss_char : tespy.tools.charactersitics.char_line, tespy.tools.data_containers.dc_cc
        Characteristic line linking heat loss to power output.

    Note
    ----
    For more information on the usage of the combustion engine see the
    examples in the tespy_examples repository.

    Example
    -------
    The combustion chamber calculates energy input due to combustion as well as
    the flue gas composition based on the type of fuel and the amount of
    oxygen supplied. In this example a mixture of methane, hydrogen and
    carbondioxide is used as fuel. There are two cooling ports, the cooling
    water will flow through them in parallel.

    >>> from tespy.components import (Sink, Source, CombustionEngine, Merge,
    ... Splitter)
    >>> from tespy.connections import Connection, Ref
    >>> from tespy.networks import Network
    >>> import numpy as np
    >>> import shutil
    >>> fluid_list = ['Ar', 'N2', 'O2', 'CO2', 'CH4', 'H2O']
    >>> nw = Network(fluids=fluid_list, p_unit='bar', T_unit='C',
    ... iterinfo=False)
    >>> amb = Source('ambient')
    >>> sf = Source('fuel')
    >>> fg = Sink('flue gas outlet')
    >>> cw_in = Source('cooling water inlet')
    >>> sp = Splitter('cooling water splitter', num_out=2)
    >>> me = Merge('cooling water merge', num_in=2)
    >>> cw_out = Sink('cooling water outlet')
    >>> chp = CombustionEngine(label='internal combustion engine')
    >>> chp.component()
    'combustion engine'
    >>> amb_comb = Connection(amb, 'out1', chp, 'in3')
    >>> sf_comb = Connection(sf, 'out1', chp, 'in4')
    >>> comb_fg = Connection(chp, 'out3', fg, 'in1')
    >>> nw.add_conns(sf_comb, amb_comb, comb_fg)
    >>> cw_sp = Connection(cw_in, 'out1', sp, 'in1')
    >>> sp_chp1 = Connection(sp, 'out1', chp, 'in1')
    >>> sp_chp2 = Connection(sp, 'out2', chp, 'in2')
    >>> chp1_me = Connection(chp, 'out1', me, 'in1')
    >>> chp2_me = Connection(chp, 'out2', me, 'in2')
    >>> me_cw = Connection(me, 'out1', cw_out, 'in1')
    >>> nw.add_conns(cw_sp, sp_chp1, sp_chp2, chp1_me, chp2_me, me_cw)

    The combustion engine produces a power output of 10 MW the oxygen to
    stoichiometric oxygen ratio is set to 1. Only pressure ratio 1 is set as
    we reconnect both cooling water streams. At the merge all pressure values
    will be identical automatically. Reference the mass flow at the splitter
    to be split in half.

    >>> chp.set_attr(pr1=0.99, P=10e6, lamb=1.0,
    ... design=['pr1'], offdesign=['zeta1'])
    >>> amb_comb.set_attr(p=5, T=30, fluid={'Ar': 0.0129, 'N2': 0.7553,
    ... 'H2O': 0, 'CH4': 0, 'CO2': 0.0004, 'O2': 0.2314})
    >>> sf_comb.set_attr(m0=0.1, T=30, fluid={'CO2': 0, 'Ar': 0, 'N2': 0,
    ... 'O2': 0, 'H2O': 0, 'CH4': 1})
    >>> cw_sp.set_attr(p=3, T=60, m=50, fluid={'CO2': 0, 'Ar': 0, 'N2': 0,
    ... 'O2': 0, 'H2O': 1, 'CH4': 0})
    >>> sp_chp2.set_attr(m=Ref(sp_chp1, 1, 0))
    >>> mode = 'design'
    >>> nw.solve(mode=mode)
    >>> nw.save('tmp')
    >>> round(chp.ti.val, 0)
    25300000.0
    >>> round(chp.Q1.val, 0)
    4980000.0
    >>> chp.set_attr(Q1=4e6, P=np.nan)
    >>> mode = 'offdesign'
    >>> nw.solve(mode=mode, init_path='tmp', design_path='tmp')
    >>> round(chp.ti.val, 0)
    17794554.0
    >>> round(chp.P.val / chp.P.design, 3)
    0.617
    >>> chp.set_attr(P=chp.P.design * 0.75, Q1=np.nan)
    >>> mode = 'offdesign'
    >>> nw.solve(mode=mode, init_path='tmp', design_path='tmp')
    >>> round(chp.ti.val, 0)
    20550000.0
    >>> round(chp.P.val / chp.P.design, 3)
    0.75
    >>> shutil.rmtree('./tmp', ignore_errors=True)
    """

    @staticmethod
    def component():
        return 'combustion engine'

    @staticmethod
    def attr():
        return {'lamb': dc_cp(min_val=1),
                'ti': dc_cp(min_val=0),
                'P': dc_cp(val=1e6, d=1, min_val=1),
                'Q1': dc_cp(min_val=1), 'Q2': dc_cp(min_val=1),
                'Qloss': dc_cp(val=1e5, d=1, min_val=1),
                'pr1': dc_cp(max_val=1),
                'pr2': dc_cp(max_val=1),
                'zeta1': dc_cp(min_val=0),
                'zeta2': dc_cp(min_val=0),
                'tiP_char': dc_cc(),
                'Q1_char': dc_cc(),
                'Q2_char': dc_cc(),
                'Qloss_char': dc_cc(),
                'S': dc_simple()}

    @staticmethod
    def inlets():
        return ['in1', 'in2', 'in3', 'in4']

    @staticmethod
    def outlets():
        return ['out1', 'out2', 'out3']

    def comp_init(self, nw):

        if not self.P.is_set:
            self.set_attr(P='var')
            msg = ('The power output of combustion engines must be set! '
                   'We are adding the power output of component ' +
                   self.label + ' as custom variable of the system.')
            logging.info(msg)

        if not self.Qloss.is_set:
            self.set_attr(Qloss='var')
            msg = ('The heat loss of combustion engines must be set! '
                   'We are adding the heat loss of component ' +
                   self.label + ' as custom variable of the system.')
            logging.info(msg)

        Component.comp_init(self, nw)

        # number of mandatroy equations for
        # cooling loops fluid balances: 2 * num_fl
        # mass flow: 3
        # pressure: 2
        # reaction balance: num_fl
        # energy balance, characteristic functions: 5
        self.num_eq = self.num_nw_fluids * 2 + 3 + 2 + self.num_nw_fluids + 5
        # P and Qloss are not included, as the equations are mandatory anyway
        for var in [self.lamb, self.ti, self.Q1, self.Q2,
                    self.pr1, self.pr2, self.zeta1, self.zeta2]:
            if var.is_set is True:
                self.num_eq += 1

        self.jacobian = np.zeros((
            self.num_eq,
            self.num_i + self.num_o + self.num_vars,
            self.num_nw_vars))

        pos = self.num_nw_fluids * 2

        self.residual = np.zeros(self.num_eq)
        self.jacobian[0:pos] = self.fluid_deriv()
        self.jacobian[pos:pos + 3] = self.mass_flow_deriv()
        self.jacobian[pos + 3:pos + 5] = self.pressure_deriv()

        self.setup_reaction_parameters()

    def equations(self):
        r"""Calculate residual vector with results of equations."""
        k = 0
        ######################################################################
        # equations for fluids in cooling loops
        self.residual[k:self.num_nw_fluids * 2] = self.fluid_func()
        k += self.num_nw_fluids * 2

        ######################################################################
        # equations for mass flow
        self.residual[k:k + 3] = self.mass_flow_func()
        k += 3

        ######################################################################
        # equations for pressure balance in combustion
        self.residual[k] = self.inl[2].p.val_SI - self.outl[2].p.val_SI
        k += 1
        self.residual[k] = self.inl[2].p.val_SI - self.inl[3].p.val_SI
        k += 1

        ######################################################################
        # equations for fluids in combustion chamber
        for fluid in self.inl[0].fluid.val.keys():
            if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                    self.always_all_equations):
                self.residual[k] = self.reaction_balance(fluid)
            k += 1

        ######################################################################
        # equation for combustion engine energy balance
        if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                self.always_all_equations):
            self.residual[k] = self.energy_balance()
        k += 1

        ######################################################################
        # equation for power to thermal input ratio from characteristic line
        if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                self.always_all_equations):
            self.residual[k] = self.tiP_char_func()
        k += 1

        ######################################################################
        # equations for heat outputs from characteristic line
        if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                self.always_all_equations):
            self.residual[k] = self.Q1_char_func()
        k += 1

        if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                self.always_all_equations):
            self.residual[k] = self.Q2_char_func()
        k += 1

        ######################################################################
        # equation for heat loss from characteristic line
        if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                self.always_all_equations):
            self.residual[k] = self.Qloss_char_func()
        k += 1

        ######################################################################
        # equation for specified lambda
        if self.lamb.is_set:
            self.residual[k] = self.lambda_func()
            k += 1

        ######################################################################
        # equation for specified thermal input
        if self.ti.is_set:
            self.residual[k] = self.ti_func()
            k += 1

        ######################################################################
        # equations for specified heat ouptputs
        if self.Q1.is_set:
            self.residual[k] = self.Q1_func()
            k += 1

        if self.Q2.is_set:
            self.residual[k] = self.Q2_func()
            k += 1

        ######################################################################
        # equations for specified pressure ratios at cooling loops
        if self.pr1.is_set:
            self.residual[k] = (
                self.pr1.val * self.inl[0].p.val_SI - self.outl[0].p.val_SI)
            k += 1

        if self.pr2.is_set:
            self.residual[k] = (
                self.pr2.val * self.inl[1].p.val_SI - self.outl[1].p.val_SI)
            k += 1

        ######################################################################
        # equations for specified zeta values at cooling loops
        if self.zeta1.is_set:
            if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                    self.always_all_equations):
                self.residual[k] = self.zeta_func(
                    zeta='zeta1', inconn=0, outconn=0)
            k += 1

        if self.zeta2.is_set:
            if (np.absolute(self.residual[k]) > err ** 2 or self.it % 4 == 0 or
                    self.always_all_equations):
                self.residual[k] = self.zeta_func(
                    zeta='zeta2', inconn=1, outconn=1)
            k += 1

    def derivatives(self, increment_filter):
        r"""Calculate matrix of partial derivatives for given equations."""
        ######################################################################
        # derivatives cooling water fluid, mass balance and pressure are static
        k = self.num_nw_fluids * 2 + 5

        ######################################################################
        # derivatives for reaction balance
        for fluid in self.nw_fluids:
            # fresh air and fuel inlets
            if not increment_filter[2, 0]:
                self.jacobian[k, 2, 0] = self.rb_numeric_deriv('m', 2, fluid)
            if not all(increment_filter[2, 3:]):
                self.jacobian[k, 2, 3:] = self.rb_numeric_deriv(
                    'fluid', 2, fluid)
            if not increment_filter[3, 0]:
                self.jacobian[k, 3, 0] = self.rb_numeric_deriv('m', 3, fluid)
            if not all(increment_filter[3, 3:]):
                self.jacobian[k, 3, 3:] = self.rb_numeric_deriv(
                    'fluid', 3, fluid)

            # combustion outlet
            if not increment_filter[6, 0]:
                self.jacobian[k, 6, 0] = self.rb_numeric_deriv('m', 6, fluid)
            if not all(increment_filter[6, 3:]):
                self.jacobian[k, 6, 3:] = self.rb_numeric_deriv(
                    'fluid', 6, fluid)
            k += 1

        ######################################################################
        # derivatives for energy balance
        f = self.energy_balance
        # mass flow cooling water
        for i in [0, 1]:
            self.jacobian[k, i, 0] = -(
                self.outl[i].h.val_SI - self.inl[i].h.val_SI)

        # mass flow and pressure for combustion reaction
        for i in [2, 3, 6]:
            if not increment_filter[i, 0]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
            if not increment_filter[i, 1]:
                self.jacobian[k, i, 1] = self.numeric_deriv(f, 'p', i)

        # enthalpy
        for i in range(4):
            self.jacobian[k, i, 2] = self.inl[i].m.val_SI
        for i in range(3):
            self.jacobian[k, i + 4, 2] = -self.outl[i].m.val_SI

        # fluid composition
        for fl in self.fuel_list:
            pos = 3 + self.nw_fluids.index(fl)
            lhv = self.fuels[fl]['LHV']
            self.jacobian[k, 2, pos] = self.inl[2].m.val_SI * lhv
            self.jacobian[k, 3, pos] = self.inl[3].m.val_SI * lhv
            self.jacobian[k, 6, pos] = -self.outl[2].m.val_SI * lhv

        # power and heat loss
        if self.P.is_var:
            self.jacobian[k, 7 + self.P.var_pos, 0] = (
                self.numeric_deriv(f, 'P', 7))
        if self.Qloss.is_var:
            self.jacobian[k, 7 + self.Qloss.var_pos, 0] = (
                self.numeric_deriv(f, 'Qloss', 7))
        k += 1

        ######################################################################
        # derivatives for thermal input to power charactersitics
        f = self.tiP_char_func
        for i in [2, 3, 6]:
            if not increment_filter[i, 0]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
            if not all(increment_filter[i, 3:]):
                self.jacobian[k, i, 3:] = self.numeric_deriv(f, 'fluid', i)

        if self.P.is_var:
            self.jacobian[k, 7 + self.P.var_pos, 0] = (
                self.numeric_deriv(f, 'P', 7))
        k += 1

        ######################################################################
        # derivatives for heat output 1 to power charactersitics
        f = self.Q1_char_func
        if not increment_filter[0, 0]:
            self.jacobian[k, 0, 0] = self.numeric_deriv(f, 'm', 0)
        if not increment_filter[0, 2]:
            self.jacobian[k, 0, 2] = self.numeric_deriv(f, 'h', 0)
        if not increment_filter[4, 2]:
            self.jacobian[k, 4, 2] = self.numeric_deriv(f, 'h', 4)
        for i in [2, 3, 6]:
            if not increment_filter[i, 0]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
            if not all(increment_filter[i, 3:]):
                self.jacobian[k, i, 3:] = self.numeric_deriv(f, 'fluid', i)

        if self.P.is_var:
            self.jacobian[k, 7 + self.P.var_pos, 0] = (
                self.numeric_deriv(f, 'P', 7))
        k += 1

        ######################################################################
        # derivatives for heat output 2 to power charactersitics
        f = self.Q2_char_func
        if not increment_filter[1, 0]:
            self.jacobian[k, 1, 0] = self.numeric_deriv(f, 'm', 1)
        if not increment_filter[1, 2]:
            self.jacobian[k, 1, 2] = self.numeric_deriv(f, 'h', 1)
        if not increment_filter[5, 2]:
            self.jacobian[k, 5, 2] = self.numeric_deriv(f, 'h', 5)
        for i in [2, 3, 6]:
            if not increment_filter[i, 0]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
            if not all(increment_filter[i, 3:]):
                self.jacobian[k, i, 3:] = self.numeric_deriv(f, 'fluid', i)

        if self.P.is_var:
            self.jacobian[k, 7 + self.P.var_pos, 0] = (
                self.numeric_deriv(f, 'P', 7))
        k += 1

        ######################################################################
        # derivatives for heat loss to power charactersitics
        f = self.Qloss_char_func
        for i in [2, 3, 6]:
            if not increment_filter[i, 0]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
            if not all(increment_filter[i, 3:]):
                self.jacobian[k, i, 3:] = self.numeric_deriv(f, 'fluid', i)

        if self.P.is_var:
            self.jacobian[k, 7 + self.P.var_pos, 0] = (
                self.numeric_deriv(f, 'P', 7))
        if self.Qloss.is_var:
            self.jacobian[k, 7 + self.Qloss.var_pos, 0] = (
                self.numeric_deriv(f, 'Qloss', 7))
        k += 1

        ######################################################################
        # derivatives for specified lambda
        if self.lamb.is_set:
            f = self.lambda_func
            if not increment_filter[2, 0]:
                self.jacobian[k, 2, 0] = self.numeric_deriv(f, 'm', 2)
            if not all(increment_filter[2, 3:]):
                self.jacobian[k, 2, 3:] = self.numeric_deriv(f, 'fluid', 2)
            if not increment_filter[3, 0]:
                self.jacobian[k, 3, 0] = self.numeric_deriv(f, 'm', 3)
            if not all(increment_filter[3, 3:]):
                self.jacobian[k, 3, 3:] = self.numeric_deriv(f, 'fluid', 3)
            k += 1

        ######################################################################
        # derivatives for specified thermal input
        if self.ti.is_set:
            f = self.ti_func
            for i in [2, 3, 6]:
                self.jacobian[k, i, 0] = self.numeric_deriv(f, 'm', i)
                self.jacobian[k, i, 3:] = (
                    self.numeric_deriv(self.ti_func, 'fluid', i))
            k += 1

        ######################################################################
        # derivatives for specified heat outputs
        if self.Q1.is_set:
            self.jacobian[k, 0, 0] = -(
                self.outl[0].h.val_SI - self.inl[0].h.val_SI)
            self.jacobian[k, 0, 2] = self.inl[0].m.val_SI
            self.jacobian[k, 4, 2] = -self.inl[0].m.val_SI
            k += 1

        if self.Q2.is_set:
            self.jacobian[k, 1, 0] = -(
                self.outl[1].h.val_SI - self.inl[1].h.val_SI)
            self.jacobian[k, 1, 2] = self.inl[1].m.val_SI
            self.jacobian[k, 5, 2] = -self.inl[1].m.val_SI
            k += 1

        ######################################################################
        # derivatives for specified pressure ratio at cooling loops
        if self.pr1.is_set:
            self.jacobian[k, 0, 1] = self.pr1.val
            self.jacobian[k, 4, 1] = -1
            k += 1

        if self.pr2.is_set:
            self.jacobian[k, 1, 1] = self.pr2.val
            self.jacobian[k, 5, 1] = -1
            k += 1

        ######################################################################
        # derivatives for specified zeta values at cooling loops
        if self.zeta1.is_set:
            f = self.zeta_func
            if not increment_filter[0, 0]:
                self.jacobian[k, 0, 0] = self.numeric_deriv(
                    f, 'm', 0, zeta='zeta1', inconn=0, outconn=0)
            if not increment_filter[0, 1]:
                self.jacobian[k, 0, 1] = self.numeric_deriv(
                    f, 'p', 0, zeta='zeta1', inconn=0, outconn=0)
            if not increment_filter[0, 2]:
                self.jacobian[k, 0, 2] = self.numeric_deriv(
                    f, 'h', 0, zeta='zeta1', inconn=0, outconn=0)
            if not increment_filter[4, 1]:
                self.jacobian[k, 4, 1] = self.numeric_deriv(
                    f, 'p', 4, zeta='zeta1', inconn=0, outconn=0)
            if not increment_filter[4, 2]:
                self.jacobian[k, 4, 2] = self.numeric_deriv(
                    f, 'h', 4, zeta='zeta1', inconn=0, outconn=0)
            k += 1

        if self.zeta2.is_set:
            f = self.zeta_func
            if not increment_filter[1, 0]:
                self.jacobian[k, 1, 0] = self.numeric_deriv(
                    f, 'm', 1, zeta='zeta2', inconn=1, outconn=1)
            if not increment_filter[1, 1]:
                self.jacobian[k, 1, 1] = self.numeric_deriv(
                    f, 'p', 1, zeta='zeta2', inconn=1, outconn=1)
            if not increment_filter[1, 2]:
                self.jacobian[k, 1, 2] = self.numeric_deriv(
                    f, 'h', 1, zeta='zeta2', inconn=1, outconn=1)
            if not increment_filter[5, 1]:
                self.jacobian[k, 5, 1] = self.numeric_deriv(
                    f, 'p', 5, zeta='zeta2', inconn=1, outconn=1)
            if not increment_filter[5, 2]:
                self.jacobian[k, 5, 2] = self.numeric_deriv(
                    f, 'h', 5, zeta='zeta2', inconn=1, outconn=1)
            k += 1

    def fluid_func(self):
        r"""
        Calculate the vector of residual values for cooling loop fluid balance.

        Returns
        -------
        residual : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = fluid_{i,in_{j}} - fluid_{i,out_{j}}\\
                \forall i \in \mathrm{fluid}, \; \forall j \in [1, 2]
        """
        residual = []
        for i in range(2):
            for fluid, x in self.inl[i].fluid.val.items():
                residual += [x - self.outl[i].fluid.val[fluid]]
        return residual

    def mass_flow_func(self):
        r"""
        Calculate the residual value for component's mass flow balance.

        Returns
        -------
        residual : list
            Vector with residual value for component's mass flow balance.

            .. math::

                0 = \dot{m}_{in,i} - \dot{m}_{out,i}\\
                \forall i \in [1, 2]\\
                0 = \dot{m}_{in,3} + \dot{m}_{in,4} - \dot{m}_{out,3}
        """
        residual = []
        for i in range(2):
            residual += [self.inl[i].m.val_SI - self.outl[i].m.val_SI]
        residual += [self.inl[2].m.val_SI + self.inl[3].m.val_SI -
                     self.outl[2].m.val_SI]
        return residual

    def fluid_deriv(self):
        r"""
        Calculate the partial derivatives for cooling loop fluid balance.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros(
            (self.num_nw_fluids * 2, 7 + self.num_vars, self.num_nw_vars))
        for i in range(self.num_nw_fluids):
            deriv[i, 0, i + 3] = 1
            deriv[i, 4, i + 3] = -1
        for j in range(self.num_nw_fluids):
            deriv[i + 1 + j, 1, j + 3] = 1
            deriv[i + 1 + j, 5, j + 3] = -1
        return deriv

    def mass_flow_deriv(self):
        r"""
        Calculate the partial derivatives for all mass flow balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((3, 7 + self.num_vars, self.num_nw_vars))
        for i in range(2):
            deriv[i, i, 0] = 1
        for j in range(2):
            deriv[j, self.num_i + j, 0] = -1
        deriv[2, 2, 0] = 1
        deriv[2, 3, 0] = 1
        deriv[2, 6, 0] = -1
        return deriv

    def pressure_deriv(self):
        r"""
        Calculate the partial derivatives for combustion pressure equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((2, 7 + self.num_vars, self.num_nw_vars))
        for k in range(2):
            deriv[k, 2, 1] = 1
        deriv[0, 6, 1] = -1
        deriv[1, 3, 1] = -1
        return deriv

    def energy_balance(self):
        r"""
        Calculate the energy balance of the combustion engine.

        Returns
        -------
        res : float
            Residual value of equation.

            .. math::

                \begin{split}
                res = & \sum_i \dot{m}_{in,i} \cdot
                \left( h_{in,i} - h_{in,i,ref} \right)\\
                & - \sum_j \dot{m}_{out,3} \cdot
                \left( h_{out,3} - h_{out,3,ref} \right)\\
                & + H_{I,f} \cdot
                \left(\sum_i \left(\dot{m}_{in,i} \cdot x_{f,i} \right)-
                \dot{m}_{out,3} \cdot x_{f,3} \right)\\
                & - \dot{Q}_1 - \dot{Q}_2 - P - \dot{Q}_{loss}\\
                \end{split}\\
                \forall i \in [3,4]

        Note
        ----
        The temperature for the reference state is set to 20 °C, thus
        the water may be liquid. In order to make sure, the state is
        referring to the lower heating value, the necessary enthalpy
        difference for evaporation is added.

        - Reference temperature: 293.15 K.
        - Reference pressure: 1 bar.
        """
        T_ref = 293.15
        p_ref = 1e5

        res = 0
        for i in self.inl[2:]:
            res += i.m.val_SI * (
                i.h.val_SI - h_mix_pT([0, p_ref, 0, i.fluid.val], T_ref))

        for o in self.outl[2:]:
            dh = 0
            n_h2o = o.fluid.val[self.h2o] / molar_masses[self.h2o]
            if n_h2o > 0:
                p = p_ref * n_h2o / molar_mass_flow(o.fluid.val)
                h = h_pT(p, T_ref, self.h2o)
                try:
                    flow = [0, p, 0, {self.h2o: 1}]
                    h_steam = h_mix_pQ(flow, 1)
                    # CP.PropsSI('H', 'P', p, 'Q', 1, self.h2o)
                except ValueError:
                    flow = [0, 615, 0, {self.h2o: 1}]
                    h_steam = h_mix_pQ(flow, 1)
                if h < h_steam:
                    dh = (h_steam - h) * o.fluid.val[self.h2o]

            res -= o.m.val_SI * (
                o.h.val_SI - h_mix_pT([0, p_ref, 0, o.fluid.val], T_ref) - dh)

        res += self.calc_ti()

        # cooling water
        for i in range(2):
            res -= self.inl[i].m.val_SI * (
                self.outl[i].h.val_SI - self.inl[i].h.val_SI)

        # power output and heat loss
        res -= self.P.val + self.Qloss.val

        return res

    def bus_func(self, bus):
        r"""
        Calculate the value of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus
            TESPy bus object.

        Returns
        -------
        val : float
            Value of energy transfer :math:`\dot{E}`. This value is passed to
            :py:meth:`tespy.components.component.Component.calc_bus_value`
            for value manipulation according to the specified characteristic
            line of the bus.

            .. math::

                \dot{E} = \begin{cases}
                LHV \cdot \dot{m}_{f} & \text{key = 'TI'}\\
                P & \text{key = 'P'}\\
                \dot{m}_1 \cdot \left( h_{1,out} - h_{1,in} \right) +
                \dot{m}_2 \cdot \left( h_{2,out} - h_{2,in} \right) &
                \text{key = 'Q'}\\
                \dot{m}_1 \cdot \left( h_{1,out} - h_{1,in} \right) &
                \text{key = 'Q1'}\\
                \dot{m}_2 \cdot \left( h_{2,out} - h_{2,in} \right) &
                \text{key = 'Q2'}\\
                \dot{Q}_{loss} & \text{key = 'Qloss'}
                \end{cases}

                \dot{Q}_1=\dot{m}_1 \cdot \left( h_{1,out} - h_{1,in} \right)\\
                \dot{Q}_2=\dot{m}_2 \cdot \left( h_{2,out} - h_{2,in} \right)
        """
        ######################################################################
        # value for bus parameter of thermal input (TI)
        if bus['param'] == 'TI':
            val = self.calc_ti()

        ######################################################################
        # value for bus parameter of power output (P)
        elif bus['param'] == 'P':
            val = self.calc_P()

        ######################################################################
        # value for bus parameter of total heat production (Q)
        elif bus['param'] == 'Q':
            val = 0
            for j in range(2):
                i = self.inl[j]
                o = self.outl[j]
                val += i.m.val_SI * (o.h.val_SI - i.h.val_SI)

        ######################################################################
        # value for bus parameter of heat production 1 (Q1)
        elif bus['param'] == 'Q1':
            i = self.inl[0]
            o = self.outl[0]
            val = i.m.val_SI * (o.h.val_SI - i.h.val_SI)

        ######################################################################
        # value for bus parameter of heat production 2 (Q2)
        elif bus['param'] == 'Q2':
            i = self.inl[1]
            o = self.outl[1]
            val = i.m.val_SI * (o.h.val_SI - i.h.val_SI)

        ######################################################################
        # value for bus parameter of heat loss (Qloss)
        elif bus['param'] == 'Qloss':
            val = self.calc_Qloss()

        ######################################################################
        # missing/invalid bus parameter
        else:
            msg = ('The parameter ' + str(bus['param']) +
                   ' is not a valid parameter for a ' + self.component() + '.')
            logging.error(msg)
            raise ValueError(msg)

        return val

    def bus_deriv(self, bus):
        r"""
        Calculate the matrix of partial derivatives of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus
            TESPy bus object.

        Returns
        -------
        deriv : ndarray
            Matrix of partial derivatives.
        """
        deriv = np.zeros((1, 7 + self.num_vars, self.num_nw_vars))
        f = self.calc_bus_value
        b = bus.comps.loc[self]

        ######################################################################
        # derivatives for bus parameter of thermal input (TI)
        if b['param'] == 'TI':
            for i in [2, 3, 6]:
                deriv[0, i, 0] = self.numeric_deriv(f, 'm', i, bus=bus)
                deriv[0, i, 3:] = self.numeric_deriv(f, 'fluid', i, bus=bus)

        ######################################################################
        # derivatives for bus parameter of power production (P) or
        # heat loss (Qloss)
        elif b['param'] == 'P' or b['param'] == 'Qloss':
            for i in [2, 3, 6]:
                deriv[0, i, 0] = self.numeric_deriv(f, 'm', i, bus=bus)
                deriv[0, i, 3:] = self.numeric_deriv(f, 'fluid', i, bus=bus)

            # variable power
            if self.P.is_var:
                deriv[0, 7 + self.P.var_pos, 0] = (
                    self.numeric_deriv(f, 'P', 7, bus=bus))

        ######################################################################
        # derivatives for bus parameter of total heat production (Q)
        elif b['param'] == 'Q':
            for i in range(2):
                deriv[0, i, 0] = self.numeric_deriv(f, 'm', i, bus=bus)
                deriv[0, i, 2] = self.numeric_deriv(f, 'h', i, bus=bus)
                deriv[0, i + 4, 2] = self.numeric_deriv(f, 'h', i + 4, bus=bus)

        ######################################################################
        # derivatives for bus parameter of heat production 1 (Q1)
        elif b['param'] == 'Q1':
            deriv[0, 0, 0] = self.numeric_deriv(f, 'm', 0, bus=bus)
            deriv[0, 0, 2] = self.numeric_deriv(f, 'h', 0, bus=bus)
            deriv[0, 4, 2] = self.numeric_deriv(f, 'h', 4, bus=bus)

        ######################################################################
        # derivatives for bus parameter of heat production 2 (Q2)
        elif b['param'] == 'Q2':
            deriv[0, 1, 0] = self.numeric_deriv(f, 'm', 1, bus=bus)
            deriv[0, 1, 2] = self.numeric_deriv(f, 'h', 1, bus=bus)
            deriv[0, 5, 2] = self.numeric_deriv(f, 'h', 5, bus=bus)

        ######################################################################
        # missing/invalid bus parameter
        else:
            msg = ('The parameter ' + str(b['param']) +
                   ' is not a valid parameter for a ' + self.component() + '.')
            logging.error(msg)
            raise ValueError(msg)

        return deriv

    def Q1_func(self):
        r"""
        Calculate residual value with specified Q1.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                val = \dot{m}_1 \cdot \left(h_{out,1} -
                h_{in,1} \right) - \dot{Q}_1
        """
        i = self.inl[0]
        o = self.outl[0]

        return self.Q1.val - i.m.val_SI * (o.h.val_SI - i.h.val_SI)

    def Q2_func(self):
        r"""
        Calculate residual value with specified Q2.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                0 = \dot{m}_2 \cdot \left(h_{out,2} - h_{in,2} \right) -
                \dot{Q}_2
        """
        i = self.inl[1]
        o = self.outl[1]

        return self.Q2.val - i.m.val_SI * (o.h.val_SI - i.h.val_SI)

    def tiP_char_func(self):
        r"""
        Calculate the relation of output power and thermal input.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                0 = P \cdot f_{TI}\left(\frac{P}{P_{ref}}\right)- LHV \cdot
                \left[\sum_i \left(\dot{m}_{in,i} \cdot
                x_{f,i}\right) - \dot{m}_{out,3} \cdot x_{f,3} \right]
                \; \forall i \in [1,2]
        """
        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return self.calc_ti() - self.tiP_char.func.evaluate(expr) * self.P.val

    def Q1_char_func(self):
        r"""
        Calculate the relation of heat output 1 and thermal input.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                \begin{split}
                0 = & \dot{m}_1 \cdot \left(h_{out,1} - h_{in,1} \right) \cdot
                f_{TI}\left(\frac{P}{P_{ref}}\right) \\
                & - LHV \cdot \left[\sum_i
                \left(\dot{m}_{in,i} \cdot x_{f,i}\right) -
                \dot{m}_{out,3} \cdot x_{f,3} \right] \cdot
                f_{Q1}\left(\frac{P}{P_{ref}}\right)\\
                \end{split}\\
                \forall i \in [3,4]
        """
        i = self.inl[0]
        o = self.outl[0]

        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return (self.calc_ti() * self.Q1_char.func.evaluate(expr) -
                self.tiP_char.func.evaluate(expr) * i.m.val_SI *
                (o.h.val_SI - i.h.val_SI))

    def Q2_char_func(self):
        r"""
        Calculate the relation of heat output 2 and thermal input.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                \begin{split}
                0 = & \dot{m}_2 \cdot \left(h_{out,2} - h_{in,2} \right) \cdot
                f_{TI}\left(\frac{P}{P_{ref}}\right) \\
                & - LHV \cdot \left[\sum_i
                \left(\dot{m}_{in,i} \cdot x_{f,i}\right) -
                \dot{m}_{out,3} \cdot x_{f,3} \right] \cdot
                f_{Q2}\left(\frac{P}{P_{ref}}\right)\\
                \end{split}\\
                \forall i \in [3,4]
        """
        i = self.inl[1]
        o = self.outl[1]

        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return (self.calc_ti() * self.Q2_char.func.evaluate(expr) -
                self.tiP_char.func.evaluate(expr) * i.m.val_SI *
                (o.h.val_SI - i.h.val_SI))

    def Qloss_char_func(self):
        r"""
        Calculate the relation of heat loss and thermal input.

        Returns
        -------
        val : float
            Residual value of equation.

            .. math::

                \begin{split}
                0 = & \dot{Q}_{loss} \cdot
                f_{TI}\left(\frac{P}{P_{ref}}\right) \\
                & - LHV \cdot \left[\sum_i
                \left(\dot{m}_{in,i} \cdot x_{f,i}\right) -
                \dot{m}_{out,3} \cdot x_{f,3} \right] \cdot
                f_{QLOSS}\left(\frac{P}{P_{ref}}\right)\\
                \end{split}\\
                \forall i \in [3,4]
        """
        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return (self.calc_ti() * self.Qloss_char.func.evaluate(expr) -
                self.tiP_char.func.evaluate(expr) * self.Qloss.val)

    def calc_ti(self):
        r"""
        Calculate the thermal input of the combustion engine.

        Returns
        -------
        ti : float
            Thermal input.

            .. math::

                ti = LHV \cdot \left[\sum_i \left(\dot{m}_{in,i} \cdot x_{f,i}
                \right) - \dot{m}_{out,3} \cdot x_{f,3} \right]

                \forall i \in [3,4]
        """
        ti = 0
        for f in self.fuel_list:
            m = 0
            for i in self.inl[2:]:
                m += i.m.val_SI * i.fluid.val[f]

            for o in self.outl[2:]:
                m -= o.m.val_SI * o.fluid.val[f]

            ti += m * self.fuels[f]['LHV']

        return ti

    def calc_P(self):
        r"""
        Calculate the power output of the combustion engine.

        Returns
        -------
        P : float
            Power output.

            .. math::

                P = \frac{LHV \cdot \dot{m}_{f}}
                {f_{TI}\left(\frac{P}{P_{ref}}\right)}

        """
        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return self.calc_ti() / self.tiP_char.func.evaluate(expr)

    def calc_Qloss(self):
        r"""
        Calculate the heat loss of the combustion engine.

        Returns
        -------
        Qloss : float
            Heat loss.

            .. math::

                \dot{Q}_{loss} = \frac{LHV \cdot \dot{m}_{f} \cdot
                f_{QLOSS}\left(\frac{P}{P_{ref}}\right)}
                {f_{TI}\left(\frac{P}{P_{ref}}\right)}
        """
        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design

        return (self.calc_ti() * self.Qloss_char.func.evaluate(expr) /
                self.tiP_char.func.evaluate(expr))

    def initialise_fluids(self):
        r"""
        Calculate reaction balance for generic starting values at outlet.

        Parameters
        ----------
        nw : tespy.networks.networks.Networks.Network
            Network using this component object.
        """
        N_2 = 0.7655
        O_2 = 0.2345

        n_fuel = 1
        lamb = 3

        fact_fuel = {}
        sum_fuel = 0
        for f in self.fuel_list:
            fact_fuel[f] = 0
            for i in self.inl:
                fact_fuel[f] += i.fluid.val[f] / 2
            sum_fuel += fact_fuel[f]

        for f in self.fuel_list:
            fact_fuel[f] /= sum_fuel

        m_co2 = 0
        m_h2o = 0
        m_fuel = 0
        for f in self.fuel_list:
            m_co2 += (n_fuel * self.fuels[f]['C'] * molar_masses[self.co2] *
                      fact_fuel[f])
            m_h2o += (n_fuel * self.fuels[f]['H'] /
                      2 * molar_masses[self.h2o] * fact_fuel[f])
            m_fuel += n_fuel * molar_masses[f] * fact_fuel[f]

        n_o2 = (m_co2 / molar_masses[self.co2] +
                0.5 * m_h2o / molar_masses[self.h2o]) * lamb

        m_air = n_o2 * molar_masses[self.o2] / O_2
        m_fg = m_air + m_fuel

        m_o2 = n_o2 * molar_masses[self.o2] * (1 - 1 / lamb)
        m_n2 = N_2 * m_air

        fg = {
            self.n2: m_n2 / m_fg,
            self.co2: m_co2 / m_fg,
            self.o2: m_o2 / m_fg,
            self.h2o: m_h2o / m_fg
        }

        o = self.outl[2]
        for fluid, x in o.fluid.val.items():
            if not o.fluid.val_set[fluid] and fluid in fg.keys():
                o.fluid.val[fluid] = fg[fluid]
        o.target.propagate_fluid_to_target(o, o.target)

    @staticmethod
    def initialise_source(c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        c : tespy.connections.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                5 \cdot 10^5 & \text{key = 'p'}\\
                10^6 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 5e5
        elif key == 'h':
            return 10e5

    @staticmethod
    def initialise_target(c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        c : tespy.connections.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                5 \cdot 10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 5e5
        elif key == 'h':
            return 5e5

    def propagate_fluid_to_target(self, inconn, start):
        r"""
        Propagate the fluids towards connection's target in recursion.

        Parameters
        ----------
        inconn : tespy.connections.connection.Connection
            Connection to initialise.

        start : tespy.components.component.Component
            This component is the fluid propagation starting point.
            The starting component is saved to prevent infinite looping.
        """
        for outconn in self.outl[:2]:
            for fluid, x in inconn.fluid.val.items():
                if (outconn.fluid.val_set[fluid] is False and
                        outconn.good_starting_values is False):
                    outconn.fluid.val[fluid] = x
            outconn.target.propagate_fluid_to_target(outconn, start)

    def propagate_fluid_to_source(self, outconn, start):
        r"""
        Propagate the fluids towards connection's source in recursion.

        Parameters
        ----------
        outconn : tespy.connections.connection.Connection
            Connection to initialise.

        start : tespy.components.component.Component
            This component is the fluid propagation starting point.
            The starting component is saved to prevent infinite looping.
        """
        for inconn in self.inl[:2]:
            for fluid, x in outconn.fluid.val.items():
                if (inconn.fluid.val_set[fluid] is False and
                        inconn.good_starting_values is False):
                    inconn.fluid.val[fluid] = x

            inconn.source.propagate_fluid_to_source(inconn, start)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i1 = self.inl[0].to_flow()
        i2 = self.inl[1].to_flow()
        o1 = self.outl[0].to_flow()
        o2 = self.outl[1].to_flow()

        v_i1 = v_mix_ph(i1, T0=self.inl[0].T.val_SI)
        v_o1 = v_mix_ph(o1, T0=self.outl[0].T.val_SI)
        v_i2 = v_mix_ph(i2, T0=self.inl[1].T.val_SI)
        v_o2 = v_mix_ph(o1, T0=self.outl[1].T.val_SI)

        self.pr1.val = o1[1] / i1[1]
        self.pr2.val = o2[1] / i2[1]
        self.zeta1.val = ((i1[1] - o1[1]) * np.pi ** 2 /
                          (8 * i1[0] ** 2 * (v_i1 + v_o1) / 2))
        self.zeta2.val = ((i2[1] - o2[1]) * np.pi ** 2 /
                          (8 * i2[0] ** 2 * (v_i2 + v_o2) / 2))
        self.Q1.val = i1[0] * (o1[2] - i1[2])
        self.Q2.val = i2[0] * (o2[2] - i2[2])
        self.P.val = self.calc_P()
        self.Qloss.val = self.calc_Qloss()

        # get bound errors for characteristic lines
        if np.isnan(self.P.design):
            expr = 1
        else:
            expr = self.P.val / self.P.design
        self.tiP_char.func.get_bound_errors(expr, self.label)
        self.Qloss_char.func.get_bound_errors(expr, self.label)
        self.Q1_char.func.get_bound_errors(expr, self.label)
        self.Q2_char.func.get_bound_errors(expr, self.label)

        CombustionChamber.calc_parameters(self)
