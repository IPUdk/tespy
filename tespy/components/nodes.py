# -*- coding: utf-8

"""Module for components of type node.

Components in this module:

    - :func:`tespy.components.nodes.node`
    - :func:`tespy.components.nodes.merge`
    - :func:`tespy.components.nodes.splitter`
    - :func:`tespy.components.nodes.separator`


This file is part of project TESPy (github.com/oemof/tespy). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location tespy/components/nodes.py

SPDX-License-Identifier: MIT
"""

import numpy as np

from tespy.components.components import component

from tespy.tools.data_containers import dc_simple
from tespy.tools.helpers import num_fluids
from tespy.tools.fluid_properties import (
        T_mix_ph, dT_mix_dph, dT_mix_pdh, dT_mix_ph_dfluid,
        h_mix_pQ
        )

# %%


class node(component):
    r"""
    The component node is the parent class for splitter, separator and merge.

    Equations

        **mandatory equations**

        - :func:`tespy.components.components.component.mass_flow_func`

        .. math::

            0 = p_{in,1} - p_{out,i}\\
            \forall i \in \mathrm{outlets}

        **additional equations**

        - :func:`tespy.components.components.splitter.additional_equations`
        - :func:`tespy.components.components.separator.additional_equations`
        - :func:`tespy.components.components.merge.additional_equations`

    Inlets/Outlets

        - specify number of outlets with :code:`num_in` (default value: 2)
        - specify number of outlets with :code:`num_out` (default value: 2)

    Image

        .. image:: _images/node.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    num_in : float/tespy.helpers.dc_simple
        Number of inlets for this component, default value: 2.

    num_out : float/tespy.helpers.dc_simple
        Number of outlets for this component, default value: 2.

    Note
    ----
    - Node: Fluid composition and enthalpy at all **outgoing** connections
      (mass flow leaves the node) is result of mixture of the properties of
      the incoming connections (mass flow enters node).
      Incoming and outgoing connections can be a result of the calculation and
      are not identical to the inlets and outlets!
    - Splitter: Fluid composition and enthalpy at all outlets is the same as
      the inlet's properties.
    - Separator: Fluid composition is variable for all outlets, temperature at
      all outlets is the same as the inlet's temperature.
    - Merge: Fluid composition and enthalpy at outlet is result of mixture of
      the inlet's properties.

    Example
    -------
    >>> from tespy import cmp, con, nwk, hlp
    >>> import numpy as np
    >>> fluid_list = ['O2', 'N2']
    >>> nw = nwk.network(fluids=fluid_list, p_unit='bar', T_unit='C',
    ...     h_unit='kJ / kg')
    >>> nw.set_printoptions(print_level='none')
    >>> so1 = cmp.source('source1')
    >>> so2 = cmp.source('source2')
    >>> si1 = cmp.sink('sink1')
    >>> si2 = cmp.sink('sink2')
    >>> n = cmp.node('node', num_in=2, num_out=2)
    >>> n.component()
    'node'
    >>> inc1 = con.connection(so1, 'out1', n, 'in1')
    >>> inc2 = con.connection(so2, 'out1', n, 'in2')
    >>> outg1 = con.connection(n, 'out1', si1, 'in1')
    >>> outg2 = con.connection(n, 'out2', si2, 'in1')
    >>> nw.add_conns(inc1, inc2, outg1, outg2)
    >>> inc1.set_attr(fluid={'O2': 1, 'N2': 0}, p=1, T=20, m=2)
    >>> inc2.set_attr(fluid={'O2': 0.5, 'N2': 0.5}, T=50, m=5)
    >>> outg1.set_attr(m=3)
    >>> nw.solve('design')
    >>> (round(outg1.fluid.val['O2'], 3), round(outg1.fluid.val['N2'], 3))
    (0.643, 0.357)
    >>> inc2.set_attr(m=np.nan)
    >>> outg1.set_attr(fluid={'O2': 0.8})
    >>> nw.solve('design')
    >>> round(inc2.m.val_SI, 3)
    1.333
    """

    def component(self):
        return 'node'

    def attr(self):
        return {'num_in': dc_simple(),
                'num_out': dc_simple()}

    def inlets(self):
        if self.num_in.val_set:
            return ['in' + str(i + 1) for i in range(self.num_in.val)]
        else:
            self.set_attr(num_in=2)
            return self.inlets()

    def outlets(self):
        if self.num_out.val_set:
            return ['out' + str(i + 1) for i in range(self.num_out.val)]
        else:
            self.set_attr(num_out=2)
            return self.outlets()

    def comp_init(self, nw):

        component.comp_init(self, nw)

        self.m_deriv = self.mass_flow_deriv()
        self.p_deriv = self.pressure_deriv()

    def equations(self):
        r"""
        Calculates vector vec_res with results of equations for this component.

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # eqation for mass flow balance
        vec_res += self.mass_flow_func()

        ######################################################################
        # equations for pressure
        inl = []
        if self.num_i > 1:
            inl = self.inl[1:]
        for c in inl + self.outl:
            vec_res += [self.inl[0].p.val_SI - c.p.val_SI]

        ######################################################################
        # additional eqations
        vec_res += self.additional_equations()

        return vec_res

    def derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given equations.

        Returns
        -------
        mat_deriv : ndarray
            Matrix of partial derivatives.
        """
        mat_deriv = []

        ######################################################################
        # derivative for mass flow balance equation
        mat_deriv += self.m_deriv

        ######################################################################
        # derivatives for pressure equations
        mat_deriv += self.p_deriv

        ######################################################################
        # additional derivatives
        mat_deriv += self.additional_derivatives()

        return np.asarray(mat_deriv)

    def additional_equations(self):
        r"""
        Calculates vector vec_res with results of additional equations for this
        component.

        Equations

            **mandatroy equations**

            - :func:`tespy.components.components.node.fluid_func`

            .. math::

                0 = \sum_i \left(\dot{m}_{i} \cdot h_{i}\right) - h_{o} \cdot
                \sum_i \dot{m}_{i}\\
                \forall o \in \text{outgoing mass flows}\\
                \text{i: incoming mass flows}

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # check for incoming/outgoing mass flows in inlets and outlets

        loc = 0
        # total incoming enthalpy
        h = 0
        # total incoming mass flow (constant within every iteration)
        self.m_inc = 0

        self.inc = []
        self.outg = []
        for c in self.inl:
            # incoming
            if c.m.val_SI >= 0:
                self.inc += [[c, loc]]
                self.m_inc += c.m.val_SI
                h += c.m.val_SI * c.h.val_SI
            # outgoing
            else:
                self.outg += [[c, loc]]
            loc += 1

        for c in self.outl:
            # inconming
            if c.m.val_SI < 0:
                self.inc += [[c, loc]]
                self.m_inc -= c.m.val_SI
                h -= c.m.val_SI * c.h.val_SI
            # outgoing
            else:
                self.outg += [[c, loc]]
            loc += 1

        ######################################################################
        # equations for fluid composition
        vec_res += self.fluid_func()

        ######################################################################
        # equations for energy balance
        for o in self.outg:
            vec_res += [h - o[0].h.val_SI * self.m_inc]

        return vec_res

    def additional_derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given additional
        equations.

        Returns
        -------
        mat_deriv : list
            Matrix of partial derivatives.
        """
        mat_deriv = []

        ######################################################################
        # derivatives for fluid balance equations
        mat_deriv += self.fluid_deriv()

        ######################################################################
        # derivatives for energy balance equations
        deriv = np.zeros((len(self.outg), self.num_i + self.num_o,
                          self.num_fl + 3))
        k = 0
        for o in self.outg:
            deriv[k, o[1], 2] = -self.m_inc
            for i in self.inc:
                deriv[k, i[1], 0] = i[0].h.val_SI - o[0].h.val_SI
                deriv[k, i[1], 2] = abs(i[0].m.val_SI)
            k += 1
        mat_deriv += deriv.tolist()

        return mat_deriv

    def fluid_func(self):
        r"""
        Calculates the vector of residual values for component's fluid balance
        equations.

        Returns
        -------
        vec_res : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = \sum_i \left(\dot{m}_{i} \cdot x_{i,j}\right) - x_{o,j}
                \cdot  \sum_i \dot{m}_{i}\\
                \forall j \in \text{fluids}\\
                \forall o \in \text{outgoing mass flows}\\
                \text{i: incoming mass flows}
        """
        vec_res = []

        for fluid in self.fluids:
            m = 0
            for i in self.inc:
                m += abs(i[0].m.val_SI) * i[0].fluid.val[fluid]
            for o in self.outg:
                vec_res += [m - o[0].fluid.val[fluid] * self.m_inc]
        return vec_res

    def fluid_deriv(self):
        r"""
        Calculates the partial derivatives for all fluid balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        num_o = len(self.outg)
        deriv = np.zeros((self.num_fl * num_o, self.num_i + self.num_o,
                          3 + self.num_fl))
        j = 0
        k = 0
        for fluid in self.fluids:
            for o in self.outg:
                deriv[k, o[1], j + 3] = -self.m_inc
                for i in self.inc:
                    deriv[k, i[1], 0] = -i[0].fluid.val[fluid]
                    deriv[k, i[1], j + 3] = -abs(i[0].m.val_SI)
                k += 1
            j += 1

        return deriv.tolist()

    def pressure_deriv(self):
        r"""
        Calculates the partial derivatives for all pressure equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((self.num_i + self.num_o - 1, self.num_i + self.num_o,
                          self.num_fl + 3))

        inl = []
        if self.num_i > 1:
            inl = self.inl[1:]
        for k in range(len(inl + self.outl)):
            deriv[k, 0, 1] = 1
            deriv[k, k + 1, 1] = -1
        return deriv.tolist()

    def initialise_fluids(self, nw):
        r"""
        Fluid initialisation for fluid mixture at outlet of the node.

        Parameters
        ----------
        nw : tespy.networks.network
            Network using this component object.
        """
        num_fl = {}
        for o in self.outl:
            num_fl[o] = num_fluids(o.fluid.val)

        for i in self.inl:
            num_fl[i] = num_fluids(i.fluid.val)

        ls = []
        if any(num_fl.values()) and not all(num_fl.values()):
            for conn, num in num_fl.items():
                if num == 1:
                    ls += [conn]

            for c in ls:
                for fluid in nw.fluids:
                    for o in self.outl:
                        if not o.fluid.val_set[fluid]:
                            o.fluid.val[fluid] = c.fluid.val[fluid]
                    for i in self.inl:
                        if not i.fluid.val_set[fluid]:
                            i.fluid.val[fluid] = c.fluid.val[fluid]

    def initialise_source(self, c, key):
        r"""
        Returns a starting value for pressure and enthalpy at component's
        outlet.

        Parameters
        ----------
        c : tespy.connections.connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            return 5e5

    def initialise_target(self, c, key):
        r"""
        Returns a starting value for pressure and enthalpy at component's
        inlet.

        Parameters
        ----------
        c : tespy.connections.connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            return 5e5

# %%


class splitter(node):
    r"""
    Equations

        **mandatory equations**

        - :func:`tespy.components.components.component.mass_flow_func`

        .. math::

            0 = p_{in} - p_{out,i} \;
            \forall i \in \mathrm{outlets}

        **additional equations**

        - :func:`tespy.components.components.splitter.additional_equations`

    Inlets/Outlets

        - in1
        - specify number of outlets with :code:`num_out` (default value: 2)

    Image

        .. image:: _images/split.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    num_out : float/tespy.helpers.dc_simple
        Number of outlets for this component, default value: 2.

    Example
    -------
    >>> from tespy import cmp, con, nwk, hlp
    >>> import numpy as np
    >>> fluid_list = ['O2', 'N2']
    >>> nw = nwk.network(fluids=fluid_list, p_unit='bar', T_unit='C',
    ...     h_unit='kJ / kg')
    >>> nw.set_printoptions(print_level='none')
    >>> so1 = cmp.source('source1')
    >>> si1 = cmp.sink('sink1')
    >>> si2 = cmp.sink('sink2')
    >>> si3 = cmp.sink('sink3')
    >>> s = cmp.splitter('splitter', num_out=3)
    >>> s.component()
    'splitter'
    >>> inc1 = con.connection(so1, 'out1', s, 'in1')
    >>> outg1 = con.connection(s, 'out1', si1, 'in1')
    >>> outg2 = con.connection(s, 'out2', si2, 'in1')
    >>> outg3 = con.connection(s, 'out3', si3, 'in1')
    >>> nw.add_conns(inc1, outg1, outg2, outg3)
    >>> inc1.set_attr(fluid={'O2': 0.23, 'N2': 0.77}, p=1, T=20, m=5)
    >>> outg1.set_attr(m=3)
    >>> outg2.set_attr(m=1)
    >>> nw.solve('design')
    >>> nw.lin_dep
    False
    >>> nw.res[-1] < 1e-3
    True
    """

    def component(self):
        return 'splitter'

    def attr(self):
        return {'num_out': dc_simple()}

    def inlets(self):
        return ['in1']

    def outlets(self):
        if self.num_out.val_set:
            return ['out' + str(i + 1) for i in range(self.num_out.val)]
        else:
            self.set_attr(num_out=2)
            return self.outlets()

    def comp_init(self, nw):

        node.comp_init(self, nw)

        self.fl_deriv = self.fluid_deriv()
        self.h_deriv = self.enthalpy_deriv()

    def additional_equations(self):
        r"""
        Calculates vector vec_res with results of additional equations for
        this component.

        Equations

            **mandatroy equations**

            .. math:: 0 = fluid_{i,in} - fluid_{i,out_{j}} \;
                \forall i \in \mathrm{fluid}, \; \forall j \in outlets

            .. math::
                0 = h_{in} - h_{out,i} \;
                \forall i \in \mathrm{outlets}\\

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # equations for fluid balance
        for o in self.outl:
            for fluid, x in self.inl[0].fluid.val.items():
                vec_res += [x - o.fluid.val[fluid]]

        ######################################################################
        # equations for energy balance
        for o in self.outl:
            vec_res += [self.inl[0].h.val_SI - o.h.val_SI]

        return vec_res

    def additional_derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given additional
        equations.

        Returns
        -------
        mat_deriv : list
            Matrix of partial derivatives.
        """
        ######################################################################
        # derivatives for fluid and energy balance equations are constant
        return self.fl_deriv + self.h_deriv

    def fluid_deriv(self):
        r"""
        Calculates the partial derivatives for all fluid balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((self.num_fl * self.num_o, 1 + self.num_o,
                          3 + self.num_fl))
        k = 0
        for o in self.outl:
            i = 0
            for fluid in self.fluids:
                deriv[i + k * self.num_fl, 0, i + 3] = 1
                deriv[i + k * self.num_fl, k + 1, i + 3] = -1
                i += 1
            k += 1
        return deriv.tolist()

    def enthalpy_deriv(self):
        r"""
        Calculates matrix of partial derivatives for enthalpy balance equation.

        Returns
        -------
        deriv : list
            Matrix of partial derivatives.
        """
        deriv = np.zeros((self.num_o, 1 + self.num_o, self.num_fl + 3))
        k = 0
        for o in self.outl:
            deriv[k, 0, 2] = 1
            deriv[k, k + 1, 2] = -1
            k += 1

        return deriv.tolist()

    def initialise_fluids(self, nw):
        r"""
        Fluid initialisation for fluid mixture at outlet of the node.

        Parameters
        ----------
        nw : tespy.networks.network
            Network using this component object.
        """
        return

# %%


class separator(node):
    r"""
    Equations

        **mandatory equations**

        - :func:`tespy.components.components.component.mass_flow_func`

        .. math::

            0 = p_{in} - p_{out,i} \;
            \forall i \in \mathrm{outlets}

        **additional equations**

        - :func:`tespy.components.components.separator.additional_equations`

    Inlets/Outlets

        - in1
        - specify number of outlets with :code:`num_out` (default value: 2)

    Image

        .. image:: _images/split.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    TODO

        - fluid separation requires power and cooling, equations have not
          been implemented!

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    num_out : float/tespy.helpers.dc_simple
        Number of outlets for this component, default value: 2.

    Example
    -------
    >>> from tespy import cmp, con, nwk, hlp
    >>> import numpy as np
    >>> fluid_list = ['O2', 'N2']
    >>> nw = nwk.network(fluids=fluid_list, p_unit='bar', T_unit='C',
    ...     h_unit='kJ / kg')
    >>> nw.set_printoptions(print_level='none')
    >>> so1 = cmp.source('source1')
    >>> si1 = cmp.sink('sink1')
    >>> si2 = cmp.sink('sink2')
    >>> s = cmp.separator('separator', num_out=2)
    >>> s.component()
    'separator'
    >>> inc1 = con.connection(so1, 'out1', s, 'in1')
    >>> outg1 = con.connection(s, 'out1', si1, 'in1')
    >>> outg2 = con.connection(s, 'out2', si2, 'in1')
    >>> nw.add_conns(inc1, outg1, outg2)
    >>> inc1.set_attr(fluid={'O2': 0.23, 'N2': 0.77}, p=1, T=20, m=5)
    >>> outg1.set_attr(fluid={'O2': 0.1, 'N2': 0.9}, m=1)
    >>> outg2.set_attr(fluid0={'O2': 0.5, 'N2': 0.5}, m0=4)
    >>> nw.solve('design')
    >>> nw.lin_dep
    False
    >>> nw.res[-1] < 1e-3
    True
    """

    def component(self):
        return 'separator'

    def attr(self):
        return {'num_out': dc_simple()}

    def inlets(self):
        return ['in1']

    def outlets(self):
        if self.num_out.val_set:
            return ['out' + str(i + 1) for i in range(self.num_out.val)]
        else:
            self.set_attr(num_out=2)
            return self.outlets()

    def additional_equations(self):
        r"""
        Calculates vector vec_res with results of additional equations for
        this component.

        Equations

            **mandatroy equations**

            .. math:: 0 = fluid_{i,in} - fluid_{i,out_{j}} \;
                \forall i \in \mathrm{fluid}, \; \forall j \in outlets

            .. math::

                0 = T_{in} - T_{out,i} \;
                \forall i \in \mathrm{outlets}

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # equations for fluid balance
        for fluid, x in self.inl[0].fluid.val.items():
            res = x * self.inl[0].m.val_SI
            for o in self.outl:
                res -= o.fluid.val[fluid] * o.m.val_SI
            vec_res += [res]

        ######################################################################
        # equations for energy balance
        for o in self.outl:
            vec_res += [
                    T_mix_ph(self.inl[0].to_flow(), T0=self.inl[0].T.val_SI) -
                    T_mix_ph(o.to_flow(), T0=o.T.val_SI)]

        return vec_res

    def additional_derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given additional
        equations.

        Returns
        -------
        mat_deriv : list
            Matrix of partial derivatives.
        """
        mat_deriv = []

        ######################################################################
        # derivatives for fluid balance equations
        mat_deriv += self.fluid_deriv()

        ######################################################################
        # derivatives for energy balance equations
        deriv = np.zeros((self.num_o, 1 + self.num_o, self.num_fl + 3))
        i = self.inl[0].to_flow()
        k = 0
        for o in self.outl:
            o = o.to_flow()
            deriv[k, 0, 1] = dT_mix_dph(i)
            deriv[k, 0, 2] = dT_mix_pdh(i)
            deriv[k, 0, 3:] = dT_mix_ph_dfluid(i)
            deriv[k, k + 1, 1] = -dT_mix_dph(o)
            deriv[k, k + 1, 2] = -dT_mix_pdh(o)
            deriv[k, k + 1, 3:] = -1 * dT_mix_ph_dfluid(o)
            k += 1
        mat_deriv += deriv.tolist()

        return mat_deriv

    def fluid_deriv(self):
        r"""
        Calculates the partial derivatives for all fluid balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((self.num_fl, 1 + self.num_o, 3 + self.num_fl))
        j = 0
        for fluid in self.fluids:
            k = 0
            for o in self.outl:
                deriv[j, k + 1, 0] = -o.fluid.val[fluid]
                deriv[j, k + 1, j + 3] = -o.m.val_SI
                k += 1
            deriv[j, 0, 0] = self.inl[0].fluid.val[fluid]
            deriv[j, 0, j + 3] = self.inl[0].m.val_SI
            j += 1
        return deriv.tolist()

    def initialise_fluids(self, nw):
        r"""
        Fluid initialisation for fluid mixture at outlet of the node.

        Parameters
        ----------
        nw : tespy.networks.network
            Network using this component object.
        """
        return

# %%


class merge(node):
    r"""
    The component node is the parent class for splitter, separator and merge.

    Equations

        **mandatory equations**

        - :func:`tespy.components.components.component.mass_flow_func`

        .. math::

            0 = p_{in} - p_{out,i} \;
            \forall i \in \mathrm{outlets}

        **additional equations**

        - :func:`tespy.components.components.merge.additional_equations`

    Inlets/Outlets

        - specify number of outlets with :code:`num_in` (default value: 2)
        - out1

    Image

        .. image:: _images/merge.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    num_in : float/tespy.helpers.dc_simple
        Number of inlets for this component, default value: 2.

    Example
    -------
    >>> from tespy import cmp, con, nwk, hlp
    >>> fluid_list = ['O2', 'N2']
    >>> nw = nwk.network(fluids=fluid_list, p_unit='bar', T_unit='C',
    ...     h_unit='kJ / kg')
    >>> nw.set_printoptions(print_level='none')
    >>> so1 = cmp.source('source1')
    >>> so2 = cmp.source('source2')
    >>> so3 = cmp.source('source3')
    >>> si1 = cmp.sink('sink1')
    >>> m = cmp.merge('merge', num_in=3)
    >>> m.component()
    'merge'
    >>> inc1 = con.connection(so1, 'out1', m, 'in1')
    >>> inc2 = con.connection(so2, 'out1', m, 'in2')
    >>> inc3 = con.connection(so3, 'out1', m, 'in3')
    >>> outg1 = con.connection(m, 'out1', si1, 'in1')
    >>> nw.add_conns(inc1, inc2, inc3, outg1)
    >>> inc1.set_attr(fluid={'O2': 0.23, 'N2': 0.77}, p=1, T=20, m=5)
    >>> inc2.set_attr(fluid={'O2': 1, 'N2':0}, T=20, m=5)
    >>> inc3.set_attr(fluid={'O2': 0, 'N2': 1}, T=20)
    >>> outg1.set_attr(fluid={'N2': 0.4})
    >>> nw.solve('design')
    >>> round(inc3.m.val_SI, 2)
    0.25
    >>> round(outg1.fluid.val['O2'], 1)
    0.6
    """

    def component(self):
        return 'merge'

    def attr(self):
        return {'num_in': dc_simple(),
                'zero_flag': dc_simple()}

    def inlets(self):
        if self.num_in.val_set:
            return ['in' + str(i + 1) for i in range(self.num_in.val)]
        else:
            self.set_attr(num_in=2)
            return self.inlets()

    def outlets(self):
        return ['out1']

    def additional_equations(self):
        r"""
        Calculates vector vec_res with results of additional equations for
        this component.

        Equations

            **mandatroy equations**

            .. math::

                0 = \dot{m}_{in_{j}} \cdot fluid_{i,in_{j}} -
                    \dot {m}_{out} \cdot fluid_{i,out} \\
                \forall i \in \mathrm{fluid}, \; \forall j \in inlets

            .. math::

                0 = h_{in} - h_{out,i} \;
                \forall i \in \mathrm{outlets}\\

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # equations for fluid balance
        for fluid, x in self.outl[0].fluid.val.items():
            res = -x * self.outl[0].m.val_SI
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            vec_res += [res]

        ######################################################################
        # equation for energy balance
        h_res = -self.outl[0].m.val_SI * self.outl[0].h.val_SI
        for i in self.inl:
            h_res += i.m.val_SI * i.h.val_SI
        vec_res += [h_res]

        return vec_res

    def additional_derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given additional
        equations.

        Returns
        -------
        mat_deriv : list
            Matrix of partial derivatives.
        """
        mat_deriv = []

        ######################################################################
        # derivatives for fluid balance equations
        mat_deriv += self.fluid_deriv()

        ######################################################################
        # derivatives for energy balance equations
        deriv = np.zeros((1, self.num_i + 1, self.num_fl + 3))
        deriv[0, self.num_i, 0] = -self.outl[0].h.val_SI
        deriv[0, self.num_i, 2] = -self.outl[0].m.val_SI
        k = 0
        for i in self.inl:
            deriv[0, k, 0] = i.h.val_SI
            deriv[0, k, 2] = i.m.val_SI
            k += 1
        mat_deriv += deriv.tolist()

        return mat_deriv

    def fluid_deriv(self):
        r"""
        Calculates the partial derivatives for all fluid balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((self.num_fl, self.num_i + 1, 3 + self.num_fl))
        j = 0
        for fluid, x in self.outl[0].fluid.val.items():
            k = 0
            for i in self.inl:
                deriv[j, k, 0] = i.fluid.val[fluid]
                deriv[j, k, j + 3] = i.m.val_SI
                k += 1
            deriv[j, k, 0] = -x
            deriv[j, k, j + 3] = -self.outl[0].m.val_SI
            j += 1
        return deriv.tolist()

# %%


class drum(component):
    r"""
    Equations

        **mandatory equations**

        - :func:`tespy.components.components.drum.fluid_func`
        - :func:`tespy.components.components.component.mass_flow_func`

        .. math::

            0 = \sum_i \left(\dot{m}_{i,in} \cdot h_{i,in} \right) -
            \sum_j \left(\dot{m}_{j,out} \cdot h_{j,out} \right)\\
            \forall i \in inlets, \; \forall j \in outlet

            0 = p_{in,1} - p_{out,i}\\
            \forall i \in \mathrm{outlets}

            0 = h_{1,out} - h\left(p, x=0 \right)

            0 = h_{2,out} - h\left(p, x=1 \right)\\
            x: \text{vapour mass fraction}

    Inlets/Outlets

        - in1, in2 (index 1: from economiser, index 2: from evaporator)
        - out1, out2 (index 1: to evaporator, index 2: to superheater)

    Image

        .. image:: _images/drum.svg
           :scale: 100 %
           :alt: alternative text
           :align: center

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    Note
    ----
    If you are using a drum in a network with multiple fluids, it is likely
    the fluid propagation causes trouble. If this is the case, try to
    specify the fluid composition at another connection of your network.

    This component assumes, that the fluid composition between outlet 1 and
    inlet 2 does not change, thus there is no equation for the fluid mass
    fraction at the inlet 2!

    Example
    -------
    >>> from tespy import cmp, con, nwk
    >>> import shutil
    >>> nw = nwk.network(fluids=['NH3', 'air'], T_unit='C', p_unit='bar',
    ...     h_unit='kJ / kg')
    >>> nw.set_printoptions(print_level='none')
    >>> f = cmp.source('feed')
    >>> ha = cmp.source('hot air')
    >>> ch = cmp.sink('chimney')
    >>> s = cmp.sink('steam')
    >>> dr = cmp.drum('drum')
    >>> dr.component()
    'drum'
    >>> ev = cmp.heat_exchanger('evaporator')
    >>> erp = cmp.pump('evaporator reciculation pump')
    >>> f_dr = con.connection(f, 'out1', dr, 'in1')
    >>> dr_erp = con.connection(dr, 'out1', erp, 'in1')
    >>> erp_ev = con.connection(erp, 'out1', ev, 'in2')
    >>> ev_dr = con.connection(ev, 'out2', dr, 'in2')
    >>> dr_s = con.connection(dr, 'out2', s, 'in1')
    >>> nw.add_conns(f_dr, dr_erp, erp_ev, ev_dr, dr_s)
    >>> ha_ev = con.connection(ha, 'out1', ev, 'in1')
    >>> ev_ch = con.connection(ev, 'out1', ch, 'in1')
    >>> nw.add_conns(ha_ev, ev_ch)
    >>> ev.set_attr(pr1=0.999, pr2=0.99, ttd_l=20, kA_char1='EVA_HOT',
    ...     kA_char2='EVA_COLD', design=['pr1', 'ttd_l'],
    ...     offdesign=['zeta1', 'kA'])
    >>> ev.set_attr(Q=-1e6)
    >>> erp.set_attr(eta_s=0.8)
    >>> f_dr.set_attr(p=5, T=-5)
    >>> erp_ev.set_attr(m=con.ref(f_dr, 4, 0), fluid={'air': 0, 'NH3': 1})
    >>> ha_ev.set_attr(fluid={'air': 1, 'NH3': 0}, T=100)
    >>> ev_ch.set_attr(p=1)
    >>> nw.solve('design')
    >>> nw.save('tmp')
    >>> round(ev.ttd_l.val, 1)
    20.0
    >>> round(f_dr.h.val, 1)
    320.2
    >>> round(dr_erp.h.val, 1)
    362.4
    >>> round(ev_dr.h.val, 1)
    684.7
    >>> round(f_dr.m.val, 2)
    0.78
    >>> ev.set_attr(Q=-0.75e6)
    >>> nw.solve('offdesign', init_path='tmp', design_path='tmp')
    >>> round(f_dr.m.val, 2)
    0.58
    >>> round(ev.ttd_l.val, 1)
    16.1
    >>> shutil.rmtree('./tmp', ignore_errors=True)
    """

    def component(self):
        return 'drum'

    def inlets(self):
        return ['in1', 'in2']

    def outlets(self):
        return ['out1', 'out2']

    def comp_init(self, nw):

        component.comp_init(self, nw)

        self.fl_deriv = self.fluid_deriv()
        self.m_deriv = self.mass_flow_deriv()
        self.p_deriv = self.pressure_deriv()

    def equations(self):
        r"""
        Calculates vector vec_res with results of equations for this component.

        Returns
        -------
        vec_res : list
            Vector of residual values.
        """
        vec_res = []

        ######################################################################
        # eqations for fluid balance
        vec_res += self.fluid_func()

        ######################################################################
        # eqations for mass flow balance
        vec_res += self.mass_flow_func()

        ######################################################################
        # eqations for pressure
        p = self.inl[0].p.val_SI
        for c in [self.inl[1]] + self.outl:
            vec_res += [p - c.p.val_SI]

        ######################################################################
        # eqations for enthalpy
        val = 0
        for i in self.inl:
            val += i.m.val_SI * i.h.val_SI
        for o in self.outl:
            val -= o.m.val_SI * o.h.val_SI
        vec_res += [val]

        ######################################################################
        # eqations for staturated fluid state at outlets
        vec_res += [h_mix_pQ(self.outl[0].to_flow(), 0) -
                    self.outl[0].h.val_SI]
        vec_res += [h_mix_pQ(self.outl[1].to_flow(), 1) -
                    self.outl[1].h.val_SI]

        return vec_res

    def derivatives(self):
        r"""
        Calculates matrix of partial derivatives for given equations.

        Returns
        -------
        mat_deriv : ndarray
            Matrix of partial derivatives.
        """
        mat_deriv = []

        ######################################################################
        # derivatives for fluid balance equations
        mat_deriv += self.fl_deriv

        ######################################################################
        # derivatives for mass flow balance equation
        mat_deriv += self.m_deriv

        ######################################################################
        # derivatives for pressure eqauations
        mat_deriv += self.p_deriv

        ######################################################################
        # derivatives for energy balance equation
        deriv = np.zeros((1, 4, self.num_fl + 3))
        k = 0
        for i in self.inl:
            deriv[0, k, 0] = i.h.val_SI
            deriv[0, k, 2] = i.m.val_SI
            k += 1
        j = 0
        for o in self.outl:
            deriv[0, j + k, 0] = -o.h.val_SI
            deriv[0, j + k, 2] = -o.m.val_SI
            j += 1
        mat_deriv += deriv.tolist()

        ######################################################################
        # derivatives of equations for saturated states at outlets
        x_deriv = np.zeros((2, 4, self.num_fl + 3))
        x_deriv[0, 2, 1] = dh_mix_dpQ(self.outl[0].to_flow(), 0)
        x_deriv[0, 2, 2] = -1
        x_deriv[1, 3, 1] = dh_mix_dpQ(self.outl[1].to_flow(), 1)
        x_deriv[1, 3, 2] = -1
        mat_deriv += x_deriv.tolist()

        return np.asarray(mat_deriv)

    def fluid_func(self):
        r"""
        Calculates the vector of residual values for component's fluid balance
        equations.

        Returns
        -------
        vec_res : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = fluid_{i,in_1} - fluid_{i,out_{j}}\\
                \forall i \in \mathrm{fluid}, \; \forall j \in inlets

        """
        vec_res = []

        for o in self.outl:
            for fluid, x in self.inl[0].fluid.val.items():
                vec_res += [x - o.fluid.val[fluid]]
        return vec_res

    def fluid_deriv(self):
        r"""
        Calculates the partial derivatives for all fluid balance equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((2 * self.num_fl, 4, 3 + self.num_fl))
        for k in range(2):
            for i in range(self.num_fl):
                deriv[i + k * self.num_fl, 0, i + 3] = 1
                deriv[i + k * self.num_fl, k + 2, i + 3] = -1
        return deriv.tolist()

    def pressure_deriv(self):
        r"""
        Calculates the partial derivatives for pressure equations.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        deriv = np.zeros((3, 4, self.num_fl + 3))
        for k in range(3):
            deriv[k, 0, 1] = 1
            deriv[k, k + 1, 1] = -1
        return deriv.tolist()

    def initialise_source(self, c, key):
        r"""
        Returns a starting value for pressure and enthalpy at component's
        outlet.

        Parameters
        ----------
        c : tespy.connections.connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                10^6 & \text{key = 'p'}\\
                h\left(p, x=0 \right) & \text{key = 'h' at outlet 1}\\
                h\left(p, x=1 \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':
            return 10e5
        elif key == 'h':
            if c.s_id == 'out1':
                return h_mix_pQ(c.to_flow(), 0)
            else:
                return h_mix_pQ(c.to_flow(), 1)

    def initialise_target(self, c, key):
        r"""
        Returns a starting value for pressure and enthalpy at component's
        inlet.

        Parameters
        ----------
        c : tespy.connections.connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                10^6 & \text{key = 'p'}\\
                h\left(p, x=0 \right) & \text{key = 'h' at inlet 1}\\
                h\left(p, x=0.7 \right) & \text{key = 'h' at inlet 2}
                \end{cases}
        """
        if key == 'p':
            return 10e5
        elif key == 'h':
            if c.t_id == 'in1':
                return h_mix_pQ(c.to_flow(), 0)
            else:
                return h_mix_pQ(c.to_flow(), 0.7)
