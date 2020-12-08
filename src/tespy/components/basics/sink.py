# -*- coding: utf-8

"""Module for class Sink.


This file is part of project TESPy (github.com/oemof/tespy). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location tespy/components/basics/sink.py

SPDX-License-Identifier: MIT
"""

import numpy as np

from tespy.components.component import Component


class Sink(Component):
    r"""
    A flow drains in a Sink.

    Equations

    This component is unconstrained.

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    design_path : str
        Path to the components design case.

    local_offdesign : boolean
        Treat this component in offdesign mode in a design calculation.

    local_design : boolean
        Treat this component in design mode in an offdesign calculation.

    char_warnings : boolean
        Ignore warnings on default characteristics usage for this component.

    printout : boolean
        Include this component in the network's results printout.

    Example
    -------
    Create a sink and specify a label.

    >>> from tespy.components import Sink
    >>> si = Sink('a labeled sink')
    >>> si.component()
    'sink'
    >>> si.label
    'a labeled sink'
    """

    @staticmethod
    def component():
        return 'sink'

    @staticmethod
    def inlets():
        return ['in1']

    def propagate_fluid_to_target(self, inconn, start):
        r"""
        Fluid propagation to target stops here.

        Parameters
        ----------
        inconn : tespy.connections.connection.Connection
            Connection to initialise.

        start : tespy.components.component.Component
            This component is the fluid propagation starting point.
            The starting component is saved to prevent infinite looping.
        """
        return

    def exergy_balance(self, Tamb):
        r"""Exergy balance calculation method of a sink.

        A sink does not destroy or produce exergy. The value of
        :math:`\dot{E}_\mathrm{F}` and :math:`\dot{E}_\mathrm{P}` are set to
        the exergy of the mass flow to make exergy balancing methods more
        simple as in general a mass flow can be fuel, product or loss.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{F} = \dot{m}_\mathrm{out} \cdot e_\mathrm{ph,out}

            \dot{E}_\mathrm{P} = \dot{E}_\mathrm{F}
        """
        self.E_P = self.inl[0].Ex_physical
        self.E_F = self.inl[0].Ex_physical
        self.E_D = np.nan
        self.epsilon = np.nan
