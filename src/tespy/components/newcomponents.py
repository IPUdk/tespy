import logging

from tespy.components import SimpleHeatExchanger, Merge, Separator, Splitter, HeatExchanger
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import SimpleDataContainer as dc_simple
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp
from tespy.tools.fluid_properties import T_mix_ph, h_mix_pT
from tespy.tools.helpers import TESPyComponentError

from tespy.components.component import Component

from tespy.tools.fluid_properties import dT_mix_dph
from tespy.tools.fluid_properties import dT_mix_pdh

from CoolProp.HumidAirProp import HAPropsSI

import warnings

import numpy as np


def get_Twb(port,T):
    M = port.fluid.val["Water"]/(port.fluid.val["Water"]+port.fluid.val["Air"])
    W = M/(1-M)
    return HAPropsSI('Twb','P',port.p.val_SI,'T',T,'W',W)

class DiabaticSimpleHeatExchanger(SimpleHeatExchanger):

    @staticmethod
    def component():
        return 'diabatic simple heat exchanger'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["eta"] = dc_cp(min_val=1e-5, val=1, max_val=1)
        variables["Q_loss"] = dc_cp(max_val=0, val=0, is_result=True)
        variables["Q_total"] = dc_cp(is_result=True)
        variables["energy_group"] = dc_gcp(
            elements=['Q_total', 'eta'],
            num_eq=1,
            latex=self.energy_balance_func_doc,
            func=self.energy_balance2_func, deriv=self.energy_balance2_deriv
        )

        return variables

    def energy_balance2_func(self):
        r"""
        Equation for pressure drop calculation.

        Returns
        -------
        residual : float
            Residual value of equation:

            .. math::

                0 =\dot{m}_{in}\cdot\left( h_{out}-h_{in}\right) -\dot{Q}
        """
        if self.Q_total.val < 0:
            return self.inl[0].m.val_SI * (
                self.outl[0].h.val_SI - self.inl[0].h.val_SI
            ) * self.eta.val - self.Q_total.val
        else:
            return self.inl[0].m.val_SI * (
                self.outl[0].h.val_SI - self.inl[0].h.val_SI
            ) - self.Q_total.val * self.eta.val

    def energy_balance2_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """


        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col] = (o.h.val_SI - i.h.val_SI)
        if i.h.is_var:
            self.jacobian[k, i.h.J_col] = -i.m.val_SI
        if o.h.is_var:
            self.jacobian[k, o.h.J_col] = i.m.val_SI
        # custom variable Q
        if self.Q_total.is_var:
            if self.Q_total.val < 0:
                self.jacobian[k, self.Q_total.J_col] = -1
            else:
                self.jacobian[k, self.Q_total.J_col] = -self.eta.val

        if self.eta.is_var:
            if self.Q_total.val < 0:
                self.jacobian[k, self.eta.J_col] = self.inl[0].m.val_SI * (
                self.outl[0].h.val_SI - self.inl[0].h.val_SI
                )
            else:
                self.jacobian[k, self.eta.J_col] = -self.Q_total.val

    def calc_parameters(self):
        super().calc_parameters()

        if self.eta.is_set:
            if self.Q.val < 0:
                self.Q_loss.val = self.Q.val * (1 - self.eta.val)
            else:
                self.Q_loss.val = -self.Q.val * (1 / self.eta.val - 1)

            self.Q_total.val = self.Q.val - self.Q_loss.val


class SimpleHeatExchangerDeltaP(SimpleHeatExchanger):

    @staticmethod
    def component():
        return 'simple heat exchanger with pressure drop'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaP"] = dc_cp(
            min_val=0,
            deriv=self.deltaP_deriv,
            func=self.deltaP_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        return variables

    def deltaP_func(self):
        r"""
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """

        return self.inl[0].p.val_SI - self.deltaP.val*1e5 - self.outl[0].p.val_SI

    def deltaP_deriv(self, increment_filter, k, pr='', inconn=0, outconn=0):
        r"""
        Calculate the partial derivatives for pressure drop.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.

        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.
        """
        
        deltaP = self.get_attr("deltaP")
        i = self.inl[inconn]
        o = self.outl[inconn]
        if i.p.is_var:
            self.jacobian[k, i.p.J_col] = 1
        if o.p.is_var:
            self.jacobian[k, o.p.J_col] = -1
        if deltaP.is_var:
            self.jacobian[k, self.pr.J_col] = 1


    def calc_parameters(self):
        super().calc_parameters()
        self.deltaP.val = (self.inl[0].p.val_SI - self.outl[0].p.val_SI)/1e5


class SimpleHeatExchangerDeltaPLossFactor(SimpleHeatExchangerDeltaP):

    @staticmethod
    def component():
        return 'diabatic simple heat exchanger'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["LF"] = dc_cp(min_val=0, val=0, max_val=1, is_result=True)
        variables["Q_loss"] = dc_cp(is_result=True)
        variables["Q_total"] = dc_cp(is_result=True)       
        variables["energy_group1"] = dc_gcp(
                elements=['LF', 'Q_total'],
                func=self.Q_total_func,
                deriv=self.Q_total_deriv,
                latex=self.energy_balance_func_doc, num_eq=1)
        variables["energy_group2"] = dc_gcp(
                elements=['Q_loss', 'Q_total'],
                func=self.Q_total_func,
                deriv=self.Q_total_deriv,
                latex=self.energy_balance_func_doc, num_eq=1)
        variables["energy_group3"] = dc_gcp(
                elements=['Q_loss', 'LF'],
                func=self.Q_total_func,
                deriv=self.Q_total_deriv,
                latex=self.energy_balance_func_doc, num_eq=1)                
        return variables

    def Q_total_func(self):
        r"""
        Equation for total heat flow rate

        """
        # self.Q_loss.val is negative and Q_total is positive (and vice versa)

        if self.energy_group2.is_set:
            self.LF.val = -self.Q_loss.val/(self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI))
        if self.energy_group3.is_set:
            self.Q_total.val = -self.Q_loss.val*(1+self.LF.val)/self.LF.val 

        return self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)*(1+self.LF.val) - self.Q_total.val
      

    def Q_total_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of Q_total

        """

        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col] = (o.h.val_SI - i.h.val_SI)*(1+self.LF.val)
        if i.h.is_var:
            self.jacobian[k, i.h.J_col] = -i.m.val_SI*(1+self.LF.val)
        if o.h.is_var:
            self.jacobian[k, o.h.J_col] = i.m.val_SI*(1+self.LF.val)
        if self.Q_total.is_var:
            self.jacobian[k, self.Q_total.J_col] = -1
        if self.LF.is_var: 
            self.jacobian[k, self.LF.J_col] = self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
        if self.Q_loss.is_var:
            self.jacobian[k, self.Q_loss.J_col] = -(1+self.LF.val)/self.LF.val

    def calc_parameters(self):
        super().calc_parameters()

        # repeat calculations to ensure variables are assigned
        if self.Q_total.is_set:
            self.Q_loss.val = self.Q.val-self.Q_total.val
            self.LF.val = -self.Q_loss.val / self.Q.val
        elif self.LF.is_set:
            self.Q_total.val = self.Q.val * (1+self.LF.val)
            self.Q_loss.val = self.Q.val-self.Q_total.val
        else:
            self.Q_total.val = self.Q.val-self.Q_loss.val
            self.LF.val = -self.Q_loss.val/self.Q.val        


class SimpleHeatExchangerDeltaPLfKpi(SimpleHeatExchangerDeltaP):

    @staticmethod
    def component():
        return 'simple heat exchanger with loss factor and KPI'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["LF"] = dc_cp(min_val=0, val=0, max_val=1, is_result=True)
        variables["Q_loss"] = dc_cp(is_result=True)
        variables["KPI"] = dc_cp(
            deriv=self.KPI_deriv,
            func=self.KPI_func,
            latex=self.pr_func_doc,
            num_eq=1)
        return variables

    def energy_balance_func(self):
        r"""
        Equation for total heat flow rate

        """

        return self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)*(1+self.LF.val) - self.Q.val

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of Q_total

        """

        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col] = (o.h.val_SI - i.h.val_SI)*(1+self.LF.val) 
        if i.h.is_var:
            self.jacobian[k, i.h.J_col] = -i.m.val_SI*(1+self.LF.val) 
        if o.h.is_var:
            self.jacobian[k, o.h.J_col] = i.m.val_SI*(1+self.LF.val) 
        if self.Q.is_var:
            self.jacobian[k, self.Q.J_col] = -1
        if self.LF.is_var: 
            self.jacobian[k, self.LF.J_col] = self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)

    def KPI_func(self):
        r"""
        Equation for total heat flow rate

        """
        return self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)*(1+self.LF.val) - self.KPI.val * self.inl[0].m.val_SI 

    def KPI_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of Q_total

        """
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col] = (o.h.val_SI - i.h.val_SI)*(1+self.LF.val) - self.KPI.val
        if i.h.is_var:
            self.jacobian[k, i.h.J_col] = -i.m.val_SI*(1+self.LF.val) 
        if o.h.is_var:
            self.jacobian[k, o.h.J_col] = i.m.val_SI*(1+self.LF.val) 
        if self.LF.is_var: 
            self.jacobian[k, self.LF.J_col] = self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
        if self.KPI.is_var:
            self.jacobian[k, self.Q_loss.J_col] = -self.inl[0].m.val_SI 

    def calc_parameters(self):
        super().calc_parameters()
        self.Q.val = self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)*(1+self.LF.val)
        # repeat calculations to ensure variables are assigned
        if self.KPI.is_set:
            self.Q.val = self.KPI.val * self.inl[0].m.val_SI 
        else:
            self.KPI.val = self.Q.val / self.inl[0].m.val_SI 
        self.Q_loss.val = - self.LF.val * self.Q.val


class TwoStreamHeatExchanger(HeatExchanger):

    @staticmethod
    def component():
        return 'two stream heat exchanger with min ttd (pinch)'

    def get_parameters(self):
        variables = super().get_parameters()
        variables['ttd_min'] = dc_cp(
                min_val=0, num_eq=1, func=self.ttd_min_func,
                deriv=self.ttd_min_deriv, latex=self.ttd_u_func_doc)
        return variables

    def _calc_dTs(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]

        T_i1 = i1.calc_T(T0=i1.T.val_SI)
        T_o1 = o1.calc_T(T0=o1.T.val_SI)
        T_i2 = i2.calc_T(T0=i2.T.val_SI)
        T_o2 = o2.calc_T(T0=o2.T.val_SI)

        if T_i1 > T_i2:
            dTa = T_i1-T_o2
            dTb = T_o1-T_i2
        else:
            dTa = -T_i1+T_o2
            dTb = -T_o1+T_i2

        return dTa,dTb

    def ttd_min_func(self):
        r"""
        Equation for minimum terminal temperature difference.
        """

        dTa,dTb = self._calc_dTs()

        if dTa < dTb:
            return self.ttd_min.val - dTa
        else:
            return self.ttd_min.val - dTb

        # T_o2 = o.calc_T(T0=o.T.val_SI)
        # return self.ttd_u.val - T_i1 + T_o2


    def ttd_min_deriv(self, increment_filter, k):
        """
        Calculate partial derivates for minimum terminal temperature difference..

        """
        f = self.ttd_min_func
        for c in [self.inl[0], self.inl[1], self.outl[0], self.outl[1]]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)

    def calc_parameters(self):
        super().calc_parameters()
        if not self.ttd_min.is_set:
            self.ttd_min.val = min(self._calc_dTs())




class MergeDeltaP(Merge):

    @staticmethod
    def component():
        return 'merge with pressure losses'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaP"] = dc_cp(
            min_val=0,
            deriv=self.deltaP_deriv,
            func=self.deltaP_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        del constraints['pressure_constraints']
        return constraints

    def deltaP_func(self):
        r"""
        Equation for pressure drop.

        """
        p_in_min = min([i.p.val_SI for i in self.inl])
        return p_in_min - self.deltaP.val*1e5 - self.outl[0].p.val_SI

    def deltaP_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for pressure drop.

        """
        p_in = [i.p.val_SI for i in self.inl]
        p_min_index = p_in.index(min(p_in))

        if self.inl[p_min_index].p.is_var:
            self.jacobian[k, self.inl[p_min_index].p.J_col] = 1 #self.pr.val
        if self.outl[0].p.is_var:
            self.jacobian[k, self.outl[0].p.J_col] = -1

    def calc_parameters(self):
        super().calc_parameters()
        Pmin = min([i.p.val_SI for i in self.inl])
        Pmax = max([i.p.val_SI for i in self.inl])
        if abs(self.outl[0].p.val_SI - Pmin) >= abs(self.outl[0].p.val_SI - Pmax):
            self.deltaP.val = (Pmin - self.outl[0].p.val_SI)/1e5
        else:
            self.deltaP.val = (Pmax - self.outl[0].p.val_SI)/1e5


class SeparatorWithSpeciesSplits(Separator):

    def __init__(self, label, **kwargs):
        #self.set_attr(**kwargs)
        # need to assign the number of outlets before the variables are set
        self.num_out = 2 # default
        for key in kwargs:
            if key == 'num_out':
                self.num_out=kwargs[key]
        super().__init__(label, **kwargs)


    @staticmethod
    def component():
        return 'separator with species flow splits'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["SFS"] = dc_cp_SFS(
            min_val=0,
            deriv=self.SFS_deriv,
            func=self.SFS_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        variables["SF"] = dc_cp_SFS(
            min_val=0,
            deriv=self.SF_deriv,
            func=self.SF_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )        
        return variables
    
    def SF_func(self):
        r"""
        Equation for SF.

        """

        fluid = self.SF.split_fluid
        out_i = int(self.SF.split_outlet[3:]) - 1
        i = self.inl[0]
        o = self.outl[out_i]

        res = self.SF.val - o.fluid.val[fluid] * o.m.val_SI

        return res

    def SF_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for SF.

        """

        fluid = self.SF.split_fluid
        out_i = int(self.SF.split_outlet[3:]) - 1

        o = self.outl[out_i]
        if o.m.is_var:
            self.jacobian[k, o.m.J_col] = -o.fluid.val[fluid]
        if fluid in o.fluid.is_var:
            self.jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI    

    def SFS_func(self):
        r"""
        Equation for SFS.

        """

        fluid = self.SFS.split_fluid
        out_i = int(self.SFS.split_outlet[3:]) - 1
        i = self.inl[0]
        o = self.outl[out_i]

        res = i.fluid.val[fluid] * i.m.val_SI * self.SFS.val \
            - o.fluid.val[fluid] * o.m.val_SI

        return res

    def SFS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for SFS.

        """

        fluid = self.SFS.split_fluid
        out_i = int(self.SFS.split_outlet[3:]) - 1

        i = self.inl[0]
        o = self.outl[out_i]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col] = i.fluid.val[fluid] * self.SFS.val
        if fluid in i.fluid.is_var:
            self.jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI * self.SFS.val
        if o.m.is_var:
            self.jacobian[k, o.m.J_col] = -o.fluid.val[fluid]
        if fluid in o.fluid.is_var:
            self.jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI

    def calc_parameters(self):
        super().calc_parameters()
        
        i = self.inl[0]
        if self.SFS.is_set:
            fluid = self.SFS.split_fluid
            self.SF.val = self.SFS.val* i.fluid.val[fluid] * i.m.val_SI
        if self.SF.is_set:
            fluid = self.SF.split_fluid
            self.SFS.val = self.SF.val / (i.fluid.val[fluid] * i.m.val_SI)

class SeparatorWithSpeciesSplitsDeltaT(SeparatorWithSpeciesSplits):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaT"] = dc_cp(
            deriv=self.energy_balance_deltaT_deriv, # same as before
            func=self.energy_balance_deltaT_func,
            latex=self.pr_func_doc,
            num_eq=self.num_out
        )
        variables["Q"] = dc_cp(is_result=True)
        #variables["Qout"] = dc_cpa()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}
        if constraints.get("energy_balance_constraints",False):
            del constraints['energy_balance_constraints']
        return constraints
    
    def energy_balance_deltaT_func(self):
        r"""
        Calculate deltaT residuals.

        """
        T_in = self.inl[0].calc_T(T0=self.inl[0].T.val_SI)
        residual = []
        for o in self.outl:
            residual += [T_in - self.deltaT.val - o.calc_T(T0=T_in)] # use T_in as guess
        return residual
    
    def energy_balance_deltaT_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.
        """
        i = self.inl[0]
        dT_dp_in = dT_mix_dph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule,T0 = i.T.val_SI,force_state=i.force_state)
        dT_dh_in = dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule,T0 = i.T.val_SI,force_state=i.force_state)
        # dT_dfluid_in = {}
        # for fluid in i.fluid.is_var:
        #     dT_dfluid_in[fluid] = dT_mix_ph_dfluid(i)
        for o in self.outl:
            if self.is_variable(i.p):
                self.jacobian[k, i.p.J_col] = dT_dp_in
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = dT_dh_in
            # for fluid in i.fluid.is_var:
            #     self.jacobian[k, i.fluid.J_col[fluid]] = dT_dfluid_in[fluid]
            args = (o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule,i.T.val_SI,o.force_state)
            if self.is_variable(o.p):
                self.jacobian[k, o.p.J_col] = -dT_mix_dph(*args)
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -dT_mix_pdh(*args)
            # for fluid in o.fluid.is_var:
            #     self.jacobian[k, o.fluid.J_col[fluid]] = -dT_mix_ph_dfluid(o)
            k += 1

        # deriv = [d for d in self.jacobian.items()]
        # [print(d) for d in deriv]
        # deriv      

    def calc_parameters(self):
        super().calc_parameters()
        i = self.inl[0]
        if not self.Q.is_set:
            self.Q.val = np.sum([o.m.val_SI * (o.h.val_SI - i.h.val_SI) for o in self.outl])

        Tmin = min([o.T.val_SI for o in self.outl])
        Tmax = max([o.T.val_SI for o in self.outl])
        if abs(i.T.val_SI - Tmin) >= abs(i.T.val_SI - Tmax):
            self.deltaT.val = i.T.val_SI - Tmin
        else:
            self.deltaT.val = i.T.val_SI - Tmax

class SeparatorWithSpeciesSplitsDeltaH(SeparatorWithSpeciesSplits):

    @staticmethod
    def component():
        return 'separator with species flow splits and dH on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaH"] = dc_cp(
            deriv=self.energy_balance_deltaH_deriv, # same as before
            func=self.energy_balance_deltaH_func,
            latex=self.pr_func_doc,
            num_eq=self.num_out
        )
        variables["Q"] = dc_cp(
            func=self.Q_func, num_eq=1,
            deriv=self.Q_deriv,
            latex=self.pr_func_doc)
        variables["KPI"] = dc_cp(
            deriv=self.KPI_deriv,
            func=self.KPI_func,
            latex=self.pr_func_doc,
            num_eq=1)        
        #variables["Qout"] = dc_cpa()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}
        if constraints.get("energy_balance_constraints",False):
            del constraints['energy_balance_constraints']        
        return constraints
    
    def energy_balance_deltaH_func(self):
        r"""
        Calculate deltaH residuals.

        """
        i = self.inl[0]
        residual = []
        for o in self.outl:
            residual += [i.h.val_SI - self.deltaH.val - o.h.val_SI]
        return residual
    
    def energy_balance_deltaH_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.
        """
        i = self.inl[0]
        for o in self.outl:
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = 1
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -1
            k += 1

    def Q_func_Tequality(self,port1,port2):
        return port1.calc_T(T0=port1.T.val_SI) - port2.calc_T(T0=port2.T.val_SI)        

    def Q_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """
        i = self.inl[0]
        o1 = self.outl[0]
        o2 = self.outl[1]
        # #res = []
        # #res += [o1.m.val_SI * (o1.h.val_SI - i.h.val_SI) + o2.m.val_SI * (o2.h.val_SI - i.h.val_SI) - self.Q.val]
        # #res += [self.Q_func_Tequality(o1,o2)]
        # #return res    
        return o1.m.val_SI * (o1.h.val_SI - i.h.val_SI) + o2.m.val_SI * (o2.h.val_SI - i.h.val_SI) - self.Q.val

    def Q_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        i = self.inl[0]
        o1 = self.outl[0]
        o2 = self.outl[1]       
        if self.is_variable(i.h):
            self.jacobian[k, i.h.J_col] = - o1.m.val_SI - o2.m.val_SI
        if self.is_variable(o1.m):
            self.jacobian[k, o1.m.J_col] = o1.h.val_SI - i.h.val_SI
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = o2.h.val_SI - i.h.val_SI            
        if self.is_variable(o1.h):
            self.jacobian[k, o1.h.J_col] = o1.m.val_SI
        if self.is_variable(o2.h):
            self.jacobian[k, o2.h.J_col] = o2.m.val_SI

        # k = k + 1 
        # for c in [self.outl[0], self.outl[1]]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.Q_func_Tequality, 'p', c, port1 = self.outl[0], port2 = self.outl[1])
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.Q_func_Tequality, 'h', c, port1 = self.outl[0], port2 = self.outl[1])

    def KPI_func(self):
        r"""
        Equation for total heat flow rate
        """
        i = self.inl[0]
        o1 = self.outl[0]
        o2 = self.outl[1]
        # res = []
        # res += [o1.m.val_SI * (o1.h.val_SI - i.h.val_SI) + o2.m.val_SI * (o2.h.val_SI - i.h.val_SI) - self.KPI.val * i.m.val_SI]
        # res += [self.Q_func_Tequality(o1,o2)]
        # return res    
        return o1.m.val_SI * (o1.h.val_SI - i.h.val_SI) + o2.m.val_SI * (o2.h.val_SI - i.h.val_SI) - self.KPI.val * i.m.val_SI

    def KPI_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        i = self.inl[0]
        o1 = self.outl[0]
        o2 = self.outl[1]       
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = - self.KPI.val
        if self.is_variable(i.h):
            self.jacobian[k, i.h.J_col] = - o1.m.val_SI - o2.m.val_SI
        if self.is_variable(o1.m):
            self.jacobian[k, o1.m.J_col] = o1.h.val_SI - i.h.val_SI
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = o2.h.val_SI - i.h.val_SI            
        if self.is_variable(o1.h):
            self.jacobian[k, o1.h.J_col] = o1.m.val_SI
        if self.is_variable(o2.h):
            self.jacobian[k, o2.h.J_col] = o2.m.val_SI
        
        # k = k + 1 
        # for c in [self.outl[0], self.outl[1]]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.Q_func_Tequality, 'p', c, port1 = self.outl[0], port2 = self.outl[1])
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.Q_func_Tequality, 'h', c, port1 = self.outl[0], port2 = self.outl[1])

    def calc_parameters(self):
        super().calc_parameters()
        i = self.inl[0]

        if not self.Q.is_set:
            self.Q.val = np.sum([o.m.val_SI * (o.h.val_SI - i.h.val_SI) for o in self.outl])
        if not self.KPI.is_set:
            self.KPI.val = np.sum([o.m.val_SI * (o.h.val_SI - i.h.val_SI) for o in self.outl]) / i.m.val_SI 

        hmin = min([o.h.val_SI for o in self.outl])
        hmax = max([o.h.val_SI for o in self.outl])
        if abs(i.h.val_SI - hmin) >= abs(i.h.val_SI - hmax):
            self.deltaH.val = i.h.val_SI - hmin
        else:
            self.deltaH.val = i.h.val_SI - hmax


class SeparatorWithSpeciesSplitsDeltaP(SeparatorWithSpeciesSplits):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT and Pr on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaP"] = dc_cp(
            min_val=0,
            deriv=self.deltaP_deriv,
            func=self.deltaP_func,
            latex=self.pr_func_doc,
            num_eq=self.num_out,
        )
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}        
        del constraints['pressure_constraints']
        return constraints   

    def deltaP_func(self):
        r"""
        Equation for pressure drop.

        """
        residual = []
        p_in = self.inl[0].p.val_SI
        for o in self.outl:
            residual += [p_in - self.deltaP.val*1e5 - o.p.val_SI]
        return residual

    def deltaP_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for pressure drop

        """
        i = self.inl[0]
        for o in self.outl:
            if i.p.is_var:
                self.jacobian[k, i.p.J_col] = 1
            if o.p.is_var:                
                self.jacobian[k, o.p.J_col] = -1
            k += 1            

    def calc_parameters(self):
        super().calc_parameters()

        Pmin = min([i.p.val_SI for i in self.outl])
        Pmax = max([i.p.val_SI for i in self.outl])
        if abs(self.inl[0].p.val_SI - Pmin) >= abs(self.inl[0].p.val_SI - Pmax):
            self.deltaP.val = (self.inl[0].p.val_SI - Pmin)/1e5
        else:
            self.deltaP.val = (self.inl[0].p.val_SI - Pmax)/1e5


class SeparatorWithSpeciesSplitsDeltaTDeltaP(SeparatorWithSpeciesSplitsDeltaT, SeparatorWithSpeciesSplitsDeltaP):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT and Pr on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        #del constraints['pressure_constraints']
        #del constraints['energy_balance_constraints']
        return constraints

class SeparatorWithSpeciesSplitsDeltaTDeltaPDeltaH(SeparatorWithSpeciesSplitsDeltaH, SeparatorWithSpeciesSplitsDeltaT, SeparatorWithSpeciesSplitsDeltaP):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT, dH and dP on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        #del constraints['pressure_constraints']
        #del constraints['energy_balance_constraints']
        return constraints

class DrierWithAir(SeparatorWithSpeciesSplitsDeltaH,SeparatorWithSpeciesSplitsDeltaT,SeparatorWithSpeciesSplitsDeltaP):

    def __init__(self, label, **kwargs):
        #self.set_attr(**kwargs)
        # need to assign the number of outlets before the variables are set
        self.num_out = 2 # default
        self.num_in = 2 # default
        for key in kwargs:
            if key == 'num_out':
                self.num_out=kwargs[key]
            if key == 'num_in':
                self.num_in=kwargs[key]                
        super().__init__(label, **kwargs)    
    
    @staticmethod
    def component():
        return 'separator with species flow splits and dT on outlets'
    
    @staticmethod
    def inlets():
        return ['in1']

    def inlets(self):
        if self.num_in.is_set:
            return ['in' + str(i + 1) for i in range(self.num_in.val)]
        else:
            self.set_attr(num_in=2)
            return self.inlets()

    def get_parameters(self):
        variables = super().get_parameters()
        variables["num_in"] = dc_simple()
        variables["dTwbProd"] = dc_cp(
            deriv=self.dTwbProd_deriv,
            func=self.dTwbProd_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )       
        variables["WBeff"] = dc_cp(
            min_val=0,max_val=1,
            deriv=self.WBeff_deriv,
            func=self.WBeff_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        variables['kA'] = dc_cp(
                min_val=0, num_eq=1, func=self.kA_func, latex=self.pr_func_doc,
                deriv=self.kA_deriv)
        variables['td_log'] = dc_cp(min_val=0, is_result=True)        
        variables['ttd_u'] = dc_cp(min_val=0, is_result=True)        
        variables['ttd_l'] = dc_cp(min_val=0, is_result=True)        
        variables['m_evap'] = dc_cp(min_val=0, is_result=True)        
        variables['Q_evap'] = dc_cp(min_val=0, is_result=True)
        variables['RH'] = dc_cp(min_val=0, max_val=100, is_result=True)
        variables["dWo"] = dc_cp(
            min_val = 0, max_val=1,
            deriv=self.dWo_deriv,
            func=self.dWo_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )           
        variables["dWo2"] = dc_cp(
            min_val = 0, max_val=1,
            deriv=self.dWo2_deriv,
            func=self.dWo2_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )                   
        variables["dfluid"] = dc_cp(
            min_val = 0, max_val=1,
            deriv=self.dfluid_deriv,
            func=self.dfluid_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )                 
        # variables['eb'] = dc_cp(
        #     min_val = 0, max_val=1,
        #     deriv=self.energy_balance_deriv,
        #     func=self.energy_balance_func,
        #     latex=self.pr_func_doc,
        #     num_eq=1,
        # )                 
        variables["deltaH"] = dc_cp(
            deriv=self.energy_balance_deltaH_deriv, # same as before
            func=self.energy_balance_deltaH_func,
            latex=self.pr_func_doc,
            num_eq=1
        )        
        return variables
    
    def energy_balance_deltaH_func(self):
        r"""
        Calculate deltaH residuals.

        """
        i = self.inl[0]
        residual = []
        for o in [self.outl[1]]:
            residual += [i.h.val_SI - self.deltaH.val - o.h.val_SI]
        return residual[0]
    
    def energy_balance_deltaH_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.
        """
        i = self.inl[0]
        for o in [self.outl[1]]:
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = 1
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -1
            k += 1    

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}
        constraints['energy_balance_constraints'] = {
                'func': self.energy_balance_func,
                'deriv': self.energy_balance_deriv,
                'constant_deriv': False, 'latex': self.energy_balance_func_doc,
                'num_eq': 1}   
        return constraints
    
    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        #i = self.inl[0]
        residual = []
        for fluid in self.variable_fluids:
            res = 0
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            for o in self.outl:
                res -= o.fluid.val[fluid] * o.m.val_SI
            residual += [res]
        
        # # additional balance equation for calculating water vapor mass fraction
        # i = self.inl[1]
        # o = self.outl[0]
        # # known imposition of water and air flows, mean we calculate o.fluid.val['Water'] by 
        # residual += [o.m.val_SI - i.m.val_SI*i.fluid.val['Air'] - o.fluid.val['Water'] * o.m.val_SI]

        # i1 = self.inl[0]
        # i2 = self.inl[1]
        # o1 = self.outl[0]
        # o2 = self.outl[1]
        # m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        # residual += [i2.m.val_SI + m_evap - o1.m.val_SI]
        return residual
    
    def fluid_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
        #i = self.inl[0]
        for fluid in self.variable_fluids:
            for o in self.outl:
                if self.is_variable(o.m):
                    self.jacobian[k, o.m.J_col] = -o.fluid.val[fluid]
                if fluid in o.fluid.is_var:
                    self.jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI

            for i in self.inl:
                if self.is_variable(i.m):
                    self.jacobian[k, i.m.J_col] = i.fluid.val[fluid]
                if fluid in i.fluid.is_var:
                    self.jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI

            k += 1    

        # i = self.inl[1]
        # o = self.outl[0]
        # if self.is_variable(o.m):
        #     self.jacobian[k, o.m.J_col] = 1 - o.fluid.val['Water']
        # if fluid in o.fluid.is_var:
        #     self.jacobian[k, o.fluid.J_col['Water']] = - o.m.val_SI
        # if self.is_variable(i.m):
        #     self.jacobian[k, i.m.J_col] = -i.fluid.val['Air']
        # if fluid in i.fluid.is_var:
        #     self.jacobian[k, i.fluid.J_col['Air']] = - i.m.val_SI

        # i1 = self.inl[0]
        # i2 = self.inl[1]
        # o1 = self.outl[0]
        # o2 = self.outl[1]

        # if self.is_variable(i2.m):
        #     self.jacobian[k, i2.m.J_col] = 1           

        # if self.is_variable(o2.m):
        #     self.jacobian[k, o2.m.J_col] = - o2.fluid.val['Water']
        # if 'Water' in o2.fluid.is_var:
        #     self.jacobian[k, o2.fluid.J_col['Water']] = - o2.m.val_SI

        # if self.is_variable(i1.m):
        #     self.jacobian[k, i1.m.J_col] = i1.fluid.val['Water']
        # if 'Water' in i1.fluid.is_var:
        #     self.jacobian[k, i1.fluid.J_col['Water']] = i1.m.val_SI

        # if self.is_variable(o1.m):
        #     self.jacobian[k, o1.m.J_col] = -1

    
    def dfluid_func(self):
        # additional balance equation for calculating water vapor mass fraction
        i = self.inl[1]
        o = self.outl[0]
        # known imposition of water and air flows, mean we calculate o.fluid.val['Water'] by 
        return o.m.val_SI - i.m.val_SI*i.fluid.val['Air'] - o.fluid.val['Water'] * o.m.val_SI
                
        # i1 = self.inl[0]
        # i2 = self.inl[1]
        # o1 = self.outl[0]
        # o2 = self.outl[1]
        # m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        # return i2.m.val_SI + m_evap - o1.m.val_SI - self.dfluid.val
    
    def dfluid_deriv(self, increment_filter, k):

        i = self.inl[1]
        o = self.outl[0]
        if self.is_variable(o.m):
            self.jacobian[k, o.m.J_col] = 1 - o.fluid.val['Water']
        if 'Water' in o.fluid.is_var:
            self.jacobian[k, o.fluid.J_col['Water']] = - o.m.val_SI
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = -i.fluid.val['Air']
        if 'Air' in i.fluid.is_var:
            self.jacobian[k, i.fluid.J_col['Air']] = - i.m.val_SI        

        # i1 = self.inl[0]
        # i2 = self.inl[1]
        # o1 = self.outl[0]
        # o2 = self.outl[1]

        # if self.is_variable(i2.m):
        #     self.jacobian[k, i2.m.J_col] = 1           

        # if self.is_variable(o2.m):
        #     self.jacobian[k, o2.m.J_col] = - o2.fluid.val['Water']
        # if 'Water' in o2.fluid.is_var:
        #     self.jacobian[k, o2.fluid.J_col['Water']] = - o2.m.val_SI

        # if self.is_variable(i1.m):
        #     self.jacobian[k, i1.m.J_col] = i1.fluid.val['Water']
        # if 'Water' in i1.fluid.is_var:
        #     self.jacobian[k, i1.fluid.J_col['Water']] = i1.m.val_SI

        # if self.is_variable(o1.m):
        #     self.jacobian[k, o1.m.J_col] = -1    

    def dTwbProd_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        i = self.inl[1]
        T_in = i.calc_T(T0=i.T.val_SI)
        T_wb = get_Twb(i,T_in)
        o = self.outl[1]
        T_out = o.calc_T(T0=o.T.val_SI)
        return T_out - T_wb - self.dTwbProd.val
    
    def dTwbProd_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
        for c in [self.inl[1]]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = dT_mix_dph(c.p.val_SI, c.h.val_SI, c.fluid_data, c.mixing_rule,T0 = c.T.val_SI,force_state=c.force_state)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = dT_mix_pdh(c.p.val_SI, c.h.val_SI, c.fluid_data, c.mixing_rule,T0 = c.T.val_SI,force_state=c.force_state)
        # T_wb is nonlinear and we cannot differentiate easily
        for c in [self.outl[1]]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.dTwbProd_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.dTwbProd_func, 'h', c)

    def dWo_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        i2 = self.inl[1]
        o1 = self.outl[0]
        T_i  = i2.calc_T(T0=i2.T.val_SI)
        T_o = o1.calc_T(T0=o1.T.val_SI)

        M_i = i2.fluid.val["Water"]
        W_i = M_i/(1-M_i)
        I_i = HAPropsSI('H','P',i2.p.val_SI,'T',T_i,'W',W_i)

        T_wb  = get_Twb(i2,T_i)
        W_wb = HAPropsSI('W','P',i2.p.val_SI,'T',T_wb,'R',1)
        I_wb = HAPropsSI('H','P',i2.p.val_SI,'T',T_wb,'R',1)

        M_o = o1.fluid.val["Water"]
        W_o = M_o/(1-M_o)

        #W_o_calc = W_i - (T_i-T_o)/(T_i-T_wb)*(W_i-W_wb)
        I_o = I_i - (T_i-T_o)/(T_i-T_wb)*(I_i-I_wb)
        W_o_calc = HAPropsSI('W','P',i2.p.val_SI,'H',I_o,'T',T_o)
              
        #T_o_linear = T_i - (T_i-T_o)/(T_i-T_wb)*(W_i-W_wb)
        #W_o_calc = W_i - (T_i-T_o)/(T_i-T_wb)*(W_i-W_wb)
        return W_o_calc - W_o - self.dWo.val

        
    
    def dWo_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """

        i2 = self.inl[1]
        o1 = self.outl[0]        
        for c in [i2, o1]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.dWo_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.dWo_func, 'h', c)
            # if self.is_variable(c.m): #, increment_filter):
            #     self.jacobian[k, c.m.J_col] = self.numeric_deriv(self.dWo_func, 'm', c, i2=i2, o1=o1)

            for fluid in self.variable_fluids:
                if fluid in c.fluid.is_var:
                    self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.dWo_func, fluid, c)

    def dWo2_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """

        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]

        Ti1  = i1.calc_T(T0=i1.T.val_SI)
        Ti2  = i2.calc_T(T0=i2.T.val_SI)
        To1  = o1.calc_T(T0=o1.T.val_SI)
        #To2  = o2.calc_T(T0=o2.T.val_SI)

        m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        Q_evap = m_evap * (o1.fluid_data['Water']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
                          -i1.fluid_data['Water']['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state))

        m_air = i2.m.val_SI*i2.fluid.val['Air'] #i2.m.val_SI # *i2.fluid.val['Air']
        Q_air = + m_air * (o1.fluid_data['Air']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
                          -i2.fluid_data['Air']['wrapper'].h_pT(i2.p.val_SI,Ti2,force_state=i2.force_state))
        
        return Q_evap + Q_air - self.dWo2.val
        
    
    def dWo2_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """

        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]

        Ti1  = i1.calc_T(T0=i1.T.val_SI)
        Ti2  = i2.calc_T(T0=i2.T.val_SI)
        To1  = o1.calc_T(T0=o1.T.val_SI)

        dh_w = (o1.fluid_data['Water']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
               -i1.fluid_data['Water']['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state))
        dh_a = (o1.fluid_data['Air']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
               -i2.fluid_data['Air']['wrapper'].h_pT(i2.p.val_SI,Ti2,force_state=i2.force_state))

        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = - o2.fluid.val['Water'] * dh_w
        if 'Water' in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col['Water']] = - o2.m.val_SI * dh_w
        if self.is_variable(i1.m):
            self.jacobian[k, i1.m.J_col] = i1.fluid.val['Water'] * dh_w
        if 'Water' in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col['Water']] = i1.m.val_SI * dh_w

        # if self.is_variable(o1.m):
        #     self.jacobian[k, o1.m.J_col] = o1.fluid.val['Air'] * dh_a
        # if 'Air' in o1.fluid.is_var:
        #     self.jacobian[k, o1.fluid.J_col['Air']] = o1.m.val_SI * dh_a

        if self.is_variable(i2.m):
            self.jacobian[k, i2.m.J_col] = i2.fluid.val['Air'] * dh_a
        if 'Air' in i2.fluid.is_var:
            self.jacobian[k, i2.fluid.J_col['Air']] = i2.m.val_SI * dh_a

            


    def res2(self,i2,o1):
        T_i  = i2.calc_T(T0=i2.T.val_SI)
        T_o = o1.calc_T(T0=o1.T.val_SI)
        T_wb  = get_Twb(i2,T_i)
        W_wb = HAPropsSI('W','P',i2.p.val_SI,'T',T_wb,'R',1)

        M_i = i2.fluid.val["Water"]
        W_i = M_i/(1-M_i)
        M_o = o1.fluid.val["Water"]
        W_o = M_o/(1-M_o)
        
        #T_o_linear = T_i - (T_i-T_o)/(T_i-T_wb)*(W_i-W_wb)
        W_o_calc = W_i - (T_i-T_o)/(T_i-T_wb)*(W_i-W_wb)
        return W_o_calc - W_o

    def energy_balance_func(self):
        r"""
        Need overwrite this function to take into account air inlet
        """
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]

        # res = []
        # res += [o1.m.val_SI * o1.h.val_SI + o2.m.val_SI * o2.h.val_SI - i1.m.val_SI * i1.h.val_SI - i2.m.val_SI * i2.h.val_SI]
        # res += [self.res2(i2,o1)]
        # return res
        return o1.m.val_SI * o1.h.val_SI + o2.m.val_SI * o2.h.val_SI - i1.m.val_SI * i1.h.val_SI - i2.m.val_SI * i2.h.val_SI
    
    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Need overwrite this function to take into account air inlet
        """
        i1 = self.inl[0]
        i2 = self.inl[1]        
        o1 = self.outl[0]
        o2 = self.outl[1]       
        
        if self.is_variable(o1.m):
            self.jacobian[k, o1.m.J_col] = o1.h.val_SI
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = o2.h.val_SI
        if self.is_variable(o1.h):
            self.jacobian[k, o1.h.J_col] = o1.m.val_SI
        if self.is_variable(o2.h):
            self.jacobian[k, o2.h.J_col] = o2.m.val_SI

        if self.is_variable(i1.m):
            self.jacobian[k, i1.m.J_col] = -i1.h.val_SI
        if self.is_variable(i2.m):
            self.jacobian[k, i2.m.J_col] = -i2.h.val_SI
        if self.is_variable(i1.h):
            self.jacobian[k, i1.h.J_col] = -i1.m.val_SI
        if self.is_variable(i2.h):
            self.jacobian[k, i2.h.J_col] = -i2.m.val_SI

        # k = k + 1

        # for c in [i2, o1]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.res2, 'p', c, i2=i2, o1=o1)
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.res2, 'h', c, i2=i2, o1=o1)
        #     if self.is_variable(c.m): #, increment_filter):
        #         self.jacobian[k, c.m.J_col] = self.numeric_deriv(self.res2, 'm', c, i2=i2, o1=o1)

        #     for fluid in self.variable_fluids:
        #         if fluid in c.fluid.is_var:
        #             self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.res2, fluid, c, i2=i2, o1=o1)

    def WBeff_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        i = self.inl[1]
        T_in = i.calc_T(T0=i.T.val_SI)
        T_wb = get_Twb(i,T_in)
        o = self.outl[0]
        T_out = o.calc_T(T0=o.T.val_SI)        
        #print ((T_in-T_out) - (T_in-T_wb)*self.WBeff.val)
        return (T_in-T_out) - (T_in-T_wb)*self.WBeff.val
    
    def WBeff_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
        for c in [self.inl[1], self.outl[0]]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.WBeff_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.WBeff_func, 'h', c)

            for fluid in self.variable_fluids:
                if fluid in c.fluid.is_var:
                    self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.WBeff_func, fluid, c)

    def KPI_func(self):
        r"""
        how much water is dried
        """
        o = self.outl[0]
        m_evap = o.m.val_SI*o.fluid.val['Water']
        return m_evap - self.KPI.val

    def KPI_deriv(self, increment_filter, k):
        o = self.outl[0]
        if self.is_variable(o.m):
            self.jacobian[k, o.m.J_col] = o.fluid.val['Water']
        if 'Water' in o.fluid.is_var:
            self.jacobian[k, o.fluid.J_col['Water']] = o.m.val_SI

    def calculate_td_log(self,T_i,T_wb,T_o):
        # 1 is with air
        i1 = self.inl[1]
        o1 = self.outl[0]

        # temperature value manipulation for convergence stability
        T_i1 = T_i
        T_o1 = T_o
        T_i2 = T_wb
        T_o2 = T_wb

        if T_i1 <= T_o2:
            T_i1 = T_o2 + 0.01
        if T_i1 <= T_o2:
            T_o2 = T_i1 - 0.01
        if T_i1 <= T_o2:
            T_o1 = T_i2 + 0.02
        if T_o1 <= T_i2:
            T_i2 = T_o1 - 0.02

        ttd_u = T_i1 - T_o2
        ttd_l = T_o1 - T_i2

        if ttd_u == ttd_l:
            td_log = ttd_l
        else:
            td_log = (ttd_l - ttd_u) / np.log((ttd_l) / (ttd_u))

        return td_log

    def kA_func(self):
        r"""
        Calculate heat transfer from heat transfer coefficient.
        """
        i = self.inl[1]
        o = self.outl[0]
        T_i = i.calc_T(T0=i.T.val_SI)
        T_wb = get_Twb(i,T_i)
        T_o = o.calc_T(T0=o.T.val_SI)

        m_air =   i.m.val_SI*i.fluid.val['Air']
        Q_air = - m_air * (o.fluid_data['Air']['wrapper'].h_pT(o.p.val_SI,T_o,force_state='g')
                          -i.fluid_data['Air']['wrapper'].h_pT(i.p.val_SI,T_i,force_state='g'))
        return Q_air - self.kA.val * self.calculate_td_log(T_i,T_wb,T_o)    

    def kA_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of heat transfer coefficient function.
        """
        i = self.inl[1]
        o = self.outl[0]
        T_i = i.calc_T(T0=i.T.val_SI)
        #T_wb = get_Twb(i,T_i)
        T_o = o.calc_T(T0=o.T.val_SI)        
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = - i.fluid.val['Air']*(o.fluid_data['Air']['wrapper'].h_pT(o.p.val_SI,T_o,force_state='g')
                                                             -i.fluid_data['Air']['wrapper'].h_pT(i.p.val_SI,T_i,force_state='g'))
        if 'Air' in i.fluid.is_var:
            self.jacobian[k, i.fluid.J_col['Air']] = - i.m.val_SI*(o.fluid_data['Air']['wrapper'].h_pT(o.p.val_SI,T_o,force_state='g')
                                                                -i.fluid_data['Air']['wrapper'].h_pT(i.p.val_SI,T_i,force_state='g'))
        for c in self.inl + self.outl:
            if self.is_variable(c.p):
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.kA_func, 'p', c)
            if self.is_variable(c.h):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.kA_func, 'h', c)

            for fluid in self.variable_fluids:
                if fluid in c.fluid.is_var:
                    self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.kA_func, fluid, c)

    def calc_parameters(self):
        super().calc_parameters()
        
        i = self.inl[0]
        o = self.outl[0]
        self.m_evap.val = o.m.val_SI*o.fluid.val['Water']
        self.Q_evap.val = self.m_evap.val * (o.fluid_data['Water']['wrapper'].h_pT(o.p.val_SI,o.T.val_SI,force_state=o.force_state)
                                            -i.fluid_data['Water']['wrapper'].h_pT(i.p.val_SI,i.T.val_SI,force_state=i.force_state))

        i = self.inl[1]
        o = self.outl[0]
        m_air  = i.m.val_SI*i.fluid.val['Air']
        Q_air = m_air * (o.fluid_data['Air']['wrapper'].h_pT(o.p.val_SI,o.T.val_SI,force_state='g')
                         -i.fluid_data['Air']['wrapper'].h_pT(i.p.val_SI,i.T.val_SI,force_state='g'))
        
        if not self.Q.is_set:
            self.Q.val = (self.outl[0].m.val_SI * self.outl[0].h.val_SI + 
                          self.outl[1].m.val_SI * self.outl[1].h.val_SI - 
                          self.inl[0].m.val_SI * self.inl[0].h.val_SI -
                          self.inl[1].m.val_SI * self.inl[1].h.val_SI)
        if not self.KPI.is_set:
            self.KPI.val = self.m_evap.val
                   
        if self.outl[1].fluid.val['Air'] > 0:
            TESPyComponentError("Air cannot go into out2")           

        T_in  = self.inl[1].T.val_SI
        T_out = self.outl[0].T.val_SI
        T_wb  = self.outl[1].T.val_SI # get_Twb(self.inl[1],T_in)

        if not self.WBeff.is_set:
            self.WBeff.val = (T_in-T_out)/(T_in-T_wb)
            if self.WBeff.val > 1.0:
                TESPyComponentError("efficiency cannot be greater than 1.0, try increase air mass flow")


        self.ttd_u.val = T_in - T_wb
        self.ttd_l.val = T_out - T_wb

        if not self.kA.is_set:
            # kA and logarithmic temperature difference
            if self.ttd_u.val < 0 or self.ttd_l.val < 0:
                self.td_log.val = np.nan
            elif self.ttd_l.val == self.ttd_u.val:
                self.td_log.val = self.ttd_l.val
            else:
                self.td_log.val = ((self.ttd_l.val - self.ttd_u.val) /
                                np.log(self.ttd_l.val / self.ttd_u.val))
            self.kA.val = -Q_air / self.td_log.val

        port_i = self.inl[1]
        # M_i = port_i.fluid.val["Water"]
        # W_i = M_i/(1-M_i)
        # I_i = HAPropsSI('H','P',port_i.p.val_SI,'T',port_i.T.val_SI,'W',W_i)
        port_o = self.outl[0]
        M_o = port_o.fluid.val["Water"]
        W_o = M_o/(1-M_o)
        # I_o = HAPropsSI('H','P',port_o.p.val_SI,'T',port_o.T.val_SI,'W',W_o)
        
        # I_wb = HAPropsSI('H','P',port_o.p.val_SI,'T',T_wb,'R',1)
        # W_wb = HAPropsSI('W','P',port_o.p.val_SI,'T',T_wb,'R',1)
        # T_o = T_in - (T_in-T_wb)/(W_i-W_wb)*(W_i-W_o)

        # T_o_2 = HAPropsSI('T','P',port_o.p.val_SI,'H',I_i,'W',W_o)
        
        # print(int(I_i),int(I_o))
        # print(int(T_o),int(T_o_2))

        # print("hey")

        Wmax = HAPropsSI('W','P',port_i.p.val_SI,'T',port_o.T.val_SI,'R',1)
        if self.WBeff.val > 1.0 or W_o > Wmax:
            self.RH.val = 100
        else:
            self.RH.val = 100 * HAPropsSI('R','P',port_i.p.val_SI,'T',port_o.T.val_SI,'W',W_o)


class SeparatorWithSpeciesSplitsDeltaTDeltaPBus(SeparatorWithSpeciesSplitsDeltaTDeltaP):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT and Pr on outlets + Bus connection on Q'

    def get_parameters(self):
        variables = super().get_parameters()
        return variables

    def bus_func(self, bus):
        r"""
        Calculate the value of the bus function.

        """
        return np.sum([o.m.val_SI * (o.h.val_SI - self.inl[0].h.val_SI) for o in self.outl])

    def bus_func_doc(self, bus):
        r"""
        Return LaTeX string of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        latex : str
            LaTeX string of bus function.
        """
        return (
            r'\dot{m}_\mathrm{in} \cdot \left(h_\mathrm{out} - '
            r'h_\mathrm{in} \right)')

    def bus_deriv(self, bus):
        r"""
        Calculate partial derivatives of the bus function.

        """

        f = self.calc_bus_value
        if self.inl[0].m.is_var:
            if self.inl[0].m.J_col not in bus.jacobian:
                bus.jacobian[self.inl[0].m.J_col] = 0
            bus.jacobian[self.inl[0].m.J_col] -= self.numeric_deriv(f, 'm', self.inl[0], bus=bus)

        if self.inl[0].h.is_var:
            if self.inl[0].h.J_col not in bus.jacobian:
                bus.jacobian[self.inl[0].h.J_col] = 0
            bus.jacobian[self.inl[0].h.J_col] -= self.numeric_deriv(f, 'h', self.inl[0], bus=bus)

        for o in self.outl:
            if o.h.is_var:
                if o.h.J_col not in bus.jacobian:
                    bus.jacobian[o.h.J_col] = 0
                bus.jacobian[o.h.J_col] -= self.numeric_deriv(f, 'h', o, bus=bus)        
            if o.m.is_var:
                if o.m.J_col not in bus.jacobian:
                    bus.jacobian[o.m.J_col] = 0
                bus.jacobian[o.m.J_col] -= self.numeric_deriv(f, 'm', o, bus=bus)        


class SeparatorWithSpeciesSplitsAndFlowSplitsDeltaTDeltaPDeltaH(SeparatorWithSpeciesSplitsDeltaT,SeparatorWithSpeciesSplitsDeltaH, SeparatorWithSpeciesSplitsDeltaP):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT, dH and dP on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        #del constraints['pressure_constraints']
        #del constraints['energy_balance_constraints']
        return constraints

    def get_parameters(self):
        variables = super().get_parameters()
        variables["FS"] = dc_cp_FS(
            min_val=0,
            deriv=self.FS_deriv,
            func=self.FS_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        return variables

    def FS_func(self):
        r"""
        Equation for flow split.

        """

        out_i = int(self.FS.split_outlet[3:]) - 1
        res = self.inl[0].m.val_SI * self.FS.val - self.outl[out_i].m.val_SI
        return res

    def FS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for flow split

        """

        out_i = int(self.FS.split_outlet[3:]) - 1

        i = self.inl[0]
        o = self.outl[out_i]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col]     = self.FS.val
        if o.m.is_var:
            self.jacobian[k, o.m.J_col]     = -1



class SplitterDeltaP(Splitter):

    def __init__(self, label, **kwargs):
        #self.set_attr(**kwargs)
        # need to assign the number of outlets before the variables are set
        for key in kwargs:
            if key == 'num_out':
                self.num_out=kwargs[key]
        super().__init__(label, **kwargs)    

    @staticmethod
    def component():
        return 'Splitter with pressure losses'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["deltaP"] = dc_cp(
            min_val=0,
            deriv=self.deltaP_deriv,
            func=self.deltaP_func,
            latex=self.pr_func_doc,
            num_eq=self.num_out,
        )
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        del constraints['pressure_constraints']
        return constraints

    def deltaP_func(self):
        r"""
        Equation for pressure drop.

        """
        #return self.inl[0].p.val_SI * self.pr.val - self.outl[0].p.val_SI
        residual = []
        p_in = self.inl[0].p.val_SI
        for o in self.outl:
            residual += [p_in - self.deltaP.val*1e5 - o.p.val_SI]
        return residual

    def deltaP_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        """

        i = self.inl[0]
        for o in self.outl:
            if i.p.is_var:
                self.jacobian[k, i.p.J_col] = 1
            if o.p.is_var:                
                self.jacobian[k, o.p.J_col] = -1
            k += 1

    def calc_parameters(self):
        super().calc_parameters()

        Pmin = min([i.p.val_SI for i in self.outl])
        Pmax = max([i.p.val_SI for i in self.outl])
        if abs(self.inl[0].p.val_SI - Pmin) >= abs(self.inl[0].p.val_SI - Pmax):
            self.deltaP.val = (self.inl[0].p.val_SI - Pmin)/1e5
        else:
            self.deltaP.val = (self.inl[0].p.val_SI - Pmax)/1e5

class SplitterWithFlowSplitter(Splitter):

    @staticmethod
    def component():
        return 'splitter with flow split ratios'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["FS"] = dc_cp_FS(
            min_val=0,
            deriv=self.FS_deriv,
            func=self.FS_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        return variables

    def FS_func(self):
        r"""
        Equation for flow split.

        """

        out_i = int(self.FS.split_outlet[3:]) - 1
        res = self.inl[0].m.val_SI * self.FS.val - self.outl[out_i].m.val_SI
        return res

    def FS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for flow split

        """

        out_i = int(self.FS.split_outlet[3:]) - 1

        i = self.inl[0]
        o = self.outl[out_i]
        if i.m.is_var:
            self.jacobian[k, i.m.J_col]     = self.FS.val
        if o.m.is_var:
            self.jacobian[k, o.m.J_col]     = -1


class SplitterWithFlowSplitterDeltaP(SplitterWithFlowSplitter, SplitterDeltaP):

    @staticmethod
    def component():
        return 'splitter with flow split ratios and pressure drop'

    def get_parameters(self):
        variables = super().get_parameters()
        return variables


#%% Class containers

class dc_cp_SFS(dc_cp):
    """
    Data container for simple properties.
    + SFS_fluid
    + SFS_outlet
    """
    @staticmethod
    def attr():
        attributes = dc_cp.attr()
        attributes.update({'split_fluid' : None, 'split_outlet' : None})
        return attributes
    
    @staticmethod
    def _serializable_keys():
        keys = dc_cp._serializable_keys()
        keys.append("split_fluid")
        keys.append("split_outlet")
        return keys

class dc_cp_FS(dc_cp):
    """
    Data container for component properties.
    + FS_outlet
    """
    @staticmethod
    def attr():
        attributes = dc_cp.attr()
        attributes.update({'split_outlet' : None})
        return attributes

    @staticmethod
    def _serializable_keys():
        keys = dc_cp._serializable_keys()
        keys.append("split_outlet")
        return keys


# class MergeWithPressureLoss(MergeDeltaP):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component MergeWithPressureLoss will change with "
#             "the next major release, please import MergeDeltaP instead."
#         )
#         warnings.warn(msg, FutureWarning)

# class SeparatorWithSpeciesSplitsAndDeltaT(SeparatorWithSpeciesSplitsDeltaT):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SeparatorWithSpeciesSplitsAndDeltaT will change with "
#             "the next major release, please import SeparatorWithSpeciesSplitsDeltaT instead."
#         )
#         warnings.warn(msg, FutureWarning)        

# class SeparatorWithSpeciesSplitsAndDeltaTAndPr(SeparatorWithSpeciesSplitsDeltaTDeltaP):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SeparatorWithSpeciesSplitsAndDeltaTAndPr will change with "
#             "the next major release, please import SeparatorWithSpeciesSplitsDeltaTDeltaP instead."
#         )
#         warnings.warn(msg, FutureWarning)   

# class SeparatorWithSpeciesSplitsAndDeltaTAndPrAndBus(SeparatorWithSpeciesSplitsDeltaTDeltaPBus):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SeparatorWithSpeciesSplitsAndDeltaTAndPrAndBus will change with "
#             "the next major release, please import SeparatorWithSpeciesSplitsDeltaTDeltaPBus instead."
#         )
#         warnings.warn(msg, FutureWarning)   
        

# class SplitterWithPressureLoss(SplitterDeltaP):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SeparatorWithSpeciesSplitsAndDeltaTAndPr will change with "
#             "the next major release, please import SeparatorWithSpeciesSplitsDeltaTDeltaP instead."
#         )
#         warnings.warn(msg, FutureWarning)   
        
# class SplitterWithPressureLoss(SplitterDeltaP):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SplitterWithPressureLoss will change with "
#             "the next major release, please import SplitterDeltaP instead."
#         )
#         warnings.warn(msg, FutureWarning)   


      

# class SplitWithFlowSplitter(SplitterWithFlowSplitter):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SplitWithFlowSplitter will change with "
#             "the next major release, please import SplitterWithFlowSplitter instead."
#         )
#         warnings.warn(msg, FutureWarning)   


# class SplitWithFlowSplitterDeltaP(SplitterWithFlowSplitterDeltaP):

#     def __init__(self, label, **kwargs):
#         super().__init__(label, **kwargs)
#         msg = (
#             "The API for the component SplitWithFlowSplitterDeltaP will change with "
#             "the next major release, please import SplitterWithFlowSplitterDeltaP instead."
#         )
#         warnings.warn(msg, FutureWarning)   
