import logging

from tespy.components import SimpleHeatExchanger, Merge, Separator, Splitter, HeatExchanger
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import SimpleDataContainer as dc_simple
from tespy.tools.data_containers import StringDataContainer as dc_string

from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp
from tespy.tools.fluid_properties import T_mix_ph, h_mix_pT
from tespy.tools.helpers import TESPyComponentError

from tespy.components.component import Component

from tespy.tools.fluid_properties import dT_mix_dph
from tespy.tools.fluid_properties import dT_mix_pdh

from CoolProp.HumidAirProp import HAPropsSI

import warnings

import numpy as np

from .newComponents import SeparatorWithSpeciesSplitsDeltaH,SeparatorWithSpeciesSplitsDeltaT,SeparatorWithSpeciesSplitsDeltaP


def get_Twb(port,T):
    M = port.fluid.val['Water']/(port.fluid.val['Water']+port.fluid.val["Air"])
    W = M/(1-M)
    return HAPropsSI('Twb','P',port.p.val_SI,'T',T,'W',W)



class TwoStreamEvaporator(SeparatorWithSpeciesSplitsDeltaH,SeparatorWithSpeciesSplitsDeltaT,SeparatorWithSpeciesSplitsDeltaP):

    def __init__(self, label, **kwargs):            
        super().__init__(label, **kwargs)    

    def outlets(self):
        return ['out1', # vapor
                'out2', # liquid
                'out3'] # steam condensate
    
    def inlets(self):
        return ['in1', 'in2']

    @staticmethod
    def component():
        return 'separator with species flow splits and dH on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["num_in"] = dc_simple()
        variables["deltaH"] = dc_cp(
            deriv=self.energy_balance_deltaH_deriv, # same as before
            func=self.energy_balance_deltaH_func,
            latex=self.pr_func_doc,
            num_eq=2
        )
        variables["Q"] = dc_cp(
            max_val=0, func=self.energy_balance_hot_func, num_eq=1,
            deriv=self.energy_balance_hot_deriv,
            latex=self.pr_func_doc)        
        variables["KPI"] = dc_cp(
            deriv=self.KPI_deriv,
            func=self.KPI_func,
            latex=self.pr_func_doc,
            num_eq=1)        
        #variables["Qout"] = dc_cpa()
        variables['kA'] = dc_cp(
                min_val=0, num_eq=1, func=self.kA_func, latex=self.pr_func_doc,
                deriv=self.kA_deriv)
        variables['td'] = dc_cp(min_val=0, is_result=True)         
        variables['dTo'] = dc_cp(
                min_val=0, num_eq=1, func=self.dTo_func, latex=self.pr_func_doc,
                deriv=self.dTo_deriv)
        variables['deltaPhot'] = dc_cp(
                min_val=0, num_eq=1, func=self.deltaPhot_func, latex=self.pr_func_doc,
                deriv=self.deltaPhot_deriv)
        variables['vaporizedfluid'] = dc_string()
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}
        constraints['energy_balance_constraints'] =  {
            'func': self.energy_balance_func, 'deriv': self.energy_balance_deriv,
            'constant_deriv': False, 'latex': self.energy_balance_func_doc,
            'num_eq': 1}      
        constraints['mass_flow_constraints'] =  {
            'func': self.mass_flow_func, 'deriv': self.mass_flow_deriv,
            'constant_deriv': True, 'latex': self.mass_flow_func_doc,
            'num_eq': 1}        
        return constraints
    

    @staticmethod
    def is_branch_source():
        # trigger start_branch
        return True

    def start_branch(self):
        branches = {}
        for outconn in [self.outl[0],self.outl[1]]:
            branch = {
                "connections": [outconn],
                "components": [self, outconn.target],
                "subbranches": {}
            }
            outconn.target.propagate_to_target(branch)

            branches[outconn.label] = branch
        return branches

    def propagate_to_target(self, branch):
        inconn = branch["connections"][-1]
        conn_idx = self.inl.index(inconn)
        if conn_idx == 1:
            # connect in2 with with out3 - othervice stop the connections 
            outconn = self.outl[2]
            branch["connections"] += [outconn]
            branch["components"] += [outconn.target]
            outconn.target.propagate_to_target(branch)

    def propagate_wrapper_to_target(self, branch):
        inconn = branch["connections"][-1]
        conn_idx = self.inl.index(inconn)
        if conn_idx == 1: 
            # connect in2 with with out3
            outconn = self.outl[2]
            branch["connections"] += [outconn]
            branch["components"] += [self]
            outconn.target.propagate_wrapper_to_target(branch)
        elif conn_idx == 0:
            # propagate wrapper to new start branches
            branch["components"] += [self]
            for outconn in [self.outl[0],self.outl[1]]:
                branch["connections"] += [outconn]
                outconn.target.propagate_wrapper_to_target(branch)            

    def deltaPhot_func(self):
        return  self.inl[1].p.val_SI - self.deltaPhot.val*1e5 - self.outl[2].p.val_SI

    def deltaPhot_deriv(self, increment_filter, k):
        if self.inl[1].p.is_var:
            self.jacobian[k, self.inl[1].p.J_col] = 1 
        if self.outl[2].p.is_var:
            self.jacobian[k, self.outl[2].p.J_col] = -1

    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        ci = [self.inl[0]]
        co = [self.outl[0],self.outl[1]]
        # hi = [self.inl[1]]
        # ho = [self.outl[2]]
        residual = []
        for fluid in self.variable_fluids:
            res = 0
            for i in ci:
                res += i.fluid.val[fluid] * i.m.val_SI
            for o in co:
                res -= o.fluid.val[fluid] * o.m.val_SI
            residual += [res]
        # for fluid in self.variable_fluids:
        #     res = 0
        #     for i in hi:
        #         res += i.fluid.val[fluid] * i.m.val_SI
        #     for o in ho:
        #         res -= o.fluid.val[fluid] * o.m.val_SI
        #     residual += [res]
        return residual
    
    def fluid_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
        ci = [self.inl[0]]
        co = [self.outl[0],self.outl[1]]
        # hi = [self.inl[1]]
        # ho = [self.outl[2]]
        for fluid in self.variable_fluids:
            for o in co:
                if self.is_variable(o.m):
                    self.jacobian[k, o.m.J_col] = -o.fluid.val[fluid]
                if fluid in o.fluid.is_var:
                    self.jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI
            for i in ci:
                if self.is_variable(i.m):
                    self.jacobian[k, i.m.J_col] = i.fluid.val[fluid]
                if fluid in i.fluid.is_var:
                    self.jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI
            k += 1    
        # for fluid in self.variable_fluids:
        #     for o in ho:
        #         if self.is_variable(o.m):
        #             self.jacobian[k, o.m.J_col] = -o.fluid.val[fluid]
        #         if fluid in o.fluid.is_var:
        #             self.jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI
        #     for i in hi:
        #         if self.is_variable(i.m):
        #             self.jacobian[k, i.m.J_col] = i.fluid.val[fluid]
        #         if fluid in i.fluid.is_var:
        #             self.jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI
        #     k += 1    

    def mass_flow_func(self):
        r"""
        Calculate the residual value for mass flow balance equation.
        """
        ci = [self.inl[0]]
        co = [self.outl[0],self.outl[1]]
        # hi = [self.inl[1]]
        # ho = [self.outl[2]]
        residuals = []
        res = 0
        for i in ci:
            res += i.m.val_SI
        for o in co:
            res -= o.m.val_SI
        residuals += [res]
        # res = 0
        # for i in hi:
        #     res += i.m.val_SI
        # for o in ho:
        #     res -= o.m.val_SI
        # residuals += [res]
        return residuals

    def mass_flow_deriv(self, k):
        r"""
        Calculate partial derivatives for mass flow equation.
        """
        ci = [self.inl[0]]
        co = [self.outl[0],self.outl[1]]
        # hi = [self.inl[1]]
        # ho = [self.outl[2]]
        for i in ci:
            if i.m.is_var:
                self.jacobian[k, i.m.J_col] = 1
        for o in co:
            if o.m.is_var:
                self.jacobian[k, o.m.J_col] = -1
        # k = k + 1
        # for i in hi:
        #     if i.m.is_var:
        #         self.jacobian[k, i.m.J_col] = 1
        # for o in ho:
        #     if o.m.is_var:
        #         self.jacobian[k, o.m.J_col] = -1

    def energy_balance_deltaH_func(self):
        r"""
        Calculate deltaH residuals.

        """
        i = self.inl[0]
        residual = []
        for o in [self.outl[0],self.outl[1]]:
            residual += [i.h.val_SI - self.deltaH.val - o.h.val_SI]
        return residual
    
    def energy_balance_deltaH_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.
        """
        i = self.inl[0]
        for o in [self.outl[0],self.outl[1]]:
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = 1
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -1
            k += 1

    def energy_balance_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """
        i = self.inl[0]
        o1 = self.outl[0]
        o2 = self.outl[1]

        hi = self.inl[1]
        ho = self.outl[2]
        return o1.m.val_SI * (o1.h.val_SI - i.h.val_SI) + o2.m.val_SI * (o2.h.val_SI - i.h.val_SI) + hi.m.val_SI * (ho.h.val_SI - hi.h.val_SI)

    def energy_balance_deriv(self, increment_filter, k):
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

        hi = self.inl[1]
        ho = self.outl[2]

        if self.is_variable(hi.m):
            self.jacobian[k, hi.m.J_col] = (ho.h.val_SI - hi.h.val_SI)
        if self.is_variable(hi.h):
            self.jacobian[k, hi.h.J_col] = -hi.m.val_SI
        if self.is_variable(ho.h):
            self.jacobian[k, ho.h.J_col] =  ho.m.val_SI


    def dTo_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """
        T0 = self.outl[0].calc_T(T0=self.outl[0].T.val_SI)
        T1 = self.outl[1].calc_T(T0=self.outl[1].T.val_SI)
        return T0 - T1 - self.dTo.val

    def dTo_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        #T0 = self.outl[0].calc_T(T0=self.outl[0].T.val_SI)
        #T1 = self.outl[1].calc_T(T0=self.outl[1].T.val_SI)
        for c in [self.outl[0], self.outl[1]]:
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.dTo_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.dTo_func, 'h', c)
            for fluid in self.variable_fluids:
                if fluid in c.fluid.is_var:
                    self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.dTo_func, fluid, c)                

    def energy_balance_hot_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """
        return self.inl[1].m.val_SI * (
            self.outl[2].h.val_SI - self.inl[1].h.val_SI
        ) + self.Q.val

    def energy_balance_hot_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        i = self.inl[1]
        o = self.outl[2]
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h):
            self.jacobian[k, i.h.J_col] = -i.m.val_SI
        if self.is_variable(o.h):
            self.jacobian[k, o.h.J_col] = i.m.val_SI

    def KPI_func(self):
        r"""
        Equation for total heat flow rate
        """
        return self.inl[1].m.val_SI * (
            self.outl[2].h.val_SI - self.inl[1].h.val_SI
        ) + self.inl[0].m.val_SI*self.inl[0].fluid.val[self.vaporizedfluid.val]*self.KPI.val


    def KPI_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        i = self.inl[1]
        o = self.outl[2]
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h):
            self.jacobian[k, i.h.J_col] = -i.m.val_SI
        if self.is_variable(o.h):
            self.jacobian[k, o.h.J_col] = i.m.val_SI
        ci = self.inl[0]
        if self.is_variable(ci.m):
            self.jacobian[k, ci.m.J_col] = ci.fluid.val[self.vaporizedfluid.val]*self.KPI.val
        if self.vaporizedfluid.val in ci.fluid.is_var:
            self.jacobian[k, ci.fluid.J_col[self.vaporizedfluid.val]] = ci.m.val_SI*self.KPI.val

    def kA_func(self):
        r"""
        Calculate heat transfer from heat transfer coefficient.
        """

        Tcold = self.outl[0].calc_T(T0=self.outl[0].T.val_SI) # vapor out
        #Thot = self.outl[2].calc_T(T0=self.outl[2].T.val_SI)
        Thot = self.inl[1].calc_T_sat() # liquid out
        self.td.val = (Thot-Tcold)

        return self.inl[1].m.val_SI * (
            self.outl[2].h.val_SI - self.inl[1].h.val_SI
        ) + self.kA.val * self.td.val

    def kA_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of heat transfer coefficient function.
        """

        i = self.inl[1]
        o = self.outl[2]
        if self.is_variable(i.m):
            self.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h):
            self.jacobian[k, i.h.J_col] = self.numeric_deriv(self.kA_func, 'h', i)
        if self.is_variable(i.p):
            self.jacobian[k, i.p.J_col] = self.numeric_deriv(self.kA_func, 'p', i)
        if self.is_variable(o.h):
            self.jacobian[k, o.h.J_col] = i.m.val_SI
        c = self.outl[0]
        if self.is_variable(c.p):
            self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.kA_func, 'p', c)
        if self.is_variable(c.h):
            self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.kA_func, 'h', c)

    def calc_parameters(self):
        super().calc_parameters()
        i = self.inl[0]

        if not self.Q.is_set:
            self.Q.val = - self.inl[1].m.val_SI * (self.outl[2].h.val_SI - self.inl[1].h.val_SI) # np.sum([o.m.val_SI * (o.h.val_SI - i.h.val_SI) for o in self.outl])
        if not self.KPI.is_set:
            self.KPI.val = - self.inl[1].m.val_SI * (self.outl[2].h.val_SI - self.inl[1].h.val_SI) / (self.inl[0].m.val_SI*self.inl[0].fluid.val[self.vaporizedfluid.val])

        hmin = min([o.h.val_SI for o in self.outl])
        hmax = max([o.h.val_SI for o in self.outl])
        if abs(i.h.val_SI - hmin) >= abs(i.h.val_SI - hmax):
            self.deltaH.val = i.h.val_SI - hmin
        else:
            self.deltaH.val = i.h.val_SI - hmax
        
        if not self.kA.is_set:
            Tcold = self.outl[0].T.val_SI # vapor out
            #Thot = self.outl[2].T.val_SI
            Thot = self.inl[1].calc_T_sat() # liquid out
            self.td.val = (Thot-Tcold)
            if Thot == Tcold:
                self.kA.val = np.NaN
            else:
                self.kA.val = -self.inl[1].m.val_SI * (self.outl[2].h.val_SI - self.inl[1].h.val_SI)/self.td.val




class TwoStreamDrier(SeparatorWithSpeciesSplitsDeltaH,SeparatorWithSpeciesSplitsDeltaT,SeparatorWithSpeciesSplitsDeltaP):

    def __init__(self, label, **kwargs):            
        super().__init__(label, **kwargs)    

    def outlets(self):
        return ['out1', # vapor 
                'out2'] # liquid

    def inlets(self):
        return ['in1', 'in2']

    @staticmethod
    def component():
        return 'separator with species flow splits and dH on outlets'

    def get_parameters(self):
        variables = super().get_parameters()
        variables["num_in"] = dc_simple()
        variables["deltaH"] = dc_cp(
            deriv=self.energy_balance_deltaH_deriv, # same as before
            func=self.energy_balance_deltaH_func,
            latex=self.pr_func_doc,
            num_eq=1
        )
        variables["dTwbProd"] = dc_cp(
            deriv=self.dTwbProd_deriv,
            func=self.dTwbProd_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )             
        variables["Q"] = dc_cp(
            max_val=0, func=self.energy_balance_hot_func, num_eq=1,
            deriv=self.energy_balance_hot_deriv,
            latex=self.pr_func_doc)        
        variables["KPI"] = dc_cp(
            deriv=self.KPI_deriv,
            func=self.KPI_func,
            latex=self.pr_func_doc,
            num_eq=1)        
        #variables["Qout"] = dc_cpa()
        variables['kA'] = dc_cp(
                min_val=0, num_eq=1, func=self.kA_func, latex=self.pr_func_doc,
                deriv=self.kA_deriv)
        variables['td_log'] = dc_cp(min_val=0, is_result=True)        
        variables["WBeff"] = dc_cp(
            min_val=0,max_val=1,
            deriv=self.WBeff_deriv,
            func=self.WBeff_func,
            latex=self.pr_func_doc,
            num_eq=1,
        ) 
        variables['RH'] = dc_cp(min_val=0, max_val=100, is_result=True)
        return variables

    def get_mandatory_constraints(self):
        constraints = super().get_mandatory_constraints()
        self.variable_fluids = set(self.inl[0].fluid.back_end.keys()) 
        #self.variable_product_fluids = [fluid for fluid in self.variable_fluids if not fluid in ['Water','Air']]
        num_fluid_eq = len(self.variable_fluids)
        constraints['fluid_constraints'] = {
            'func': self.fluid_func, 'deriv': self.fluid_deriv,
            'constant_deriv': False, 'latex': self.fluid_func_doc,
            'num_eq': num_fluid_eq}
        constraints['energy_balance_constraints'] =  {
            'func': self.energy_balance_func, 'deriv': self.energy_balance_deriv,
            'constant_deriv': False, 'latex': self.energy_balance_func_doc,
            'num_eq': 1}      
        constraints['mass_flow_constraints'] =  {
            'func': self.mass_flow_func, 'deriv': self.mass_flow_deriv,
            'constant_deriv': True, 'latex': self.mass_flow_func_doc,
            'num_eq': 2}        
        return constraints
    

    # @staticmethod
    # def is_branch_source():
    #     # trigger start_branch
    #     return True

    # def start_branch(self):
    #     branches = {}
    #     for outconn in [self.outl[0],self.outl[1]]:
    #         branch = {
    #             "connections": [outconn],
    #             "components": [self, outconn.target],
    #             "subbranches": {}
    #         }
    #         outconn.target.propagate_to_target(branch)

    #         branches[outconn.label] = branch
    #     return branches

    # def propagate_to_target(self, branch):
    #     inconn = branch["connections"][-1]
    #     conn_idx = self.inl.index(inconn)
    #     if conn_idx == 1:
    #         # connect in2 with with out3 - othervice stop the connections 
    #         outconn = self.outl[2]
    #         branch["connections"] += [outconn]
    #         branch["components"] += [outconn.target]
    #         outconn.target.propagate_to_target(branch)

    # def propagate_wrapper_to_target(self, branch):
    #     inconn = branch["connections"][-1]
    #     conn_idx = self.inl.index(inconn)
    #     if conn_idx == 1: 
    #         # connect in2 with with out3
    #         outconn = self.outl[2]
    #         branch["connections"] += [outconn]
    #         branch["components"] += [self]
    #         outconn.target.propagate_wrapper_to_target(branch)
    #     elif conn_idx == 0:
    #         # propagate wrapper to new start branches
    #         branch["components"] += [self]
    #         for outconn in [self.outl[0],self.outl[1]]:
    #             branch["connections"] += [outconn]
    #             outconn.target.propagate_wrapper_to_target(branch)            


    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        residual = []
        for fluid in self.variable_fluids:
            res = 0
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            for o in self.outl:
                res -= o.fluid.val[fluid] * o.m.val_SI
            residual += [res]        
        return residual
    
    def fluid_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
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

    def mass_flow_func(self):
        r"""
        Calculate the residual value for mass flow balance equation.
        """
        i1 = self.inl[0]
        i2 = self.inl[1]  # air
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid

        m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        residuals = []
        residuals += [i1.m.val_SI - o2.m.val_SI - m_evap]
        residuals += [i2.m.val_SI - o1.m.val_SI + m_evap]

        return residuals

    def mass_flow_deriv(self, k):
        r"""
        Calculate partial derivatives for mass flow equation.
        """
        i1 = self.inl[0]
        i2 = self.inl[1]  # air
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid

        fluid = 'Water'

        if i1.m.is_var:
            self.jacobian[k, i1.m.J_col] = 1 - i1.fluid.val[fluid]
        if o2.m.is_var:
            self.jacobian[k, o2.m.J_col] = -1 + o2.fluid.val[fluid]
        if fluid in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col[fluid]] = - i1.m.val_SI
        if fluid in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col[fluid]] = + o2.m.val_SI

        k = k + 1

        if i2.m.is_var:
            self.jacobian[k, i2.m.J_col] = 1
        if o1.m.is_var:
            self.jacobian[k, o1.m.J_col] = -1
        if i1.m.is_var:
            self.jacobian[k, i1.m.J_col] = + i1.fluid.val[fluid]
        if o2.m.is_var:
            self.jacobian[k, o2.m.J_col] = - o2.fluid.val[fluid]
        if fluid in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col[fluid]] = + i1.m.val_SI
        if fluid in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col[fluid]] = - o2.m.val_SI

    def energy_balance_deltaH_func(self):
        r"""
        Calculate deltaH residuals.

        """
        i = self.inl[0]
        residual = []
        for o in [self.outl[1]]: # liquid -- let energy balance solve the other
            residual += [i.h.val_SI - self.deltaH.val - o.h.val_SI]
        return residual
    
    def energy_balance_deltaH_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.
        """
        i = self.inl[0]
        for o in [self.outl[1]]: # liquid -- let energy balance solve the other
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = 1
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -1
            k += 1

    def energy_balance_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """
        i1 = self.inl[0]
        i2 = self.inl[1]  # air
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid

        return i1.m.val_SI * i1.h.val_SI + i2.m.val_SI * i2.h.val_SI \
               - o2.m.val_SI * o2.h.val_SI - o1.m.val_SI * o1.h.val_SI

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """
        i1 = self.inl[0]
        i2 = self.inl[1]  # air
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid

        for i in [i1,i2]:
            if self.is_variable(i.m):
                self.jacobian[k, i.m.J_col] = i.h.val_SI
            if self.is_variable(i.h):
                self.jacobian[k, i.h.J_col] = i.m.val_SI
        for o in [o2,o1]:
            if self.is_variable(o.m):
                self.jacobian[k, o.m.J_col] = -o.h.val_SI
            if self.is_variable(o.h):
                self.jacobian[k, o.h.J_col] = -o.m.val_SI


    def dTwbProd_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        i = self.inl[1]
        T_in = i.calc_T(T0=i.T.val_SI)
        T_wb = get_Twb(i,T_in)
        o = self.outl[1] # liquid
        T_out = o.calc_T(T0=o.T.val_SI)
        return T_out - T_wb - self.dTwbProd.val
    
    def dTwbProd_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """
        # for c in [self.inl[1]]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = dT_mix_dph(c.p.val_SI, c.h.val_SI, c.fluid_data, c.mixing_rule,T0 = c.T.val_SI,force_state=c.force_state)
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = dT_mix_pdh(c.p.val_SI, c.h.val_SI, c.fluid_data, c.mixing_rule,T0 = c.T.val_SI,force_state=c.force_state)
        # T_wb is nonlinear and we cannot differentiate easily
        for c in [self.inl[1],self.outl[1]]: # in and liquid
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.dTwbProd_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.dTwbProd_func, 'h', c)


    def energy_balance_hot_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.
        """

        i1 = self.inl[0]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid        

        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)
        m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        Q_evap = m_evap * (o1.fluid_data['Water']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
                          -i1.fluid_data['Water']['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state))

        return (Q_evap - self.Q.val)/(self.Q.val+1e-6)

        # i1 = self.inl[0]
        # i2 = self.inl[1]  # air
        # o1 = self.outl[0] # vapor
        # o2 = self.outl[1] # liquid           

        # Ti2 = i2.calc_T(T0=i2.T.val_SI)
        # To1 = o1.calc_T(T0=o1.T.val_SI)

        # m_air = i2.m.val_SI*i2.fluid.val['Air'] 
        # Q_air = m_air * (i2.fluid_data['Air']['wrapper'].h_pT(i2.p.val_SI,Ti2,force_state=i2.force_state)
        #                 -o1.fluid_data['Air']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state))
        
        # return (Q_air - self.Q.val)/self.Q.val

    def energy_balance_hot_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """

        # i1 = self.inl[0]
        # i2 = self.inl[1]  # air
        # o1 = self.outl[0] # vapor
        # o2 = self.outl[1] # liquid   

        # fluid = 'Air'
        # Ti2 = i2.calc_T(T0=i2.T.val_SI)
        # To1 = o1.calc_T(T0=o1.T.val_SI)        
        # hi2 = i2.fluid_data[fluid]['wrapper'].h_pT(i2.p.val_SI,Ti2,force_state=i2.force_state)
        # ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)

        # # Q_air =  i2.m.val_SI*i2.fluid.val['Air']  * (hi2 - ho1)
        
        # if self.is_variable(i2.m):
        #     self.jacobian[k, i2.m.J_col] = i2.fluid.val[fluid] * (hi2 - ho1)
        # if fluid in i2.fluid.is_var:
        #     self.jacobian[k, i2.fluid.J_col[fluid]] = i2.m.val_SI * (hi2 - ho1)

        # for c in [i2,o1]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'p', c)
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'h', c)
        # for c in [i2,o1]:
        #     if fluid in c.fluid.is_var:
        #         self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.energy_balance_hot_func, fluid, c)

        # # let's just do numerical derivative of them all
        # i1 = self.inl[0]
        # i2 = self.inl[1]
        # o1 = self.outl[0] # vapor
        # o2 = self.outl[1] # liquid       

        # for c in [i1,i2,o2,o1]:
        #     if self.is_variable(c.m): #, increment_filter):
        #         self.jacobian[k, c.m.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'm', c)
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'p', c)
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'h', c)
        #     for fluid in self.variable_fluids:
        #         if fluid in c.fluid.is_var:
        #             self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.energy_balance_hot_func, fluid, c)  

        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid   

        fluid = 'Water'
        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)

        ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
        hi1 = i1.fluid_data[fluid]['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state)                                                     

        # Q_evap = (i1.m.val_SI*i1.fluid.val[fluid] - o2.m.val_SI*o2.fluid.val[fluid]) * (ho1 - hi1)

        if self.is_variable(i1.m):
            self.jacobian[k, i1.m.J_col] = i1.fluid.val[fluid] * (ho1 - hi1)/(self.Q.val+1e-6)
        if fluid in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col[fluid]] = i1.m.val_SI * (ho1 - hi1)/(self.Q.val+1e-6)
        
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = - o2.fluid.val[fluid] * (ho1 - hi1)/(self.Q.val+1e-6)
        if fluid in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col[fluid]] = - o2.m.val_SI * (ho1 - hi1)/(self.Q.val+1e-6)

        # for c in [i1,i2,o2,o1]:
        #     if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
        #         self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'p', c)
        #     if self.is_variable(c.h): #, increment_filter):
        #         self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'h', c)
        #     if self.is_variable(c.m): #, increment_filter):
        #         self.jacobian[k, c.m.J_col] = self.numeric_deriv(self.energy_balance_hot_func, 'm', c)
        # #for c in [i1,o2,o1]:
        #     for fluid in self.variable_fluids:
        #         if fluid in c.fluid.is_var:
        #             self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.energy_balance_hot_func, fluid, c)    


    def KPI_func(self):
        r"""
        Equation for total heat flow rate
        """

        i1 = self.inl[0]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid        

        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)
        m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        Q_evap = m_evap * (o1.fluid_data['Water']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
                          -i1.fluid_data['Water']['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state))

        Q = i1.m.val_SI*i1.fluid.val['Water']*self.KPI.val
        return (Q_evap - Q)/(Q+1e-6)

    def KPI_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.
        """

        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid  

        fluid = 'Water'
        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)

        ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
        hi1 = i1.fluid_data[fluid]['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state)                                                     

        # Q_evap = (i1.m.val_SI*i1.fluid.val[fluid] - o2.m.val_SI*o2.fluid.val[fluid]) * (ho1 - hi1)
        Q = i1.m.val_SI*i1.fluid.val['Water']*self.KPI.val

        if self.is_variable(i1.m):
            self.jacobian[k, i1.m.J_col] = (i1.fluid.val[fluid] - i1.m.val_SI*self.KPI.val) * (ho1 - hi1)/(Q+1e-6)
        if fluid in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col[fluid]] = (i1.m.val_SI - i1.fluid.val['Water']*self.KPI.val)  * (ho1 - hi1)/(Q+1e-6)
        
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = - o2.fluid.val[fluid] * (ho1 - hi1)/(Q+1e-6)
        if fluid in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col[fluid]] = - o2.m.val_SI * (ho1 - hi1)/(Q+1e-6)

    def calculate_td_log(self,T_i,T_wb,T_o):
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
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid 

        Ti2 = i2.calc_T(T0=i2.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)
        Twb = get_Twb(i2,Ti2)
        self.td_log.val = self.calculate_td_log(Ti2,Twb,To1)

        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        m_evap = i1.m.val_SI*i1.fluid.val['Water'] - o2.m.val_SI*o2.fluid.val['Water']
        Q_evap = m_evap * (o1.fluid_data['Water']['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
                          -i1.fluid_data['Water']['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state))

        Q = self.kA.val * self.td_log.val
        return (Q_evap - Q)/(Q+1e-6)

    def kA_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of heat transfer coefficient function.
        """
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid 

        fluid = 'Water'
        Ti1 = i1.calc_T(T0=i1.T.val_SI)
        To1 = o1.calc_T(T0=o1.T.val_SI)

        ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,To1,force_state=o1.force_state)
        hi1 = i1.fluid_data[fluid]['wrapper'].h_pT(i1.p.val_SI,Ti1,force_state=i1.force_state)                                                     

        # Q_evap = (i1.m.val_SI*i1.fluid.val[fluid] - o2.m.val_SI*o2.fluid.val[fluid]) * (ho1 - hi1)
        Q =  self.kA.val * self.td_log.val

        if self.is_variable(i1.m):
            self.jacobian[k, i1.m.J_col] = i1.fluid.val[fluid] * (ho1 - hi1)/(Q+1e-6)
        if fluid in i1.fluid.is_var:
            self.jacobian[k, i1.fluid.J_col[fluid]] = i1.m.val_SI * (ho1 - hi1)/(Q+1e-6)
        
        if self.is_variable(o2.m):
            self.jacobian[k, o2.m.J_col] = - o2.fluid.val[fluid] * (ho1 - hi1)/(Q+1e-6)
        if fluid in o2.fluid.is_var:
            self.jacobian[k, o2.fluid.J_col[fluid]] = - o2.m.val_SI * (ho1 - hi1)/(Q+1e-6)

        # i = self.inl[1]
        # o = self.outl[2]
        # if self.is_variable(i.m):
        #     self.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        # if self.is_variable(i.h):
        #     self.jacobian[k, i.h.J_col] = self.numeric_deriv(self.kA_func, 'h', i)
        # if self.is_variable(i.p):
        #     self.jacobian[k, i.p.J_col] = self.numeric_deriv(self.kA_func, 'p', i)
        # if self.is_variable(o.h):
        #     self.jacobian[k, o.h.J_col] = i.m.val_SI
        # c = self.outl[1]
        # if self.is_variable(c.p):
        #     self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.kA_func, 'p', c)
        # if self.is_variable(c.h):
        #     self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.kA_func, 'h', c)


    def WBeff_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.
        """
        i2 = self.inl[1]
        Ti2 = i2.calc_T(T0=i2.T.val_SI)
        T_wb = get_Twb(i2,Ti2)
        o2 = self.outl[0] # vapor
        To2 = o2.calc_T(T0=o2.T.val_SI)

        return (Ti2-To2) - (Ti2-T_wb)*self.WBeff.val
    
    def WBeff_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.
        """

        # let's just do numerical derivative of them all
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid        

        for c in [i1,i2,o1,o2]:
            if self.is_variable(c.m): #, increment_filter):
                self.jacobian[k, c.m.J_col] = self.numeric_deriv(self.WBeff_func, 'm', c)
            if self.is_variable(c.p): #, increment_filter): increment filter may detect no change on the wrong end 
                self.jacobian[k, c.p.J_col] = self.numeric_deriv(self.WBeff_func, 'p', c)
            if self.is_variable(c.h): #, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(self.WBeff_func, 'h', c)
            for fluid in self.variable_fluids:
                if fluid in c.fluid.is_var:
                    self.jacobian[k, c.fluid.J_col[fluid]] = self.numeric_deriv(self.WBeff_func, fluid, c)  

    def calc_parameters(self):
        super().calc_parameters()
        i = self.inl[0]

        i1 = self.inl[0]
        i2 = self.inl[1]  # air
        o1 = self.outl[0] # vapor
        o2 = self.outl[1] # liquid

        # m_air = i2.m.val_SI*i2.fluid.val['Air'] #i2.m.val_SI # *i2.fluid.val['Air']
        # Q_air = + m_air * (o1.fluid_data['Air']['wrapper'].h_pT(o1.p.val_SI,o1.T.val_SI,force_state=o1.force_state)
        #                   -i2.fluid_data['Air']['wrapper'].h_pT(i2.p.val_SI,i2.T.val_SI,force_state=i2.force_state))

        if not self.Q.is_set:
            fluid = 'Water'
            ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,o1.T.val_SI,force_state=o1.force_state)
            hi1 = i1.fluid_data[fluid]['wrapper'].h_pT(i1.p.val_SI,i1.T.val_SI,force_state=i1.force_state)                                                     
            self.Q.val = (i1.m.val_SI*i1.fluid.val[fluid] - o2.m.val_SI*o2.fluid.val[fluid]) * (ho1 - hi1)
        if not self.KPI.is_set:
            fluid = 'Water'
            ho1 = o1.fluid_data[fluid]['wrapper'].h_pT(o1.p.val_SI,o1.T.val_SI,force_state=o1.force_state)
            hi1 = i1.fluid_data[fluid]['wrapper'].h_pT(i1.p.val_SI,i1.T.val_SI,force_state=i1.force_state)                                                     
            self.KPI.val = (i1.m.val_SI*i1.fluid.val[fluid] - o2.m.val_SI*o2.fluid.val[fluid]) * (ho1 - hi1) / (i1.m.val_SI*i1.fluid.val['Water'])

        hmin = min([o.h.val_SI for o in self.outl])
        hmax = max([o.h.val_SI for o in self.outl])
        if abs(i.h.val_SI - hmin) >= abs(i.h.val_SI - hmax):
            self.deltaH.val = i.h.val_SI - hmin
        else:
            self.deltaH.val = i.h.val_SI - hmax

        Thot_in = i2.T.val_SI 
        Thot_out = o1.T.val_SI
        Tcold = get_Twb(i2,Thot_in)
        if not self.kA.is_set:
            ttd_u = Thot_in - Tcold
            ttd_l = Thot_out - Tcold
            if ttd_u < 0 or ttd_l < 0:
                self.td_log.val = np.nan
            elif ttd_l == ttd_u:
                self.td_log.val = ttd_l
            else:
                self.td_log.val = ((ttd_l - ttd_u) / np.log(ttd_l / ttd_u))
            self.kA.val = self.Q.val / self.td_log.val

        if not self.WBeff.is_set:
            self.WBeff.val = (Thot_in-Thot_out)/(Thot_in-Tcold)
            if self.WBeff.val > 1.0:
                TESPyComponentError("efficiency cannot be greater than 1.0, try increase air mass flow")            

        M_o = o1.fluid.val['Water']
        W_o = M_o/(1-M_o)
        Wmax = HAPropsSI('W','P',o1.p.val_SI,'T',o1.T.val_SI,'R',1)
        if W_o > Wmax:
            self.RH.val = W_o/Wmax*100
            TESPyComponentError("Relative humidity cannot be greater than 100 %, try increase air mass flow")            
        else:
            self.RH.val = 100 * HAPropsSI('R','P',o1.p.val_SI,'T',o1.T.val_SI,'W',W_o)                