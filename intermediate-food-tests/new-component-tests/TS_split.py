# %%

import logging


from tespy.components import HeatExchangerSimple, Source, Sink, Merge, Separator 
from tespy.tools import ComponentProperties
from tespy.connections import Connection
from tespy.networks import Network
import numpy as np

from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp


class DiabaticSimpleHeatExchanger(HeatExchangerSimple):

    @staticmethod
    def component():
        return 'diabatic simple heat exchanger'

    def get_variables(self):
        variables = super().get_variables()
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
        self.jacobian[k, 0, 0] = (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI)
        self.jacobian[k, 0, 2] = -self.inl[0].m.val_SI
        self.jacobian[k, 1, 2] = self.inl[0].m.val_SI
        # custom variable Q
        if self.Q_total.is_var:
            if self.Q_total.val < 0:
                self.jacobian[k, 2 + self.Q.var_pos, 0] = -1
            else:
                self.jacobian[k, 2 + self.Q.var_pos, 0] = -self.eta.val

        if self.eta.is_var:
            if self.Q_total.val < 0:
                self.jacobian[k, 2 + self.eta.var_pos, 0] = self.inl[0].m.val_SI * (
                self.outl[0].h.val_SI - self.inl[0].h.val_SI
                )
            else:
                self.jacobian[k, 2 + self.eta.var_pos, 0] = -self.Q_total.val

    def calc_parameters(self):
        super().calc_parameters()

        if self.eta.is_set:
            if self.Q.val < 0:
                self.Q_loss.val = self.Q.val * (1 - self.eta.val)
            else:
                self.Q_loss.val = -self.Q.val * (1 / self.eta.val - 1)

            self.Q_total.val = self.Q.val - self.Q_loss.val


class MergeWithPressureLoss(Merge):

    @staticmethod
    def component():
        return 'merge with pressure losses'

    def get_variables(self):
        variables = super().get_variables()
        variables["pr"] = dc_cp(
            min_val=0,
            deriv=self.pr_deriv,
            func=self.pr_func,
            latex=self.pr_func_doc,
            num_eq=1
        )
        return variables

    def get_mandatory_constraints(self):
        return {
            'mass_flow_constraints': {
                'func': self.mass_flow_func, 'deriv': self.mass_flow_deriv,
                'constant_deriv': True, 'latex': self.mass_flow_func_doc,
                'num_eq': 1},
            'fluid_constraints': {
                'func': self.fluid_func, 'deriv': self.fluid_deriv,
                'constant_deriv': False, 'latex': self.fluid_func_doc,
                'num_eq': self.num_nw_fluids},
            'energy_balance_constraints': {
                'func': self.energy_balance_func,
                'deriv': self.energy_balance_deriv,
                'constant_deriv': False, 'latex': self.energy_balance_func_doc,
                'num_eq': 1}
        }

    def pr_func(self):
        r"""
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """
        return self.inl[0].p.val_SI * self.pr.val - self.outl[0].p.val_SI

    def pr_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.
        """
        self.jacobian[k, 0, 1] = self.pr.val
        self.jacobian[k, self.num_i, 1] = -1

    def calc_parameters(self):

        self.pr.val = self.outl[0].p.val_SI / self.inl[0].p.val_SI
        for i in range(self.num_i):
            if self.inl[i].p.val < self.outl[0].p.val:
                msg = (
                    f"The pressure at inlet {i + 1} is lower than the pressure "
                    f"at the outlet of component {self.label}."
                )
                logging.warning(msg)

class SeparatorWithCompositionSplits(Separator):

    @staticmethod
    def component():
        return 'separator with species flow splits'

    def get_variables(self):
        variables = super().get_variables()
        variables["SFS"] = self.dc_cp_SFS(
            min_val=0,
            deriv=self.SFS_deriv,
            func=self.SFS_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        return variables

    class dc_cp_SFS(ComponentProperties):
        """
        Data container for component properties. 
        + SFS_fluid
        + SFS_outlet
        """
        @staticmethod
        def attr():
            """
            Return the available attributes for a ComponentProperties type object.

            Returns
            -------
            out : dict
                Dictionary of available attributes (dictionary keys) with default
                values.
            """
            return {
                'val': 1, 'val_SI': 0, 'is_set': False, 'd': 1e-4,
                'min_val': -1e12, 'max_val': 1e12, 'is_var': False,
                'val_ref': 1, 'design': np.nan, 'is_result': False,
                'num_eq': 0, 'func_params': {}, 'func': None, 'deriv': None,
                'latex': None, 'split_fluid' : str, 'split_outlet' : str}

    def SFS_func(self):
        r"""
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """
        # residual = []
        # for fluid, x in self.inl[0].fluid.val.items():
        #     res = x * self.inl[0].m.val_SI
        #     for o in self.outl:
        #         res -= o.fluid.val[fluid] * o.m.val_SI
        #     residual += [res]
        # return residual

        fluid = self.SFS.split_fluid
        out_i = int(self.SFS.split_outlet[3:]) - 1

        res = self.inl[0].fluid.val[fluid] * self.inl[0].m.val_SI * self.SFS.val \
            - self.outl[out_i].fluid.val[fluid] * self.outl[out_i].m.val_SI 

        #print(res)
        return res

    def SFS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.
        """

        # j=0 
        # self.jacobian[k, j, 0] = self.inl[j].fluid.val[self.split_fluid] * self.TS.val
        # self.jacobian[k, j, i + 3] = self.inl[j].m.val_SI * self.TS.val             

        # i = 0
        # for fluid, x in self.outl[0].fluid.val.items():
        #     j = 0
        #     for inl in self.inl:
        #         self.jacobian[k, j, 0] = inl.fluid.val[fluid]
        #         self.jacobian[k, j, i + 3] = inl.m.val_SI
        #         j += 1
        #     self.jacobian[k, j, 0] = -x
        #     self.jacobian[k, j, i + 3] = -self.outl[0].m.val_SI
        #     i += 1
        #     k += 1

        fluid_index = list(self.inl[0].fluid.val.keys()).index(self.SFS.split_fluid)
        fluid = self.SFS.split_fluid
        out_i = int(self.SFS.split_outlet[3:]) - 1

        i = fluid_index
        j = 0 
        self.jacobian[k, j, 0]     = self.inl[0].fluid.val[fluid] * self.SFS.val 
        self.jacobian[k, j, i + 3] = self.inl[0].m.val_SI * self.SFS.val 
        j = 1 + out_i 
        self.jacobian[k, j, 0]     = -self.outl[out_i].fluid.val[fluid]
        self.jacobian[k, j, i + 3] = -self.outl[out_i].m.val_SI 
        
        #print(self.jacobian)
        #print(self.jacobian[k,:,:])




# %%

# caution, must write "Water" (capital W) in INCOMP backend -> CoolProp bug? Intentional?
fluids = ["INCOMP::Water", "INCOMP::T66"]
nw = Network(fluids=fluids, p_unit="bar", T_unit="C")

so = Source("Source")
se = SeparatorWithCompositionSplits("Separator")
si1 = Sink("Sink 1")
si2 = Sink("Sink 2")

c1 = Connection(so, "out1", se, "in1", label="1")
c2 = Connection(se, "out1", si1, "in1", label="2")
c3 = Connection(se, "out2", si2, "in1", label="3")

nw.add_conns(c1, c2, c3)

# set some generic data for starting values
c1.set_attr(m=1, p=1.2, T=30, fluid={"Water": 0.9, "T66": 0.1})

# set compositions 
c2.set_attr(fluid={"Water": 0.85, "T66": 0.15})
# or others 
#c3.set_attr(fluid={"Water": 0.85, "T66": 0.15})
# or others
#c2.set_attr(fluid={"Water": 0.85})
#c3.set_attr(fluid={"Water": 1})

# This one produce error because T66 is 0 and the equation cannot be solved.. 
#c2.set_attr(fluid={"Water": 1, "T66": 0.0})

# specify this one to avoid using the species flow split : SFS  
#c2.set_attr(m=0.5)
# set the species flow split, specify the fluid and the outlet too.. (we might need some checks of this)
se.set_attr(SFS={
    'val': 0.6, 'is_set': True, 
    'split_fluid' : 'T66', 'split_outlet' : "out1"})

# add some guess values
c2.set_attr(m0=0.5,p0=1.2,h0=1e5,T0=50,fluid0={"Water": 0.5, "T66": 0.5})
c3.set_attr(m0=0.5,p0=1.2,h0=1e5,T0=50,fluid0={"Water": 0.5, "T66": 0.5})

nw.solve("design")
nw.print_results()

print(nw.results['Connection'])

m_T66_c1 = c1.m.val * c1.fluid.val['T66']
m_T66_c2 = c2.m.val * c2.fluid.val['T66']
print(f"Species flow split is {m_T66_c2/m_T66_c1}")

