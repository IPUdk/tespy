import logging

from tespy.components import HeatExchangerSimple, Merge, Separator, Splitter
from tespy.tools.data_containers import ComponentProperties as dc_cp
from tespy.tools.data_containers import ComponentPropertiesArray as dc_cpa
from tespy.tools.data_containers import GroupedComponentProperties as dc_gcp
from tespy.tools.fluid_properties import T_mix_ph

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

class SeparatorWithSpeciesSplits(Separator):

    @staticmethod
    def component():
        return 'separator with species flow splits'

    def get_variables(self):
        variables = super().get_variables()
        variables["SFS"] = dc_cp_SFS(
            min_val=0,
            deriv=self.SFS_deriv,
            func=self.SFS_func,
            latex=self.pr_func_doc,
            num_eq=1,
        )
        variables["Q"] = dc_cp(is_result=True)       
        variables["Qout"] = dc_cpa()       
        return variables

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

    def calc_parameters(self):
        super().calc_parameters()

        self.Qout.val = []
        for o in self.outl:
            self.Qout.val += [o.m.val_SI * (o.h.val_SI - self.inl[0].h.val_SI)]        

        self.Q.val = np.sum(self.Qout.val)


class SeparatorWithSpeciesSplitsAndDeltaT(SeparatorWithSpeciesSplits):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT on outlets'

    def get_variables(self):
        variables = super().get_variables()
        variables["deltaT"] = dc_cpa(
            deriv=self.energy_balance_deriv, # same as before
            func=self.energy_balance_deltaT_func,
            latex=self.pr_func_doc
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
            # 'energy_balance_constraints': {
            #     'func': self.energy_balance_func,
            #     'deriv': self.energy_balance_deriv,
            #     'constant_deriv': False, 'latex': self.energy_balance_func_doc,
            #     'num_eq': self.num_o},
            'pressure_constraints': {
                'func': self.pressure_equality_func,
                'deriv': self.pressure_equality_deriv,
                'constant_deriv': True,
                'latex': self.pressure_equality_func_doc,
                'num_eq': self.num_i + self.num_o - 1}
        }

    def energy_balance_deltaT_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : list
            Residual value of energy balance.

            .. math::

                0 = T_{in} - T_{out,j}\\
                \forall j \in \text{outlets}
        """
        residual = []
        T_in = T_mix_ph(self.inl[0].get_flow(), T0=self.inl[0].T.val_SI)
        i=0
        for o in self.outl:
            residual += [T_in + self.deltaT.val[i] - T_mix_ph(o.get_flow(), T0=o.T.val_SI)]
            i+=1
        return residual

class SplitWithFlowSplitter(Splitter):

    @staticmethod
    def component():
        return 'splitter with flow split ratios'

    def get_variables(self):
        variables = super().get_variables()
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
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """

        out_i = int(self.FS.split_outlet[3:]) - 1
        res = self.inl[0].m.val_SI * self.FS.val - self.outl[out_i].m.val_SI 

        #print(res)
        return res

    def FS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.
        """

        out_i = int(self.FS.split_outlet[3:]) - 1

        j = 0 
        self.jacobian[k, j, 0]     = self.FS.val 
        j = 1 + out_i 
        self.jacobian[k, j, 0]     = -1
        
        #print(self.jacobian)
        #print(self.jacobian[k,:,:])


class SeparatorWithSpeciesSplitsAndDeltaTAndPr(SeparatorWithSpeciesSplits):

    @staticmethod
    def component():
        return 'separator with species flow splits and dT on outlets'

    def get_variables(self):
        variables = super().get_variables()
        variables["deltaT"] = dc_cpa(
            deriv=self.energy_balance_deriv, # same as before
            func=self.energy_balance_deltaT_func,
            latex=self.pr_func_doc
        )
        variables["pr"] = dc_cp(
            min_val=0,
            deriv=self.pr_deriv,
            func=self.pr_func,
            latex=self.pr_func_doc,
            num_eq=1
        )        
        return variables

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
        super().calc_parameters()
        self.pr.val = self.outl[0].p.val_SI / self.inl[0].p.val_SI
        for i in range(self.num_i):
            if self.inl[i].p.val < self.outl[0].p.val:
                msg = (
                    f"The pressure at inlet {i + 1} is lower than the pressure "
                    f"at the outlet of component {self.label}."
                )
                logging.warning(msg)
        

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
            # 'energy_balance_constraints': {
            #     'func': self.energy_balance_func,
            #     'deriv': self.energy_balance_deriv,
            #     'constant_deriv': False, 'latex': self.energy_balance_func_doc,
            #     'num_eq': self.num_o},
            # 'pressure_constraints': {
            #     'func': self.pressure_equality_func,
            #     'deriv': self.pressure_equality_deriv,
            #     'constant_deriv': True,
            #     'latex': self.pressure_equality_func_doc,
            #     'num_eq': self.num_i + self.num_o - 1}
        }

    def energy_balance_deltaT_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : list
            Residual value of energy balance.

            .. math::

                0 = T_{in} - T_{out,j}\\
                \forall j \in \text{outlets}
        """
        residual = []
        T_in = T_mix_ph(self.inl[0].get_flow(), T0=self.inl[0].T.val_SI)
        i=0
        for o in self.outl:
            residual += [T_in + self.deltaT.val[i] - T_mix_ph(o.get_flow(), T0=o.T.val_SI)]
            i+=1
        return residual

class SplitWithFlowSplitter(Splitter):

    @staticmethod
    def component():
        return 'splitter with flow split ratios'

    def get_variables(self):
        variables = super().get_variables()
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
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """

        out_i = int(self.FS.split_outlet[3:]) - 1
        res = self.inl[0].m.val_SI * self.FS.val - self.outl[out_i].m.val_SI 

        #print(res)
        return res

    def FS_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.
        """

        out_i = int(self.FS.split_outlet[3:]) - 1

        j = 0 
        self.jacobian[k, j, 0]     = self.FS.val 
        j = 1 + out_i 
        self.jacobian[k, j, 0]     = -1
        
        #print(self.jacobian)
        #print(self.jacobian[k,:,:])


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
        attributes.update({'split_fluid' : str, 'split_outlet' : str})
        return attributes

class dc_cp_FS(dc_cp):
    """
    Data container for component properties. 
    + FS_outlet
    """
    @staticmethod
    def attr():
        attributes = dc_cp.attr()
        attributes.update({'split_outlet' : str})
        return attributes
