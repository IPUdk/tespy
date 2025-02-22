import CoolProp as CP
from CoolProp.CoolProp import AbstractState

from tespy.tools.global_vars import ERR


class FluidPropertyWrapper:

    def __init__(self, fluid, back_end=None) -> None:
        self.back_end = back_end
        self.fluid = fluid
        if "[" in self.fluid:
            self.fluid, self._fractions = self.fluid.split("[")
            self._fractions = self._fractions.replace("]", "")
        else:
            self._fractions = None

    def _not_implemented(self) -> None:
        raise NotImplementedError(
            f"Method is not implemented for {self.__class__.__name__}."
        )

    def _is_below_T_critical(self, T):
        self._not_implemented()

    def _make_p_subcritical(self, p):
        self._not_implemented()

    def T_ph(self, p, h):
        self._not_implemented()

    def T_ps(self, p, s):
        self._not_implemented()

    def h_pT(self, p, T):
        self._not_implemented()

    def h_QT(self, Q, T):
        self._not_implemented()

    def s_QT(self, Q, T):
        self._not_implemented()

    def T_sat(self, p):
        self._not_implemented()

    def p_sat(self, T):
        self._not_implemented()

    def Q_ph(self, p, h):
        self._not_implemented()

    def d_ph(self, p, h):
        self._not_implemented()

    def d_pT(self, p, T):
        self._not_implemented()

    def d_QT(self, Q, T):
        self._not_implemented()

    def viscosity_ph(self, p, h):
        self._not_implemented()

    def viscosity_pT(self, p, T):
        self._not_implemented()

    def s_ph(self, p, h):
        self._not_implemented()

    def s_pT(self, p, T):
        self._not_implemented()


class CoolPropWrapper(FluidPropertyWrapper):

    def __init__(self, fluid, back_end=None) -> None:
        super().__init__(fluid, back_end)

        if self.back_end is None:
            self.back_end = "HEOS"

        self.AS = AbstractState(self.back_end, self.fluid)
        self._set_constants()

    def _set_constants(self):
        self._T_min = self.AS.trivial_keyed_output(CP.iT_min)
        self._T_max = self.AS.trivial_keyed_output(CP.iT_max)
        try:
            self._aliases = CP.CoolProp.get_aliases(self.fluid)
        except RuntimeError:
            self._aliases = [self.fluid]

        if self.back_end == "INCOMP":
            if self._fractions is not None:
                try:
                    self.AS.set_volu_fractions([float(self._fractions)])
                except ValueError:
                    self.AS.set_mass_fractions([float(self._fractions)])
            self._p_min = 1e2
            self._p_max = 1e8
            self._p_crit = 1e8
            self._T_crit = None
            self._molar_mass = 1
            try:
                # how to know that we have a binary mixture?
                self._T_min = self.AS.trivial_keyed_output(CP.iT_freeze)
            except ValueError:
                pass
        else:
            self._p_min = self.AS.trivial_keyed_output(CP.iP_min)
            self._p_max = self.AS.trivial_keyed_output(CP.iP_max)
            self._p_crit = self.AS.trivial_keyed_output(CP.iP_critical)
            self._T_crit = self.AS.trivial_keyed_output(CP.iT_critical)
            self._molar_mass = self.AS.trivial_keyed_output(CP.imolar_mass)

    def _is_below_T_critical(self, T):
        return T < self._T_crit

    def _make_p_subcritical(self, p):
        if p > self._p_crit:
            p = self._p_crit * 0.99
        return p

    def get_T_max(self, p):
        if self.back_end == "INCOMP":
            return self.T_sat(p)
        else:
            return self._T_max

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.T()

    def T_ps(self, p, s):
        self.AS.update(CP.PSmass_INPUTS, p, s)
        return self.AS.T()

    def h_pQ(self, p, Q):
        self.AS.update(CP.PQ_INPUTS, p, Q)
        return self.AS.hmass()

    def h_ps(self, p, s):
        self.AS.update(CP.PSmass_INPUTS, p, s)
        return self.AS.hmass()

    def h_pT(self, p, T):
        if self.back_end == "INCOMP":
            if T == (self._T_max + self._T_min) / 2:
                T += ERR
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.hmass()

    def h_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.hmass()

    def s_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.smass()

    def T_sat(self, p):
        p = self._make_p_subcritical(p)
        self.AS.update(CP.PQ_INPUTS, p, 0)
        return self.AS.T()

    def p_sat(self, T):
        if T > self._T_crit:
            T = self._T_crit * 0.99

        self.AS.update(CP.QT_INPUTS, 0, T)
        return self.AS.p()

    def Q_ph(self, p, h):
        p = self._make_p_subcritical(p)
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.Q()

    def d_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.rhomass()

    def d_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.rhomass()

    def d_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.rhomass()

    def viscosity_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.viscosity()

    def viscosity_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.viscosity()

    def s_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.smass()

    def s_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.smass()


class IAPWSWrapper(FluidPropertyWrapper):


    def __init__(self, fluid, back_end=None) -> None:
        # avoid unncessary loading time if not used
        try:
            import iapws
        except ModuleNotFoundError:
            msg = (
                "To use the iapws fluid properties you need to install "
                "iapws."
            )
            raise ModuleNotFoundError(msg)

        super().__init__(fluid, back_end)
        self._aliases = CP.CoolProp.get_aliases("H2O")

        if self.fluid not in self._aliases:
            msg = "The iapws wrapper only supports water as fluid."
            raise ValueError(msg)

        if self.back_end is None:
            self.back_end = "IF97"

        if self.back_end == "IF97":
            self.AS = iapws.IAPWS97
        elif self.back_end == "IF95":
            self.AS = iapws.IAPWS95
        else:
            msg = f"The specified back_end {self.back_end} is not available."
            raise NotImplementedError(msg)
        self._set_constants(iapws)

    def _set_constants(self, iapws):
        self._T_min = iapws._iapws.Tt
        self._T_max = 2000
        self._p_min = iapws._iapws.Pt * 1e6
        self._p_max = 100e6
        self._p_crit = iapws._iapws.Pc * 1e6
        self._T_crit = iapws._iapws.Tc
        self._molar_mass = iapws._iapws.M

    def _is_below_T_critical(self, T):
        return T < self._T_crit

    def _make_p_subcritical(self, p):
        if p > self._p_crit:
            p = self._p_crit * 0.99
        return p

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        return self.AS(h=h / 1e3, P=p / 1e6).T

    def T_ps(self, p, s):
        self.AS.update(CP.PSmass_INPUTS, p, s)
        return self.AS.T()

    def h_pQ(self, p, Q):
        self.AS.update(CP.PQ_INPUTS, p, Q)
        return self.AS.hmass()

    def h_ps(self, p, s):
        return self.AS(P=p / 1e6, s=s / 1e3).h * 1e3

    def h_pT(self, p, T):
        return self.AS(P=p / 1e6, T=T).h * 1000

    def h_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.hmass()

    def s_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.smass()

    def T_sat(self, p):
        p = self._make_p_subcritical(p)
        return self.AS(P=p / 1e6, x=0).T

    def p_sat(self, T):
        if T > self._T_crit:
            T = self._T_crit * 0.99

        self.AS.update(CP.QT_INPUTS, 1, T)
        return self.AS.p()

    def Q_ph(self, p, h):
        p = self._make_p_subcritical(p)
        return self.AS(h=h / 1e3, P=p / 1e6).x

    def d_ph(self, p, h):
        return self.AS(h=h / 1e3, P=p / 1e6).rho

    def d_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.rhomass()

    def d_QT(self, Q, T):
        self.AS.update(CP.QT_INPUTS, Q, T)
        return self.AS.rhomass()

    def viscosity_ph(self, p, h):
        self.AS.update(CP.HmassP_INPUTS, h, p)
        return self.AS.viscosity()

    def viscosity_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.viscosity()

    def s_ph(self, p, h):
        return self.AS(P=p / 1e6, h=h / 1e3).s * 1e3

    def s_pT(self, p, T):
        self.AS.update(CP.PT_INPUTS, p, T)
        return self.AS.smass()


class PyromatWrapper(FluidPropertyWrapper):

    def __init__(self, fluid, back_end=None) -> None:
        # avoid unnecessary loading time if not used
        try:
            import pyromat as pm
            pm.config['unit_energy'] = "J"
            pm.config['unit_pressure'] = "Pa"
            pm.config['unit_molar'] = "mol"
        except ModuleNotFoundError:
            msg = (
                "To use the pyromat fluid properties you need to install "
                "pyromat."
            )
            raise ModuleNotFoundError(msg)

        super().__init__(fluid, back_end)
        self._create_AS(pm)
        self._set_constants()

    def _create_AS(self, pm):
        self.AS = pm.get(f"{self.back_end}.{self.fluid}")

    def _set_constants(self):
        self._p_min, self._p_max = 100, 1000e5
        self._T_min, self._T_max = self.AS.Tlim()
        self._molar_mass = self.AS.mw()

    def isentropic(self, p_1, h_1, p_2):
        return self.h_ps(p_2, self.s_ph(p_1, h_1))

    def T_ph(self, p, h):
        return self.AS.T(p=p, h=h)[0]

    def T_ps(self, p, s):
        return self.AS.T(p=p, s=s)[0]

    def h_pT(self, p, T):
        return self.AS.h(p=p, T=T)[0]

    def T_ph(self, p, h):
        return self.AS.T(p=p, h=h)[0]

    def T_ps(self, p, s):
        return self.AS.T(p=p, s=s)[0]

    def h_pT(self, p, T):
        return self.AS.h(p=p, T=T)[0]

    def h_ps(self, p, s):
        return self.AS.h(p=p, s=s)[0]

    def d_ph(self, p, h):
        return self.AS.d(p=p, h=h)[0]

    def d_pT(self, p, T):
        return self.AS.d(p=p, T=T)[0]

    def s_ph(self, p, h):
        return self.AS.s(p=p, h=h)[0]

    def s_pT(self, p, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.s(p=p, T=T)[0]

    def h_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.h(x=Q, T=T)[0]

    def s_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.s(x=Q, T=T)[0]

    def T_boiling(self, p):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.T(x=1, p=p)[0]

    def p_boiling(self, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.p(x=1, T=T)[0]

    def Q_ph(self, p, h):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.x(p=p, h=h)[0]

    def d_QT(self, Q, T):
        if self.back_end == "ig":
            self._not_implemented()
        return self.AS.d(x=Q, T=T)[0]
