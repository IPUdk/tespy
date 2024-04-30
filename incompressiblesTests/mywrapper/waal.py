import sympy as sp

# Define symbols
x = sp.Symbol('x')
p, R, T, a, b, c  = sp.symbols('p R T a b c')
V = 1 / x

# Define the equation after substitution
eq = R*T/(V-b) - a/V**2 - p
# eq_sub = eq.subs(V, 1/x)
eq = V**3 - a*V**2 + b*V - c 
#eq = V**3 - (b+R*T/p)*V**2 + a/p*V - a*b/p 
# https://www.quora.com/From-the-Van-der-Waals-equation-how-can-I-make-V-the-subject-of-the-formula
# Solve the transformed equation
solutions = sp.solve(eq, V)
#solutions = sp.solve(eq_sub, x)
solutions_V = [1 / sol for sol in solutions]  # Convert back to V
print(solutions_V)


import sympy as sp

# Define symbols
D, V, T, a, b, R = sp.symbols('D V T a b R')

# Define the van der Waals equation of state
P = R * T / (V - b) - a / V**2

# Differentiate with respect to volume
dP_dT = sp.diff(P, T)
print("dP/dT =", dP_dT)

# Differentiate with respect to volume
dP_dV = sp.diff(P, V)
print("dP/dV =", dP_dV)

Vint = sp.integrate(T*dP_dT - P,V)

P = R * T / (V - b) - a / V**2
_P = P.subs(V,1/D)
_dP_dT = sp.diff(_P, T)
Dint = sp.integrate(1/D**2*(_P-T*_dP_dT), D)


P = R * T / V
_P = P.subs(V,1/D)
_dP_dT = sp.diff(_P, T)
Dint = sp.integrate(1/D**2*(_P-T*_dP_dT), D)



T=300
G = [4.6e4,1.011249e3,8.3893e-1,-2.19989e-4,2.466190e-7,-9.7047e-11]
cv = sum([G[i]*T**(i+1-2) for i in range(6)])
