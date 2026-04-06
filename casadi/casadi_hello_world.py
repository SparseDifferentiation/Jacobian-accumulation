import casadi as ca
x = ca.MX.sym("x")
print(ca.jacobian(ca.sin(x),x))
