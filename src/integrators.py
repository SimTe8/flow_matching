# integrators to solve the ODEs defined by the vector field


def euler(x, v_func, t, h):
    v = v_func(x, t)
    return x + h * v


def midpoint(y, f, t, h):
    k1 = f(y, t)
    k2 = f(y + h / 2 * k1, t + h / 2)
    y_new = y + h * k2
    return y_new


def rk2(y, f, t, h):
    k1 = f(y, t)
    k2 = f(y + h * k1, t + h)
    y_new = y + h / 2 * (k1 + k2)
    return y_new


def rk4(y, f, t, h):
    k1 = f(y, t)
    k2 = f(y + h / 2 * k1, t + h / 2)
    k3 = f(y + h / 2 * k2, t + h / 2)
    k4 = f(y + h * k3, t + h)
    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new
