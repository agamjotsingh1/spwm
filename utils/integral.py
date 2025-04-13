'''
a -> b: integration limits for x 
c -> d: integration limits for y 
f: function of two variables, f(x, y)
'''
def double_integral(a, b, c, d, f, precision = 1e-3):
    x_vals = np.arange(a, b + precision, precision)
    y_vals = np.arange(c, d + precision, precision)

    # Integrate f(x, y) over x for each y
    inner_integral = [
        np.trapezoid([f(x, y) for x in x_vals], dx=precision)
        for y in y_vals
    ]

    # Integrate the result over y
    return np.trapezoid(inner_integral, dx=precision)
