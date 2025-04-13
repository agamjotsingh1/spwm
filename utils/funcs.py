
def carrier(a, w, t): # triangular wave
    return (a) - (2*a/(np.pi)*np.arccos(np.cos(w*t -(np.pi/2))))

def modulating(a, w, t): # sin wave
    return a*np.sin(w*t)

def pwm(t, wc, wm, ac, am):
    if (carrier(ac, wc, t) > modulating(am, wm, t) ):
        return 1
    else:
        return 0

def f(x, y, wc, wm, ac, am):
    if (carrier(ac, wc, x/wc) > modulating(am, wm, y/wm) ):
        return 1
    else:
        return 0
