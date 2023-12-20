import scipy.special
import numpy as np
import mpmath
from matplotlib import pyplot
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class Gamma:
    """Creates a gamma distribution. mean is the mean of the initial distribution,
    and k is the shape parameter."""
    def __init__(self, mean, k):
        self.mean0 = mean
        self.k = k

    def __call__(self, epsilon):
        Ns = 1000.
        return Ns * epsilon**(self.k - 1.) * np.exp(-self.k * epsilon)

    def get_ebar(self, ki, C):
        """Returns the mean value of the distribution when a cumulative infection
        pressure of C has occured, with rate constant ki."""
        return 1. / (1. / self.mean0 + ki * C / self.k)

    def get_distribution(self, ki, C, epsilon, xS=1.):
        """Returns the value of the distribution at value epsilon,
        given the cumulative infection pressure of C and infection rate ki.
        if xS=1., returns the normalized distribution; you can alternately
        provide xS to see the shrinking distribution."""
        k, mean0 = self.k, self.mean0
        nS = (k / mean0 + ki * C)**k
        nS *= epsilon**(k - 1)
        nS *= np.exp(- (k / mean0 + ki * C) * epsilon)
        nS /= scipy.special.gamma(k)
        return nS


class Beta:
    """Creates callable beta function with alpha, beta values specified."""
    def __init__(self, alpha, beta, Ns=1000.):
        self.alpha = alpha
        self.beta = beta
        self.Ns = Ns
        mean = alpha / (alpha + beta)
        self.scale = 1. / mean  # Will be used to set the initial mean to 1.
        self.max_value = self.scale  # Upper bound.
        self.mean = mean

    def __call__(self, x):
        alpha, beta, scale = self.alpha, self.beta, self.scale
        y = x / scale
        pdf = y**(alpha - 1.) * (1. - y)**(beta - 1.)
        pdf *= scipy.special.beta(alpha, beta)
        pdf /= scale
        return pdf * self.Ns

    def getK(self, phi):
        """Returns the apparent order paramter K at a given "time" phi."""
        alpha, beta, scale = self.alpha, self.beta, self.scale
        num = 1. + alpha
        num *= reghyp1f1(alpha, alpha + beta, -phi * scale)
        num *= reghyp1f1(2. + alpha, 2. + alpha + beta, -phi * scale)
        den = alpha * reghyp1f1(1. + alpha, 1. + alpha + beta, -phi * scale)**2
        K = -1 + num / den
        return K

    def get_ebar(self, ki, C):
        """Returns the mean value of the distribution when a cumulative infection
        pressure of C has occured, with rate constant ki."""
        phi = ki * C
        alpha, beta, scale = self.alpha, self.beta, self.scale
        epbar = alpha * reghyp1f1(1. + alpha, 1. + alpha + beta, -phi * scale)
        epbar /= reghyp1f1(alpha, alpha + beta, -phi * scale)
        epbar *= scale
        return epbar


def reghyp1f1(a, b, z):
    """Regularized Hypergeometric 1F1 function."""
    return float(mpmath.hyp1f1(a, b, z) / mpmath.gamma(b))


def get_power_results(power, ki, kd, times, xI0):
    """Perform the numerical integration with the power-law model,
    with power being the order of the rate term with respect to xS."""
    dydt = MakePower_dydt(ki, kd, power)
    answer = solve_ivp(fun=dydt, t_span=(times[0], times[-1]),
                       y0=[1. - xI0, xI0], dense_output=True)
    return answer.sol(times)


class MakePower_dydt:
    """Makes a callable function to provide dydt for our system of equation,
    under the power-law model, where power is typically 1 or 2 to indicate
    the order of the rate with respect to xS."""
    def __init__(self, ki, kd, power):
        self.ki = ki
        self.kd = kd
        self.power = power

    def __call__(self, t, y):
        xS, xI = y
        rate = self.ki * xS**self.power * xI
        dxS_dt = - rate
        dxI_dt = rate - self.kd * xI
        return [dxS_dt, dxI_dt]


def get_gamma_results(k, ki, kd, times, xI0):
    """Perform the numerical integration with the gamma distribution,
    with shape parameter k."""
    distribution = Gamma(mean=1., k=k)
    dydt = MakeDistro_dydt(ki, kd, distribution.get_ebar)
    answer = solve_ivp(fun=dydt, t_span=(times[0], times[-1]),
                       y0=[1. - xI0, xI0, 0.], dense_output=True)
    return answer.sol(times)


def get_beta_results(alpha, beta, ki, kd, times, xI0):
    """Perform the numerical integration with the uniform distribution."""
    distribution = Beta(alpha=alpha, beta=beta)
    dydt = MakeDistro_dydt(ki, kd, distribution.get_ebar)
    answer = solve_ivp(fun=dydt, t_span=(times[0], times[-1]),
                       y0=[1. - xI0, xI0, 0.], dense_output=True)
    return answer.sol(times)


class MakeDistro_dydt:
    """Makes a callable function to provide dydt for our system of equations.
    Here, I'm using ki and kd to indicate the infection and decay rate constant
    (to avoid using confusing ourselves with two things called gamma!).
    get_ebar is a function that returns the current mean value of the
    distribution given ki and C, where C is the cumulative infection pressure.
    """
    def __init__(self, ki, kd, get_ebar):
        self.ki = ki
        self.kd = kd
        self.get_ebar = get_ebar

    def __call__(self, t, y):
        xS, xI, C = y
        ebar = self.get_ebar(self.ki, C)
        rate = self.ki * ebar * xI * xS
        dxS_dt = - rate
        dxI_dt = rate - self.kd * xI
        dC_dt = xI
        return [dxS_dt, dxI_dt, dC_dt]


fig, axes = pyplot.subplots(ncols=2, figsize=(8., 4.))
fig.subplots_adjust(left=0.08, right=0.99, top=0.80)
dax, kax = axes
linestyles = (':', '--', '-.', (0, (3, 1, 1, 1)))

phis = np.linspace(0., 8., num=1000)

# Dynamics plot.

ki = 2.  # R0
kd = 1.  # making this unitless
xI0 = 0.0001  # initial infected population; rest assumed susceptible
times = np.linspace(0., 20., num=1000)

# Power models.
power = 1.
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=6., color='0.8', label='1st-order (singular)')
dax.text(times[-1] + 1.7, 1. - ys[0, -1],
         r'$\beta I S^{:g}$'.format(power), va='center')
power = 1.5
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=6., color='0.8', label='2nd-order')
dax.text(times[-1] + 1.7, 1. - ys[0, -1],
         r'$\beta I S^{{{:g}}}$'.format(power), va='center')
power = 2.
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=6., color='0.8', label='3rd-order')
dax.text(times[-1] + 1.7, 1. - ys[0, -1],
         r'$\beta I S^{:g}$'.format(power), va='center')
power = 3.
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=6., color='0.8', label='4th-order')
dax.text(times[-1] + 1.7, 1. - ys[0, -1],
         r'$\beta I S^{:g}$'.format(power), va='center')
power = 5.
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=6., color='0.8', label='5th-order')
dax.text(times[-1] + 1.7, 1. - ys[0, -1],
         r'$\beta I S^{:g}$'.format(power), va='center')

# Homogeneous model.
power = 1.
ys = get_power_results(power, ki, kd, times, xI0)
dax.plot(times, 1. - ys[0], lw=1., color='k', label='homogeneous')

final_phis = {}

# Gamma models.
kax.set_prop_cycle(None)
k = 0.25
ys = get_gamma_results(k, ki, kd, times, xI0)
final_phis['gamma,{:g}'.format(k)] = ys[-1][-1] * ki
dax.plot(times, 1. - ys[0], label='gamma k={:g}'.format(k))
k = 0.5
ys = get_gamma_results(k, ki, kd, times, xI0)
final_phis['gamma,{:g}'.format(k)] = ys[-1][-1] * ki
dax.plot(times, 1. - ys[0], label='gamma k={:g}'.format(k))
k = 0.5
k = 1.
ys = get_gamma_results(k, ki, kd, times, xI0)
final_phis['gamma,{:g}'.format(k)] = ys[-1][-1] * ki
dax.plot(times, 1. - ys[0], label='gamma k={:g}'.format(k))
k = 2.
ys = get_gamma_results(k, ki, kd, times, xI0)
final_phis['gamma,{:g}'.format(k)] = ys[-1][-1] * ki
dax.plot(times, 1. - ys[0], label='gamma k={:g}'.format(k))

# Beta model.
dax.set_prop_cycle(None)
linestyle = iter(linestyles)

a, b = 0.25, 35.
ys = get_beta_results(a, b, ki, kd, times, xI0)
final_phis['beta,{:g},{:g}'.format(a, b)] = ys[-1][-1] * ki
curve = dax.plot(times, 1. - ys[0], linestyle=next(linestyle),
                 label='beta a={:g},b={:g}'.format(a, b))[0]
dax.plot(times[-1], 1. - ys[0][-1], '.', color=curve.get_color())

a, b = 0.5, .5
ys = get_beta_results(a, b, ki, kd, times, xI0)
final_phis['beta,{:g},{:g}'.format(a, b)] = ys[-1][-1] * ki
curve = dax.plot(times, 1. - ys[0], linestyle=next(linestyle),
                 label='beta a={:g},b={:g}'.format(a, b))[0]
dax.plot(times[-1], 1. - ys[0][-1], '.', color=curve.get_color())

a, b = 1., 1.
ys = get_beta_results(a, b, ki, kd, times, xI0)
final_phis['beta,{:g},{:g}'.format(a, b)] = ys[-1][-1] * ki
curve = dax.plot(times, 1. - ys[0], linestyle=next(linestyle),
                 label='beta a={:g},b={:g}'.format(a, b))[0]
dax.plot(times[-1], 1. - ys[0][-1], '.', color=curve.get_color())

a, b = 2., 2.
ys = get_beta_results(a, b, ki, kd, times, xI0)
final_phis['beta,{:g},{:g}'.format(a, b)] = ys[-1][-1] * ki
curve = dax.plot(times, 1. - ys[0], linestyle=next(linestyle),
                 label='beta a={:g},b={:g}'.format(a, b))[0]
dax.plot(times[-1], 1. - ys[0][-1], '.', color=curve.get_color())

dax.set_ylabel('cumulative infected ($I + R$)')
dax.set_xlabel(r'time, dimensionless ($t/\gamma$)')

dax.set_xlim(0., 25.)
dax.set_ylim(bottom=0.)

linestyle = iter(linestyles)
betacurves = []

a, b = .25, 35.
beta = Beta(alpha=a, beta=b)
Ks = [beta.getK(phi) for phi in phis]
curve = kax.plot(phis, Ks, linestyle=next(linestyle),
                 label='Beta: $a$={:g},$b$={:g} (gamma-like)'.format(a, b))[0]
betacurves += [curve]
f = interp1d(phis, Ks)
final_phi = final_phis['beta,{:g},{:g}'.format(a, b)]
kax.plot(final_phi, f(final_phi), '.', color=curve.get_color())
kax.annotate('gamma-like',
             xy=(0.1, f(0.1)), xycoords='data',
             xytext=(0.1 - 0.10, f(0.1) - 0.8), textcoords='data',
             arrowprops={'arrowstyle': '->',
                         'color': curve.get_color(),
                         'connectionstyle': 'arc3,rad=0.3'})

a, b = .5, .5
beta = Beta(alpha=a, beta=b)
Ks = [beta.getK(phi) for phi in phis]
curve = kax.plot(phis, Ks, linestyle=next(linestyle),
                 label='Beta: $a$={:g},$b$={:g} (bimodal)'.format(a, b))[0]
betacurves += [curve]
f = interp1d(phis, Ks)
final_phi = final_phis['beta,{:g},{:g}'.format(a, b)]
kax.plot(final_phi, f(final_phi), '.', color=curve.get_color())
kax.annotate('bimodal',
             xy=(1.8, f(1.8)), xycoords='data',
             xytext=(1.8 - 0.10, f(1.8) - 0.5), textcoords='data',
             arrowprops={'arrowstyle': '->',
                         'color': curve.get_color(),
                         'connectionstyle': 'arc3,rad=0.3'})

a, b = 1., 1.
beta = Beta(alpha=a, beta=b)
Ks = [beta.getK(phi) for phi in phis]
curve = kax.plot(phis, Ks, linestyle=next(linestyle),
                 label='Beta: $a$={:g},$b$={:g} (uniform)'.format(a, b))[0]
betacurves += [curve]
f = interp1d(phis, Ks)
final_phi = final_phis['beta,{:g},{:g}'.format(a, b)]
kax.plot(final_phi, f(final_phi), '.', color=curve.get_color())
kax.annotate('uniform',
             xy=(2.5, f(2.1)), xycoords='data',
             xytext=(2.1 + 1.80, f(2.1) - 0.1), textcoords='data',
             arrowprops={'arrowstyle': '->',
                         'color': curve.get_color(),
                         'connectionstyle': 'arc3,rad=-0.1'})

a, b = 2., 2.
beta = Beta(alpha=a, beta=b)
Ks = [beta.getK(phi) for phi in phis]
curve = kax.plot(phis, Ks, linestyle=next(linestyle),
                 label='Beta: $a$={:g},$b$={:g} (unimodal)'.format(a, b))[0]
betacurves += [curve]
f = interp1d(phis, Ks)
final_phi = final_phis['beta,{:g},{:g}'.format(a, b)]
kax.plot(final_phi, f(final_phi), '.', color=curve.get_color())
kax.annotate('unimodal',
             xy=(2.1 + 0.3, f(2.1)), xycoords='data',
             xytext=(2.1 + 1.80, f(2.1) - 0.25), textcoords='data',
             arrowprops={'arrowstyle': '->',
                         'color': curve.get_color(),
                         'connectionstyle': 'arc3,rad=-0.1'})

# Gammas.
kax.set_prop_cycle(None)
gammacurves = []
k = 0.25
c = kax.plot([phis[0], phis[-1]], [1. / k]*2, '-',
             label='Gamma: $k$={:g}'.format(k))[0]
gammacurves += [c]
k = 0.5
c = kax.plot([phis[0], phis[-1]], [1. / k]*2, '-',
             label='Gamma: $k$={:g}'.format(k))[0]
gammacurves += [c]
k = 1.
c = kax.plot([phis[0], phis[-1]], [1. / k]*2, '-',
             label='Gamma: $k$={:g}'.format(k))[0]
gammacurves += [c]
k = 2.
c = kax.plot([phis[0], phis[-1]], [1. / k]*2, '-',
             label='Gamma: $k$={:g}'.format(k))[0]
gammacurves += [c]

# Homogeneous.
kax.plot([phis[0], phis[-1]], [0.]*2, '-k')
kax.annotate('homogeneous',
             xy=(1., 0.), xycoords='data',
             xytext=(2.1,  -0.4), textcoords='data',
             arrowprops={'arrowstyle': '->',
                         'color': 'k',
                         'connectionstyle': 'arc3,rad=-0.2'})

kax.set_ylabel(r'$\mathcal{K}(t)$, general order parameter')
kax.set_xlabel(r'$\phi(t)$, progress variable')
kax.set_ylim(-0.7, 4.5)
kax.set_yticks(range(0, 5))

dax.text(0.07, 0.95, 'A', fontsize=12., weight='bold', transform=dax.transAxes,
         ha='center', va='center')
kax.text(0.07, 0.95, 'B', fontsize=12., weight='bold', transform=kax.transAxes,
         ha='center', va='center')


# Custom legend.
def add_entry(column, y, label, style, color, lw=1.):
    lax.plot([column, column + 0.2], [y]*2, style, color=color, lw=lw)
    lax.text(column + 0.25, y, label, va='center', fontsize=9.)


lax = fig.add_axes((0.1, 0.83, 0.8, 0.99 - 0.83))
y = 0.
add_entry(0., y, 'homogeneous', '-', 'k')
y -= 1.
add_entry(0., y, 'power law', '-', '0.8', 6.)
y = 1.
for curve in betacurves:
    y -= 1.
    add_entry(1.7, y, curve.get_label(), curve.get_linestyle(),
              curve.get_color(), lw=2.0)
y = 1.
for curve in gammacurves:
    y -= 1.
    add_entry(0.8, y, curve.get_label(), curve.get_linestyle(),
              curve.get_color(), lw=2.0)

lax.set_xlim(-0.1, 3.)
lax.set_ylim(-3.5, 0.5)
lax.set_xticks([])
lax.set_yticks([])
for spine in lax.spines.values():
    spine.set_edgecolor('0.8')

fig.savefig('fig3.pdf')
