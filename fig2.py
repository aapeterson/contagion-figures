#!/usr/bin/env python

import mpmath
import numpy as np
import scipy.special
from scipy.integrate import trapz
from matplotlib import pyplot


class Evolution:
    """Evolves a distribution with chi.
    The initial distribution must be given by initial_distribution
    which takes in epsilon and outputs the distribution value.
    """
    def __init__(self, initial_distribution, beta=1.):
        self.dist0 = initial_distribution
        self.beta = beta

    def __call__(self, epsilon, chi):
        """Returns the value of the evolved distribution at
        'time' chi and value epsilon."""
        fxn = self.dist0(epsilon)
        exponential = np.exp(-self.beta * chi * epsilon)
        return fxn * exponential

    def get_mean(self, chi, epsilon_range):
        """At 'time' chi, returns the mean epsilon of the distribution
        over the range in epsilon_range (a list of numbers from min
        to max that gives the spacing for numerical integration).
        Does it numerically with scipy's trapz; that is, the trapezoid
        rule."""
        epsilon_range = np.array(epsilon_range)
        ns = np.array([self(epsilon, chi) for epsilon in epsilon_range])
        num = trapz(ns * epsilon_range, epsilon_range)
        den = trapz(ns, epsilon_range)
        return num / den, den


class Gamma:
    """Creates callable gamma function with k value specified."""
    def __init__(self, k=1., Ns=1000.):
        self.k = k
        self.Ns = Ns

    def __call__(self, epsilon):
        k = self.k
        mean0 = 1.
        pdf = self.Ns * epsilon**(k - 1.) * np.exp(-k * epsilon / mean0)
        pdf /= (mean0 / k)**k * scipy.special.gamma(k)
        return pdf


class Beta:
    """Creates callable beta function with alpha, beta values specified."""
    def __init__(self, alpha, beta, Ns=1000.):
        self.alpha = alpha
        self.beta = beta
        self.Ns = Ns
        gamma = scipy.special.gamma
        self.gamma_factor = gamma(alpha + beta) / gamma(alpha) / gamma(beta)
        mean = alpha / (alpha + beta)
        self.scale = 1. / mean  # Will be used to set the initial mean to 1.
        self.max_value = self.scale  # Upper bound.
        self.mean = mean

    def __call__(self, x):
        alpha, beta, scale = self.alpha, self.beta, self.scale
        y = x / scale
        pdf = y**(alpha - 1.) * (1. - y)**(beta - 1.) * self.gamma_factor
        pdf /= scale
        return pdf * self.Ns

    def getK(self, phi):
        """Returns the apparent order paramter K at a given "time" phi."""
        alpha, beta = self.alpha, self.beta
        num = 1. + alpha
        num *= reghyp1f1(alpha, alpha + beta, -phi)
        num *= reghyp1f1(2. + alpha, 2. + alpha + beta, -phi)
        den = alpha * reghyp1f1(1. + alpha, 1. + alpha + beta, -phi)**2
        K = -1 + num / den
        return K


def reghyp1f1(a, b, z):
    """Regularized Hypergeometric 1F1 function."""
    return float(mpmath.hyp1f1(a, b, z) / mpmath.gamma(b))


def makefig(figsize=(8., 4.), nrows=4, ncols_left=4, ncols_right=4,
            lm=0.03, rm=0.01, hg=0.03, arrow_ratio=.5,
            tm=0.08, bm=0.05, vg=0.03,
            ):
    """Custom figure for this plot."""
    fig = pyplot.figure(figsize=figsize)
    axes = np.ndarray((nrows, ncols_left + ncols_right), dtype=object)
    axwidth = ((1. - lm - rm - hg * (ncols_left + ncols_right)) /
               (ncols_left + ncols_right + arrow_ratio))
    axheight = (1. - tm - bm - vg * (nrows - 1.)) / nrows
    bottompoint = 1. - tm - axheight
    for irow in range(nrows):
        leftpoint = lm
        for icol in range(ncols_left):
            ax = fig.add_axes((leftpoint, bottompoint, axwidth, axheight))
            axes[irow, icol] = ax
            leftpoint += axwidth + hg
        leftpoint += hg + arrow_ratio * axwidth
        for icol in range(ncols_right):
            ax = fig.add_axes((leftpoint, bottompoint, axwidth, axheight))
            axes[irow, icol + ncols_left] = ax
            leftpoint += axwidth + hg
        bottompoint -= axheight + vg
    # Arrow axes.
    bottompoint = 0. + bm
    leftpoint = lm + axwidth * ncols_left + hg * ncols_left
    arrow_width = arrow_ratio * axwidth
    arrow_height = axheight * nrows + vg * (nrows - 1.)
    ax_arrow = fig.add_axes((leftpoint, bottompoint,
                             arrow_width, arrow_height))
    return fig, axes, ax_arrow


fig, axes, ax_arrow = makefig()
fill_color = '0.8'
labels = {}
meanheight = 0.2
ax_number = -1
chis = [0., 0.5, 2., 8.]
letters = 'ABCDEFGHIJKLMNOP'
betalabel = r'$\mathbf{{{:s}}}$ beta\n$a$={:g},$b$={:g}'
gammalabel = r'$\mathbf{{{:s}}}$ gamma\n $k$={:g}'
ytop = 1050.  # top ylim

ax_number += 1
a, b = 0.25, 35.
labels[ax_number] = betalabel.format(letters[ax_number], a, b)
beta = Beta(a, b)
evolve = Evolution(beta)
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., 2.)
    epsilons = np.logspace(np.log10(1e-8), np.log10(beta.max_value), num=1000)
    mean, count = evolve.get_mean(chi, epsilons)
    k = 1. / beta.getK(1000.)
    fits = [(count / scipy.special.gamma(k) / (mean / k)**k * epsilon**(k-1) *
             np.exp(-epsilon * k / mean)) for epsilon in epsilons]
    ax.plot(epsilons, fits, ':')
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)

ax_number += 1
a, b = .5, .5
labels[ax_number] = betalabel.format(letters[ax_number], a, b)
beta = Beta(a, b)
evolve = Evolution(beta)
local_axes = axes[:, ax_number]
ylim = None
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    if ylim is None:
        ylim = ax.get_ylim()
    epsilons = np.logspace(np.log10(1e-8), np.log10(beta.max_value / 2.))
    upper_epsilons = beta.max_value - np.flip(epsilons)[:-1]
    epsilons = np.concatenate((epsilons, upper_epsilons))
    mean, count = evolve.get_mean(chi, epsilons)
    k = 1. / beta.getK(1000.)
    fits = [(count / scipy.special.gamma(k) / (mean / k)**k * epsilon**(k-1) *
             np.exp(-epsilon * k / mean)) for epsilon in epsilons]
    ax.plot(epsilons, fits, ':')
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)

ax_number += 1
a, b = 1., 1.
labels[ax_number] = betalabel.format(letters[ax_number], a, b)
beta = Beta(a, b)
evolve = Evolution(beta)
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    mean, count = evolve.get_mean(chi, epsilons)
    fits = [count / mean * np.exp(-epsilon / mean) for epsilon in epsilons]
    k = 1. / beta.getK(1000.)
    fits = [(count / scipy.special.gamma(k) / (mean / k)**k * epsilon**(k-1) *
             np.exp(-epsilon * k / mean)) for epsilon in epsilons]
    ax.plot(epsilons, fits, ':')
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)

ax_number += 1
a, b = 2., 2.
labels[ax_number] = betalabel.format(letters[ax_number], a, b)
beta = Beta(a, b)
evolve = Evolution(beta)
epsilons = np.linspace(0., beta.scale)
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    mean, count = evolve.get_mean(chi, epsilons)
    k = 1. / beta.getK(1000.)
    fits = [(count / scipy.special.gamma(k) / (mean / k)**k * epsilon**(k-1) *
             np.exp(-epsilon * k / mean)) for epsilon in epsilons]
    ax.plot(epsilons, fits, ':')
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)

ax_number += 1
k = 0.25
labels[ax_number] = gammalabel.format(letters[ax_number], k)
evolve = Evolution(Gamma(k=k))
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    ax.plot(epsilons, ys, ':')
    epsilons = np.logspace(np.log10(1e-8), np.log10(100.), num=1000)
    mean, count = evolve.get_mean(chi, epsilons)
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)
    ax.set_xlim(0., 2.)

ax_number += 1
k = 0.5
labels[ax_number] = gammalabel.format(letters[ax_number], k)
evolve = Evolution(Gamma(k=k))
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    ax.plot(epsilons, ys, ':')
    epsilons = np.logspace(np.log10(1e-8), np.log10(100.), num=1000)
    mean, count = evolve.get_mean(chi, epsilons)
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)
    ax.set_xlim(0., 2.)

ax_number += 1
k = 1.0
labels[ax_number] = gammalabel.format(letters[ax_number], k)
evolve = Evolution(Gamma(k=k))
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    ax.plot(epsilons, ys, ':')
    epsilons = np.logspace(np.log10(1e-8), np.log10(100.), num=1000)
    mean, count = evolve.get_mean(chi, epsilons)
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)
    ax.set_xlim(0., 2.)

ax_number += 1
k = 2.0
labels[ax_number] = gammalabel.format(letters[ax_number], k)
evolve = Evolution(Gamma(k=k))
local_axes = axes[:, ax_number]
for chi, ax in zip(chis, local_axes):
    epsilons = np.linspace(0., 2., num=100)
    ys = [evolve(epsilon, chi) for epsilon in epsilons]
    ax.plot(epsilons, ys)
    ax.fill_between(epsilons, ys, color=fill_color)
    ax.set_xlim(0., epsilons[-1])
    ax.plot(epsilons, ys, ':')
    epsilons = np.logspace(np.log10(1e-8), np.log10(100.), num=1000)
    mean, count = evolve.get_mean(chi, epsilons)
    ax.set_ylim(0., ytop)
    ylim = ax.get_ylim()
    ax.plot([mean] * 2, [0., ylim[1] * meanheight], '-', color='0.5')
    ax.set_ylim(ylim)
    ax.set_xlim(0., 2.)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


for ax_number, label in labels.items():
    box = axes[0, ax_number].get_position()
    x = np.average((box.x0, box.x1))
    y = box.y1 + 0.07
    fig.text(x, y, label, ha='center', va='top', fontsize=9.)

axes[0, 0].set_ylabel(r'$n_\mathrm{S}(\varepsilon,0)$')
for number, ax in enumerate(axes[1:, 0]):
    ax.set_ylabel(r'$n_\mathrm{{S}}(\varepsilon,t_{:d})$'.format(number + 1))
for ax in axes[-1, :]:
    ax.set_xlabel(r'$\varepsilon$')

bbox_props = dict(boxstyle="rarrow,pad=0.3", fc='0.9',  ec="k", lw=1.0)
ax_arrow.text(0.5, 0,
              r'increasing time [$\phi(t)$]',
              ha="center", va="center", rotation=-90,
              bbox=bbox_props)
ax_arrow.set_ylim(1., -1.)
ax_arrow.set_xlim(0., 1.)

if True:
    ax_arrow.spines['top'].set_visible(False)
    ax_arrow.spines['bottom'].set_visible(False)
    ax_arrow.spines['left'].set_visible(False)
    ax_arrow.spines['right'].set_visible(False)
    ax_arrow.set_xticks([])
    ax_arrow.set_yticks([])

# Annotations.
axes[0, 0].annotate(r'$\bar{\varepsilon}_0$',
                    xy=(1.0, ytop*(meanheight*1.1)), xycoords='data',
                    xytext=(1.5, ytop*(meanheight*2.0)), textcoords='data',
                    arrowprops={'arrowstyle': '->',
                                'color': 'k',
                                'connectionstyle': 'arc3,rad=0.3'})
axes[1, 0].annotate(r'$\bar{\varepsilon}(t)$',
                    xy=(0.3, ytop*(meanheight*1.1)), xycoords='data',
                    xytext=(1.3, ytop*(meanheight*2.0)), textcoords='data',
                    arrowprops={'arrowstyle': '->',
                                'color': 'k',
                                'connectionstyle': 'arc3,rad=0.3'})
axes[1, 2].annotate(r'gamma',
                    xy=(0.2, ytop*0.6), xycoords='data',
                    xytext=(0.9, ytop*0.7), textcoords='data',
                    arrowprops={'arrowstyle': '->',
                                'color': 'C1',
                                'connectionstyle': 'arc3,rad=0.3'})

fig.savefig('fig2.pdf')
