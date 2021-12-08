Thermal Mechanisms
==================

<!-- [![arXiv](https://shields.io/badge/arXiv-xxx-b31b1b)](http://arxiv.org/abs/xxx) -->

Usage
-----

The programs and scripts in this repository require a C++11 compiler,
the standard Python scientific stack ([NumPy][numpy] + [SciPy][scipy])
and [Numba][numba].  Plotting requires [Matplotlib][mpl] and
[mmmpl][mmmpl].

Description
-----------

A [bar-joint mechanism][mech] is a deformable assembly of
freely-rotating joints connected by stiff bars.  The free-energy
landscape of such mechanisms is dominated by the neighborhoods of points
that correspond to singularities in its shape space—the set of its
lowest-energy states with distinct shapes.  This repository contains
a set of programs to find the free-energy landscapes of bar-joint
mechanisms using Monte Carlo methods.  It also contains a set of scripts
to numerically parameterize one-dimensional shape spaces.  Also included
are Mathematica notebooks to simplify some of the analytical
calculations detailed in the paper.

<p align="center">
  <img src="https://raw.githubusercontent.com/manu-mannattil/assets/master/thermmech/4bar.gif" alt="Four-bar linkage animation"/>
</p>

The above animation illustrates the shape space of a [four-bar
linkage][4bar] showing two modes of deformation (blue and red curves)
and two singularities (black dots) corresponding to situations where the
bars become collinear and support a state of self stress.

License
-------

Licensed under the 3-clause BSD license. See the file LICENSE for more
details.

[4bar]: https://en.wikipedia.org/wiki/Four-bar_linkage
[mech]: https://en.wikipedia.org/wiki/Mechanism_(engineering)
[mmmpl]: https://github.com/manu-mannattil/mmmpl
[mpl]: https://matplotlib.org
[numba]: https://numba.pydata.org
[numpy]: https://numpy.org
[scipy]: https://scipy.org
