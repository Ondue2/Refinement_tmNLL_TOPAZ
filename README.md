# Refinement_tmNLL_TOPAZ
Refinement program designed for neutron diffraction data obtained from TOPAZ at ORNL

This program can refine nuclear and magnetic structure for **k** = (0,0,0).

The program is written and tested in Python 3.13.

**Required packages**: `numpy`, `scipy.optimize`, `pandas`, `matplotlib.pyplot`, `sympy`, `pymatgen.symmetry.maggroups`


For the optimization method, this program supports global optimization using the differential evolution algorithm as well as local optimization using L-BFGS-B method.
Global optimization typically takes tens of minutes, while local optimization only takes a few minutes. However, the global optimizer is generally better than a human using a local optimizer, even with smart initial value selection, in terms of finding the true global minimum in the fitting parameter space. 

This program uses twice the mean negative log-Gaussian likelihood (tmNLL) as the objective to minimize instead of the conventional Chi2 objective. Using the tmNLL objective might mitigate statistical bias that occurs when using the conventional Chi2 objective with a weighting scheme. 

This program uses the _SHELX_ weighting scheme with parameters _a_ and _b_ with _P_ = _I<sub>c</sub>_. Using _P_ = _I<sub>c_ instead of _P_ = (2 _I<sub>c</sub>_ + _I<sub>o</sub>_)/3 minimizes the statistical bias when the objective is tmNLL instead of Chi2. The parameters _a_ and _b_ are to be optimized, rather than fixed, and their effect flattens Chi2 when the objective is tmNLL.

This program supports extinction correction based on the isotropic Becker-Coppens Type I Gaussian correction following _JANA_.

For guidance on using this program, please see .
For details on the theoretical background of this program, please see .



