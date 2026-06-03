"""
Independent verification package for SDAS (State-Dependent Adaptive Synergy).

This package provides a completely independent implementation of the cable-driven
origami hand quasi-static solver, using:
- Classic Capstan friction formula (not Della Santina M-matrix framework)
- Pinocchio library for rigid body dynamics (different codebase)
- Iterative Newton solver (different from Schur complement)

The purpose is cross-validation: if this independent solver produces the same
joint angles as SDAS, it confirms that SDAS's constraint selection and Schur
complement solver are correct — without circular dependence on the same
theoretical framework.
"""

__version__ = "0.1.0"
