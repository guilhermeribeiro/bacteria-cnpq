# -*- coding: utf-8 -*-
"""
test_harness
~~~~~~~~~~~~
Minimally-invasive persistence layer for GeneNet experiments.
Captures network state, plots, terminal output and simulation data.

Usage::

    from test_harness import TestRun
    run = TestRun(auto_capture=True)
    # ... your experiment code ...
    run.save(network=network, simulation_output=output)
"""
from .capture import TestRun

__all__ = ['TestRun']
