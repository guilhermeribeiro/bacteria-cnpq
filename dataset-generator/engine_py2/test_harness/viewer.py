# -*- coding: utf-8 -*-
"""
test_harness/viewer.py
~~~~~~~~~~~~~~~~~~~~~~
Command-line utility to inspect saved test runs.

Usage::

    python -m test_harness.viewer              # list all runs
    python -m test_harness.viewer <run_name>   # inspect a specific run
    python -m test_harness.viewer latest        # inspect the latest run
"""
import sys
import json
import os
import numpy as np

from .capture import TestRun


def _print_separator(char='-', width=60):
    print(char * width)


def _format_array_summary(arr):
    """One-line summary of a numpy array."""
    return 'shape={}, dtype={}, min={:.4f}, max={:.4f}, mean={:.4f}'.format(
        arr.shape, arr.dtype, float(arr.min()), float(arr.max()),
        float(arr.mean()))


def list_all_runs():
    """Print a table of all saved runs."""
    runs = TestRun.list_runs()
    if not runs:
        print('No saved runs found.')
        return

    print('\n  Saved Test Runs')
    _print_separator('=')
    for i, name in enumerate(runs):
        run = TestRun.load(name)
        duration = run.metadata.get('duration_seconds', '?')
        notes = run.metadata.get('notes', '')
        n_plots = len(run.metadata.get('plots_saved', []))
        has_net = 'yes' if run.network_state else 'no'
        print('  {:>3d}. {}  |  {:.0f}s  |  net:{}  |  plots:{}  |  {}'.format(
            i + 1, name, float(duration) if duration != '?' else 0,
            has_net, n_plots, notes[:40]))
    _print_separator('=')
    print('  Total: {} runs\n'.format(len(runs)))


def inspect_run(run_name):
    """Print detailed info about a specific run."""
    if run_name == 'latest':
        run = TestRun.load_latest()
    else:
        run = TestRun.load(run_name)

    print('\n  Run: {}'.format(run._timestamp))
    _print_separator('=')

    # Metadata
    if run.metadata:
        print('  Duration: {}s'.format(
            run.metadata.get('duration_seconds', '?')))
        print('  Notes: {}'.format(
            run.metadata.get('notes', '(none)')))
        plots = run.metadata.get('plots_saved', [])
        print('  Plots saved: {}'.format(len(plots)))
        for p in plots:
            print('    - {}'.format(os.path.basename(p)))

    _print_separator()

    # Network state
    if run.network_state:
        print('  Network State:')
        for name, arr in run.network_state.items():
            print('    {}: {}'.format(name, _format_array_summary(arr)))
    else:
        print('  Network State: (not saved)')

    _print_separator()

    # Cost history
    if run.cost_history is not None:
        print('  Cost History: {}'.format(
            _format_array_summary(run.cost_history)))
    else:
        print('  Cost History: (not saved)')

    # Simulation output
    if run.simulation_output is not None:
        print('  Simulation Output: {}'.format(
            _format_array_summary(run.simulation_output)))
    else:
        print('  Simulation Output: (not saved)')

    _print_separator()

    # Circuit design
    if run.circuit_design:
        print('  Circuit Design:')
        for slot in run.circuit_design.get('slots', []):
            print('    Slot {}: {} ({:.1f}%)'.format(
                slot['slot'], slot['winner'], slot['confidence_pct']))
    else:
        print('  Circuit Design: (not saved)')

    _print_separator('=')
    print()


def main():
    if len(sys.argv) < 2:
        list_all_runs()
    else:
        inspect_run(sys.argv[1])


if __name__ == '__main__':
    main()
