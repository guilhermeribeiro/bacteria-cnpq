# -*- coding: utf-8 -*-
"""
test_harness/capture.py
~~~~~~~~~~~~~~~~~~~~~~~
Main TestRun class for capturing and persisting experiment state.
"""
import os
import sys
import json
import time
import numpy as np
from datetime import datetime

from .hooks import StdoutTee, PlotCapture
from .plotting import generate_circuit_plots

# Resolve base directory relative to this file's location
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNS_DIR = os.path.join(_BASE_DIR, 'test_runs')


class TestRun:
    """Captures and persists all state from a GeneNet experiment run.

    Usage (in exampleScript.py)::

        from test_harness import TestRun
        run = TestRun(auto_capture=True)
        # ... your existing code ...
        run.save(network=network, simulation_output=output)

    To load a previous run::

        run = TestRun.load_latest()
        run.network_state['W']  # numpy array
    """

    def __init__(self, auto_capture=False, notes='', run_dir=None):
        self._start_time = time.time()
        self._timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if run_dir is not None:
            self._run_dir = os.path.abspath(run_dir)
        else:
            self._run_dir = os.path.join(_RUNS_DIR, self._timestamp)
        try:
            os.makedirs(self._run_dir)
        except OSError:
            pass

        self._notes = notes
        self._stdout_tee = None
        self._plot_capture = None

        # Loaded state (populated by load())
        self.network_state = None
        self.simulation_output = None
        self.cost_history = None
        self.circuit_design = None
        self.metadata = None

        if auto_capture:
            self._install_hooks()

    def _install_hooks(self):
        """Install stdout tee and matplotlib capture hooks."""
        # Stdout tee
        log_path = os.path.join(self._run_dir, 'terminal_output.txt')
        self._stdout_tee = StdoutTee(log_path)
        sys.stdout = self._stdout_tee

        # Plot capture
        self._plot_capture = PlotCapture(self._run_dir)
        self._plot_capture.install()

    def _uninstall_hooks(self):
        """Restore original stdout and plt.show."""
        if self._stdout_tee:
            self._stdout_tee.close()
            self._stdout_tee = None
        if self._plot_capture:
            self._plot_capture.uninstall()
            self._plot_capture = None

    def save(self, network=None, simulation_output=None, cost_history=None,
             notes=None, seed=None):
        """Save all experiment state to disk.

        Parameters
        ----------
        network : object, optional
            The GeneNet network object. Its Theano shared variables are
            extracted via .get_value() and saved.
        simulation_output : numpy.ndarray, optional
            The full simulation output array from simulateNetwork().
        cost_history : list or numpy.ndarray, optional
            Cost/loss values over training iterations.
        notes : str, optional
            Free-form notes about this run (overrides constructor notes).
        seed : int, optional
            The random seed used for this run.
        """
        if notes is not None:
            self._notes = notes
        if seed is not None:
            self._seed = seed

        # --- LIMPEZA PROFUNDA DA REDE (THEANO) ANTES DO EXPORT ---
        if network is not None:
            try:
                alpha_vals = network.alpha.get_value()
                W_vals = network.W.get_value()
                for i in range(len(alpha_vals)):
                    winner_idx = int(np.argmax(alpha_vals[i]))
                    # Vazio = 6, YFP = 7
                    if winner_idx == 6 or winner_idx == 7:
                        W_vals[:, i] = -100.0 # Peso negativo alto zera a força pós-softplus (saídas)
                    if winner_idx == 6:
                        W_vals[i, :] = -100.0 # Zera entradas do Vazio
                
                # Zera as entradas (Slot 0 e 1) para que não sejam reprimidas por ninguém
                W_vals[0, :] = -100.0
                W_vals[1, :] = -100.0
                
                network.W.set_value(W_vals)
            except Exception as e:
                print("[test_harness] Erro na limpeza da rede:", e)
                
        # --- Network state ---
        if network is not None:
            self._save_network_state(network)

        # --- Simulation output ---
        if simulation_output is not None:
            np.save(os.path.join(self._run_dir, 'simulation_output.npy'),
                    simulation_output)

        # --- Cost history ---
        if cost_history is not None:
            cost_arr = np.array(cost_history, dtype=np.float64)
            np.save(os.path.join(self._run_dir, 'cost_history.npy'), cost_arr)

        # --- Circuit design (re-extract from network) ---
        if network is not None:
            self._save_circuit_design(network)

        # --- Generate circuit plots ---
        generated_plots = []
        if network is not None:
            try:
                # Load state directly for plotting to avoid disk roundtrip
                state = {}
                for param in network.parameters:
                    state[param.name] = param.get_value()
                
                design_path = os.path.join(self._run_dir, 'circuit_design.json')
                design = None
                if os.path.exists(design_path):
                    with open(design_path, 'r') as f:
                        design = json.load(f)
                
                generated_plots = generate_circuit_plots(self._run_dir, state, design)
            except Exception as e:
                print('[test_harness] Error generating plots: {}'.format(e))

        # --- Metadata ---
        elapsed = time.time() - self._start_time
        meta_plots = (self._plot_capture.saved_paths if self._plot_capture else [])
        meta_plots.extend(generated_plots)
        
        meta = {
            'timestamp': self._timestamp,
            'duration_seconds': round(elapsed, 2),
            'notes': self._notes,
            'seed': getattr(self, '_seed', None),
            'plots_saved': meta_plots,
        }
        with open(os.path.join(self._run_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # --- Cleanup hooks ---
        self._uninstall_hooks()

        print('[test_harness] Run saved to: {}'.format(self._run_dir))

    def _save_network_state(self, network):
        """Extract and save all Theano shared variable values."""
        state = {}
        # Save all items in network.parameters
        for param in network.parameters:
            name = param.name
            state[name] = param.get_value()

        # Also try to get protein library if present (static, but useful)
        for attr_name in ['y_min_t', 'y_max_t', 'Kd_t', 'n_t']:
            attr = getattr(network, attr_name, None)
            if attr is not None:
                try:
                    state[attr_name] = attr.get_value()
                except Exception:
                    pass

        np.savez(os.path.join(self._run_dir, 'network_state.npz'), **state)

    def _save_circuit_design(self, network):
        """Re-compute and save the circuit design (slot winners)."""
        alpha_attr = getattr(network, 'alpha', None)
        if alpha_attr is None:
            return

        try:
            alpha_vals = alpha_attr.get_value()
        except Exception:
            return

        exp_a = np.exp(alpha_vals - np.max(alpha_vals, axis=1, keepdims=True))
        probs = exp_a / np.sum(exp_a, axis=1, keepdims=True)

        # Try to get protein names from the model module
        prot_names = None
        try:
            import GeneNet_model
            prot_names = getattr(GeneNet_model, 'PROT_NAMES', None)
        except ImportError:
            pass
        if prot_names is None:
            try:
                import GeneNet_comprotein
                prot_names = getattr(GeneNet_comprotein, 'PROT_NAMES', None)
            except ImportError:
                pass
        if prot_names is None:
            prot_names = ['Protein_{}'.format(i) for i in range(probs.shape[1])]

        design = {'slots': []}
        for i in range(len(alpha_vals)):
            winner_idx = int(np.argmax(probs[i]))
            design['slots'].append({
                'slot': i,
                'winner': prot_names[winner_idx],
                'confidence_pct': round(float(probs[i][winner_idx]) * 100, 2),
                'probabilities': {
                    prot_names[j]: round(float(probs[i][j]) * 100, 2)
                    for j in range(len(prot_names))
                }
            })

        with open(os.path.join(self._run_dir, 'circuit_design.json'), 'w') as f:
            json.dump(design, f, indent=2)

        # --- Save Wiring Matrix (W) as a labeled CSV ---
        W_attr = getattr(network, 'W', None)
        if W_attr is not None:
            try:
                W_vals = W_attr.get_value()
                # Binarizacao topologica estrita (0.0 ou 1.0):
                # No forward pass do Theano, W_bin = round(sigmoid(W)), que eh 1.0 se e somente se W >= 0.0.
                # Transposicao (.T): Converte de (target, source) do Theano para (source, target) esperado pelo Graphs.jl no Julia.
                W_forces = (W_vals.T >= 0.0).astype(np.float64)
                
                csv_path = os.path.join(self._run_dir, 'wiring_matrix.csv')
                
                # Use standard file open (io.open for compatibility)
                import io
                with io.open(csv_path, 'w', encoding='utf-8') as f:
                    # Header: empty top-left cell, then column names
                    col_headers = ['Slot {} ({})'.format(s['slot'], s['winner']) for s in design['slots']]
                    f.write(u',' + u','.join(col_headers) + u'\n')
                    
                    # Rows
                    for i in range(W_forces.shape[0]):
                        row_label = u'Slot {} ({})'.format(design['slots'][i]['slot'], design['slots'][i]['winner'])
                        # Format weights to 4 decimal places
                        row_data = [u'{:.4f}'.format(val) for val in W_forces[i]]
                        f.write(row_label + u',' + u','.join(row_data) + u'\n')
            except Exception as e:
                print("[test_harness] Error saving wiring_matrix.csv:", e)

    # ------------------------------------------------------------------
    # Loading previous runs
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, run_name):
        """Load a specific run by its directory name (timestamp).

        Parameters
        ----------
        run_name : str
            The run directory name, e.g. '2026-07-28_17-30-00'.

        Returns
        -------
        TestRun
            A TestRun instance with loaded data as attributes.
        """
        run_dir = os.path.join(_RUNS_DIR, run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                'Run not found: {}'.format(run_dir))
        return cls._load_from_dir(run_dir)

    @classmethod
    def load_latest(cls):
        """Load the most recent run.

        Returns
        -------
        TestRun
            A TestRun instance with loaded data.
        """
        if not os.path.isdir(_RUNS_DIR):
            raise FileNotFoundError(
                'No runs directory found at: {}'.format(_RUNS_DIR))
        runs = sorted(os.listdir(_RUNS_DIR))
        if not runs:
            raise FileNotFoundError('No runs found.')
        return cls._load_from_dir(os.path.join(_RUNS_DIR, runs[-1]))

    @classmethod
    def list_runs(cls):
        """List all saved run names.

        Returns
        -------
        list of str
            Sorted list of run directory names.
        """
        if not os.path.isdir(_RUNS_DIR):
            return []
        return sorted(os.listdir(_RUNS_DIR))

    @classmethod
    def _load_from_dir(cls, run_dir):
        """Internal: populate a TestRun from saved files."""
        instance = cls.__new__(cls)
        instance._run_dir = run_dir
        instance._stdout_tee = None
        instance._plot_capture = None

        # Metadata
        meta_path = os.path.join(run_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                instance.metadata = json.load(f)
        else:
            instance.metadata = {}

        # Network state
        state_path = os.path.join(run_dir, 'network_state.npz')
        if os.path.exists(state_path):
            instance.network_state = dict(np.load(state_path,
                                                  allow_pickle=False))
        else:
            instance.network_state = None

        # Simulation output
        sim_path = os.path.join(run_dir, 'simulation_output.npy')
        if os.path.exists(sim_path):
            instance.simulation_output = np.load(sim_path)
        else:
            instance.simulation_output = None

        # Cost history
        cost_path = os.path.join(run_dir, 'cost_history.npy')
        if os.path.exists(cost_path):
            instance.cost_history = np.load(cost_path)
        else:
            instance.cost_history = None

        # Circuit design
        design_path = os.path.join(run_dir, 'circuit_design.json')
        if os.path.exists(design_path):
            with open(design_path, 'r') as f:
                instance.circuit_design = json.load(f)
        else:
            instance.circuit_design = None

        instance._notes = instance.metadata.get('notes', '')
        instance._timestamp = instance.metadata.get('timestamp', '')

        return instance

    def __repr__(self):
        return '<TestRun "{}" dir="{}">'.format(
            self._timestamp, self._run_dir)
