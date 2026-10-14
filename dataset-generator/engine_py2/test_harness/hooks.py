# -*- coding: utf-8 -*-
"""
test_harness/hooks.py
~~~~~~~~~~~~~~~~~~~~~
Non-invasive hooks that capture matplotlib figures and stdout output
without disrupting normal program behavior.
"""
import os
import sys
import matplotlib.pyplot as plt


class StdoutTee:
    """Duplicates stdout to both the terminal and a file."""

    def __init__(self, filepath):
        import io
        self._file = io.open(filepath, 'w', encoding='utf-8')
        self._original = sys.stdout

    def write(self, data):
        self._original.write(data)
        if isinstance(data, bytes):
            data = data.decode('utf-8', 'replace')
        self._file.write(data)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._original

    # Proxy attribute access to original stdout
    def __getattr__(self, name):
        return getattr(self._original, name)


class PlotCapture:
    """Monkey-patches plt.show() to save figures as PNGs before displaying."""

    def __init__(self, output_dir):
        self._output_dir = output_dir
        self._counter = 0
        self._original_show = plt.show
        self._saved_paths = []

    def install(self):
        """Replace plt.show with the capturing version."""
        original = self._original_show
        capture = self

        def _capturing_show(*args, **kwargs):
            # Save all open figures
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                fname = 'plot_{:03d}.png'.format(capture._counter)
                path = os.path.join(capture._output_dir, fname)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                capture._saved_paths.append(path)
                capture._counter += 1
            
            # Fecha as figuras para liberar memória e EVITAR pop-ups bloqueantes
            plt.close('all')

        plt.show = _capturing_show

    def uninstall(self):
        """Restore original plt.show."""
        plt.show = self._original_show

    @property
    def saved_paths(self):
        return list(self._saved_paths)
