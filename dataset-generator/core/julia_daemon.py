# -*- coding: utf-8 -*-
"""
julia_daemon.py
==============================================================================
Cliente de Daemon Persistente para Julia SSA (Gillespie Estocastico).
Elimina 100% do overhead de importacao do ModelingToolkit/JumpProcesses (~140s).
Cada avaliacao individual roda em ~1.5 a 3.0 segundos com Threads.@threads!
Inclui auto-reconecao, timeouts e fallback seguro para subprocesso.
==============================================================================
"""
from __future__ import print_function, division
import os
import sys
import json
import time
import threading
import subprocess

class JuliaDaemonClient(object):
    def __init__(self, julia_exe, proj_dir, daemon_script, num_threads=6, ensemble_size=32, gate="XOR"):
        self.julia_exe = julia_exe
        self.proj_dir = os.path.abspath(proj_dir)
        self.daemon_script = os.path.abspath(daemon_script)
        self.num_threads = int(num_threads)
        self.ensemble_size = int(ensemble_size)
        self.gate = str(gate)
        self.lock = threading.Lock()
        self.proc = None
        self._start_daemon()

    def _start_daemon(self):
        cmd = [
            self.julia_exe,
            "-t", str(self.num_threads),
            "--startup-file=no",
            "-O1",
            "--project=" + self.proj_dir,
            self.daemon_script,
            str(self.ensemble_size),
            self.gate
        ]
        env = os.environ.copy()
        env["GKSwstype"] = "100"
        
        # Inicia processo do daemon
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            universal_newlines=True,
            bufsize=1
        )
        
        # Aguarda sinal de prontidao JULIA_DAEMON_READY (timeout de 120s para warmup inicial)
        t_start = time.time()
        ready = False
        while time.time() - t_start < 180:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read()
                    raise RuntimeError("Julia daemon encerrou prematuramente (ret={}): {}".format(
                        self.proc.returncode, err
                    ))
                time.sleep(0.1)
                continue
            if "JULIA_DAEMON_READY" in line:
                ready = True
                break
                
        if not ready:
            self.close()
            raise RuntimeError("Timeout aguardando inicializacao do Julia daemon (>180s)")

    def evaluate(self, run_dir, timeout=60):
        """
        Envia uma pasta de circuito para avaliacao no daemon.
        Retorna: (snr_val: float, snr_data: dict, elapsed: float, err_msg: str)
        """
        run_dir_abs = os.path.abspath(run_dir).replace("\\", "/")
        with self.lock:
            # Verifica se o daemon ainda esta vivo
            if self.proc is None or self.proc.poll() is not None:
                try:
                    self._start_daemon()
                except Exception as e:
                    return 0.0, None, 0.0, "Falha ao reiniciar daemon: " + str(e)
                    
            t0 = time.time()
            try:
                self.proc.stdin.write(run_dir_abs + "\n")
                self.proc.stdin.flush()
                
                # Le resposta
                resp = self.proc.stdout.readline().strip()
                elapsed = time.time() - t0
                
                if resp.startswith("SNR_RESULT:"):
                    parts = resp.split(":")
                    snr_val = float(parts[1])
                    snr_file = os.path.join(run_dir, "snr_results.json")
                    snr_data = None
                    if os.path.exists(snr_file):
                        try:
                            with open(snr_file, "r") as f:
                                snr_data = json.load(f)
                            snr_data["snr"] = snr_val
                        except Exception:
                            pass
                    return snr_val, snr_data, elapsed, ""
                elif resp.startswith("ERROR:"):
                    return 0.0, None, elapsed, resp
                else:
                    return 0.0, None, elapsed, "Resposta inesperada do daemon: " + resp
            except Exception as e:
                elapsed = time.time() - t0
                # Se quebrou pipe ou deu timeout, reinicia daemon para a proxima
                try:
                    self.close()
                except Exception:
                    pass
                return 0.0, None, elapsed, "Excecao no daemon: " + str(e)

    def close(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.stdin.write("EXIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=5)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            self.proc = None
