# -*- coding: utf-8 -*-
"""
julia_daemon.py
==============================================================================
Cliente de Daemon Persistente para Julia SSA (Gillespie Estocastico).
Elimina 100% do overhead de importacao do ModelingToolkit/JumpProcesses (~140s).
Cada avaliacao individual roda em ~1.5 a 3.0 segundos com Threads.@threads!

Engenharia Defensiva de Produção:
1. Leitura não-bloqueante via background thread + queue.Queue com timeout estrito.
2. Redirecionamento de stderr diretamente para arquivo de log em disco (evita
   deadlock no kernel do Windows por esgotamento do buffer anônimo de pipe de 4KB).
3. Watchdog com auto-recuperação: se a simulação SSA travar (>timeout), o processo
   Julia emperrado é terminado à força (taskkill /F /T /PID) e um novo daemon
   limpo é inicializado sem travar o pool de workers do Optuna.
==============================================================================
"""
from __future__ import print_function, division
import os
import sys
import math
import json
import time
import threading
import subprocess

try:
    import queue
except ImportError:
    import Queue as queue


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
        self.stdout_queue = None
        self.reader_thread = None
        self.stderr_file = None
        
        # Garante pasta de logs dedicada
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(os.path.dirname(script_dir), "logs")
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except Exception:
                pass
        self.stderr_path = os.path.join(self.log_dir, "julia_daemon_stderr.log")
        self._start_daemon()

    def _start_daemon(self):
        self._cleanup_process()
        
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
        
        # Abre arquivo de log para stderr (nunca trava buffers anônimos de pipe)
        try:
            self.stderr_file = open(self.stderr_path, "a")
        except Exception:
            self.stderr_file = None
        
        cflags = 0
        if sys.platform == "win32" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            cflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file if self.stderr_file is not None else subprocess.PIPE,
            env=env,
            universal_newlines=True,
            bufsize=1,
            creationflags=cflags
        )
        
        # Fila assíncrona para leitura do stdout
        self.stdout_queue = queue.Queue()
        
        def _stdout_reader(proc, out_q):
            try:
                for line in iter(proc.stdout.readline, ''):
                    out_q.put(line)
            except Exception:
                pass
            finally:
                out_q.put(None)
                
        self.reader_thread = threading.Thread(target=_stdout_reader, args=(self.proc, self.stdout_queue))
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        # Aguarda sinal de prontidão JULIA_DAEMON_READY (timeout de 180s para warmup inicial)
        t_start = time.time()
        ready = False
        while time.time() - t_start < 180:
            try:
                line = self.stdout_queue.get(timeout=1.0)
                if line is None:
                    break
                if "JULIA_DAEMON_READY" in line:
                    ready = True
                    break
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
                continue
                
        if not ready:
            self._cleanup_process()
            raise RuntimeError("Timeout ou falha aguardando inicializacao do Julia daemon (>180s). Verifique {}".format(self.stderr_path))

    def _cleanup_process(self):
        if self.proc is not None:
            pid = self.proc.pid
            try:
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    import signal
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
            
        if self.stderr_file is not None:
            try:
                self.stderr_file.flush()
                self.stderr_file.close()
            except Exception:
                pass
            self.stderr_file = None

    def evaluate(self, run_dir, timeout=60):
        """
        Envia uma pasta de circuito para avaliacao no daemon com timeout estrito.
        Retorna: (snr_val: float, snr_data: dict, elapsed: float, err_msg: str)
        """
        run_dir_abs = os.path.abspath(run_dir).replace("\\", "/")
        with self.lock:
            # Verifica se o daemon ainda está ativo
            if self.proc is None or self.proc.poll() is not None:
                try:
                    self._start_daemon()
                except Exception as e:
                    return 0.0, None, 0.0, "Falha ao reiniciar daemon: " + str(e)
                    
            # Drena quaisquer resíduos antigos na fila
            while not self.stdout_queue.empty():
                try:
                    self.stdout_queue.get_nowait()
                except Exception:
                    break

            t0 = time.time()
            try:
                self.proc.stdin.write(run_dir_abs + "\n")
                self.proc.stdin.flush()
                
                # Leitura não-bloqueante com timeout estrito!
                resp_line = self.stdout_queue.get(timeout=float(timeout))
                elapsed = time.time() - t0
                
                if resp_line is None:
                    self._cleanup_process()
                    return 0.0, None, elapsed, "Processo Julia encerrou prematuramente durante a avaliacao"
                    
                resp = resp_line.strip()
                if resp.startswith("SNR_RESULT:"):
                    parts = resp.split(":")
                    try:
                        snr_val = float(parts[1])
                        if math.isnan(snr_val) or math.isinf(snr_val):
                            snr_val = 0.0
                    except (ValueError, IndexError):
                        snr_val = 0.0
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
            except queue.Empty:
                elapsed = time.time() - t0
                # Circuito estocástico rígido (stiff loop) atingiu timeout! Mata o Julia travado e reinicia limpo
                self._cleanup_process()
                return 0.0, None, elapsed, "TIMEOUT_JULIA_SSA_EXCEEDED (>{}s)".format(timeout)
            except Exception as e:
                elapsed = time.time() - t0
                self._cleanup_process()
                return 0.0, None, elapsed, "Excecao no daemon: " + str(e)

    def close(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.stdin.write("EXIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
            self._cleanup_process()
