#!/usr/bin/env python3
"""
gerar_csv_snr_real.py — gera um CSV com o SNR REAL de cada circuito.

Le o circuit_dataset.jsonl e, para cada circuito, recalcula o SNR correto
(definicao do Lewis = signal/sigma) a partir das amostras estocasticas ja
guardadas, colocando lado a lado com o `snr` reportado (inflado):

    SNR_real = |media_high - media_low| / raiz(var_high + var_low)

Saida: snr_real.csv

Uso:
    python3 gerar_csv_snr_real.py
"""
import csv
import json
from pathlib import Path
from statistics import mean, stdev, variance

BASE = Path(__file__).parent
ENTRADA = BASE / "circuit_dataset.jsonl"
SAIDA = BASE / "snr_real.csv"

COLUNAS = [
    "circuit_id", "gate", "n_nodes",
    "snr_reportado",     # o valor inflado que esta no dataset
    "snr_real",          # o correto = signal / sigma (def. do Lewis)
    "media_low", "media_high", "desvio_low", "desvio_high",
    "sinal", "ruido",    # intermediarios, para poder conferir a conta
]


def main():
    n = 0
    with open(ENTRADA) as entrada, open(SAIDA, "w", newline="") as saida:
        escritor = csv.DictWriter(saida, fieldnames=COLUNAS)
        escritor.writeheader()

        for linha in entrada:
            linha = linha.strip()
            if not linha:
                continue
            reg = json.loads(linha)

            low = reg["raw_samples_low"]    # saidas com input BAIXO
            high = reg["raw_samples_high"]  # saidas com input ALTO

            # estatisticas (desvio amostral, ddof=1, igual ao Lewis)
            m_lo, m_hi = mean(low), mean(high)
            s_lo, s_hi = stdev(low), stdev(high)

            sinal = abs(m_hi - m_lo)
            # ruido = desvio da diferenca; amostras independentes ->
            # var(high - low) = var(high) + var(low)
            ruido = (variance(high) + variance(low)) ** 0.5
            snr_real = sinal / ruido if ruido > 1e-9 else 0.0

            escritor.writerow({
                "circuit_id": reg["id"],
                "gate": reg.get("gate", ""),
                "n_nodes": len(reg["matrix_W"]),
                "snr_reportado": round(reg["snr"], 6),
                "snr_real": round(snr_real, 6),
                "media_low": round(m_lo, 4),
                "media_high": round(m_hi, 4),
                "desvio_low": round(s_lo, 4),
                "desvio_high": round(s_hi, 4),
                "sinal": round(sinal, 4),
                "ruido": round(ruido, 4),
            })
            n += 1

    print(f"OK: {n} circuitos escritos em {SAIDA.name}")


if __name__ == "__main__":
    main()
