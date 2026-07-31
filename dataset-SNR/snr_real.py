#!/usr/bin/env python3
"""
Uso (no terminal):
    python3 snr_real.py 38279916588943071_1841
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev, variance

DATASET = Path(__file__).parent / "circuit_dataset.jsonl"


def encontrar_circuito(circuit_id):
    """Percorre o .jsonl linha a linha e devolve o circuito com aquele 'id'."""
    with open(DATASET) as arquivo:
        for linha in arquivo:
            registro = json.loads(linha)
            if registro["id"] == circuit_id:
                return registro
    return None  # nao encontrou


def main():
    # 1) le o id passado no terminal
    if len(sys.argv) != 2:
        print("uso: python3 snr_real.py <circuit_id>")
        sys.exit(1)
    circuit_id = sys.argv[1]

    # 2) acha o circuito no dataset
    circ = encontrar_circuito(circuit_id)
    if circ is None:
        print(f"circuito '{circuit_id}' nao encontrado em {DATASET.name}")
        sys.exit(1)

    # 3) as amostras estocasticas de cada condicao (saida do reporter YFP)
    low = circ["raw_samples_low"]    # com input BAIXO
    high = circ["raw_samples_high"]  # com input ALTO

    # 4) estatisticas de cada condicao (desvio amostral = ddof=1, igual ao Lewis)
    media_low, media_high = mean(low), mean(high)
    desvio_low, desvio_high = stdev(low), stdev(high)

    # 5) SINAL = distancia entre as medias das duas condicoes
    sinal = abs(media_high - media_low)

    # 6) RUIDO = desvio-padrao da DIFERENCA (high - low).
    #    Como as amostras sao independentes, a variancia da diferenca e a
    #    soma das variancias: var(high - low) = var(high) + var(low).
    ruido = (variance(high) + variance(low)) ** 0.5

    # 7) SNR real = sinal / ruido  (protege contra divisao por ~zero)
    snr_real = sinal / ruido if ruido > 1e-9 else 0.0

    # 8) imprime tudo de forma clara
    print(f"circuito: {circuit_id}   (porta={circ.get('gate','?')}, "
          f"N={len(circ['matrix_W'])} nos)")
    print("-" * 56)
    print(f"  input BAIXO : media = {media_low:8.2f}   desvio = {desvio_low:8.2f}")
    print(f"  input ALTO  : media = {media_high:8.2f}   desvio = {desvio_high:8.2f}")
    print("-" * 56)
    print(f"  SINAL  |media_alto - media_baixo|      = {sinal:8.2f}")
    print(f"  RUIDO  raiz(var_alto + var_baixo)      = {ruido:8.2f}")
    print("-" * 56)
    print(f"  >> SNR REAL (sinal / ruido)            = {snr_real:7.2f}")
    print(f"     SNR reportado no dataset  = {circ['snr']:7.2f}")


if __name__ == "__main__":
    main()
