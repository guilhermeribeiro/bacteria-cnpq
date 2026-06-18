import numpy as np
import random
import matplotlib.pyplot as plt


class GeneNetwork:
    def __init__(self, edges):
        """
        edges: lista de pares (i, j) = i reprime j
        """
        self.edges = edges

    def repression_strength(self, node):
        # número de repressores que chegam ao nó
        print(self.edges)
        print(node)
        return sum(1 for (_, j) in self.edges if j == node)


PARAMS = {
    "k_tx": 0.02,     # transcrição basal MUITO fraca
    "k_deg": 0.01,    # degradação
    "k_bind": 0.05,
    "k_unbind": 0.1
}


def gillespie_gene(network, input_level, T=1000):
    """
    SSA com binding/unbinding explícito
    """
    X = 0          # proteína de saída
    bound = 0      # promotor reprimido ou não
    t = 0.0

    repression = network.repression_strength("out")
    leak = 0.001 + 0.002 * repression

    while t < T:
        rates = []

        # transcrição
        if bound == 0:
            rates.append(("prod", PARAMS["k_tx"]))
        else:
            rates.append(("prod", leak))  # vazamento mínimo

        # degradação
        rates.append(("deg", PARAMS["k_deg"] * X))

        # binding
        rates.append(("bind", PARAMS["k_bind"] * repression * input_level * (1 - bound)))

        # unbinding
        rates.append(("unbind", PARAMS["k_unbind"] * bound))

        total = sum(r for _, r in rates)
        if total == 0:
            break

        t += np.random.exponential(1 / total)

        r = random.random() * total
        acc = 0
        for name, rate in rates:
            acc += rate
            if r < acc:
                if name == "prod":
                    X += 1
                elif name == "deg":
                    X = max(X - 1, 0)
                elif name == "bind":
                    bound = 1
                elif name == "unbind":
                    bound = 0
                break

    return X


def sample_distribution(net, input_level, n=1024):
    return np.array([
        gillespie_gene(net, input_level)
        for _ in range(n)
    ])


def compute_snr(high, low):
    z = high - low
    return z.mean() / (z.std(ddof=1) + 1e-9)



human_inverter = GeneNetwork(edges=[("in", "out")])

low  = sample_distribution(human_inverter, input_level=1)
high = sample_distribution(human_inverter, input_level=10)

snr_human = compute_snr(high, low)
print("Human inverter SNR:", snr_human)


def mutate_topology(topology, n_edges, n_mutations=4):
    new_edges = topology.edges.copy()

    for _ in range(n_mutations):
        i = random.randrange(len(new_edges))
        new_edges[i] = ("n" + str(random.randint(0, 3)), "out")

    return GeneNetwork(new_edges)



def random_topology(n_edges=3):
    return GeneNetwork(
        edges=[("n" + str(random.randint(0,3)), "out") for _ in range(n_edges)]
    )


def evolve_inverter(generations=64):
    parent = random_topology()
    best = None
    history = []

    for g in range(generations):
        child = mutate_topology(parent, n_edges=len(parent.edges), n_mutations=4)

        low  = sample_distribution(child, 1)
        high = sample_distribution(child, 10)

        snr = compute_snr(high, low)

        if best is None or snr >= best:
            parent = child
            best = snr

        history.append(best)

    return history


snr_curve = evolve_inverter()

plt.plot(snr_curve)
plt.xlabel("Generation")
plt.ylabel("SNR")
plt.title("Evolution of genetic inverter (Python reproduction)")
plt.show()
