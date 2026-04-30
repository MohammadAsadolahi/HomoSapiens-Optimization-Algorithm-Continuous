"""
Generate all publication-quality plots for the Homo Sapiens Optimization (HSO) Algorithm.
Includes: convergence curves, benchmark comparisons, population dynamics,
algorithm mechanism illustrations, and 3D surface visualizations.
"""

import numpy as np
from numpy import random, abs, cos, exp, mean, pi, prod, sin, sqrt, sum, arange, e
import matplotlib.pyplot as plt
import matplotlib
import copy as c
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['figure.dpi'] = 150

# ──────────────────────────────────────────────────────────────────
# Benchmark Functions
# ──────────────────────────────────────────────────────────────────


def levy(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (sin(pi * z[0])**2
            + sum((z[:-1] - 1)**2 * (1 + 10 * sin(pi * z[:-1] + 1)**2))
            + (z[-1] - 1)**2 * (1 + sin(2 * pi * z[-1])**2))


def ackley_nd(x):
    x = np.asarray(x)
    n = len(x)
    return -20.0 * exp(-0.2 * sqrt(sum(x**2) / n)) - exp(sum(cos(2 * pi * x)) / n) + e + 20


def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x**2 - A * np.cos(2 * pi * x)) for x in X])


def sphere(x):
    return sum(np.array(x)**2)


def rosenbrock(x):
    x = np.asarray(x)
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def griewank(x):
    x = np.asarray(x)
    s = sum(x**2) / 4000
    p = np.prod(cos(x / sqrt(arange(1, len(x) + 1))))
    return s - p + 1

# ──────────────────────────────────────────────────────────────────
# Organism Class (HSO Core)
# ──────────────────────────────────────────────────────────────────


class Organism:
    def __init__(self, location, direction, health_decay, loss, optimization_dims):
        self.direction = direction
        self.optimization_dims = optimization_dims
        self.location = location
        self.health = 1.0
        self.health_decay = health_decay
        self.speed = 0.001
        self.average_return = 0
        self.gamma = 0.10
        self.loss = loss(self.location)
        self.relative_loss = 0
        self.absolute_loss = self.loss

    def reduceStamina(self):
        self.health *= self.health_decay
        if self.health < 0.01:
            return True
        return False

    def update_loss(self, similarity, beta):
        self.absolute_loss = (
            (1 - beta) * self.relative_loss) + (beta * similarity)

    def move(self, loss):
        difference = loss(self.location) - loss(self.location + self.direction)
        self.average_return += self.gamma * (self.average_return - difference)
        self.change_direction()
        self.location += self.direction
        self.loss = loss(self.location)
        return self.reduceStamina()

    def change_direction(self):
        if self.average_return <= 0:
            randomness_scale = np.interp(
                1 - self.health, (0, 1), (0.0005, 0.01))
            direction_change = np.random.laplace(
                loc=0, scale=randomness_scale, size=self.optimization_dims)
            self.direction += direction_change

    def clone(self):
        self.health = min(self.health / self.health_decay, 1)
        return self.location

    def offspting_cunductor(self, radicalism):
        direction = np.zeros(self.optimization_dims)
        indexes = np.random.randint(0, direction.size, size=(
            np.random.randint(1, max(2, int(self.optimization_dims / 10) + 1))))
        direction[indexes] = random.uniform(-radicalism, +
                                            radicalism, size=indexes.shape)
        return direction

# ──────────────────────────────────────────────────────────────────
# HSO Runner
# ──────────────────────────────────────────────────────────────────


def run_hso(loss_function, optimization_dims=30, initial_population=2000,
            beta_init=0.25, beta_decay=0.99, health_decay=0.976,
            rounds=300, offspring_birth_rate_init=15, search_range=4.0,
            verbose=False, max_population=500):
    """Run HSO and return convergence history + population history."""
    beta = beta_init
    population = []
    initial_locations = [random.uniform(-search_range, search_range,
                                        size=optimization_dims) for _ in range(initial_population)]
    initial_directions = [
        random.uniform(-beta, beta, size=optimization_dims) for _ in range(initial_population)]
    for l, d in zip(initial_locations, initial_directions):
        population.append(Organism(c.deepcopy(l), c.deepcopy(
            d), health_decay, loss_function, optimization_dims))

    elites = []
    pop_sizes = []
    offspring_birth_rate = offspring_birth_rate_init

    for r in range(rounds):
        dead_flag = []
        offsprings = []
        beta *= beta_decay

        population.sort(key=lambda organism: organism.loss)
        elites.append(population[0].loss)
        pop_sizes.append(len(population))

        # Cap population to prevent O(n^2) blowup on similarity
        if len(population) > max_population:
            population = population[:max_population]

        locations = np.array([p.location for p in population])
        losses = np.array([p.loss for p in population])
        similarity = np.inner(locations, locations)
        i = 0
        for s in range(similarity.shape[0]):
            similarity[s][i] = 0
            i += 1
        similarity = np.sum(similarity, axis=1)

        sim_range = np.max(similarity) - np.min(similarity)
        if sim_range > 0:
            similarity = (similarity - np.min(similarity)) / sim_range
        else:
            similarity = np.zeros_like(similarity)

        loss_range = np.max(losses) - np.min(losses)
        if loss_range > 0:
            losses_norm = (losses - np.min(losses)) / loss_range
        else:
            losses_norm = np.zeros_like(losses)

        for h, s, l in zip(population, similarity, losses_norm):
            h.relative_loss = l
            h.update_loss(s, beta)

        if verbose and r % 50 == 0:
            print(f"  iter:{r} pop:{len(population)} best:{elites[-1]:.6f}")

        population.sort(key=lambda organism: organism.absolute_loss)

        spring_counts = offspring_birth_rate
        for organism in population:
            if spring_counts > 0:
                base_location = organism.clone()
                for i in range(spring_counts):
                    offsprings.append(Organism(c.deepcopy(base_location),
                                               c.deepcopy(
                                                   organism.offspting_cunductor(beta)),
                                               health_decay, loss_function, optimization_dims))
                spring_counts -= 1
                dead_flag.append(False)
            else:
                dead_flag.append(organism.move(loss_function))

        offspring_birth_rate = max(10, offspring_birth_rate - 1)
        dead_flag = np.where(np.array(dead_flag, dtype=bool) == True)
        population = [i for j, i in enumerate(
            population) if j not in dead_flag[0]] + offsprings

    return elites, pop_sizes, population[0].location

# ──────────────────────────────────────────────────────────────────
# Comparison: Simple PSO
# ──────────────────────────────────────────────────────────────────


def run_pso(loss_function, dims=30, n_particles=200, rounds=300, search_range=4.0):
    w, c1, c2 = 0.7, 1.5, 1.5
    positions = random.uniform(-search_range,
                               search_range, (n_particles, dims))
    velocities = random.uniform(-0.1, 0.1, (n_particles, dims))
    pbest = positions.copy()
    pbest_scores = np.array([loss_function(p) for p in positions])
    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    history = []
    for _ in range(rounds):
        r1, r2 = random.rand(n_particles, dims), random.rand(n_particles, dims)
        velocities = w * velocities + c1 * r1 * \
            (pbest - positions) + c2 * r2 * (gbest - positions)
        positions += velocities
        scores = np.array([loss_function(p) for p in positions])
        improved = scores < pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = scores[improved]
        if np.min(pbest_scores) < gbest_score:
            gbest_idx = np.argmin(pbest_scores)
            gbest = pbest[gbest_idx].copy()
            gbest_score = pbest_scores[gbest_idx]
        history.append(gbest_score)
    return history

# ──────────────────────────────────────────────────────────────────
# Comparison: Differential Evolution
# ──────────────────────────────────────────────────────────────────


def run_de(loss_function, dims=30, pop_size=200, rounds=300, search_range=4.0, F=0.8, CR=0.9):
    population = random.uniform(-search_range, search_range, (pop_size, dims))
    scores = np.array([loss_function(p) for p in population])
    best_idx = np.argmin(scores)
    history = [scores[best_idx]]
    for _ in range(rounds):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, cc = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - cc), -search_range, search_range)
            cross_points = random.rand(dims) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dims)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_score = loss_function(trial)
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score
        history.append(np.min(scores))
    return history[:rounds]

# ──────────────────────────────────────────────────────────────────
# Comparison: Genetic Algorithm
# ──────────────────────────────────────────────────────────────────


def run_ga(loss_function, dims=30, pop_size=200, rounds=300, search_range=4.0, mutation_rate=0.1):
    population = random.uniform(-search_range, search_range, (pop_size, dims))
    history = []
    for _ in range(rounds):
        scores = np.array([loss_function(p) for p in population])
        history.append(np.min(scores))
        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            parent1 = population[i] if scores[i] < scores[j] else population[j]
            i, j = np.random.choice(pop_size, 2, replace=False)
            parent2 = population[i] if scores[i] < scores[j] else population[j]
            # Crossover
            mask = random.rand(dims) < 0.5
            child = np.where(mask, parent1, parent2)
            # Mutation
            if random.rand() < mutation_rate:
                mut_idx = np.random.randint(0, dims, size=max(1, dims // 10))
                child[mut_idx] += random.normal(0, 0.5, size=len(mut_idx))
            new_pop.append(child)
        population = np.array(new_pop)
    return history

# ──────────────────────────────────────────────────────────────────
# Comparison: Simulated Annealing
# ──────────────────────────────────────────────────────────────────


def run_sa(loss_function, dims=30, rounds=300, search_range=4.0, T_init=100, cooling=0.995):
    current = random.uniform(-search_range, search_range, dims)
    current_score = loss_function(current)
    best = current.copy()
    best_score = current_score
    T = T_init
    history = []
    steps_per_round = 200  # match eval budget roughly
    for r in range(rounds):
        for _ in range(steps_per_round):
            neighbor = current + random.normal(0, 0.3, dims)
            neighbor_score = loss_function(neighbor)
            delta = neighbor_score - current_score
            if delta < 0 or random.rand() < np.exp(-delta / max(T, 1e-10)):
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            T *= cooling
        history.append(best_score)
    return history

# ──────────────────────────────────────────────────────────────────
# PLOT GENERATION
# ──────────────────────────────────────────────────────────────────


os.makedirs("assets", exist_ok=True)

COLORS = {
    'hso': '#1a73e8',       # Google Blue
    'pso': '#ea4335',       # Google Red
    'de': '#fbbc04',        # Google Yellow
    'ga': '#34a853',        # Google Green
    'sa': '#9334e6',        # Purple
    'bg': '#0d1117',
    'grid': '#30363d',
    'text': '#f0f6fc',
}


def style_dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS['bg'])
    ax.set_title(title, fontsize=15, fontweight='bold',
                 color=COLORS['text'], pad=12)
    ax.set_xlabel(xlabel, fontsize=12, color=COLORS['text'])
    ax.set_ylabel(ylabel, fontsize=12, color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'], which='both')
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])


# ═══════════════════════════════════════════════════════════════════
# PLOT 1: 3D Benchmark Surface Visualizations
# ═══════════════════════════════════════════════════════════════════
print("Generating 3D benchmark surfaces...")


def levy_2d(x, y):
    z1 = 1 + (x - 1) / 4
    z2 = 1 + (y - 1) / 4
    return (np.sin(np.pi * z1)**2 + (z1 - 1)**2 * (1 + 10 * np.sin(np.pi * z1 + 1)**2)
            + (z2 - 1)**2 * (1 + np.sin(2 * np.pi * z2)**2))


def rastrigin_2d(x, y):
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def ackley_2d(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def rosenbrock_2d(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


fig = plt.figure(figsize=(20, 5), facecolor=COLORS['bg'])
surfaces = [
    ("Levy Function", levy_2d, (-4, 4)),
    ("Rastrigin Function", rastrigin_2d, (-5.12, 5.12)),
    ("Ackley Function", ackley_2d, (-5, 5)),
    ("Rosenbrock Function", rosenbrock_2d, (-3, 3)),
]
for idx, (name, func, bounds) in enumerate(surfaces, 1):
    ax = fig.add_subplot(1, 4, idx, projection='3d', facecolor=COLORS['bg'])
    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.85,
                    edgecolor='none', antialiased=True)
    ax.set_title(name, fontsize=11, fontweight='bold',
                 color=COLORS['text'], pad=0)
    ax.tick_params(colors=COLORS['text'], labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(COLORS['grid'])
    ax.yaxis.pane.set_edgecolor(COLORS['grid'])
    ax.zaxis.pane.set_edgecolor(COLORS['grid'])
    ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.savefig("assets/benchmark_surfaces.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> benchmark_surfaces.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Multi-Benchmark Convergence Comparison (30D)
# ═══════════════════════════════════════════════════════════════════
print("Running 30D benchmark comparisons (this may take a few minutes)...")

benchmarks_30d = {
    'Levy': levy,
    'Rastrigin': rastrigin,
    'Sphere': sphere,
    'Rosenbrock': rosenbrock,
}

dims = 10
rounds = 150

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLORS['bg'])
axes = axes.flatten()

for idx, (bench_name, bench_func) in enumerate(benchmarks_30d.items()):
    ax = axes[idx]
    print(f"  Running {bench_name} {dims}D...")

    print(f"    HSO...")
    hso_hist, hso_pop, _ = run_hso(
        bench_func, dims, initial_population=200, rounds=rounds, verbose=False)
    print(f"    PSO...")
    pso_hist = run_pso(bench_func, dims, n_particles=80, rounds=rounds)
    print(f"    DE...")
    de_hist = run_de(bench_func, dims, pop_size=80, rounds=rounds)
    print(f"    GA...")
    ga_hist = run_ga(bench_func, dims, pop_size=80, rounds=rounds)

    ax.semilogy(hso_hist, color=COLORS['hso'],
                linewidth=2.5, label='HSO (Ours)', zorder=5)
    ax.semilogy(pso_hist, color=COLORS['pso'],
                linewidth=1.5, label='PSO', alpha=0.85)
    ax.semilogy(de_hist, color=COLORS['de'],
                linewidth=1.5, label='DE', alpha=0.85)
    ax.semilogy(ga_hist, color=COLORS['ga'],
                linewidth=1.5, label='GA', alpha=0.85)

    style_dark_ax(ax, f"{bench_name} (D={dims})",
                  "Iteration", "Best Fitness (log)")
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'], loc='upper right')

plt.tight_layout(pad=2.0)
plt.savefig("assets/convergence_30d.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> convergence_30d.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: High-Dimensional Scalability (Levy: 10D, 30D, 50D, 100D)
# ═══════════════════════════════════════════════════════════════════
print("Running scalability analysis (Levy function at multiple dimensions)...")

fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['bg'])
dim_colors = ['#4fc3f7', '#1a73e8', '#1565c0', '#0d47a1']
dim_configs = [
    (5,   150,  100),
    (10,  200,  120),
    (20,  300,  150),
    (30,  400,  180),
]

for (d, pop, rnds), color in zip(dim_configs, dim_colors):
    print(f"  HSO Levy D={d}...")
    hist, _, _ = run_hso(levy, d, initial_population=pop,
                         rounds=rnds, verbose=False)
    ax.semilogy(hist, color=color, linewidth=2.2, label=f'D={d}')

style_dark_ax(ax, "HSO Scalability — Levy Function Across Dimensions",
              "Iteration", "Best Fitness (log)")
ax.legend(fontsize=13, facecolor='#161b22', edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'], loc='upper right')
plt.tight_layout()
plt.savefig("assets/scalability.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> scalability.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: Population Dynamics
# ═══════════════════════════════════════════════════════════════════
print("Running population dynamics analysis...")

hist, pop_sizes, _ = run_hso(
    levy, 10, initial_population=200, rounds=120, verbose=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=COLORS['bg'])

# Population size over time
ax1.fill_between(range(len(pop_sizes)), pop_sizes,
                 alpha=0.3, color=COLORS['hso'])
ax1.plot(pop_sizes, color=COLORS['hso'], linewidth=2)
style_dark_ax(ax1, "Population Size Dynamics", "", "Population Size")

# Convergence
ax2.semilogy(hist, color='#00e676', linewidth=2.5)
ax2.fill_between(range(len(hist)), hist, alpha=0.15, color='#00e676')
style_dark_ax(ax2, "Convergence Trajectory — Levy 10D",
              "Iteration", "Best Fitness (log)")

plt.tight_layout(pad=2.0)
plt.savefig("assets/population_dynamics.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> population_dynamics.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 5: Algorithm Mechanism - Exploration vs Exploitation
# ═══════════════════════════════════════════════════════════════════
print("Generating mechanism illustrations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=COLORS['bg'])

# (a) Beta decay schedule
ax = axes[0]
beta_vals = [0.25]
for i in range(499):
    beta_vals.append(beta_vals[-1] * 0.99)
ax.plot(beta_vals, color='#ff6d00', linewidth=2.5)
ax.fill_between(range(len(beta_vals)), beta_vals, alpha=0.2, color='#ff6d00')
ax.axhline(y=0.05, color='#ff1744', linestyle='--', alpha=0.5, linewidth=1)
ax.text(350, 0.06, 'Exploitation Dominant',
        fontsize=10, color='#ff1744', alpha=0.7)
ax.text(10, 0.22, 'Exploration Phase', fontsize=10, color='#ff6d00', alpha=0.8)
style_dark_ax(ax, "β Decay Schedule\n(Exploration → Exploitation)",
              "Iteration", "β Value")

# (b) Laplace perturbation scaling with health
ax = axes[1]
healths = np.linspace(1, 0.01, 500)
sigmas = np.interp(1 - healths, (0, 1), (0.0005, 0.01))
ax.plot(1 - healths, sigmas, color='#00e5ff', linewidth=2.5)
ax.fill_between(1 - healths, sigmas, alpha=0.15, color='#00e5ff')
ax.set_xlim(0, 1)
style_dark_ax(ax, "Laplace Perturbation Scale\n(Age-Adaptive Exploration)",
              "Age (1 - health)", "σ (perturbation scale)")
ax.annotate('Young: precise\nsearch', xy=(0.1, 0.001), fontsize=9, color='#b0bec5',
            ha='center')
ax.annotate('Old: desperate\nexploration', xy=(0.85, 0.008), fontsize=9, color='#b0bec5',
            ha='center')

# (c) Laplace vs Gaussian distributions
ax = axes[2]
x_range = np.linspace(-0.05, 0.05, 1000)
laplace_pdf = (1 / (2 * 0.005)) * np.exp(-np.abs(x_range) / 0.005)
gaussian_pdf = (1 / (0.005 * np.sqrt(2 * np.pi))) * \
    np.exp(-x_range**2 / (2 * 0.005**2))
ax.plot(x_range, laplace_pdf, color='#00e5ff',
        linewidth=2.5, label='Laplace (HSO)')
ax.plot(x_range, gaussian_pdf, color='#ff6d00',
        linewidth=2, linestyle='--', label='Gaussian')
ax.fill_between(x_range, laplace_pdf, alpha=0.1, color='#00e5ff')
style_dark_ax(ax, "Laplace vs Gaussian Perturbation\n(Heavy-Tailed Exploration)",
              "Perturbation Δ", "Density")
ax.legend(fontsize=11, facecolor='#161b22',
          edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

plt.tight_layout(pad=2.0)
plt.savefig("assets/mechanism_exploration.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> mechanism_exploration.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 6: Algorithm Architecture Diagram (text-based)
# ═══════════════════════════════════════════════════════════════════
print("Generating algorithm flow diagram...")

fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

boxes = [
    (5, 9.2, "HOMO SAPIENS OPTIMIZATION (HSO)", '#1a73e8', 16),
    (2.5, 7.5,
     "1. Initialize Population\nN organisms in [-R, R]ᴰ", '#34a853', 11),
    (7.5, 7.5, "2. Evaluate Fitness\nf(x) for each organism", '#ea4335', 11),
    (2.5, 5.5, "3. Compute Diversity\nInner-product similarity\nMin-max normalization", '#fbbc04', 11),
    (7.5, 5.5, "4. Blended Ranking\nL = (1-β)·loss + β·similarity\nSort by absolute fitness", '#9334e6', 11),
    (2.5, 3.5, "5. Elite Reproduction\nTop-K clone + sparse mutation\nHealth restoration", '#00bcd4', 11),
    (7.5, 3.5, "6. Population Movement\nLaplace perturbation\nAge-adaptive σ", '#ff6d00', 11),
    (5, 1.8, "7. Selection Pressure\nRemove dead (health < 0.01) | β ← β × decay\nBirth rate ← max(10, rate - 1)", '#e91e63', 11),
]

for x, y, text, color, fs in boxes:
    bbox = dict(boxstyle='round,pad=0.6', facecolor=color,
                alpha=0.15, edgecolor=color, linewidth=2)
    ax.text(x, y, text, fontsize=fs, ha='center', va='center', color=COLORS['text'],
            fontweight='bold', bbox=bbox, family='monospace' if fs < 14 else 'sans-serif')

# Arrows
arrow_props = dict(arrowstyle='->', color='#8b949e', lw=1.5)
connections = [
    (2.5, 7.0, 2.5, 6.1),
    (7.5, 7.0, 7.5, 6.1),
    (2.5, 6.95, 7.5, 7.95),  # init -> eval
    (7.5, 4.95, 7.5, 4.1),
    (2.5, 4.95, 2.5, 4.1),
    (2.5, 5.0, 7.5, 5.9),    # diversity -> ranking
    (2.5, 2.95, 5, 2.3),
    (7.5, 2.95, 5, 2.3),
]

plt.tight_layout()
plt.savefig("assets/algorithm_flow.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> algorithm_flow.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 7: Final Results Summary Heatmap
# ═══════════════════════════════════════════════════════════════════
print("Generating results summary heatmap...")
print("  Running all benchmarks for summary table (30D)...")

results = {}
bench_funcs = {'Levy': levy, 'Rastrigin': rastrigin,
               'Sphere': sphere, 'Rosenbrock': rosenbrock}
algo_names = ['HSO', 'PSO', 'DE', 'GA']

for bench_name, bench_func in bench_funcs.items():
    print(f"  {bench_name}...")
    hso_h, _, _ = run_hso(bench_func, 10, 200, rounds=120)
    pso_h = run_pso(bench_func, 10, 80, rounds=120)
    de_h = run_de(bench_func, 10, 80, rounds=120)
    ga_h = run_ga(bench_func, 10, 80, rounds=120)
    results[bench_name] = [hso_h[-1], pso_h[-1], de_h[-1], ga_h[-1]]

fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

data = np.array([results[k] for k in bench_funcs.keys()])
# Log-scale for visualization
log_data = np.log10(np.clip(data, 1e-15, None))

im = ax.imshow(log_data, cmap='RdYlGn_r', aspect='auto', alpha=0.85)

ax.set_xticks(range(len(algo_names)))
ax.set_yticks(range(len(bench_funcs)))
ax.set_xticklabels(algo_names, fontsize=13,
                   fontweight='bold', color=COLORS['text'])
ax.set_yticklabels(list(bench_funcs.keys()), fontsize=13,
                   fontweight='bold', color=COLORS['text'])

for i in range(len(bench_funcs)):
    for j in range(len(algo_names)):
        val = data[i, j]
        text_color = 'white' if log_data[i, j] > np.median(
            log_data) else 'black'
        ax.text(j, i, f'{val:.2e}', ha='center', va='center', fontsize=11,
                fontweight='bold', color=text_color)

ax.set_title("Final Best Fitness Across Benchmarks (D=10)", fontsize=15,
             fontweight='bold', color=COLORS['text'], pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('log₁₀(fitness)', color=COLORS['text'], fontsize=11)
cbar.ax.tick_params(colors=COLORS['text'])

plt.tight_layout()
plt.savefig("assets/results_heatmap.png", dpi=200,
            bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  -> results_heatmap.png")

# Save results to file for README table
with open("assets/results_table.txt", "w") as f:
    f.write("| Benchmark | HSO (Ours) | PSO | DE | GA |\n")
    f.write("|-----------|-----------|-----|----|----|  \n")
    for bench_name in bench_funcs:
        vals = results[bench_name]
        row = f"| {bench_name} |"
        for v in vals:
            row += f" {v:.4e} |"
        f.write(row + "\n")

print("\n=== All plots generated successfully! ===")
print("Files saved in assets/ directory.")
