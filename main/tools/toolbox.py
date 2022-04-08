import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from numpy.typing import ArrayLike
from typing import Callable, Tuple

from main.individual.individual import Individual


def selection_rank_with_population_replacement_elite(
    population: list[Individual], elite_size=0.1, new_pop=0.2
) -> list[Individual]:
    """Select the best individuals for the previous generation method.

    Args:
        population: The current population.
        elite_size: The best n percent of individuals to include in the next generation.
        new_pop: The n percent of migrants who will join the population. (new individuals)

    Returns:
        selected: Selected individuals.

    """
    sorted_individuals = sorted(population, key=lambda ind: ind.get_sharpe(), reverse=True)
    best_n_individuals = int(np.floor(len(sorted_individuals) * elite_size))
    new_individuals = int(np.floor(len(sorted_individuals) * new_pop))
    rank_distance = 1 / len(population)
    ranks = [(1 - i * rank_distance) for i in range(len(population))]
    ranks_sum = sum(ranks)
    selected = sorted_individuals[:best_n_individuals]

    for i in range(len(sorted_individuals) - best_n_individuals - new_individuals):
        shave = random.random() * ranks_sum
        rank_sum = 0
        for i in range(len(sorted_individuals)):
            rank_sum += ranks[i]
            if rank_sum > shave:
                selected.append(sorted_individuals[i])
                break
    new_individuals = [Individual.create_random() for _ in range(new_individuals)]
    selected.extend(new_individuals)

    return selected


def mutation_operation(population: list[Individual], method: Callable, prob: float) -> list[Individual]:
    """Decides if the current individual must mutate or not.

    Args:
        population: Current population.
        method: Function to use in crossover operation.
        prob: float between 0 and 1.

    Returns:
        mutated_offspring: Mutated population.

    """
    mutated_offspring = []
    for mutant in population:
        if random.random() < prob:
            new_mutant = method(mutant)
            mutated_offspring.append(new_mutant)
        else:
            mutated_offspring.append(mutant)
    return mutated_offspring


def mutation_stocks_fitness_driven(ind: Individual, max_tries: int = 3) -> Individual:
    """Mutation operation.

    This mutation approach selects n random stocks from the Individual
    and replaces them with the same number of assets randomly chosen
    from the asset universe. If the resultan individual is better than
    the initial individual it will replace the initial one. Otherwise,
    it will be reattempted as many times as indicated by the argument
    max_retries.

    Args:
        ind: The given individual to mutate.
        max_tries: Max number of retries.

    Returns:
        Initial or mutated individual.

    """
    for t in range(0, max_tries):
        mut_s = copy.deepcopy(ind.portfolio_idx)
        mut_w = copy.deepcopy(ind.portfolio_weights)
        n = random.randint(1, len(ind.portfolio_idx))
        stocks_to_mutate = random.sample(range(len(ind.portfolio_idx)), n)
        new_stocks = random.sample(range(1, ind.universe.shape[1]), n)
        mut_s[stocks_to_mutate] = new_stocks
        mutated = Individual(portfolio_idx=mut_s, portfolio_weights=mut_w)
        if mutated.get_sharpe() > ind.get_sharpe():
            return mutated
    return ind


def crossover_operation(population: list[Individual], method: Callable, prob: float) -> list[Individual]:
    """Decides if the current individual must be selected for crossover operation.

    Args:
        population: Current population.
        method: Function to use in crossover operation.
        prob: float between 0 and 1.

    Returns:
        crossed_offspring: Crossed offspring population.

    """
    crossed_offspring = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < prob:
            kid1, kid2 = method(ind1, ind2)
            crossed_offspring.append(kid1)
            crossed_offspring.append(kid2)
        else:
            crossed_offspring.append(ind1)
            crossed_offspring.append(ind2)

    best_of_population = max(population, key=lambda ind: ind.get_sharpe())
    best_of_offspring = max(crossed_offspring, key=lambda ind: ind.get_sharpe())

    if best_of_population.get_sharpe() > best_of_offspring.get_sharpe():  # TODO: It's not an ordered list!!!
        crossed_offspring[-1] = best_of_population

    return crossed_offspring


def arithmetic_roulette_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """Crossover operation.

    Since this approach allows for mutations among the assets of the individuals,
    I have limited the maximum number of assets that an individual can have to 20.
    This means that mod 20 will be used to prevent the number of assets of the individual
    from getting larger and larger.

    For each pair of individuals whose number of assets adds up to less than 20 assets,
    the offspring will have the sum of both parents (avoiding duplicities) and the weights
    will be inherited according to the following equation:

    > Offspring A = α ∗ Parent1 + (1 -α) ∗ Parent2.
    > Offspring B = (1 -α) ∗ Parent1 + α ∗ Parent2

    Where α is a random number between 0 and 1.

    On the other hand, if two parents have 14 and 17 assets respectively due to work in
    module 20 their offspring will have 11 assets.

    Args:
        parent1: First individual.
        parent2: Second individual.

    Returns:
        child1 & child2 both crossed offspring.

    """
    alpha = np.random.rand()
    l1 = len(parent1.portfolio_idx)
    l2 = len(parent2.portfolio_idx)
    p1w = parent1.portfolio_weights
    p2w = parent2.portfolio_weights
    p1f = parent1.portfolio_idx
    p2f = parent2.portfolio_idx

    if l1 + l2 > 20:
        from_p1 = int(np.floor(l1 * 0.5)) if l1 % 2 == 0 else int(np.ceil(l1 * 0.5))
        from_p2 = int(np.floor(l2 * 0.5)) if l2 % 2 == 0 else int(np.ceil(l2 * 0.5))

        p1_idx = random.sample(range(0, l1), from_p1)
        p2_idx = random.sample(range(0, l2), from_p2)

        c1pf = list(p1f[p1_idx])
        c1pf.extend(list(p2f[p2_idx]))

        c1w = list(p1w[p1_idx])
        c1w.extend(list(p2w[p2_idx]))
        c1w /= sum(c1w)

        l = (l1 + l2) % 20

        if l == 0:

            p1_idx = random.sample(range(0, l1), from_p1)
            p2_idx = random.sample(range(0, l2), from_p2)

            c2pf = list(p1f[p1_idx])
            c2pf.extend(list(p2f[p2_idx]))

            c2w = list(p1w[p1_idx])
            c2w.extend(list(p2w[p2_idx]))
            c2w /= sum(c2w)

        elif l == 1:

            if alpha > 0.5:
                p_idx = random.sample(range(0, l1), 1)
                c2pf = p1f[p_idx]
                c2w = [1]
            else:
                p_idx = random.sample(range(0, l2), 1)
                c2pf = p2f[p_idx]
                c2w = [1]
        else:
            p = int(np.floor(l * 0.5)) if l1 % 2 == 0 else int(np.ceil(l * 0.5))
            delta = l - p
            p1_idx = random.sample(range(0, l1), p)
            p2_idx = random.sample(range(0, l2), delta)

            c2pf = list(p1f[p1_idx])
            c2pf.extend(list(p2f[p2_idx]))

            c2w = list(p1w[p1_idx])
            c2w.extend(list(p2w[p2_idx]))
            c2w /= sum(c2w)

        c1pf = np.array(c1pf)
        c1w = np.array(c1w)
        c2pf = np.array(c2pf)
        c2w = np.array(c2w)

        c1pf, c1w = merge_duplicates(portfolio_idx=c1pf, porfolio_weights=c1w)
        c2pf, c2w = merge_duplicates(portfolio_idx=c2pf, porfolio_weights=c2w)

        child1 = Individual(portfolio_idx=c1pf, portfolio_weights=c1w)
        child2 = Individual(portfolio_idx=c2pf, portfolio_weights=c2w)

    else:
        cf = list(p1f)
        cf.extend(list(p2f))

        c1w1 = p1w * alpha
        c1w2 = p2w * (1 - alpha)
        c1w = list(c1w1)
        c1w.extend(c1w2)
        c1w = np.array(c1w)

        c2w1 = (1 - alpha) * p1w
        c2w2 = alpha * p2w
        c2w = list(c2w1)
        c2w.extend(c2w2)
        c2w = np.array(c2w)

        cf_ = cf.copy()
        cf_ = np.array(cf_)
        cf = np.array(cf)

        cf, c1w = merge_duplicates(portfolio_idx=cf, porfolio_weights=c1w)
        cf_, c2w = merge_duplicates(portfolio_idx=cf_, porfolio_weights=c2w)

        child1 = Individual(portfolio_idx=cf, portfolio_weights=c1w)
        child2 = Individual(portfolio_idx=cf_, portfolio_weights=c2w)

    return child1, child2


def stats(
    population: list[Individual],
    best_ind: Individual,
    fit_avg: list[float],
    fit_best: list[float],
    fit_best_ever: list[float],
) -> Tuple[Individual, list[float], list[float], list[float]]:
    """Return the most relevant individuals.

    Args:
        population: Current population.
        best_ind: Previous population bet individual.
        fit_avg: Fitness average of current population.
        fit_best: Fitness of the best individual for the current population.
        fit_best_ever: The best fittness ever registered.

    Returns:
        Tuple[best_ind, fit_avg, fit_best, fit_best_ever]: updated values for the current population.

    """
    best_of_generation = max(population, key=lambda ind: ind.get_sharpe())
    if best_ind.get_sharpe() < best_of_generation.get_sharpe():
        best_ind = best_of_generation
    fit_avg.append(sum([ind.get_sharpe() for ind in population]) / len(population))
    fit_best.append(best_of_generation.get_sharpe())
    fit_best_ever.append(max(fit_best + fit_best_ever))

    return best_ind, fit_avg, fit_best, fit_best_ever


def plot_stats(fit_avg: list[float], fit_best_ever: list[float], title: str):
    """Plot the best and the average fitness for each generation.

    Args:
        fit_avg: List of fitness averages.
        fit_best_ever: The best fitness for each generation.
        title: Just a title.

    """
    plt.plot(fit_avg, label="Average Fitness of Gen")
    plt.plot(fit_best_ever, label="Best Fitness")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_frontier(risk: list[float], ret: list[float], lr: Individual, br: Individual, bs: Individual):
    """Plot Markowitz efficient frontier.

    Args:
        risk: List of all individuals risk.
        ret: List of all individuals returns.
        lr: Lowest risk individual.
        br: Best return individual.
        bs: Best sharpe ratio individual.

    """
    plt.figure(figsize=(20, 15))
    plt.scatter(risk, ret, c=list(np.divide(ret, risk)), cmap="viridis", linewidths=1)
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.scatter(lr.risk(), lr.expected_return(), s=500, c="red", marker=(5, 2))
    plt.scatter(bs.risk(), bs.expected_return(), s=300, c="green", marker=(5, 1))
    plt.scatter(br.risk(), br.expected_return(), s=300, c="yellow", marker=(5, 0))
    plt.show()


def dupl_pmcguire(seq: ArrayLike) -> dict:
    """Optimal way to classify repeated values.

    Args:
        seq: NumPy Array object with numeric values.

    Returns:
        Dictionary where keys are the repeated element in the array and value
        is a list with the position of the kye in the given array.

    """
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return dict([(key, locs) for key, locs in tally.items() if len(locs) > 1])


def merge_duplicates(portfolio_idx: ArrayLike, porfolio_weights: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Merge duplicated values.

    Due to the crossover operation some individuals may have repeated stocks indices,
    this functions merge all of them into one single stock where the weight is the
    aggregated sum of all repeated instances.

    Args:
        portfolio_idx: Indices of the current allocations.
        porfolio_weights: Weights for each allocation.

    Returns:
        pfi: Indices if the stocks in the asset universe.
        pfw: Weights for each asset.

    """
    where = dupl_pmcguire(portfolio_idx)
    pf = []
    w = []

    pf.extend(list(set(where.keys())))
    idx = [l for sublist in list(where.values()) for l in sublist]
    portfolio_idx = np.delete(portfolio_idx, idx)

    for v in where.values():
        weights = sum(porfolio_weights[v])
        w.append(weights)

    porfolio_weights = np.delete(porfolio_weights, idx)
    pf.extend(portfolio_idx)
    w.extend(porfolio_weights)

    pfi = np.array(pf)
    pfw = np.array(w)

    return pfi, pfw
