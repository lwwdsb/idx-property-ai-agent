"""Shared exact tests for the tuning loop. ONE implementation, so the gate that decides
"keep this change" and the gate that decides "the annotators prefer this side" cannot drift
apart — the same reason the metrics library is shared between the TS and Python runners.
"""
import math


def binom_two_sided(k, n):
    """Two-sided exact binomial p under H0: p=0.5, for k successes in n trials.

    Serves both tests the loop uses, because they are the same test:
      - McNemar on paired binary outcomes: n = discordant pairs, k = min(b, c)
      - Sign test on pairwise preferences: n = non-tied judgements, k = wins for one side
    Ties / concordant pairs carry no information about the difference and are excluded from n,
    which is precisely why this is stronger than comparing two point estimates.
    """
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_exact(b, c):
    """b = A wins the pair, c = B wins it, concordant pairs excluded."""
    return binom_two_sided(min(b, c), b + c)


def sign_test(wins_a, wins_b):
    """Pairwise preferences with ties already dropped."""
    return binom_two_sided(wins_a, wins_a + wins_b)
