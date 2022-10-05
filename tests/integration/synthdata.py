import numpy as np

from src.chronostar.component.spacetimecomponent \
    import apply_age_constraints


def generate_association(mean, covariance, age, nstars=100, rng=None):
    aged_mean, aged_covariance = apply_age_constraints(
        mean, covariance, age,
    )
    if rng is None:
        rng = np.random.default_rng()
    # import ipdb; ipdb.set_trace()
    return rng.multivariate_normal(aged_mean, aged_covariance, size=nstars)


def generate_two_overlapping(
    age1,
    age2,
    nstars1=1_000,
    nstars2=1_000,
    rng=None
):
    dim = 6
    X_OFFSET = 50.
    V_OFFSET = 5.
    DV = 3.
    mean1 = np.zeros(dim)
    mean2 = np.copy(mean1)
    mean2[0] = X_OFFSET
    mean2[4] = V_OFFSET

    stdevs1 = np.array([age1*DV] * 3 + [DV] * 3)
    cov1 = np.eye(dim) * stdevs1

    stdevs2 = np.array([age2*DV] * 3 + [DV] * 3)
    cov2 = np.eye(dim) * stdevs2

    if rng is None:
        rng = np.random.default_rng()

    stars1 = generate_association(mean1, cov1, age1, nstars=nstars1, rng=rng)
    stars2 = generate_association(mean2, cov2, age2, nstars=nstars2, rng=rng)

    all_stars = np.vstack((stars1, stars2))
    return all_stars
