import numpy as np

from src.chronostar.component.spacetimecomponent \
    import apply_age_constraints


def generate_association(mean, covariance, age, nstars=100):
    aged_mean, aged_covariance = apply_age_constraints(
        mean, covariance, age,
    )
    rng = np.random.default_rng()
    # import ipdb; ipdb.set_trace()
    return rng.multivariate_normal(aged_mean, aged_covariance, size=nstars)


def generate_two_overlapping(age1, age2):
    dim = 6
    X_OFFSET = 30.
    V_OFFSET = 4.
    DV = 3.
    mean1 = np.zeros(dim)
    mean2 = np.copy(mean1)
    mean2[0] = X_OFFSET
    mean2[4] = V_OFFSET

    stdevs1 = np.array([age1*DV] * 3 + [DV] * 3)
    cov1 = np.eye(dim) * stdevs1

    stdevs2 = np.array([age2*DV] * 3 + [DV] * 3)
    cov2 = np.eye(dim) * stdevs2

    stars1 = generate_association(mean1, cov1, age1)
    stars2 = generate_association(mean2, cov2, age2)
    all_stars = np.vstack((stars1, stars2))
    return all_stars
