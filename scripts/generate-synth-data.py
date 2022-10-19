import numpy as np

from chronostar.synthdata import generate_association
from chronostar.traceorbit import trace_epicyclic_orbit

if __name__ == '__main__':
    ages = [10., 35., 75.]

    final_means = [
        np.zeros(6),
        np.array([100., 0., 0., 5., 0., 0.]),
        np.array([0., 100., 0., 0., 5., 0.]),
    ]

    birth_means = []
    for m, a in zip(final_means, ages):
        birth_means.append(trace_epicyclic_orbit(m[np.newaxis], -a).squeeze())

    birth_covs = np.array(np.broadcast_to(np.eye(6), (3, 6, 6)))

    birth_dxyz = 10.
    birth_duvw = 2.

    birth_covs[:, :3, :3] *= birth_dxyz**2
    birth_covs[:, 3:, 3:] *= birth_duvw**2

    star_counts = [150, 350, 500]

    data = np.vstack([
        generate_association(m, bc, a, n) for m, bc, a, n in zip(
            final_means, birth_covs, ages, star_counts
        )
    ])

    true_membership = np.zeros(data.shape)
    true_membership[:star_counts[0], 0] = 1.
    true_membership[star_counts[0]:star_counts[0] + star_counts[1], 1] = 1.
    true_membership[star_counts[0] + star_counts[1]:, 2] = 1.

    np.save('data.npy', data)
    np.save('true_means.npy', birth_means)
    np.save('true_covs.npy', birth_covs)
    np.save('true_ages.npy', ages)
    np.save('true_membership.npy', true_membership)
