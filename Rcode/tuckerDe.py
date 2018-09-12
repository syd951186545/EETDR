from tensorly.decomposition import  _tucker

def partial_tucker(tensor, rank=None, n_iter_max=100, init='svd', tol=10e-5,
                   random_state=None, verbose=False, ranks=None):
    """Partial tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition exclusively along the provided modes.

    Parameters
    ----------
    tensor : ndarray
    modes : int list
            list of the modes on which to perform the decomposition
    ranks : None or int list
            size of the core tensor, ``(len(ranks) == len(modes))``
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity

    Returns
    -------
    core : ndarray
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``
    """
    modes = list(range(_tucker.T.ndim(tensor)))
    if ranks is not None:
        message = "'ranks' is depreciated, please use 'rank' instead"
        _tucker.warnings.warn(message, DeprecationWarning)
        rank = ranks

    if rank is None:
        rank = [_tucker.T.shape(tensor)[mode] for mode in modes]
    elif isinstance(rank, int):
        message = "Given only one int for 'rank' intead of a list of {} modes. Using this rank for all modes.".format(len(modes))
        _tucker.warnings.warn(message, DeprecationWarning)
        rank = [rank for _ in modes]

    # SVD init
    print("SVD init")
    if init == 'svd':
        factors = []
        for index, mode in enumerate(modes):
            eigenvecs, _, _ = _tucker.T.partial_svd(_tucker.unfold(tensor, mode), n_eigenvecs=rank[index])
            factors.append(eigenvecs)
    else:
        rng = _tucker.check_random_state(random_state)
        core = _tucker.T.tensor(rng.random_sample(rank), **_tucker.T.context(tensor))
        factors = [_tucker.T.tensor(rng.random_sample((_tucker.T.shape(tensor)[mode], rank[index])), **_tucker.T.context(tensor)) for (index, mode) in enumerate(modes)]

    rec_errors = []
    norm_tensor = _tucker.T.norm(tensor, 2)

    print("decomposition")
    for iteration in range(n_iter_max):
        for index, mode in enumerate(modes):
            core_approximation = _tucker.multi_mode_dot(tensor, factors, modes=modes, skip=index, transpose=True)
            eigenvecs, _, _ = _tucker.T.partial_svd(_tucker.unfold(core_approximation, mode), n_eigenvecs=rank[index])
            factors[index] = eigenvecs

        core = _tucker.multi_mode_dot(tensor, factors, modes=modes, transpose=True)

        # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
        rec_error = _tucker.sqrt(abs(norm_tensor**2 - _tucker.T.norm(core, 2)**2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return core, factors