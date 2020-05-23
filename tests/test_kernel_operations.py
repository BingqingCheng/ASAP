"""
Test for the ml_kernel_operations.py module.
"""

import numpy as np
import pytest

from asaplib.kernel.ml_kernel_operations import kerneltodis, kerneltodis_linear, kerneltorho, normalizekernel


@pytest.mark.parametrize("kernel_matrix", [
    np.array([(2, 2), (2, 2)], dtype=np.float32),
    np.array([(1, 2), (1, 2)], dtype=np.float32),
])
def test_normalize_kernel(kernel_matrix):
    """
    Test for the normalize_kernel function.

    TODO: Second input is not a valid kernel matrix yet the function still returns a value
    """
    normalized_kernel = normalizekernel(kernel_matrix)

    # Check that ones are placed on the diagonal
    assert np.sum(np.diag(normalized_kernel)) == len(normalized_kernel)

    # Check that all kernel elements are less than or equal to 1
    assert np.all(normalized_kernel <= 1)


@pytest.mark.parametrize("kernel_matrix", [
    np.array([(2, 2), (2, 2)], dtype=np.float32),
    np.array([(1, 2), (2, 1)], dtype=np.float32),
])
def test_kernel_to_dis(kernel_matrix):
    """
    Test for the kernel to distance matrix function which uses the transformation: d_ij = sqrt(2 - 2*k_ij) where k_ij
    is an element of the normalized kernel. This function assumes that the input is a valid kernel matrix i.e. that
    the diagonal elements are equal to each other.
    """
    distance_matrix = kerneltodis(kernel_matrix)
    normalized_kernel = normalizekernel(kernel_matrix)
    assert np.all(distance_matrix == 2 - 2*normalized_kernel)


@pytest.mark.parametrize("kernel_matrix", [
    np.array([(2, 2), (2, 2)], dtype=np.float32),
    np.array([(1, 2), (2, 1)], dtype=np.float32),
])
def test_kernel_to_dis_linear(kernel_matrix):
    """
    Test for the kernel to distance matrix function which uses the transformation: d_ij = 1-k_ij where k_ij
    is an element of the normalized kernel. This function assumes that the input is a valid kernel matrix i.e. that
    the diagonal elements are equal to each other.
    """
    distance_matrix = kerneltodis_linear(kernel_matrix)
    normalized_kernel = normalizekernel(kernel_matrix)
    assert np.all(distance_matrix == 1 - normalized_kernel)


@pytest.mark.parametrize("kernel_matrix", [
    np.array([(2, 2), (2, 2)], dtype=np.float32),
    np.array([(1, 2), (2, 1)], dtype=np.float32),
])
def test_kernel_to_rho(kernel_matrix):
    """
    Test for the kernel to density matrix function. Assumes that the input is a valid kernel matrix.
    """
    delta = 1
    rho_matrix = kerneltorho(kernel_matrix, delta)

    # Assert that all elements of the density matrix are greater than zero.
    assert np.all(rho_matrix >= 0)
