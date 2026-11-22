"""Gate tensors in interleaved output and input axis order."""

from __future__ import annotations

import jax.numpy as jnp


def createRXX(theta):
    anti_diag_sin = -1j * jnp.sin(theta / 2) * jnp.array([[0, 1], [1, 0]])
    return jnp.array(
        [
            [jnp.cos(theta / 2) * jnp.identity(2), anti_diag_sin],
            [anti_diag_sin, jnp.cos(theta / 2) * jnp.identity(2)],
        ]
    )


def createRZZ(theta):
    return (
        jnp.diag(
            jnp.exp(jnp.array([-theta * 1j / 2, theta * 1j / 2, theta * 1j / 2, -theta * 1j / 2]))
        )
        .reshape(2, 2, 2, 2)
        .transpose(1, 3, 2, 0)
    )


def createRX(theta):
    return jnp.array(
        [[jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)], [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)]]
    )


def createRZ(theta):
    return jnp.diag(jnp.exp(jnp.array([-1j * theta / 2, 1j * theta / 2])))


def _controlled_rotation(rotation):
    """Apply the rotation to the top qubit when the bottom qubit is one."""
    control_zero = jnp.array([[1, 0], [0, 0]])
    control_one = jnp.array([[0, 0], [0, 1]])
    matrix = jnp.kron(jnp.eye(2, dtype=rotation.dtype), control_zero) + jnp.kron(rotation, control_one)
    return matrix.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)


def createCRX(theta):
    return _controlled_rotation(createRX(theta))


def createCRZ(theta):
    return _controlled_rotation(createRZ(theta))


def createCNOT(dtype=jnp.complex128):
    return jnp.array(
        [
            [[[1, 0], [0, 0]], [[0, 0], [0, 1]]],
            [[[0, 0], [0, 1]], [[1, 0], [0, 0]]],
        ],
        dtype=dtype,
    )


def InitState(theta):
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def PaulyZ():
    return jnp.array([[1, 0], [0, -1]])


def createBMERABlock(type=1, params=None):
    """Build local RX/RZ rotations followed by CNOT (1), CRX (2), or CRZ (3)."""
    I = jnp.eye(4).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
    if type not in (1, 2, 3):
        raise ValueError("type can be only 1 to 3")
    if params is None:
        params = jnp.zeros((4 if type == 1 else 5,))

    first_mat = jnp.matmul(createRZ(params[2]), createRX(params[0]))
    second_mat = jnp.matmul(createRZ(params[3]), createRX(params[1]))
    if type == 1:
        entangler = createCNOT(first_mat.dtype)
    elif type == 2:
        entangler = createCRX(params[4])
    else:
        entangler = createCRZ(params[4])

    result = jnp.einsum("ijkl, jm -> imkl", I, first_mat)
    result = jnp.einsum("ijkl, lm -> ijkm", result, second_mat)
    return jnp.einsum("ijkl, jmln -> imkn", entangler, result)


def H(block: jnp.array):
    if block.shape != (2, 2, 2, 2):
        raise ValueError("for BMERABlocks only")
    block = jnp.einsum("ijkl -> jilk", block)
    return jnp.conjugate(block)
