"""Run a four-qubit BMERA circuit on CPU."""

import jax
import jax.numpy as jnp

from bmera import build_bmera_meta, forward_single


def main():
    with jax.default_device(jax.devices("cpu")[0]):
        meta = build_bmera_meta(n_qubits=4, type=2)
        params = jnp.full((meta["depth"], 4, meta["n_per_block"]), 0.1)
        angles = jnp.array([0.1, -0.2, 0.3, -0.4])
        print("Pauli-Z expectations", forward_single(params, angles, meta))


if __name__ == "__main__":
    main()
