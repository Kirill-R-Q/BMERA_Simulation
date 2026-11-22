"""Compatibility imports for the PennyLane block builders."""

from bmera.backends.pennylane import (
    BMERA_CNOT_block,
    BMERA_CRX_block,
    BMERA_CRZ_block,
    BMERA_RXX_block,
    RXX,
)

__all__ = ["BMERA_CNOT_block", "BMERA_CRX_block", "BMERA_CRZ_block", "BMERA_RXX_block", "RXX"]
