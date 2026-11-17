"""HEOM -> spatiotemporal Pauli process (SPP) -> error mitigation.

circuit     logical gates, learning layers, ideal layer unitaries
pulse       Segment / Frame / Slot and the pulse compiler
bath        exponential bath terms (classical and complex), noise presets
heom, tdvp  hierarchy engines on the [system, bath...] chain
spp         parallel construction of the time MPS, checks, sampling
mitigation  noise amplification (PEA/ZNE) and variational inverse (PEC)
"""
from .mps import MPS, Truncation
from .pauli import characters, paulis, walsh
