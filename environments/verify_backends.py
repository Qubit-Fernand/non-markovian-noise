"""Check TensorCircuit Bell state and automatic differentiation per backend."""
import argparse
import json

import numpy as np
import tensorcircuit as tc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["numpy"])
    args = parser.parse_args()
    results = []
    for backend in args.backends:
        with tc.runtime_backend(backend), tc.runtime_dtype("complex128"):
            k = tc.backend
            circuit = tc.Circuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            zz = float(np.real(k.numpy(circuit.expectation_ps(z=[0, 1]))))
            np.testing.assert_allclose(zz, 1, atol=1e-7)
            record = dict(backend=backend, bell_zz=zz)
            if backend != "numpy":
                def energy(theta):
                    c = tc.Circuit(1)
                    c.rx(0, theta=theta)
                    return k.real(c.expectation_ps(z=[0]))
                theta = k.cast(k.convert_to_tensor(.3), "float64")
                value, derivative = k.value_and_grad(energy)(theta)
                value, derivative = float(k.numpy(value)), float(k.numpy(derivative))
                np.testing.assert_allclose(value, np.cos(.3), atol=1e-7)
                np.testing.assert_allclose(derivative, -np.sin(.3), atol=1e-7)
                record.update(energy=value, derivative=derivative)
                if backend in ("jax", "tensorflow"):
                    jitted = float(k.numpy(k.jit(energy)(theta)))
                    np.testing.assert_allclose(jitted, value, atol=1e-7)
                    record["jitted_energy"] = jitted
            results.append(record)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
