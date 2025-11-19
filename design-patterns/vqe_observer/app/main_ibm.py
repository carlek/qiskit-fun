import numpy as np
from vqe_observer.subjects.vqe_evaluator_ibm import IBMVQEEvaluator as Evaluator
from vqe_observer.observers.logger import Logger
from vqe_observer.observers.best_tracker import BestTracker
from vqe_observer.observers.visualizer import Visualizer
from vqe_observer.observers.param_shift_gd import ParamShiftGD
from vqe_observer.observers.param_shift_adam import ParamShiftAdam
from vqe_observer.utils.problem import initial_params, build_2qubit_problem
from vqe_observer.utils.verify import verify_final_state
from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    
    pauli_str = "2 I0I1 - 2 X0X1 + 3 Y0Y1 - 3 Z0Z1"
    H, ansatz = build_2qubit_problem(pauli_str)
    theta0 = initial_params(ansatz, mode="zeros")  # zeros | random

    service = QiskitRuntimeService(name="vqe-instance")
    backend_name = service.least_busy(operational=True, simulator=False)
    instance_name="crn:v1:bluemix:public:quantum-computing:us-east:a/537cf98871e141e3a56e559ed3c0e8ec:f68bcf88-13f2-4e8d-b1ce-fb88efa9a44f::",

    print(f"Using IBM Quantum backend: {backend_name}")
    evaluator = Evaluator(
        H=H,
        ansatz=ansatz,
        theta0=theta0,
        backend=backend_name,
        instance=instance_name,
        shots=1000,
        optimization_level=1,
        resilience_level=1,
        seed=42,
    )

    # create optimizers (observers)
    # Gradient-descent optimizer using parameter shift
    optimizer_GD = ParamShiftGD(
        evaluator=evaluator,
        lr=0.05,
        grad_clip=None,
        momentum=0.0,
        shift=np.pi/2,
        epsilon=1e-07,
        limit=3,
    )
    
    # Adam optimizer
    optimizer_Adam = ParamShiftAdam(
        evaluator=evaluator,
        lr=0.05,
        beta1=0.7,
        beta2=0.999,
        shift=np.pi/2,
        epsilon=1e-07,
        limit=3,
    )

    optimizer = optimizer_GD  # or optimizer_GD

    # create other observers and attach all
    logger = Logger()
    best_tracker = BestTracker()
    visualizer = Visualizer(f"{optimizer.name()} with {evaluator.name()}")

    for obs in [optimizer, logger, best_tracker, visualizer]:
        evaluator.attach(obs)

    # Run optimization loop, may stop earlier (epsilon, limit))
    print("\nStarting VQE optimization...\n")
    evaluator.execute(max_iters=20)

    # Post-run reports and cleanup
    logger.dump()
    best_tracker.dump()
    visualizer.dump()

    print("\nFinal theta:", np.array2string(evaluator.params, precision=6, floatmode="fixed"))
    verify_final_state(ansatz, evaluator.params, H)

    if hasattr(evaluator, "cleanup"):
        evaluator.cleanup()


if __name__ == "__main__":
    main()
