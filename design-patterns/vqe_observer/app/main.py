import numpy as np
from vqe_observer.subjects.vqe_evaluator import VQEEvaluator as Evaluator
from vqe_observer.observers.logger import Logger
from vqe_observer.observers.best_tracker import BestTracker
from vqe_observer.observers.visualizer import Visualizer
from vqe_observer.observers.param_shift_gd import ParamShiftGD
from vqe_observer.observers.param_shift_adam import ParamShiftAdam
from vqe_observer.utils.problem import initial_params, build_2qubit_problem
from vqe_observer.utils.verify import verify_final_state

def main() -> None:

    pauli_str = "2 I0I1 - 2 X0X1 + 3 Y0Y1 - 3 Z0Z1"
    H, ansatz = build_2qubit_problem(pauli_str)
    theta0 = initial_params(ansatz, mode="zeros")  # zeros | random

    # create evaluator (subject)
    evaluator = Evaluator(H=H, ansatz=ansatz, theta0=theta0, max_iters=1000 )

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

    # create other observers
    optimizer = optimizer_Adam
    logger = Logger()
    best_tracker = BestTracker()
    visualizer = Visualizer(f"{optimizer.name()} with {evaluator.name()}")
    
    # attach all observers to subject
    for o in [optimizer, logger, best_tracker, visualizer,]:
        evaluator.attach(o)
  
    # start subject, a loop which notifies all observers and may stop earlier (epsilon, limit)
    evaluator.execute()

    # post-run reports and cleanup
    logger.dump()
    best_tracker.dump()
    visualizer.dump()

    print("Final theta:", np.array2string(evaluator.params, precision=6, floatmode='fixed'))
    verify_final_state(ansatz, evaluator.params, H)

if __name__ == "__main__":
    main()
