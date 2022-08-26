import logging
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import tfs

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from knobtimizer.codes.base_code_class import AcceleratorCode

LOGGER = logging.getLogger(__name__)
class KnobOptimization(ElementwiseProblem):

    def __init__(self,
                 knobs: list[str],
                 max_knob_val: float,
                 assessment_method: AcceleratorCode,
                 repair_method: AcceleratorCode=None,
                 **kwargs):

        no_sextupole_circuits = len(knobs)

        # define max and min integrated sextupole strength as upper and lower bounds
        xl = np.ones(no_sextupole_circuits)*-1*max_knob_val
        xu = np.ones(no_sextupole_circuits)*max_knob_val

        self.knobs=knobs
        self.assessment_method = assessment_method
        self.repair_method = repair_method

        # single object is product of DA areas
        super().__init__(n_var=no_sextupole_circuits, n_obj=1, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        strengths = pd.Series(index=self.knobs, data=x).to_dict()

        with tempfile.TemporaryDirectory(prefix='scoring') as temp_directory:
            score=self.assessment_method.return_score(strengths, Path(temp_directory))
        
        out["F"] = -score
        # implement chroma not as constraints, but as repair operator, since for chroma, one needs specific combinations and not a range per variable
        # out["G"] = 0


def save_results(result: list, knobs: list[str], filename:Path) -> None:
    strengths = pd.DataFrame(data={'Knob':knobs,'KnobStrength':result.X})
    LOGGER.info(f'Saving results to {filename}.')
    tfs.write(filename, strengths)


def repair_fun(X, problem):
    with tempfile.TemporaryDirectory(prefix='repairing') as temp_directory:
        try:
            X = problem.repair_method.repair(
                    pd.Series(index=problem.knobs, data=X).to_dict(),
                    Path(temp_directory)
                    )
        except NotImplementedError:
            pass
        return X


class KnobRepair(Repair):
    
    def _do(self, problem, X, **kwargs):
        
        return problem.elementwise_runner(lambda x: repair_fun(x, problem), X)
        