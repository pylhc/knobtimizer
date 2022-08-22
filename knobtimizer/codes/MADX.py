
from pathlib import Path
import tfs
import numpy as np
import pandas as pd
from knobtimizer.codes.base_code_class import AcceleratorCode

class MADX(AcceleratorCode):
    TFS_DA = 'da.tfs'
    TFS_MA = 'ma.tfs'

    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        super().__init__(executable, template_file, template_directory, repair_mask)


    def assess_score(self, working_directory: Path) -> float:
        da = tfs.read(Path(working_directory) / self.TFS_DA)
        da_area = np.sum(-0.5 * (da['Y0SURV'][1:] - da['Y0SURV'][:-1])
                  * (da['X0SURV'][1:] - da['X0SURV'][:-1])
                  - da['Y0SURV'][:-1]
                  * (da['X0SURV'][1:] - da['X0SURV'][:-1]))/1.0E-3/1.0E-6

        ma = tfs.read(Path(working_directory) / self.TFS_MA)
        ma_area = np.sum(-0.5 * (ma['X0SURV'][:-1] - ma['X0SURV'][1:])
                  * (ma['D0SURV'][:-1] - ma['D0SURV'][1:])
                  - ma['X0SURV'][1:]
                  * (ma['D0SURV'][:-1] - ma['D0SURV'][1:]))/1.0E-3/1.0E-3

        return da_area*ma_area


    def repair(self, strengths:dict, working_directory:Path) -> np.array:
        # use madx mask to correct chroma
        # Note: check in MAD-X mask if K2 or K2L is used

        self.fill_template(strengths, working_directory, self.repair_mask, True)
        self.run_script(working_directory, self.repair_mask)
        knob_strengths=pd.read_csv(Path(working_directory)/'knob_strength.str',
                                sep='=|;',
                                usecols=[1],
                                header=None,
                                dtype=float,
                                engine='python',
                                names=['K2'])

        return knob_strengths['K2'].to_numpy()