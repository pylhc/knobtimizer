
from pathlib import Path
import tfs
from knobtimizer.codes.base_code_class import AcceleratorCode
import numpy as np

class MADXCHROM(AcceleratorCode):
    TFS = 'twiss.tfs'

    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        super().__init__(executable, template_file, template_directory, repair_mask)

    def assess_score(self, working_directory: Path) -> float:
        tw = tfs.read(Path(working_directory) / self.TFS)        
        return -(np.abs(tw.headers['DQ1'])+np.abs(tw.headers['DQ2']))

    def repair(self, strengths:dict, working_directory:Path) -> np.array:

        return np.array(list(strengths.values()))