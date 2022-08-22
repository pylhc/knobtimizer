from pathlib import Path
import numpy as np
import re
from knobtimizer.codes.base_code_class import AcceleratorCode

SAD_SCORE_SYNTAX = r'Score: *([0-9]*)$\n'

class SAD(AcceleratorCode):
    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        super().__init__(executable, template_file, template_directory, repair_mask)


    def assess_score(self, working_directory:Path) -> float:
        logfiles =[f for f in working_directory.glob('*.log') ] 
        assert len(list(logfiles)) == 1
        with open(logfiles[0], 'r') as f:
            data = f.read()
        scores = re.findall(SAD_SCORE_SYNTAX, data, flags=re.MULTILINE)
        return np.prod([float(score) for score in scores])
