from pathlib import Path
from knobtimizer.codes.base_code_class import AcceleratorCode

class TEST(AcceleratorCode):
    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        super().__init__(executable, template_file, template_directory, repair_mask)

    def assess_score(self, working_directory:Path) -> float:
        return 1.