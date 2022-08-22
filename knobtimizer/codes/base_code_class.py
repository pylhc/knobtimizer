from pathlib import Path
import abc
import subprocess
import jinja2

class AcceleratorCode(abc.ABC):
    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        self.executable = executable
        self.template_file = template_file
        self.repair_mask = repair_mask
        # Note: assumes both templates are stored in one template_directory
        self.template_directory = template_directory


    def run_script(self, working_directory: Path, template_file: str) -> None:
        with open(working_directory/self.return_log_name(template_file), 'w') as f: 
            subprocess.run([self.executable,
                            template_file],
                            check=True,
                            stdout=f,
                            cwd=working_directory)


    def return_log_name(self, template_file) -> Path:
        return Path(template_file).with_suffix('').with_suffix('.log')


    def fill_template(self, fill_dictionary: dict, working_directory: Path, template_file:str, strict: bool) -> Path:
        loader = jinja2.FileSystemLoader(searchpath=self.template_directory)
        if strict:
            env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
        else:
            env = jinja2.Environment(loader=loader)

        template = env.get_template(template_file)
        template_path = working_directory/Path(template_file)
        with open(template_path, 'w') as f:
            f.write(template.render(**fill_dictionary))    
        return template_path


    def return_score(self, strengths: dict, working_directory: Path):
        self.fill_template(strengths, working_directory, self.template_file, True)
        self.run_script(working_directory, self.template_file)
        return self.assess_score(working_directory)


    def assess_score(self, working_directory: Path = None) -> float:
        return NotImplementedError


    def repair(self, strengths: dict, working_directory: Path = None):
        raise NotImplementedError
        