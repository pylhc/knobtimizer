# Knobtimizer

This package is a wrapper around the multi-objective optimization framework [pymoo](https://pymoo.org/),
with [dask](https://www.dask.org/) as scheduler, to support use with different codes and via templates.

A custom-created class with a scoring function has to be implemented, as well as providing a template file and the path to an executable.
With those, optimization of the provided knobs will be performed using a number of different algorithms (PSO, NelderMead, GA, ..).

## Installation

To install the package, run

```bash
git clone https://github.com/pylhc/knobtimizer
python -m pip install --editable "knobtimizer[all]"
```

## Example use

The package allows the find values for a set of variable/knobs, for which the scoring function is maximized.
To do so, several preparatory steps are required beforehand.

First, a template file for use with the associated codes needs to be prepared.
The template needs to contain the names of the knobs, which will be replaced before execution.
The template has to follow the [jinja2](https://palletsprojects.com/p/jinja/) template convention.
For example, if the knobnames are `Kn1` and `Kn2`, the template has to contain `{{Kn1}}` and `{{Kn2}}`.

Second, an Accelerator Code class has to be written and put in `knobtimizer/codes`.
The class name and module name should be the same.
This class should provide an `assess_score` method, which loads the output of the run and returns the score.

A simple accelerator class can look like this:

```python
class TEST(AcceleratorCode):
    def __init__(self, executable: str, template_file: str, template_directory: str, repair_mask: str = None, **kwargs) -> None:
        super().__init__(executable, template_file, template_directory, repair_mask)

    def assess_score(self, working_directory:Path) -> float:
        # load file1 and file2
        return file1[Score] + file2[Score]
```

To launch the optimization, the easiest way is to call the `main` function in a python script, like so:

```python
import knobtimizer.run_optimization

if __name__ == "__main__":
    knobtimizer.run_optimization.main(
    cluster='local',
    codes={'TEST': PathToExecutable},
    algorithm='PSO',
    working_directory=PathToWorkingDirectory,
    knobs= ['Kn1', 'Kn2'],
    template_file=PathToTemplate,
    assessment_method='TEST',
    population=20,
    generations=5
    )
```

Launching this script will perform an optimization using the ParticleSwarmOptimization (PSO) algorithm, running with 20 indivduals for 5 generations, and using multiprocessing on the local machine.
After the optimization is finished, a results file will be saved in the `workingdirectory`.

## License

This project is licensed under the `MIT License` - see the [LICENSE](LICENSE) file for details.
