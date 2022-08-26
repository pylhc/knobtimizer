
import yaml
import shutil
from pathlib import Path
import tfs
import logging

from generic_parser import EntryPointParameters, entrypoint
from knobtimizer.iotools import save_config, PathOrStr, timeit
from knobtimizer.run_optimization import (MADX_EXECUTABLE, fill_in_replace_dict, create_code_classes)

LOGGER = logging.getLogger(__name__)
REPOSITORY_TOP_LEVEL = Path(__file__).resolve().parent.parent

def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name='codes',
        type=dict,
        default={
            'TEST':{'executable':MADX_EXECUTABLE},
        },
        help='Defines codes and executables to use.'
    )
    params.add_parameter(
        name='code_path',
        type=PathOrStr,
        default=REPOSITORY_TOP_LEVEL/'knobtimizer'/'codes',
        help='Define path where code classes are located.'
    )
    params.add_parameter(
        name='working_directory',
        type=PathOrStr,
        required=True,
        help='Directory where template and results will be saved to.'
    )
    params.add_parameter(
        name='results_file',
        type=PathOrStr,
        required=True,
        help='Result file with knob values.'
    )
    params.add_parameter(
        name='template_file',
        type=PathOrStr,
        required=True,
        help='Which template to use for DA evaluation.'
    )
    params.add_parameter(
        name='repair_mask',
        type=PathOrStr,
        default=None,
        help='Which template to use for repair operation.'
    )
    params.add_parameter(
        name='replace_file',
        type=PathOrStr,
        help='Path to yaml config file, containing additional information to fill into the template.'
    )
    params.add_parameter(
        name='assessment_method',
        type=str,
        required=True,
        help='Which code to use. Should be defined in the codes parameter, and a corresponding code class added to the codes directory.'
    )
    params.add_parameter(
        name='repair_method',
        type=str,
        default=None,
        help='Which code to use for repair method. Should be defined in the codes parameter, and a corresponding code class added to the codes directory.'
    )
    
    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOGGER.info('Verify results started.')

    # check options and convert types where necessary
    save_config(Path(opt.working_directory), opt, "verify_results")
    LOGGER.info(f'Configuration saved to {opt.working_directory}.')
    opt = check_opt(opt)
    
    # take replace dict from yaml and fill in template in place
    if opt.replace_file is not None:
        fill_in_replace_dict(opt)

    LOGGER.info('Create code classes.')
    assess_methods = create_code_classes(opt)

    LOGGER.info(f'Load results file from {opt.results_file}.')
    results = tfs.read(opt.results_file, index='Knob')

    LOGGER.info('Start run results.')
    run_results(opt, assess_methods=assess_methods, results=results)


def check_opt(opt: dict) -> dict:
   
    opt.working_directory = Path(opt.working_directory)
    opt.working_directory.mkdir(parents=True, exist_ok=True)

    opt.code_path = Path(opt.code_path)
    
    if opt.replace_file is not None:
        LOGGER.info(f'Load YAML file {opt.replace_file} with replace dict.')
        with open(Path(opt.replace_file)) as f:
            opt.config_dict = yaml.safe_load(f)

    # copy template files to working directory and keep only the name for jinja
    if not Path(opt.template_file).is_file():
        raise AttributeError(f'Template file {opt.template_file} not found.')
    opt.template_file = Path(opt.template_file)
    shutil.copy(opt.template_file, opt.working_directory/opt.template_file.name)
    opt.template_file = opt.template_file.name
    LOGGER.debug(f'Template file {opt.template_file} copied to {opt.working_directory}.')

    # copy template files to working directory and keep only the name for jinja
    if opt.repair_mask is not None:
        if not (Path(opt.repair_mask)).is_file():
            raise AttributeError(f'Repair mask {opt.repair_mask} not found.')
    
        opt.repair_mask = Path(opt.repair_mask)
        shutil.copy(opt.repair_mask, opt.working_directory/opt.repair_mask.name)
        opt.repair_mask = opt.repair_mask.name
        LOGGER.debug(f'Template file {opt.repair_mask} copied to {opt.working_directory}.')
        
    if opt.repair_method is None:
        opt.repair_method = opt.assessment_method
        LOGGER.debug(f'No repair method selected, assessment method is used.')
    
    return opt


def run_results(opt: dict, assess_methods: dict, results:tfs.TfsDataFrame) -> None:
    with timeit(lambda t: LOGGER.info(f'Time for assess score: {t} s')):
        work_directory = opt.working_directory/'Assess_score'
        work_directory.mkdir(parents=True, exist_ok=True)
        score=assess_methods[opt.assessment_method].return_score(
            results.squeeze().to_dict(),
            work_directory
            )
        LOGGER.info(f'Assessment score: {score}')
    if opt.repair_mask is not None:
        with timeit(lambda t: LOGGER.info(f'Time for repair method: {t} s')):
            work_directory = opt.working_directory/'Repair_method'
            work_directory.mkdir(parents=True, exist_ok=True)
            score=assess_methods[opt.repair_method].repair(
                results.squeeze().to_dict(),
                    work_directory
                    )
            LOGGER.info(f'Repair return values: {score}')

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()