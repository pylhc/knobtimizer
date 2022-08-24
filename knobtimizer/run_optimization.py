#!/usr/bin/env python
"""
The knobtimization toolkit allows a parameter optimization using a number of different algorithms provided by `pymoo` and combining it with a number of different distributed computing options using `dask`.
The evaluation is performed by using template files, which are executed by a given code.

The main requirement is a list of knobs/variables and a template file, where these knobs can be substitute in.
The template file should follow the jinja convention, e.g. for a knob named 'KB1', the template file should include {{KB1}}.
This filled-in file will then be executed using a code defined in codes.
Each defined code should have an accompanying Codeclass, located in the codes folder.
Based on the ouput of the code, the `assess_score` method will then return the score for each run.

Additionally, a repair mask can be provided, which allows to adjust the knob value to account for other constraints.

"""
import sys
import yaml
import importlib
from pathlib import Path
import socket
import shutil
import numpy as np
import jinja2
import pandas as pd
import dill
import logging

from generic_parser import EntryPointParameters, entrypoint
from knobtimizer.iotools import save_config, PathOrStr, timeit

from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.optimize import minimize
from pymoo.core.problem import DaskParallelization
from pymoo.core.problem import LoopedElementwiseEvaluation
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from dask.distributed import Client, LocalCluster
from dask_jobqueue import HTCondorCluster

import knobtimizer.optimization_toolkit as toolkit

LOGGER = logging.getLogger(__name__)

REPOSITORY_TOP_LEVEL = Path(__file__).resolve().parent.parent

MADX_EXECUTABLE = REPOSITORY_TOP_LEVEL/'knobtimizer'/'codes'/'madx'

CHECKPOINT_FILE="checkpoint.pkl"
RESULTS_FILE="results.tfs"

CLUSTER={
    'HTC':{'dask_queue':HTCondorCluster,
           'cluster_parameter':{
                # 'n_workers':10,
                'cores':1,
                'memory':'2000MB',
                'disk':'1000MB',
                'death_timeout' : '60',
                'nanny' : False,
                'scheduler_options':{
                    'port': 8786,
                    'host': socket.gethostname()
                    },
                'job_extra':{
                    'log': 'dask_job_output.log',
                    'output': 'dask_job_output.out',
                    'error': 'dask_job_output.err',
                    'should_transfer_files': 'Yes',
                    'when_to_transfer_output': 'ON_EXIT',
                    '+JobFlavour': '"tomorrow"',
                    },
                'extra' : ['--worker-port 10000:10100']
                }
            },
    'local':{
        'dask_queue':LocalCluster,
        'cluster_parameter':{
            # 'n_workers':10,
            'nanny' : False,
            'processes':False,
        }
    },
}

ALGORITHMS={
    'PSO':PSO,
    'DE':DE,
    'NelderMead':NelderMead, 
    'GA':GA,
    'NSGA2':NSGA2, 
}


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name='cluster',
        type=str,
        required=True,
        choices=list(CLUSTER.keys()),
        help='Which cluster to use.'
    )
    params.add_parameter(
        name='codes',
        type=dict,
        default={
            'TEST':{'executable':MADX_EXECUTABLE},
        },
        help='Defines code classes to use and possible extra arguments required in the initialization.'
    )
    params.add_parameter(
        name='code_path',
        type=PathOrStr,
        default=REPOSITORY_TOP_LEVEL/'knobtimizer'/'codes',
        help='Define path where code classes are located.'
    )
    params.add_parameter(
        name='algorithm',
        type=str,
        required=True,
        help='Which optimization algorithm to use.'
    )
    params.add_parameter(
        name='generations',
        type=int,
        default=10,
        help='Number of generations to use for evolutionary algorithms.'
    )
    params.add_parameter(
        name='population',
        type=int,
        default=10,
        help='Populationsize.'
    )
    params.add_parameter(
        name='working_directory',
        type=PathOrStr,
        required=True,
        help='Directory where template and results will be saved to.'
    )
    params.add_parameter(
        name='knobs',
        type=str,
        nargs='+',
        default=None,
        required=True,
        help='List of knob names.'
    )
    params.add_parameter(
        name='max_knob_value',
        type=float,
        default=2., # coincidently, this is the K2L max for the FCC-ee arc sextupoles
        help='Maximum knobvalue.'
    )
    params.add_parameter(
        name='start_values',
        type=float,
        nargs='+',
        default=None,
        help='Starting values for each knob.'
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
    params.add_parameter(
        name='dryrun',
        type=bool,
        default=False,
        help='Flag to enable a dryrun. During dryrun, the assess_score and repair method will be run once in the working directory to check timing.'
    )
    params.add_parameter(
        name='checkpoint',
        type=bool,
        default=False,
        help=f'Flag to enable a checkpointing. If enabled, the status of the algorithm will be saved in a pickle file. If a {CHECKPOINT_FILE} is present in the working_directory, it will be loaded as initialization.'
    )

    return params


@entrypoint(get_params(), strict=True)
def main(opt):
    LOGGER.info('Knobtimization started!')

    # check options and convert types where necessary
    save_config(Path(opt.working_directory), opt, "knobtimizer")
    LOGGER.info(f'Configuration saved to {opt.working_directory}.')
    opt = check_opt(opt)

    # take replace dict from yaml and fill in template in place
    if opt.replace_file is not None:
        fill_in_replace_dict(opt)

    LOGGER.info('Create code classes.')
    assess_methods = create_code_classes(opt)
    
    if opt.dryrun:
        LOGGER.info('Dryrun started.')
        dryrun(opt, assess_methods=assess_methods)
    else:
        LOGGER.info('Run optimization.')
        result = run_optimization(opt, assess_methods=assess_methods)
        return result.X


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

    if opt.start_values is None:
        opt.start_values = [0]*len(opt.knobs)
        LOGGER.debug(f'No start values given, initialized with zeros.')

    if opt.algorithm not in ALGORITHMS:
        raise AttributeError(f"Algorithm {opt.algorithm} not available.")

    if opt.population < 3:
        raise AttributeError("Population size smaller 3 should be avoided due to issues in pymoo display.")
    
    return opt


def fill_in_replace_dict(opt: dict) -> dict:
        LOGGER.debug('Fill template file with replace dict.')
        fill_template(opt.config_dict['REPLACE_DICT'],
                      opt.working_directory,
                      opt.template_file,
                      opt.working_directory,
                      strict=False)
        if opt.repair_mask is not None:
            LOGGER.debug('Fill repair mask with replace dict.')
            fill_template(opt.config_dict['REPLACE_DICT'],
                          opt.working_directory,
                          opt.repair_mask,
                          opt.working_directory,
                          strict=False)


def fill_template(fill_dictionary: dict, template_directory: Path, template_file: str, working_directory: Path, strict: bool) -> Path:
        loader = jinja2.FileSystemLoader(searchpath=template_directory)
        if strict:
            env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
        else:
            env = jinja2.Environment(loader=loader)

        template = env.get_template(template_file)
        template_path = working_directory/Path(template_file)
        with open(template_path, 'w') as f:
            f.write(template.render(**fill_dictionary))    
        return template_path


def create_code_classes(opt: dict) -> dict:
    assess_methods={}
    for code in opt.codes:
        LOGGER.info(f'Loading code class {code}.py from {opt.code_path}.')
        spec = importlib.util.spec_from_file_location(code, opt.code_path/f'{code}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[code] = mod
        spec.loader.exec_module(mod)

        code_class = getattr(mod, code)
        assess_methods[code]=code_class(
            template_file=opt.template_file,
            template_directory=opt.working_directory,
            repair_mask=opt.repair_mask,
            **opt.codes[code],
            )
        LOGGER.info(f'Accelerator code class {code} created.')

    return assess_methods


def dryrun(opt: dict, assess_methods: dict) -> None:
    with timeit(lambda t: LOGGER.info(f'Time for assess score: {t} s')):
        work_directory = opt.working_directory/'Assess_score'
        work_directory.mkdir(parents=True, exist_ok=True)
        score=assess_methods[opt.assessment_method].return_score(
            pd.Series(index=opt.knobs, data=opt.start_values).to_dict(),
            work_directory
            )
        LOGGER.info(f'Assessment score: {score}')
    if opt.repair_mask is not None:
        with timeit(lambda t: LOGGER.info(f'Time for repair method: {t} s')):
            work_directory = opt.working_directory/'Repair_method'
            work_directory.mkdir(parents=True, exist_ok=True)
            score=assess_methods[opt.repair_method].repair(
                pd.Series(index=opt.knobs, data=opt.start_values).to_dict(),
                    work_directory
                    )
            LOGGER.info(f'Repair return values: {score}')


def run_optimization(opt: dict, assess_methods: dict):
    with CLUSTER[opt.cluster]['dask_queue'](**CLUSTER[opt.cluster]['cluster_parameter']) as cluster:

        with Client(cluster) as client:
            cluster.scale(opt.population)
            if opt.algorithm == 'NelderMead':
                LOGGER.info('NelderMead does not with dask and cluster setting is ignored. Optimization will run locally.')
                runner = LoopedElementwiseEvaluation() # Simplex broken when used in conjunction with dask
            else:
                LOGGER.info(f'Running optimization on {opt.cluster}.')
                runner = DaskParallelization(client)

            problem = toolkit.KnobOptimization(
                knobs=opt.knobs,
                max_knob_val=opt.max_knob_value,
                assessment_method=assess_methods[opt.assessment_method],
                repair_method=assess_methods[opt.repair_method],
                elementwise_runner=runner, 
            )

            if opt.checkpoint and (opt.working_directory/CHECKPOINT_FILE).is_file():
                LOGGER.info(f'Loading checkpoint from {opt.working_directory/CHECKPOINT_FILE}.')
                with open(opt.working_directory/CHECKPOINT_FILE, 'rb') as f:
                    algorithm = dill.load(f)
                    algorithm.termination = MaximumGenerationTermination(algorithm.n_gen+opt.generations)
                    algorithm.problem = None # needed to reload problem, otherwise uses old/prev. dask runner which has timed out
            else:
                start_values = np.zeros((opt.population, len(opt.knobs)))
                start_values[0, :] = np.array(opt.start_values)
                for i in range(1, opt.population):
                    start_values[i, :] = start_values[0, :] + np.random.normal(scale=0.1*opt.max_knob_value, size=problem.n_var)

                algorithm=ALGORITHMS[opt.algorithm](
                    pop_size=opt.population,
                    adaptive=True,
                    sampling=start_values,
                    repair=toolkit.KnobRepair()
                    )
                
                algorithm.termination = MaximumGenerationTermination(opt.generations)
                LOGGER.info(f'Set up algorithm {opt.algorithm} with Population {opt.population} and with {opt.generations} Generations.')

            LOGGER.info('Optimization started.')
            res = minimize(
                problem,
                algorithm=algorithm,
                seed=1,
                save_history=False,
                verbose=True,
                copy_algorithm=False,
            )
            LOGGER.info('Optimization finished.')

            if opt.checkpoint:
                LOGGER.info(f'Save checkpoint to {opt.working_directory/CHECKPOINT_FILE}.')
                with open(opt.working_directory/CHECKPOINT_FILE, "wb") as f:
                    dill.dump(algorithm, f)

        toolkit.save_results(res, opt.knobs, opt.working_directory/RESULTS_FILE)
    return res


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()