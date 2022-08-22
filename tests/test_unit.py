import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import dill
import tfs

import knobtimizer.sextupole_circuits
import knobtimizer.run_optimization
import knobtimizer.optimization_toolkit
import knobtimizer.codes.TEST
import knobtimizer.codes.SAD
import knobtimizer.codes.MADX
import knobtimizer.codes.MADXCHROM

REPOSITORY_TOP_LEVEL = Path(__file__).resolve().parent.parent
TEST_INPUT =  REPOSITORY_TOP_LEVEL/'tests'/'input'

TEST_KNOBS = [f'K2S{i}' for i in range(1, 2*73+1)]
RUN_MODE = 'local'

def test_run_da(tmp_path):
    TestClass = knobtimizer.codes.TEST.TEST(
        executable=knobtimizer.run_optimization.MADX_EXECUTABLE,
        template_file='FCCee_t_529.chroma.madx.template',
        template_directory=TEST_INPUT,
        )
    strengths=pd.Series(index=TEST_KNOBS, data=np.zeros(len(TEST_KNOBS))).to_dict()
    score=TestClass.return_score(
        strengths=strengths,
        working_directory=tmp_path
        )
    assert score == 1


def test_run_madxchrom(tmp_path):
    MadXChrom = knobtimizer.codes.MADXCHROM.MADXCHROM(
        executable=knobtimizer.run_optimization.MADX_EXECUTABLE,
        template_file='FODO_chrom.madx.template',
        template_directory=TEST_INPUT,
        )
    strengths=pd.Series(index=['K2S1', 'K2S2'], data=np.zeros(2)).to_dict()
    score=MadXChrom.return_score(
        strengths=strengths,
        working_directory=tmp_path
        )
    assert score+(0.4368240649+0.3247152235)<1e-12


def test_score_sad(tmp_path):
    SadClass = knobtimizer.codes.SAD.SAD(
        knobtimizer.run_optimization.SAD_EXECUTABLE,
        template_file='FCCee_t_529.sad.template',
        template_directory=TEST_INPUT,
        )
    strengths=pd.Series(index=TEST_KNOBS, data=np.zeros(len(TEST_KNOBS))).to_dict()

    score=SadClass.return_score(
        strengths=strengths,
        working_directory=tmp_path
        )
    assert score == np.prod([239, 205, 445, 238])


def test_chroma_sad(tmp_path):
    SadClass = knobtimizer.codes.SAD.SAD(
        knobtimizer.run_optimization.SAD_EXECUTABLE,
        template_file='FCCee_t_529.sad.template',
        template_directory=TEST_INPUT,
        )
    strengths=pd.Series(index=TEST_KNOBS, data=np.zeros(len(TEST_KNOBS))).to_dict()
    with pytest.raises(NotImplementedError):
        SadClass.repair(
            strengths,
            tmp_path
        )


def test_chroma_madx(tmp_path):
    MadxClass = knobtimizer.codes.MADX.MADX(
        executable=knobtimizer.run_optimization.MADX_EXECUTABLE,
        template_file='FCCee_t_529.chroma.madx.template',
        template_directory=TEST_INPUT,
        repair_mask='FCCee_t_529.chroma.madx.template'
        )
    strengths=pd.Series(index=TEST_KNOBS, data=np.zeros(len(TEST_KNOBS))).to_dict()
    MadxClass.repair(
        strengths,
        tmp_path
    )


@pytest.mark.parametrize("algorithm", knobtimizer.run_optimization.ALGORITHMS.keys())
def test_algorithms(tmp_path, algorithm):
    knobtimizer.run_optimization.main(
        cluster=RUN_MODE,
        codes={'MADXCHROM':{'executable':knobtimizer.run_optimization.MADX_EXECUTABLE}},
        algorithm=algorithm,
        working_directory=tmp_path,
        knobs=['K2S1', 'K2S2'],
        max_knob_value=1.e-2,
        template_file=TEST_INPUT/'FODO_chrom.madx.template',
        replace_file=None,
        assessment_method='MADXCHROM',
        population=10,
        generations=30,
    )
    res=tfs.read(tmp_path/'results.tfs')
    MadXChrom = knobtimizer.codes.MADXCHROM.MADXCHROM(
        executable=knobtimizer.run_optimization.MADX_EXECUTABLE,
        template_file='FODO_chrom.madx.template',
        template_directory=TEST_INPUT,
        )
    strengths=pd.Series(index=['K2S1', 'K2S2'], data=res['KnobStrength']).to_dict()
    score=MadXChrom.return_score(
        strengths=strengths,
        working_directory=tmp_path
        )
    assert np.abs(score)<1e-2


@pytest.mark.parametrize("algorithm", knobtimizer.run_optimization.ALGORITHMS.keys())
def test_run_checkpoint(tmp_path, algorithm):
    generations=3
    seed =1
    population=3
    knobtimizer.run_optimization.main(
        cluster=RUN_MODE,
        codes={'MADXCHROM':{'executable':knobtimizer.run_optimization.MADX_EXECUTABLE}},
        algorithm=algorithm,
        working_directory=tmp_path,
        knobs=['K2S1', 'K2S2'],
        max_knob_value=1.e-2,
        template_file=TEST_INPUT/'FODO_chrom.madx.template',
        replace_file=None,
        assessment_method='MADXCHROM',
        population=population,
        generations=generations,
        checkpoint=True
    )
    assert (tmp_path/knobtimizer.run_optimization.CHECKPOINT_FILE).is_file()
    with open(tmp_path/knobtimizer.run_optimization.CHECKPOINT_FILE, 'rb') as f:
        checkpoint = dill.load(f)
    assert checkpoint.n_gen == generations+1
    assert len(checkpoint.pop) == population
    assert checkpoint.seed == seed

    knobtimizer.run_optimization.main(
        cluster=RUN_MODE,
        codes={'MADXCHROM':{'executable':knobtimizer.run_optimization.MADX_EXECUTABLE}},
        algorithm=algorithm,
        working_directory=tmp_path,
        knobs=['K2S1', 'K2S2'],
        max_knob_value=1.e-2,
        template_file=TEST_INPUT/'FODO_chrom.madx.template',
        replace_file=None,
        assessment_method='MADXCHROM',
        population=population,
        generations=generations,
        checkpoint=True
    )


def test_run_repair(tmp_path):
    knobtimizer.run_optimization.main(
        cluster=RUN_MODE,
        algorithm='PSO',
        working_directory=tmp_path,
        knobs=TEST_KNOBS,
        template_file=TEST_INPUT/'FCCee_t_529.sad.template',
        repair_mask=TEST_INPUT/'FCCee_t_529.chroma.madx.template',
        replace_file=None,
        assessment_method='SAD',
        repair_method='MADX',
        population=3,
        generations=3,
    )