from pathlib import Path
import traceback

import pandas as pd

from iblutil.util import setup_logger
import ephys_atlas.rawephys
from ephys_atlas.data import atlas_pids
from one.api import ONE

ROOT_PATH = Path("/mnt/s1/ephys-atlas")
logger = setup_logger(name='paper-ephys-atlas', level='INFO')

TASKS = {
    'destripe_ap': {
        'version': '1.3.0',
    },
    'destripe_lf': {
        'version': '1.3.0',
    },
    'localise': {
        'version': '1.2.0',
        'depends_on': ['destripe_ap'],
    },
}

def report():
    """
    Looks at the folder and flag files according to the task specifications
    Builds a dataframe where each row is a pid, and each column is a task
    :return:
    """
    pids, alyx_pids = atlas_pids(one)
    pids = sorted(pids)
    flow = pd.DataFrame(index=pids, columns=list(TASKS.keys())).fillna('')

    for pid in flow.index:
        for t in TASKS:
            task_file = next(ROOT_PATH.joinpath(pid).glob(f'.{t}*'), None)
            if task_file:
                flow.loc[pid, t] = task_file.name
    return flow


def run_flow(pids=None):
    if pids is None:
        pids, alyx_pids = atlas_pids(one)
        pids = sorted(pids)

    for pid in pids:
        destripe_ap(pid)
        destripe_lf(pid)
        localise(pid)


def task(version='', depends_on=None, path_task=None, **kwargs):
    depends_on = [depends_on] if isinstance(depends_on, str) else depends_on
    path_task = path_task or ROOT_PATH
    def inner(func):
        def wrapper(pid, *args, **kwargs):
            # if the task has already run with the same version number, skip and exit
            flag_file = path_task.joinpath(pid).joinpath(f'.{func.__name__}_{version}')
            # remove an eventual error file on a previous run (TODO; rerun errors or not, here we assume yes)
            error_file = path_task.joinpath(pid).joinpath(f'.{func.__name__}_ERROR')
            if flag_file.exists():
                logger.info(f'skipping task {func.__name__} for pid {pid}')
                return
            # now remove all error files or previous version files
            for f in path_task.joinpath(pid).glob(f'.{func.__name__}_*'):
                f.unlink()
            # check that dependencies are met, exit if not
            if depends_on is not None:
                for parent_task in depends_on:
                    flag_parent = next(path_task.joinpath(pid).glob(f".{parent_task}*"), None)
                    if flag_parent is None or 'ERROR' in flag_parent.name:
                        logger.info(f'unmet dependencies for task {func.__name__} for pid {pid}')
                        return
            # try and run the task with error catching
            logger.info(f'running task {func.__name__} for pid {pid}')
            path_task.joinpath(pid).mkdir(exist_ok=True, parents=True)
            try:
                func(pid, *args, **kwargs)
            except Exception:
                str_exception = traceback.format_exc()
                logger.error(str_exception)
                with open(error_file, 'w+') as fp:
                    fp.write(str_exception)
                return
            # if everything went well, exit with the flag file in place
            flag_file.touch(exist_ok=True)
            logger.info(f'completed task {func.__name__} for pid {pid}')
        return wrapper
    return inner


@task(**TASKS['destripe_ap'])
def destripe_ap(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)

@task(**TASKS['destripe_lf'])
def destripe_lf(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf', clobber=False)


@task(**TASKS['localise'])
def localise(pid):
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.localisation(destination, clobber=False)
