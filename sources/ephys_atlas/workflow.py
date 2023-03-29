from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from iblutil.util import setup_logger
from brainbox.io.one import SpikeSortingLoader

import ephys_atlas.rawephys
from ephys_atlas.data import atlas_pids


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
    'compute_sorted_features': {
        'version': '1.0.0',
        'depends_on': [],
    },
    'compute_raw_features': {
        'version': '1.0.0',
        'depends_on': ['destripe_lf', 'localise'],
    }
}


def report(one=None, pids=None):
    """
    Looks at the folder and flag files according to the task specifications
    Builds a dataframe where each row is a pid, and each column is a task
    :return:
    """
    if pids is None:
        pids, _ = atlas_pids(one)
        pids = sorted(pids)
    flow = pd.DataFrame(index=pids, columns=list(TASKS.keys())).fillna('')

    for pid in flow.index:
        for t in TASKS:
            task_file = next(ROOT_PATH.joinpath(pid).glob(f'.{t}*'), None)
            if task_file:
                flow.loc[pid, t] = task_file.name
    return flow


def get_pids_for_task(task_name, flow=None, include_errors=True, n_workers=1, worker_id=0):
    """
    From a flow dataframe, returns the PIDS that have errored or not been run
    :param task_name:
    :param flow:
    :param include_errors:
    :param n_workers:
    :param worker_id:
    :return:
    """
    if include_errors:
        pids = flow.index[flow[task_name] != f".{task_name}_{TASKS[task_name]['version']}"]
    else:
        raise NotImplementedError
    if n_workers > 1:
        pids = np.array_split(pids, n_workers)[worker_id]
    return pids


def run_flow(pids=None, one=None):
    if pids is None:
        pids, alyx_pids = atlas_pids(one)
        pids = sorted(pids)

    for pid in pids:
        destripe_ap(pid)
        destripe_lf(pid)
        # localise(pid) TODO this is in a different environment / or run everything in torch ?


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
def destripe_ap(pid, one):
    """
    :param pid:
    :param one:
    :return:
        TXXXX/ap.npy
        TXXXX/ap.yml
        TXXXX/ap_raw.npy
    """
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='ap', clobber=False)


@task(**TASKS['destripe_lf'])
def destripe_lf(pid, one):
    """
    :param pid:
    :param one:
    :return:
        TXXXX/lf.npy
        TXXXX/lf.yml
        TXXXX/lf_raw.npy
    """
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.destripe(pid, one=one, destination=destination, typ='lf', clobber=False)

@task(**TASKS['compute_sorted_features'])
def compute_sorted_features(pid, one, root_path=None):
    """
    :param pid:
    :param one:
    :param root_path:
    :return:
        clusters.pqt
        spikes.pqt
        channels.pqt
    """
    root_path = root_path or ROOT_PATH
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    pd.DataFrame(clusters).set_index('uuids').to_parquet(root_path.joinpath(pid, 'clusters.pqt'))
    pd.DataFrame(spikes).to_parquet(root_path.joinpath(pid, 'spikes.pqt'))
    pd.DataFrame(channels).to_parquet(root_path.joinpath(pid, 'channels.pqt'))


@task(**TASKS['localise'])
def localise(pid):
    """
    :param pid:
    :param root_path:
    :return:
        TXXXX/spikes.pqt
        TXXXX/waveforms.npy
    """
    destination = ROOT_PATH.joinpath(pid)
    ephys_atlas.rawephys.localisation(destination, clobber=False)


@task(**TASKS['compute_raw_features'])
def compute_raw_features(pid, root_path=None):
    """
    :param pid:
    :param root_path:
    :return:
        raw_ephys_features.pqt
    """
    root_path = root_path or ROOT_PATH
    ap_features = ephys_atlas.rawephys.compute_ap_features(pid, root_path=root_path)
    lf_features = ephys_atlas.rawephys.compute_lf_features(pid, root_path=root_path)
    spikes_features = ephys_atlas.rawephys.compute_spikes_features(pid, root_path=root_path)

    channels_features = spikes_features.groupby('trace').agg(
        alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha", aggfunc="std"),
        spike_count=pd.NamedAgg(column="alpha", aggfunc="count"),
        cloud_x_std=pd.NamedAgg(column="x", aggfunc="std"),
        cloud_y_std=pd.NamedAgg(column="y", aggfunc="std"),
        cloud_z_std=pd.NamedAgg(column="z", aggfunc="std"),
        peak_trace_idx=pd.NamedAgg(column="peak_trace_idx", aggfunc="mean"),
        peak_time_idx=pd.NamedAgg(column="peak_time_idx", aggfunc="mean"),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_idx=pd.NamedAgg(column="trough_time_idx", aggfunc="mean"),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
    )

    channels_features = pd.merge(channels_features, ap_features, left_index=True, right_index=True)
    channels_features = pd.merge(channels_features, lf_features, left_index=True, right_index=True)
    # add the pid as the main index to prepare for concatenation
    channels_features = pd.concat({pid: channels_features}, names=['pid'])
    channels_features.to_parquet(root_path.joinpath(pid, 'raw_ephys_features.pqt'))
    return channels_features
