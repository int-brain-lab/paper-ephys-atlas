"""
Worfklow management for ephys atlas features computations

For each pid a set of dependent tasks computes features and saves them.
Each of the tasks are defined in the TASKS dictionary and correspond to a function below

To have an overview of the tasks dependencies, here is the workflow graph:
>>> from ephys_atlas.workflow import graph, ROOT_PATH
>>> graph(ROOT_PATH / 'workflow.png')

A decorator handles the file flagging and the dependencies between tasks to allow re-runs and backfills.
Each task writes a small flag file in the pid folder to indicate that it has been run successfully or not.

.{TASK_NAME}.{TASK_VERSION}.{TASK_STATUS}
.compute_raw_features-1.0.0-complete
.compute_sorted_features-1.0.1-error

The decorator also checks the dependencies between tasks and will not run a task if its dependencies are not met.
Dependencies are defined in the TASKS dictionary as a list of task names, optionally with a version number.
This is the minimum version number that will not trigger a re-run of the task.
depends_on': ['destripe_ap', 'destripe_lf-1.2.0']

Check-out the tests for more details on how to use the decorator in ./tests/test_workflow.py

The current workflow is described in the TASKS dictionary below.

To change the default path of execution, set the TASK_ROOT_PATH environment variable.
>>> os.environ['TASK_ROOT_PATH'] = '/path/to/ephys-atlas'
"""

from pathlib import Path
import traceback
import packaging.version
from collections import OrderedDict
import os

import numpy as np
import pandas as pd

from iblutil.util import setup_logger
from brainbox.io.one import SpikeSortingLoader
import phylib.stats

import elts.parede_raw_ephys_features.rawephys
from ephys_atlas.data import atlas_pids

ROOT_PATH = Path(os.getenv("TASK_ROOT_PATH", Path("/mnt/s0/ephys-atlas")))
logger = setup_logger(name="paper-ephys-atlas", level="INFO")
"""
Tasks versioning notes:
-   compute_sorted_features
    - 1.4.0 add the correlogram.npy output
    - 1.5.0 remove correlogram.npy output and replaces by
    correlograms_time_scale.npy and correlograms_refractory_period.npy
"""
TASKS = OrderedDict(
    {
        "destripe_ap": {
            "version": "1.3.0",
        },
        "destripe_lf": {
            "version": "1.3.0",
        },
        "localise": {
            "version": "1.2.0",
            "depends_on": ["destripe_ap"],
        },
        "compute_sorted_features": {
            "version": "1.5.0",
        },
        "compute_raw_features": {
            "version": "1.4.1",
            "depends_on": ["destripe_lf", "localise"],
        },
    }
)
assert all(["-" not in TASKS for t in TASKS]), "Task names cannot contain -"


def graph(output_file=None):
    """
    Display and optionally save the workflow graph
    Needs graphviz to be installed
    :param output_file:
    :return:
    """
    import graphviz
    from string import ascii_letters

    dot = graphviz.Digraph(comment="Ephys atlas workflow")
    tletters = {t: ascii_letters[i] for i, t in enumerate(TASKS)}
    edges = []
    for t in TASKS:
        dot.node(tletters[t], t)
        for dep in TASKS[t].get("depends_on", []):
            edges.append(tletters[dep] + tletters[t])
    dot.edges(edges)
    if output_file is not None:
        dot.render(output_file, view=True)


def report(one=None, pids=None, path_task=ROOT_PATH):
    """
    Looks at the folder and flag files according to the task specifications
    Builds a dataframe where each row is a pid, and each column is a task
    :param one:
    :param pids:
    :param path_task:
    :return:
    """
    if pids is None:
        pids, _ = atlas_pids(one)
        pids = sorted(pids)
    tasks = pd.DataFrame(index=pids, columns=list(TASKS.keys())).fillna("")

    for pid in tasks.index:
        for t in TASKS:
            task_file = next(path_task.joinpath(pid).glob(f".{t}*"), None)
            if task_file:
                tasks.loc[pid, t] = task_file.name
    return tasks


@pd.api.extensions.register_dataframe_accessor("flow")
class Flow:
    """This adds a few methods to the Dataframe object for convenience"""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def versions(self):
        return self.applymap(lambda x: (x + "-").split("-")[1])

    def print_report(self):
        """Prints a command line report of current status of tasks"""
        flow = self._obj
        npids = flow.shape[0]
        print("complete=  error:*   unmet deps:/   old version:+  new-")
        for task_name, task in TASKS.items():
            nnew = sum(flow[task_name] == "")
            nold = sum(
                flow[task_name].apply(
                    lambda x: x.endswith("complete")
                    & ((x + "-").split("-")[1] != task["version"])
                )
            )
            nerr = sum(flow[task_name] == f".{task_name}-{task['version']}-error")
            ncomplete = sum(
                flow[task_name] == f".{task_name}-{task['version']}-complete"
            )
            nnr = sum(
                flow[task_name] == f".{task_name}-{task['version']}-unmet_dependencies"
            )
            print(
                f"{ncomplete:4d}= {nerr:4d}* {nnr:4d}/ {nold:4d}+ {nnew:4d}-", task_name
            )
            print(
                "=" * int(ncomplete / npids * 100)
                + "*" * int(nerr / npids * 100)
                + "/" * int(nnr / npids * 100)
                + "+" * int(nold / npids * 100)
                + "-" * int(nnew / npids * 100)
            )

    def get_pids_ready(self, task_name=None, include_errors=False):
        """For a given task name, returns pids ready to run"""
        assert task_name
        flow = self._obj
        # rejects tasks that ran with the same version
        include = ~flow[task_name].apply(
            lambda x: x.startswith(f".{task_name}-{TASKS[task_name]['version']}")
        )
        if include_errors:
            include = np.logical_or(
                flow[task_name].apply(lambda x: x.endswith("error")), include
            )
        pids = flow.index[include]
        return pids


def run_flow(pids=None, one=None):
    if pids is None:
        pids, alyx_pids = atlas_pids(one)
        pids = sorted(pids)

    for pid in pids:
        destripe_ap(pid)
        destripe_lf(pid)
        # localise(pid) TODO this is in a different environment / or run everything in torch ?


def task(version=None, depends_on=None, path_task=None):
    """
    Decorator to mark a function as a task
    :param version:
    :param depends_on:
    :param path_task:
    :return:
    """
    depends_on = [depends_on] if isinstance(depends_on, str) else depends_on
    path_task = path_task or ROOT_PATH

    def inner(func):
        def wrapper(
            pid,
            *args,
            version=version,
            depends_on=depends_on,
            force_run=False,
            **kwargs,
        ):
            # if the task has already run with the same version number, skip and exit
            current_path = (
                kwargs.pop("path_task") if "path_task" in kwargs else path_task
            )
            flag_file = current_path.joinpath(pid).joinpath(
                f".{func.__name__}-{version}-complete"
            )
            # remove an eventual error file on a previous run (here we ru-run on errors)
            error_file = current_path.joinpath(pid).joinpath(
                f".{func.__name__}-{version}-error"
            )
            unmet_deps_file = current_path.joinpath(pid).joinpath(
                f".{func.__name__}-{version}-unmet_dependencies"
            )

            def should_i_run():
                if force_run:
                    return True
                if flag_file.exists():
                    logger.info(f"skipping task {func.__name__} for pid {pid}")
                    return False
                # now remove all error files or previous version files
                for f in current_path.joinpath(pid).glob(f".{func.__name__}-*"):
                    f.unlink()
                # check that dependencies are met, exit if not
                if depends_on is not None:
                    unmet_dependencies = False
                    # loop on parent tasks and check if they have run
                    for parent_task in depends_on:
                        # gets an empty string version if no minumum version is specified
                        parent_task, required_parent_version = (
                            parent_task + "------"
                        ).split("-", maxsplit=2)[:2]
                        flag_parent = next(
                            current_path.joinpath(pid).glob(f".{parent_task}*"), None
                        )
                        # the parent task hasn't run
                        if flag_parent is None:
                            logger.info(
                                f"unmet dependencies for task {func.__name__} for pid {pid}: {parent_task} has not run"
                            )
                            unmet_dependencies = True
                        else:
                            _, parent_version, status = flag_parent.name.split("-")
                            # the parent task has errored
                            if status == "error":
                                logger.info(
                                    f"unmet dependencies for task {func.__name__} for pid {pid}: {parent_task} has errored"
                                )
                                unmet_dependencies = True
                            # the parent task has run with a now deprecated version
                            if required_parent_version and (
                                packaging.version.parse(required_parent_version)
                                > packaging.version.parse(parent_version)
                            ):
                                logger.info(
                                    f"unmet dependencies for task {func.__name__} for pid {pid}: {parent_task} "
                                    f"ran with a deprecated version {parent_version} < minimum required: {required_parent_version}"
                                )
                                unmet_dependencies = True
                    if unmet_dependencies:
                        if unmet_deps_file.parent.exists():
                            unmet_deps_file.touch()
                        return False
                return True

            if not should_i_run():
                return
            # try and run the task with error catching
            logger.info(f"running task {func.__name__} for pid {pid}")
            current_path.joinpath(pid).mkdir(exist_ok=True, parents=True)
            try:
                func(pid, *args, **kwargs)
            except Exception:
                str_exception = traceback.format_exc()
                logger.error(str_exception)
                with open(error_file, "w+") as fp:
                    fp.write(str_exception)
                return
            # if everything went well, exit with the flag file in place
            flag_file.touch(exist_ok=True)
            logger.info(f"completed task {func.__name__} for pid {pid}")

        return wrapper

    return inner


@task(**TASKS["destripe_ap"])
def destripe_ap(pid, one, data_path=ROOT_PATH):
    """
    :param pid:
    :param one:
    :return:
        TXXXX/ap.npy
        TXXXX/ap.yml
        TXXXX/ap_raw.npy
    """
    destination = data_path.joinpath(pid)
    elts.parede_raw_ephys_features.rawephys.destripe(
        pid, one=one, destination=destination, typ="ap", clobber=False
    )


@task(**TASKS["destripe_lf"])
def destripe_lf(pid, one, data_path=ROOT_PATH):
    """
    :param pid:
    :param one:
    :return:
        TXXXX/lf.npy
        TXXXX/lf.yml
        TXXXX/lf_raw.npy
    """
    destination = data_path.joinpath(pid)
    elts.parede_raw_ephys_features.rawephys.destripe(
        pid, one=one, destination=destination, typ="lf", clobber=False
    )


@task(**TASKS["compute_sorted_features"])
def compute_sorted_features(pid, one, data_path=ROOT_PATH):
    """
    :param pid:
    :param one:
    :param data_path:
    :return:
        clusters.pqt
        spikes.pqt
        channels.pqt
    """
    AP_SAMPLING_RATE = 30_000
    ssl = SpikeSortingLoader(one=one, pid=pid)
    spikes, clusters, channels = ssl.load_spike_sorting(
        dataset_types=["spikes.samples"]
    )
    if spikes == clusters == channels == {}:
        raise ValueError("No spike sorting found for this session")
    # here the channels object comes in two flavours: raw channels ('localCoordinates', 'rawInd')
    # and processed channels ('x', 'y', 'z', 'acronym', 'atlas_id', 'axial_um', 'lateral_um')
    if "localCoordinates" in channels:
        channels = dict(
            axial_um=channels["localCoordinates"][:, 1],
            lateral_um=channels["localCoordinates"][:, 0],
        )
    # compute the correlograms
    corr_bin_ts_secs, corr_win_ts_secs = (0.001, 1)  # time-scale long autocorrelogram
    corr_bin_rf_secs, corr_win_rf_secs = (
        1 / AP_SAMPLING_RATE,
        0.02,
    )  # refractory period short range
    ns_ts = int(np.ceil((corr_win_ts_secs / corr_bin_ts_secs + 1) / 2))
    correlograms_ts = np.zeros((ns_ts, clusters["uuids"].size), dtype=np.int32)
    ns_rf = int(np.ceil((corr_win_rf_secs / corr_bin_rf_secs + 1) / 2))
    correlograms_rf = np.zeros((ns_rf, clusters["uuids"].size), dtype=np.int32)
    for c, s in pd.DataFrame(spikes).groupby("clusters"):
        correlograms_ts[:, c] = phylib.stats.correlograms(
            s["times"],
            s["clusters"],
            c,
            sample_rate=AP_SAMPLING_RATE,
            bin_size=corr_bin_ts_secs,
            window_size=corr_win_ts_secs,
            symmetrize=False,
        )
        correlograms_rf[:, c] = phylib.stats.correlograms(
            s["times"],
            s["clusters"],
            c,
            sample_rate=AP_SAMPLING_RATE,
            bin_size=corr_bin_rf_secs,
            window_size=corr_win_rf_secs,
            symmetrize=False,
        )
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    data_path.joinpath(pid).mkdir(exist_ok=True, parents=True)
    # the concat syntax sets a higher level index on the dataframe as pid
    np.save(data_path.joinpath(pid, "correlograms_time_scale.npy"), correlograms_ts)
    np.save(
        data_path.joinpath(pid, "correlograms_refractory_period.npy"), correlograms_rf
    )
    pd.concat({pid: pd.DataFrame(clusters)}, names=["pid"]).to_parquet(
        data_path.joinpath(pid, "clusters.pqt")
    )
    pd.concat({pid: pd.DataFrame(spikes)}, names=["pid"]).to_parquet(
        data_path.joinpath(pid, "spikes_sorted.pqt")
    )
    df_channels = pd.concat({pid: pd.DataFrame(channels)}, names=["pid"])
    df_channels["histology"] = ssl.histology
    # get the spike sorter version field
    dset = one.alyx.rest(
        "datasets",
        "list",
        collection=ssl.collection,
        name="spikes.times.npy",
        session=ssl.eid,
    )
    df_channels["version"] = dset[0]["version"]
    df_channels.to_parquet(data_path.joinpath(pid, "channels.pqt"))


@task(**TASKS["localise"])
def localise(pid, clobber=False, data_path=ROOT_PATH):
    """
    :param pid:
    :param path_task:
    :return:
        TXXXX/spikes.pqt
        TXXXX/waveforms.npy
    """
    destination = data_path.joinpath(pid)
    elts.parede_raw_ephys_features.rawephys.localisation(destination, clobber=clobber)


@task(**TASKS["compute_raw_features"])
def compute_raw_features(pid, data_path=None):
    """
    :param pid:
    :param path_task:
    :return:
        raw_ephys_features.pqt
    """
    root_path = data_path or ROOT_PATH
    ap_features, fs = elts.parede_raw_ephys_features.rawephys.compute_ap_features(
        pid, root_path=root_path
    )
    lf_features = elts.parede_raw_ephys_features.rawephys.compute_lf_features(
        pid, root_path=root_path, current_source=False
    )
    cs_features = elts.parede_raw_ephys_features.rawephys.compute_lf_features(
        pid, root_path=root_path, current_source=True
    )
    spikes_features = elts.parede_raw_ephys_features.rawephys.compute_spikes_features(
        pid, root_path=root_path
    )
    # need to rename and cast this column to have a consistent merge with ap and lf features later
    spikes_features["channel"] = spikes_features["trace"].astype(np.int16)

    fcn_mean_time = (
        lambda x: np.mean((x - elts.parede_raw_ephys_features.rawephys.TROUGH_OFFSET))
        / fs
    )

    channels_features = spikes_features.groupby("channel").agg(
        alpha_mean=pd.NamedAgg(column="alpha", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha", aggfunc=lambda x: np.std(x, ddof=0)),
        spike_count=pd.NamedAgg(column="alpha", aggfunc="count"),
        peak_time_secs=pd.NamedAgg(column="peak_time_idx", aggfunc=fcn_mean_time),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_secs=pd.NamedAgg(column="trough_time_idx", aggfunc=fcn_mean_time),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
        tip_time_secs=pd.NamedAgg(column="tip_time_idx", aggfunc=fcn_mean_time),
        tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
        recovery_time_secs=pd.NamedAgg(
            column="recovery_time_idx", aggfunc=fcn_mean_time
        ),
        depolarisation_slope=pd.NamedAgg(column="depolarisation_slope", aggfunc="mean"),
        repolarisation_slope=pd.NamedAgg(column="repolarisation_slope", aggfunc="mean"),
        recovery_slope=pd.NamedAgg(column="recovery_slope", aggfunc="mean"),
        polarity=pd.NamedAgg(column="invert_sign_peak", aggfunc=lambda x: -x.mean()),
    )
    channels_features = pd.merge(
        channels_features, ap_features, left_index=True, right_index=True
    )
    channels_features = pd.merge(
        channels_features, lf_features, left_index=True, right_index=True
    )
    channels_features = pd.merge(
        channels_features,
        cs_features,
        left_index=True,
        right_index=True,
        suffixes=("", "_csd"),
    )
    # add the pid as the main index to prepare for concatenation
    channels_features = pd.concat({pid: channels_features}, names=["pid"])
    channels_features.to_parquet(root_path.joinpath(pid, "raw_ephys_features.pqt"))
    return channels_features
