import ephys_atlas.workflow as workflow
import tempfile
from pathlib import Path
import unittest


# fake pids
pids = [
    '00007da5-08ea-4d23-873a-2a43e6b10966',
    '0000fc49-9d30-4929-8788-c3ff137b5b42'
]
pid = pids[0]


class TestWorkflows(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_task = Path(self.temp_dir.name)
        print(self.path_task)

    def test_simple_task(self):
        @workflow.task(version='1.0.0', path_task=self.path_task)
        def task_a(pid):
            pass
        task_a(pid=pid)
        assert self.path_task.joinpath(pid, '.task_a_1.0.0').exists()

    def test_error_task(self):
        @workflow.task(version='1.0.0', path_task=self.path_task)
        def task_b(pid=None, error=True):
            if error:
                raise ValueError("Dont'feel like completing today")

        # this will error, we expect an error flag as an output
        task_b(pid=pid)
        assert self.path_task.joinpath(pid, '.task_b_ERROR').exists()
        # this time on a re-run this will succeed, here the error flag is not there anymore and we have a task flag
        task_b(pid=pid, error=False)
        assert not self.path_task.joinpath(pid, '.task_b_ERROR').exists()
        assert self.path_task.joinpath(pid, '.task_b_1.0.0').exists()

    def test_error_dependencies(self):

        @workflow.task(version='1.0.0', path_task=self.path_task)
        def task_a(pid):
            raise ValueError("Dont'feel like completing today")
        @workflow.task(version='1.0.0', path_task=self.path_task, depends_on='task_a')
        def task_b(pid):
            pass
        task_a(pid=pid)
        assert self.path_task.joinpath(pid, '.task_a_ERROR').exists()
        task_b(pid=pid)
        assert not self.path_task.joinpath(pid, '.task_b_1.0.0').exists()

    def test_simple_dependencies(self):
        @workflow.task(version='1.0.0', path_task=self.path_task)
        def task_a(pid):
            pass
        @workflow.task(version='1.0.0', path_task=self.path_task, depends_on='task_a')
        def task_b(pid):
            pass

        task_b(pid=pid)
        assert not self.path_task.joinpath(pid, '.task_b_1.0.0').exists()
        task_a(pid=pid)
        assert self.path_task.joinpath(pid, '.task_a_1.0.0').exists()
        task_b(pid=pid)
        assert self.path_task.joinpath(pid, '.task_b_1.0.0').exists()

    def test_new_version_rerun_and_dependencies(self):
        @workflow.task(version='1.0.0', path_task=self.path_task)
        def task_a(pid):
            pass
        @workflow.task(version='1.0.0', path_task=self.path_task, depends_on='task_a_2.0.0')
        def task_b(pid):
            pass

        # run old version task_a
        task_a(pid=pid)
        # run task b, with strict dependency will not execute
        task_b(pid=pid)
        assert not self.path_task.joinpath(pid, '.task_b_1.0.0').exists()
        @workflow.task(version='2.0.0', path_task=self.path_task)
        def task_a(pid):
            pass

        # run new task a, the old flag disappear and a new one gets created
        task_a(pid=pid)
        assert self.path_task.joinpath(pid, '.task_a_2.0.0').exists()
        assert not self.path_task.joinpath(pid, '.task_a_1.0.0').exists()

        # now task b dependency is met
        task_b(pid=pid)
        assert self.path_task.joinpath(pid, '.task_b_1.0.0').exists()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
