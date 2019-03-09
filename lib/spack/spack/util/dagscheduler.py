from copy import copy, deepcopy
import time
from multiprocessing import Pool, cpu_count
import spack
from spack.spec import Spec
import llnl.util.tty as tty
from llnl.util.lang import memoized
from spack.util.install_time_estimator import AsymptoticTimeEstimator

# Open questions
# If a package is not profiled. What should the default behavior be? Max jobs?

# TODO: Returns logical processors, which will be 1x, 2x, or 4x the
#  number of available hardware cores
@memoized
def get_cpu_count():
    return int(cpu_count()/2)


class SpecNode:
    """Represents the position of one Spec in the DAG"""

    def __init__(self, spec):
        self.spec = spec if spec.concrete else spec.concretized()
        self.dependencies = list(self._immediate_deps_gen(spec))
        self.hash = spec.full_hash()
        self.dependents = set()

    def __repr__(self):
        return self.hash

    @staticmethod
    def _immediate_deps_gen(spec):
        """Generator that yields all non-transient dependencies for spec"""
        for dep_spec in spec._dependencies.values():
            yield dep_spec.spec


class ParallelConcretizer:
    """Concretizes Specs using parallel workers
    For best utilization, pass a large list of specs to concrete_specs_gen
    and iterate over the results in a loop body.

    The right worker count is decided by available cores, loss of
    spec memoization between worker processes, and the number of specs to
    concretize.
    """

    def __init__(self, workers=1, ignore_error=False):
        adjusted_workers = int(max(1, min(workers, int(get_cpu_count()))))

        self.ignore_error = ignore_error
        self.workers = adjusted_workers
        self.pool = Pool(adjusted_workers)

        if adjusted_workers != workers:
            tty.warn('Parallel concretizer adjusted worker '
                     'count from %s to %s' % (workers, adjusted_workers))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()
        self.pool.terminate()

    @staticmethod
    def _concretize_spec(ser_spec):
        try:
            spec = Spec.from_yaml(ser_spec)
            try:
                return ser_spec, spec.concretized().to_yaml(all_deps=True)
            except Exception as e:
                tty.warn('Could not concretize %s: %s' % (spec.name, e))
        except Exception as e:
            tty.warn('Could not deserialize spec %s' % e.message)
        return None

    def concrete_specs_gen(self, specs, print_time=False):
        """Concretizes specs across a pool of processes

        Returns:
             A generator that yields a tuple with the original spec and a
             concretized spec. Output is not necessarily ordered.
        """

        start_time = time.time()

        abs_list, conc_list = [], []
        for s in specs:
            conc_list.append(s) if s.concrete else abs_list.append(s)

        # Note: Json serialization does not support 'all_deps'
        yaml_specs = [s.to_yaml(all_deps=True) for s in abs_list]

        # Spin off work to the pool
        conc_spec_gen = self.pool.imap_unordered(
            ParallelConcretizer._concretize_spec, yaml_specs)

        # Hand back any specs that are already concrete while the pool works
        for concrete_spec in conc_list:
            yield (concrete_spec, concrete_spec)

        # Yield specs as the pool concretizes them
        for spec_yaml_tuple in conc_spec_gen:
            if spec_yaml_tuple is None:
                if not self.ignore_error:
                    raise Exception("Parallel concretization Failed!")
            else:
                orig_spec_yaml, conc_spec_yaml = spec_yaml_tuple
                yield (Spec.from_yaml(orig_spec_yaml),
                       Spec.from_yaml(conc_spec_yaml))

        if print_time:
            tot_time = time.time() - start_time
            spec_per_second = len(specs) / tot_time
            tty.msg('Added %s specs in %s seconds (%s Spec/s)' % (
                len(specs), round(tot_time, 2), round(spec_per_second, 2)))


class DagSchedulerBase:
    """Base class for a DAG Scheduler
    Defines methods for manipulating a spec DAG.
    Generating a schedule is left to derived classes"""

    def __init__(self):
        self.tree = dict()
        self._build_schedule_called = False

    def add_spec(self, spec):
        """Add a Spec to the DAG"""
        self.add_specs([spec])

    def prune_installed(self, verbose=False):
        """Removes specs that are already installed"""
        # expand a list instead of iterating the tree directly
        # so items can be removed by the inner loop
        s_list = list(self.tree.items())
        initial_spec_count = len(s_list)

        for _, node in s_list:
            if len(spack.store.db.query(node.spec)):
                self.install_successful(node.spec)

        spec_count = len(self.tree.items())
        if verbose:
            tty.msg('%d specs already installed, %d not yet installed' % (
                initial_spec_count - spec_count, spec_count))

    def _install_successful(self, spec):
        """Removes a spec from the DAG and returns any new specs that no
        longer have dependencies."""
        new_no_deps = set()

        spec_node = self.tree[spec.full_hash()]

        for dpt in spec_node.dependents:
            if dpt in self.tree.keys():
                dspec_node = self.tree[dpt]
                dspec_node.dependencies.remove(spec_node.spec)

                if len(dspec_node.dependencies) == 0 and \
                        dspec_node.hash in self.tree.keys():
                    new_no_deps.add(dspec_node.spec)

        self.tree.pop(spec_node.hash)
        return new_no_deps

    def install_successful(self, spec):
        raise NotImplementedError()

    def install_failed(self, spec):
        """Removes a spec and its dependents. Returns any dependent specs that
        can no longer build"""
        unreachable_specs = set()

        spec_node = self.tree[spec.full_hash()]
        untraversed = copy(spec_node.dependents)

        while len(untraversed) > 0:
            untraversed_hash = untraversed.pop()
            if untraversed_hash not in self.tree.keys():
                continue

            node = self.tree[untraversed_hash]
            for dpt_hash in node.dependents:
                untraversed.add(dpt_hash)
            unreachable_specs.add(node.spec)
            self.tree.pop(node.hash)

        return unreachable_specs

    def ready_specs(self):
        """Returns a list of specs that have no dependencies and are ready to
        install"""
        return {node.spec for _, node in self.tree.items()
                if not len(node.dependencies)}

    def count(self):
        return len(self.tree)

    # TODO: Track specs to mark as explicit?
    def add_specs(self, specs, workers=1, verbose=False):
        """Add a list of Specs to the DAG"""

        with ParallelConcretizer(workers, ignore_error=True) as pc:
            for _, spec in pc.concrete_specs_gen(specs, print_time=verbose):
                # tuple list of [(parent, spec)]
                unresolved_specs = [(None, spec)]

                while len(unresolved_specs) > 0:
                    parent, cur_spec = unresolved_specs.pop()
                    cur_hash = cur_spec.full_hash()

                    # Add a node when it is not in the dictionary
                    if cur_hash not in self.tree:
                        s = SpecNode(cur_spec)
                        self.tree[cur_hash] = s
                        unresolved_specs.extend(
                            [(s, dep) for dep in s.dependencies])
                        # print('Processed %s' % cur_spec.name)

                    # Add a dependent when a parent is defined
                    if parent is not None:
                        self.tree[cur_hash].dependents.add(parent.hash)

    def _check_build_schedule_called(self):
        """Ensure build schedule is called before doing work"""
        if not self._build_schedule_called:
            raise Exception('DagScheduler Error: build_schedule() not called')

    def build_schedule(self, print_time=False):
        """Allow static schedulers to construct a schedule"""
        raise NotImplementedError()

    def pop_spec(self):
        """Returns a ready spec and the number of build jobs"""
        raise NotImplementedError()


class SimpleDagScheduler(DagSchedulerBase):
    """Implements a Dag Scheduler
    - No dependency ordering
    - Supports single node builds
    - Make jobs are set to cores/workers"""

    def __init__(self, workers=1):
        super().__init__()
        # Allows for slight over-provisioning
        self.build_jobs = int(round(get_cpu_count()/workers))
        self._ready_to_install = set()

    def build_schedule(self, print_time=False):
        self._build_schedule_called = True
        self.prune_installed()
        self._ready_to_install = set(self.ready_specs())

    # def pop_read_specs(self):
    #     self._check_build_schedule_called()
    #
    #     if len(self._ready_to_install) > 0:
    #         return self.build_jobs, self._ready_to_install.pop()
    #     else:
    #         return None

    def pop_all_ready_specs(self):
        for spec in self._ready_to_install:
            yield self.build_jobs, spec

        self._ready_to_install.clear()

    def install_successful(self, spec):
        for spec in self._install_successful(spec):
            self._ready_to_install.add(spec)


class TwoStepSchedulerBase(DagSchedulerBase):
    """Contains common methods for two step scheduling implementation.
    The two step schedulers implemented here will try to allocate a number of
    cores that will improve the makespan of the schedule, then use a list
    scheduling algorithm to create the schedule"""

    class Task:
        """Internal task object used for scheduling"""

        def __init__(self, estimator, spec_node):
            self.start_time = 0
            self.end_time = 0
            self.n = 1
            self.dependencies = []
            self.dependents = []
            self._estimator = estimator
            self.spec_node = spec_node
            self._t_level = 0
            self._b_level = self.exec_time()
            # DAG can be multi-rooted. This helps traversing loops prune
            # tasks already visited
            self.visited = False
            # Caches a value of a heavily used inner-loop value
            self.criticality = self._b_level + self._t_level
            # Track the number of unscheduled dependencies
            # to improve lookup time of ready tasks
            self.unsched_deps = len(self.dependencies)
            # Identifies which compute node the task
            # will run when scheduling across multiple nodes
            self.exec_node_id = None

        def add_dependent(self, dept):
            self.dependents.append(dept)

        def add_dependency(self, depc):
            self.dependencies.append(depc)
            self.unsched_deps = len(self.dependencies)

        def is_entry(self):
            """Returns whether this task is an entrance task"""
            return len(self.dependencies) == 0

        def is_exit(self):
            """Returns whether this task is an exit task"""
            return len(self.dependents) == 0

        def init_unsched_deps(self):
            self.unsched_deps = len(self.dependencies)

        def b_level(self, recalculate=False):
            """Longest path to an exit node including this node"""

            if recalculate:
                self._b_level = self.exec_time() + max(
                    [t._b_level for t in self.dependents], default=0)

            return self._b_level

        def t_level(self, recalculate=False):
            """Longest path to and entrance node excluding this node"""

            if recalculate:
                self._t_level = max(
                    (t._t_level + t.exec_time() for t in self.dependencies),
                    default=0)

            return self._t_level

        # TODO: consider removing. It is more efficient to
        #  update b/t-levels in one batch
        def set_procs(self, nproc):
            """Sets the allocated procs"""
            if self.n == nproc:
                return

            self.n = nproc

        def exec_time(self, procs=None):
            if procs is None:
                procs = self.n
            # Using Amdahl's law
            # return ((1 - self.parallel_portion) + (
            #            self.parallel_portion / procs)) * self.serial_exec
            return self._estimator(procs)

        @staticmethod
        def link_tasks(dpt, dpdc):
            dpt.add_dependency(dpdc)
            dpdc.add_dependent(dpt)

    def __init__(self, timing_profile):
        super.__init__()
        self.timing_profile = timing_profile

    @staticmethod
    def calculate_levels(tasks):
        """Traverses task list to update b-levels and t-levels"""

        # This function gets called a lot and has to traverse the entire DAG
        # Steps
        #  - Create a BFS ordering starting with the exit tasks
        #  - Traverse forward, updating b-level
        #  - Traverse in reverse, updating t-level and
        #  criticality (b-level + t-level)

        # Create a list
        t_len = len(tasks)
        idx = 0
        l_queue = [None] * t_len

        for t in tasks:
            t.visited = False

        # Add exit nodes
        for t in tasks:
            if t.is_exit():
                l_queue[idx] = t
                t.visited = True
                idx += 1

        # Update b-level BF traversal
        i = 0
        while i < t_len:
            t = l_queue[i]
            t.b_level(True)
            for td in t.dependencies:
                if not td.visited:
                    td.visited = True
                    l_queue[idx] = td
                    idx += 1
            i += 1

        # Update t-level and criticality in reverse traversal
        while i > 0:
            i -= 1
            t = l_queue[i]
            t.t_level(True)
            t.criticality = t._b_level + t._t_level

    @staticmethod
    def get_ready_tasks(tasks):
        """A subset of tasks whose dependencies are scheduled"""
        return [t for t in tasks if t.unsched_deps == 0]

    @staticmethod
    def get_makespan(tasks):
        """Makespan is determined by the longest end time"""
        return max([t.end_time for t in tasks])

    @staticmethod
    def schedule_task(t, proc_idle_time, p_list, start_time):
        """Schedules a task, mutates processor idle times"""
        new_end_time = start_time + t.exec_time()
        for p in p_list:
            proc_idle_time[p] = new_end_time

        t.start_time = start_time
        t.end_time = new_end_time

    @staticmethod
    def find_idle_hole(n, earliest_start, proc_idle_time):
        """Finds the earliest processor idle for a given number of cores
        returns tuple: (processor list, start time)
        This implementation will also try to minimize holes in scheduling"""

        # sort process ranks by idle time and select the first n
        p_list = sorted(range(len(proc_idle_time)),
                        key=lambda k: proc_idle_time[k])

        stop = n
        while stop < len(p_list) and proc_idle_time[
            p_list[stop]] <= earliest_start:
            stop += 1

        p_list = p_list[stop - n: stop]

        return p_list, max(earliest_start, proc_idle_time[p_list[-1]])

    @staticmethod
    def critical_tasks(t_list):
        """Returns a list of tasks on the critical path"""

        crit_tasks = []
        max_cost = 0

        for t in t_list:
            cost = t.criticality
            if cost > max_cost:
                max_cost = cost
                crit_tasks = [t]
            elif cost == max_cost:
                crit_tasks.append(t)

        return crit_tasks

    def mls(self, t_list, nproc):
        """M-task list scheduler
        Creates a schedule from a set of tasks which have been assigned cores.
        Works by iteratively selecting a task with the highest priority and
        scheduling it as soon as possible.

        Returns a list of tasks with start and stop times"""

        proc_idle_time = [0] * nproc
        unsched = set(t_list)
        scheduled = set()

        # Make sure any previous scheduling information is wiped
        for t in unsched:
            t.start_time = 0
            t.end_time = 0
            t.init_unsched_deps()

        while len(unsched) > 0:
            # Priority is determined by the ready task with the highest b-level
            t = max(self.get_ready_tasks(unsched), key=lambda x: x._b_level)

            # Get the tasks earliest start time
            earliest_start = max([d.end_time for d in t.dependencies],
                                 default=0)

            # Give it a schedule
            self.schedule_task(t, proc_idle_time,
                               *self.find_idle_hole(t.n, earliest_start,
                                                    proc_idle_time))
            unsched.remove(t)
            scheduled.add(t)

            for dpdt in t.dependents:
                dpdt.unsched_deps -= 1

        return list(scheduled)


class CPRDagScheduler(TwoStepSchedulerBase):
    """Critical Path Reduction scheduler.
    CPR is a two step scheduler that iteratively to allocates more cores
    to critical tasks and tests the resulting schedule for a decrease in
    makespan.

    It's time complexity is high, and becomes very costly after about 200 specs:

    --> O(EV^2P + V^3P(logV + PlogP)) Where:
    P is the number of cores
    V is the number of specs
    E is the number of dependencies between specs
    """

    def __init__(self, timing_profile):
        super.__init__(timing_profile)
        self.tasks = []

        # A list task end times for tasks that have not yet completed. Used
        # for selecting the next task to schedule
        self.incomplete_task_end_times = []

        # spec -> task lookup table
        self.spec_to_task = {}

    def _build_task_list(self):
        """Translates the spec tree defined in the parent into a task list
        usable for this algorithm"""

        self.tasks = []

        # DFS traversal starting with the top-level dependency
        nds = [n for n in self.tree.values() if len(n.dependents) == 0]

        # the DAG can be multi-rooted, so avoid
        # visiting the same dependency twice
        visited = set(n.hash for n in nds)

        while len(nds) > 0:
            nd = nds.pop(0)

            for s in nd.dependencies:
                if s.hash not in visited:
                    nds.append(s)
                    visited.add(s.hash)

            # Create the task
            t = self.Task(None, nd)

            # Make edges between tasks according to Spec's dependencies
            for dpt in nd.depenents:
                # This is a BF traversal, so the parent
                # must have been added recently
                for task in reversed(self.tasks):
                    if task.spec_node.hash == dpt.hash:
                        self.Task.link_tasks(task, t)
                        break

            # Add task to the list
            self.tasks.append(t)

    def build_schedule(self, print_time=False):

        # Initializing tasks and schedule
        self._build_task_list()

        for t in self.tasks:
            t.set_procs(1)

        self.calculate_levels(self.tasks)

        nproc = get_cpu_count()
        sched = self.mls(self.tasks, nproc)

        # Keep looping until a better schedule can't be created
        sched_modified = True
        while sched_modified:
            sched_modified = False
            resizable_tasks = [t for t in self.tasks if t.n < nproc]

            while not sched_modified and len(resizable_tasks) > 0:
                # select a task on the critical path
                # and increase its processor allocation
                ct = self.critical_tasks(resizable_tasks)[0]
                old_makespan = self.get_makespan(sched)
                ct.set_procs(ct.n + 1)
                self.calculate_levels(self.tasks)

                # create a new schedule
                sched = self.mls(self.tasks, nproc)

                # if the makespan decreased, use the new schedule
                new_makespan = self.get_makespan(sched)
                if new_makespan < old_makespan:
                    sched_modified = True
                else:
                    # Otherwise, revert the schedule and remove
                    # the task from the list
                    ct.set_procs(ct.n - 1)
                    self.calculate_levels(self.tasks)
                    resizable_tasks.remove(ct)
                    sched = self.mls(self.tasks, nproc)

        self.incomplete_task_end_times = sorted(t.end_time for t in self.tasks)
        self.spec_to_task = {t.spec_node.spec: t for t in self.tasks}

    def install_successful(self, spec):
        task = self.spec_to_task[spec]

        # propagate to the base class
        self._install_successful(spec)

        # remove end time from the list
        self.incomplete_task_end_times.remove(task.end_time)

    def install_failed(self, spec):

        # Get all of the tasks that cannot execute
        unreachable_tasks = [self.spec_to_task[t]
                             for t in super().install_failed(spec)]
        unreachable_tasks.append(self.spec_to_task[spec])

        # remove from task list
        for t in unreachable_tasks:
            self.tasks.remove(t)
            self.incomplete_task_end_times.remove(t.end_time)

    def pop_all_ready_specs(self):
