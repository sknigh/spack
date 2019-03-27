import time
from copy import copy
from datetime import datetime
from multiprocessing import Pool, cpu_count

import llnl.util.tty as tty
import spack
from llnl.util.lang import memoized
from spack.spec import Spec
from spack.util.install_time_estimator import AsymptoticTimeEstimator


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


class DagManager:
    """Base class for a DAG Scheduler
    Defines methods for manipulating a spec DAG.
    Generating a schedule is left to derived classes"""

    def __init__(self):
        self.tree = dict()

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

    def install_successful(self, spec):
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
        return {node.spec for node in self.tree.values()
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

                    # Add a dependent when a parent is defined
                    if parent is not None:
                        self.tree[cur_hash].dependents.add(parent.hash)


class DagSchedulerBase:
    def __init__(self, dag_manager):
        if not dag_manager:
            dag_manager = DagManager()

        self._dag_manager = dag_manager
        self._build_schedule_called_already = False

    def build_schedule(self, print_time=False):
        """Constructs a schedule, should not be called multiple times"""
        raise NotImplementedError()

    def add_spec(self, spec):
        """Add a Spec to the DAG"""
        self._dag_manager.add_spec(spec)

    def add_specs(self, specs):
        """Add multiple Specs to the DAG"""
        self._dag_manager.add_specs(specs)

    def install_successful(self, spec):
        """Indicate a spec was successfully installed"""
        raise NotImplementedError()

    def install_failed(self, spec):
        """Indicate a spec failed to install, return list of specs that will
        not be installed"""
        for spec in self._dag_manager.install_failed(spec):
            yield spec

    def schedule_is_built(self):
        return self._build_schedule_called_already

    def _build_schedule_called(self):
        """Ensure build_schedule is only called once"""
        if not self._build_schedule_called_already:
            self._build_schedule_called_already = True
        else:
            raise Exception('Build schedule already called')

    def _check_build_schedule_called(self):
        """Ensure build schedule is called before doing work"""
        if not self._build_schedule_called:
            raise Exception('Dag Scheduler Error: build_schedule() not called')

    def prune_installed(self, verbose=False):
        self._dag_manager.prune_installed(verbose)

    def pop_ready_specs(self):
        """Returns (Build Jobs, Spec) for every Spec ready to install. Only
        returns each spec once."""
        raise NotImplementedError()

    def count(self):
        return self._dag_manager.count()

    def get_makespan(self):
        """Gets the makespan"""
        raise NotImplementedError()


class SimpleDagScheduler(DagSchedulerBase):
    """Implements a Dag Scheduler
    Serially unwinds DAG like standard Spec dependency traversal"""

    def __init__(self, workers=get_cpu_count(), dag_manager=None):
        super().__init__(dag_manager)

        self.make_jobs = workers
        self._ready_to_install = set()
        self._outstanding_spec = None

    def build_schedule(self, print_time=False):
        self._build_schedule_called()
        self.prune_installed()
        self._ready_to_install = set(self._dag_manager.ready_specs())

    def install_failed(self, spec):
        self._outstanding_spec = None
        return super().install_failed(spec)

    def pop_ready_specs(self):
        # This DAG Scheduler builds one spec at a time with all cores
        # Do not pop another Spec when one is outstanding
        if self._outstanding_spec or len(self._ready_to_install) == 0:
            return []

        spec = self._ready_to_install.pop()
        self._outstanding_spec = spec
        return [(None, spec)]

    def install_successful(self, spec):
        if spec.full_hash() != self._outstanding_spec.full_hash():
            raise Exception('SimpleDagScheduler does not recognize this spec')

        self._outstanding_spec = None

        for spec in self._dag_manager.install_successful(spec):
            self._ready_to_install.add(spec)


class TwoStepSchedulerBase(DagSchedulerBase):
    """Contains common methods for two step scheduling implementation.
    The two step schedulers implemented here will try to allocate a number of
    cores that will improve the makespan of the schedule, then use a list
    scheduling algorithm to create the schedule"""

    class Task:
        """Internal task object used for scheduling"""

        def __init__(self, timing_db, spec_node):
            self.start_time = 0
            self.end_time = 0
            self.n = 1
            self.dependencies = []
            self.dependents = []
            self._estimator = self._init_estimator(spec_node.spec.name,
                                                   timing_db)
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
            self.precedence_level=None
            # Identifies which compute node the task
            # will run when scheduling across multiple nodes
            self.exec_node_id = None
            self.name = spec_node.spec.name

            # tty.warn('Creating task for %s' % self.spec_node.spec.name)
            # for i in [1, 2, 4, 8]:
            #     tty.warn(' %s: %s' % (
            #     i, round(self._estimator.estimate(i), 2)))

        def is_scalable(self, threshold=0.8):
            """Use some threshold for whether a task is scalable. Filtering
            tasks that are not scalable improves scheduler execution time by
            limiting the number of options
            """
            return self.exec_time(8)/self.exec_time(1) < threshold

        def phase_decomposition(self, timing_db):
            """Decomposes task into its phases
            NOTE: used for theoretical DAG schedules, phase tasks will not
            actually work in a real installation"""

            package = self.spec_node.spec.name
            phase_list = []
            last_task = None
            for phase in list(timing_db.phases_gen(package)):
                timing_tups = list(timing_db.phase_timings(package, phase))
                est = AsymptoticTimeEstimator()
                est.add_measurements(*zip(*timing_tups))

                t = copy(self)
                t._estimator = est
                t.name = '%s::%s' % (t.name, phase)

                t.dependents = []
                t.dependencies = []
                if last_task is not None:
                    self.link_tasks(t, last_task)
                last_task = t
                phase_list.append(t)

            return phase_list

        @staticmethod
        def phase_task_dag(task_list, timing_db):
            """Creates a DAG with phase tasks from a DAG with package tasks"""

            visited = {}
            tasks = [t for t in task_list if len(t.dependents) == 0]

            def process_package_task(t):
                if t in visited:
                    return visited[t]

                p_list = t.phase_decomposition(timing_db)

                visited[t] = p_list
                tasks.extend(t.dependencies)

                for dep in t.dependents:
                    # recurse on dependents that have not been processed
                    dep_lst = visited[dep] if \
                        dep in visited else process_package_task(dep)

                    TwoStepSchedulerBase.Task.link_tasks(dep_lst[0],
                                                         p_list[-1])
                return p_list

            while len(tasks) > 0:
                t = tasks.pop(0)

                if t not in visited:
                    process_package_task(t)

            # The dictionary values still partition by package task. Flatten it
            return [idx for pt in visited.values() for idx in pt]

        @staticmethod
        def _init_estimator(spec_name, timing_db):
            # get list of (job, time) tuples generator
            tup_list = timing_db.package_timings(spec_name)

            # insert into an estimator
            est = AsymptoticTimeEstimator()

            try:
                est.add_measurements(*zip(*tup_list))
            except:
                tty.die("Could not initialize task '%s', database missing "
                        "timing information" % spec_name)
            return est

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

        def work_change(self):
            """Returns the improvement in execution time by increasing
            processor count divided by number of processors used"""
            initial = self.exec_time() / self.n
            final = self.exec_time(self.n + 1) / (self.n + 1)
            return initial - final

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
            return self._estimator.estimate(procs)

        @staticmethod
        def link_tasks(dpt, dpdc):
            dpt.add_dependency(dpdc)
            dpdc.add_dependent(dpt)

    def __init__(self, dag_manager, timing_db):
        super().__init__(dag_manager)
        self.timing_db = timing_db
        self.tasks = []
        self.tasks_by_prec_level = []
        self._exit_tasks = None
        self._entry_tasks = None
        self._decompose_phase_tasks = False

        # A list task end times for tasks that have not yet completed. Used
        # for selecting the next task to schedule
        self.incomplete_task_end_times = []

        # spec -> task lookup table
        self.spec_to_task = {}

        self._popped_tasks = set()

    def test_schedule_integrity(self):
        """Checks that schedule does not violate dependency ordering"""
        for t in self.tasks:
            for depc in t.dependencies:
                if depc.start_time > t.start_time:
                    raise Exception(
                        '%s ends after %s starts' % (depc.name, t.name))

    def sched_build_time(self):
        """Returns the time it took to create the schedule"""
        raise NotImplementedError()

    def entry_tasks(self):
        """Returns a list of entrance tasks from the current DAG"""
        if self._entry_tasks:
            return self._entry_tasks
        else:
            self._entry_tasks = [t for t in self.tasks if t.is_entry()]
            return self._entry_tasks

    def exit_tasks(self):
        """Returns a list of exit tasks from the current DAG"""
        if self._exit_tasks:
            return self._exit_tasks
        else:
            self._exit_tasks = [t for t in self.tasks if t.is_exit()]
            return self._exit_tasks

    def init_precedence_levels(self):
        """Configures precedence level, an integer identifier for each task
        that spans the graph.
        Returns the count of processors already allocated at each level"""

        # O(V + VE)
        #previous_layer = [t for t in self.tasks if t.is_exit()]
        previous_layer = self.exit_tasks()
        next_layer = []
        prec = 0
        p_level_procs = []

        while len(previous_layer) > 0:
            p_sum = 0
            for t in previous_layer:
                t.precedence_level = prec
                next_layer.extend(t.dependencies)
                p_sum += t.n

            p_level_procs.append(p_sum)
            previous_layer = next_layer
            next_layer = []
            prec += 1

        return p_level_procs

    def decompose_task_phases(self, decompose):
        """Whether to substitute package tasks with equivalent phase tasks."""
        self._decompose_phase_tasks = decompose

    def calculate_levels(self):
        """Traverses task list to update b-levels, t-levels and criticality"""

        for t in self.tasks_by_prec_level:
            t.b_level(True)

        for t in reversed(self.tasks_by_prec_level):
            t.t_level(True)
            t.criticality = t._b_level + t._t_level

    @staticmethod
    def get_ready_tasks(tasks):
        """A subset of tasks whose dependencies are scheduled"""
        return [t for t in tasks if t.unsched_deps == 0]

    def get_makespan(self):
        """Makespan is determined by the longest end time"""
        return max([t.end_time for t in self.exit_tasks()])

    @staticmethod
    def schedule_task(t, proc_idle_time, p_list, start_time, node_idx):
        """Schedules a task, mutates processor idle times"""
        new_end_time = start_time + t.exec_time()
        for p in p_list:
            proc_idle_time[p] = new_end_time

        t.start_time = start_time
        t.end_time = new_end_time
        t.exec_node_id = node_idx

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
            # computers are supposed to replace us, yet they can't add...
            if abs(cost - max_cost) < 0.00001:
                crit_tasks.append(t)
            elif cost > max_cost:
                max_cost = cost
                crit_tasks = [t]

        return crit_tasks

    def print_schedule(self):
        tty.msg('Task schedule')

        name_len = max(len(t.spec_node.spec.name) for t in self.tasks)
        fmt_str = '%-LENs %-3s %-8s %-5s %-5s'.replace('LEN', str(name_len))

        tty.msg(fmt_str % ('Name', 'Jobs', 'blevel', 'Start', 'End'))
        for t in sorted(self.tasks, key=lambda task: task.start_time):
            tty.msg(fmt_str % (
                t.name,
                t.n,
                round(t.b_level(), 1),
                round(t.start_time, 1),
                round(t.end_time, 1)))

        tty.msg('')
        tty.msg('Estimated execution time: %s' %
                round(self.get_makespan(), 2))

    def _build_task_list(self):
        """Translates the spec tree defined in the Dag Manager into a task list
        usable for this algorithm"""

        # Create the tasks
        self.tasks = [self.Task(self.timing_db, nd)
                      for nd in self._dag_manager.tree.values()]

        # lazy way O(V^2E)
        for t in self.tasks:
            for dep in t.spec_node.dependencies:
                for candidate_dep in self.tasks:
                    if dep.full_hash() == candidate_dep.spec_node.hash:
                        self.Task.link_tasks(t, candidate_dep)
                        break

        if self._decompose_phase_tasks:
            self.tasks = self.Task.phase_task_dag(self.tasks, self.timing_db)

        self.init_precedence_levels()
        self.tasks_by_prec_level = sorted(
            self.tasks, key=lambda _task: _task.precedence_level)

    def mn_mls(self, t_list, nproc, nnode):
        """Multi-Node M-task list scheduler
        A variation of MLS that allows schedule creation across multiple
        nodes."""

        # don't use * for outer list
        node_proc_idle_time = [[0] * nproc for _ in range(nnode)]
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

            # Get the task's earliest possible start time from dependencies
            earliest_start = max([d.end_time for d in t.dependencies],
                                 default=0)

            # Get earliest start time for each node
            hole_choices = [(self.find_idle_hole(
                t.n, earliest_start, pit), pit) for pit in node_proc_idle_time]
            # TODO: find the smallest increase in sched gap for tie breaker
            (best_hole, best_node_procs), node_idx = hole_choices[0], 0
            for idx, (hole, pit) in enumerate(hole_choices[1:]):
                if hole[1] < best_hole[1]:
                    best_hole = hole
                    best_node_procs = pit
                    node_idx = idx + 1

            # schedule task
            self.schedule_task(t, best_node_procs, *best_hole, node_idx)
            unsched.remove(t)
            scheduled.add(t)

            for dpdt in t.dependents:
                dpdt.unsched_deps -= 1

        return list(scheduled)

    def sequential_estimate(self):
        """Gets expected execution time if the tasks were run in serial"""
        return sum(t.exec_time(get_cpu_count()) for t in self.tasks)

    def install_successful(self, spec):
        task = self.spec_to_task[spec]

        # propagate to the base class
        self._dag_manager.install_successful(spec)

        # remove end time from the list
        self.incomplete_task_end_times.remove(task.end_time)

    def install_failed(self, spec):

        # Get all of the tasks that cannot execute
        failed = list(super().install_failed(spec))
        failed.append(spec)

        # Remove tasks from schedule
        for t in [self.spec_to_task[t] for t in failed]:
            self.tasks.remove(t)
            self.incomplete_task_end_times.remove(t.end_time)

        # Return the list to the caller
        return failed

    def pop_ready_specs(self):
        """ Generates a list of ready tasks

        The schedule is generated with concrete start and end times. A task may
        begin when all tasks with an end time earlier than its start time have
        completed"""

        earliest_end_time = min(self.incomplete_task_end_times,
                                default=float('inf'))
        for t in self.tasks:
            if t.start_time < earliest_end_time and t not in self._popped_tasks:
                # ensure the task is only returned from this generator once
                self._popped_tasks.add(t)
                yield t.n, t.spec_node.spec


class CPRDagScheduler(TwoStepSchedulerBase):
    """Critical Path Reduction scheduler.
    CPR is a two step scheduler that iteratively to allocates more cores
    to critical tasks and tests the resulting schedule for a decrease in
    makespan.

    It's time complexity is high, and becomes very costly after about 200 specs:

    --> O(EV^2P + V^3P(logV + PlogP)) Where:
    P is the number of cores
    V is the number of specs
    E is the total number of dependencies
    """

    def __init__(self, timing_db, dag_manager=None):
        super().__init__(dag_manager, timing_db)
        self._schedule_build_time = None

    def sched_build_time(self):
        """Returns time it took to build the schedule in seconds"""
        return self._schedule_build_time

    def build_schedule(self, print_time=False,
                       nproc=get_cpu_count(),
                       nnode=1,
                       scalability_filter=float("inf")):

        start = datetime.now()

        self._build_schedule_called()

        # Initializing tasks and schedule
        self._build_task_list()
        self.calculate_levels()
        self.mn_mls(self.tasks, nproc, nnode)

        # Keep looping until a better schedule can't be created
        sched_modified = True
        scalable_tasks = [t for t in self.tasks
                          if t.is_scalable(scalability_filter)]
        while sched_modified:
            sched_modified = False
            resizable_tasks = [t for t in scalable_tasks if t.n < nproc]

            self.calculate_levels()
            self.mn_mls(self.tasks, nproc, nnode)
            old_makespan = self.get_makespan()
            while not sched_modified and len(resizable_tasks) > 0:
                critical_tasks = self.critical_tasks(resizable_tasks)
                # greatest improvement in work change is used as tie-breaker
                ct = max(critical_tasks, key=lambda task: task.work_change())
                ct.n += 1

                self.calculate_levels()
                self.mn_mls(self.tasks, nproc, nnode)
                new_makespan = self.get_makespan()

                # if the makespan decreased, use the new schedule
                if new_makespan < old_makespan:
                    # print(new_makespan)
                    sched_modified = True
                else:
                    # Otherwise, revert the allotment and remove
                    # the task from the list
                    ct.n -= 1
                    resizable_tasks.remove(ct)

        self.incomplete_task_end_times = sorted(t.end_time for t in self.tasks)
        self.spec_to_task = {t.spec_node.spec: t for t in self.tasks}

        self._schedule_build_time = (datetime.now() - start)


class MCPADagScheduler(TwoStepSchedulerBase):
    """Modified Critical Path and Allocation scheduler.
    MCPA is a two step scheduler that iteratively to allocates more cores
    to critical tasks that maximize the reduction to work area.

    Where:
    P is the number of cores
    V is the number of specs
    E is the number of dependencies between specs
    """

    def __init__(self, timing_db, dag_manager=None):
        super().__init__(dag_manager, timing_db)
        self._schedule_build_time = None

    def sched_build_time(self):
        return self._schedule_build_time

    def compute_area(self, tot_procs):
        """Calculates the computing area, which is the total computed area"""
        return sum(t.exec_time() * t.n for t in self.tasks) / tot_procs

    def build_schedule(self,
                       print_time=False,
                       nproc=get_cpu_count(),
                       nnode=1):

        start = datetime.now()

        self._build_schedule_called()

        # Initializing tasks and schedule
        self._build_task_list()
        num_prec_levels = len(self.init_precedence_levels())
        self.calculate_levels()

        # Used to calculate precedence level
        visited = set()

        # continue while Critical path exceeds average compute area
        while (max(t._b_level for t in self.entry_tasks()) >
               self.compute_area(nproc)):

            # list of tasks on the critical path that can allocate more cores
            crit_tasks = [t for t in
                          self.critical_tasks(
                              tt for tt in self.tasks if tt.n < nproc)]

            # calculate precedence levels, but only on visited tasks
            prec_levels = [0] * num_prec_levels
            for t in visited:
                prec_levels[t.precedence_level] += t.n

            for t in crit_tasks:
                prec_levels[t.precedence_level] += t.n

            # select the task with the greatest improvement
            opt_task = max(
                (t for t in crit_tasks
                 if prec_levels[t.precedence_level] < nproc),
                key=lambda x: x.work_change(), default=None)

            if not opt_task:
                # print('Warn: No improvement found at iteration', i)
                break

            # Update the task, precedence/b/t-levels, and visited set
            opt_task.n += 1
            self.calculate_levels()
            visited.add(opt_task)

        self.mn_mls(self.tasks, nproc, nnode)
        self.incomplete_task_end_times = sorted(t.end_time for t in self.tasks)
        self.spec_to_task = {t.spec_node.spec: t for t in self.tasks}

        self._schedule_build_time = (datetime.now() - start)


class CPADagScheduler(TwoStepSchedulerBase):
    """Critical Path and Allocation Scheduling algorithm.
    Low time complexity two-step scheduling algorithm, O(V(V + E)P"""

    def __init__(self, timing_db, dag_manager=None):
        super().__init__(dag_manager, timing_db)
        self._schedule_build_time = None

    def sched_build_time(self):
        return self._schedule_build_time

    def compute_area(self, tot_procs):
        """Calculates the computing area, which is the total computed area"""
        return sum(t.exec_time() * t.n for t in self.tasks) / tot_procs

    def build_schedule(self,
                       print_time=False,
                       nproc=get_cpu_count(),
                       nnode=1):

        start = datetime.now()
        self._build_schedule_called()

        # Initializing tasks and schedule
        self._build_task_list()
        self.calculate_levels()

        # continue while Critical path exceeds average compute area
        while (max(t._b_level for t in self.entry_tasks()) >
               self.compute_area(nproc)):

            # list of tasks on the critical path that can allocate more cores
            crit_tasks = [t for t in
                          self.critical_tasks(
                              tt for tt in self.tasks if tt.n < nproc)]

            # select the task with the greatest work improvement
            opt_task = max(
                (t for t in crit_tasks),
                key=lambda x: x.work_change(), default=None)

            if not opt_task:
                # print('Warn: No improvement found at iteration', i)
                break

            # Update the task, precedence/b/t-levels, and visited set
            opt_task.n += 1
            self.calculate_levels()

        self.mn_mls(self.tasks, nproc, nnode)
        self.incomplete_task_end_times = sorted(t.end_time for t in self.tasks)
        self.spec_to_task = {t.spec_node.spec: t for t in self.tasks}

        self._schedule_build_time = (datetime.now() - start)


def schedule_selector(specs,
                      timing_db=None,
                      preferred_scheduler=None,
                      nproc=get_cpu_count(),
                      nnode=1):
    """Selects and initializes the best scheduler from the provided
    information"""

    dm = DagManager()
    dm.add_specs(specs, 2)
    dm.prune_installed()
    tty.msg('Created DAG with %s Specs' % dm.count())

    if not timing_db or preferred_scheduler == 'SimpleDagScheduler':
        # No timing information prevents sophisticated scheduling
        tty.msg('Selected SimpleDagScheduler')
        return SimpleDagScheduler(dag_manager=dm)

    mcpa_sched = MCPADagScheduler(timing_db, dag_manager=dm)
    mcpa_sched.build_schedule(nproc=nproc, nnode=nnode)

    # CPR is costly, do not use when there too many Specs
    if dm.count() > 200:
        tty.msg('Selected MCPADagScheduler')
        return mcpa_sched

    cpr_sched = CPRDagScheduler(timing_db, dag_manager=dm)
    cpr_sched.build_schedule(nproc=nproc, nnode=nnode)

    mcpa_makespan = mcpa_sched.get_makespan()
    cpr_makespan = cpr_sched.get_makespan()

    # When the user has explicitly requested a scheduler
    if preferred_scheduler == 'CPRDagScheduler':
        tty.msg('Selected CPRDagScheduler')
        return cpr_sched
    elif preferred_scheduler == 'MCPADagScheduler':
        tty.msg('Selected MCPADagScheduler')
        return mcpa_sched

    # select the schedule with the shortest makespan
    if cpr_makespan < mcpa_makespan or preferred_scheduler == 'CPRDagScheduler':
        tty.msg('Selected CPRDagScheduler')
        return cpr_sched
    if mcpa_sched.sequential_estimate() < mcpa_makespan:
        tty.msg('Selected SimpleDagScheduler')
        return SimpleDagScheduler(dag_manager=dm)
    else:
        tty.msg('Selected MCPADagScheduler')
        return mcpa_sched


def compare_schedules(spec,
                      timing_db=None,
                      phase_tasks=False,
                      nproc=get_cpu_count(),
                      nnode=1):
    dm = DagManager()
    dm.add_specs([spec], 4)
    dm.prune_installed()
    # tty.msg('Created DAG with %s Specs' % dm.count())

    def mk_sched(sched_type):
        sched = sched_type(timing_db, dag_manager=dm)
        sched.decompose_task_phases(phase_tasks)
        sched.build_schedule(nproc=nproc, nnode=nnode)
        sched.test_schedule_integrity()
        return sched

    cpa_sched = mk_sched(CPADagScheduler)
    mcpa_sched = mk_sched(MCPADagScheduler)
    cpr_sched = mk_sched(CPRDagScheduler)

    filt80_cpr_sched = CPRDagScheduler(timing_db, dag_manager=dm)
    filt80_cpr_sched.decompose_task_phases(phase_tasks)
    filt80_cpr_sched.build_schedule(nproc=nproc, scalability_filter=0.8,
                                    nnode=nnode)
    filt80_cpr_sched.test_schedule_integrity()

    cpa_makespan = cpa_sched.get_makespan()
    mcpa_makespan = mcpa_sched.get_makespan()
    cpr_makespan = cpr_sched.get_makespan()
    filt80_cpr_makespan = filt80_cpr_sched.get_makespan()

    tty.msg('CPA makespan:           %6ss creation time: %s' % (
        round(cpa_makespan, 1), cpa_sched.sched_build_time()))
    tty.msg('MCPA makespan:          %6ss creation time: %s' % (
        round(mcpa_makespan, 1), mcpa_sched.sched_build_time()))
    tty.msg('CPR  makespan:          %6ss creation time: %s' % (
        round(cpr_makespan, 1), cpr_sched.sched_build_time()))
    tty.msg('Filtered CPR  makespan: %6ss creation time: %s' % (
        round(filt80_cpr_makespan, 1), filt80_cpr_sched.sched_build_time()))
    tty.msg('Serial makespan:        %6ss' %
            round(mcpa_sched.sequential_estimate(), 1))

    # mcpa_sched.print_schedule()
    #cpr_sched.print_schedule()
    #filt80_cpr_sched.print_schedule()

    s = spec.name
    print([s, 'Filtered CPR', filt80_cpr_sched.get_makespan(),
           filt80_cpr_sched.sched_build_time().total_seconds()], ',')
    print([s, 'CPR', cpr_sched.get_makespan(),
          cpr_sched.sched_build_time().total_seconds()], ',')
    print([s, 'MCPA', mcpa_sched.get_makespan(),
           mcpa_sched.sched_build_time().total_seconds()], ',')
    print([s, 'CPA', cpa_sched.get_makespan(),
           cpa_sched.sched_build_time().total_seconds()], ',')
    print([s, 'Simple Parallel', cpr_sched.sequential_estimate(), 0], ',')


def compare_large_schedules(specs,
                            timing_db=None,
                            phase_tasks=False,
                            nproc=get_cpu_count(),
                            nnode=1):
    dm = DagManager()
    dm.add_specs(specs, 4)
    dm.prune_installed()
    # tty.msg('Created DAG with %s Specs' % dm.count())

    def mk_sched(sched_type):
        sched = sched_type(timing_db, dag_manager=dm)
        sched.decompose_task_phases(phase_tasks)
        sched.build_schedule(nproc=nproc, nnode=nnode)
        sched.test_schedule_integrity()
        return sched

    cpa_sched = mk_sched(CPADagScheduler)
    mcpa_sched = mk_sched(MCPADagScheduler)

    print(len(cpa_sched.tasks), 'tasks')
    print(['sched', 'MCPA', mcpa_sched.get_makespan(),
           mcpa_sched.sched_build_time().total_seconds()], ','),
    print(['sched', 'CPA', cpa_sched.get_makespan(),
           cpa_sched.sched_build_time().total_seconds()], ',')
    print(['sched', 'Simple Parallel', mcpa_sched.sequential_estimate(), 0], ',')

    def print_sched_node_allotment(sched, name):
        print(name)
        assigned_nodes = set(t.exec_node_id for t in sched.tasks)
        print("Scheduled Nodes:", assigned_nodes)
        for a in assigned_nodes:
            print("Node",
                  a,
                  sum(1 for t in sched.tasks if t.exec_node_id == a),
                  "Tasks")

    print_sched_node_allotment(mcpa_sched, 'MCPA Schedule')
    print_sched_node_allotment(cpa_sched, 'CPA Schedule')
