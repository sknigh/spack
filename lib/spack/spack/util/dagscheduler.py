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
        self.weight_curve = None
        self.jobs = None
        self.optimal_jobs = 1

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
        # TODO: This check assumes hyperthreading is enabled
        adjusted_workers = int(max(1, min(workers, int(get_cpu_count() / 2))))

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
    Defines methods for manipulating the DAG
    Generating a weighting and/or job priority is left to derived classes"""

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
        """Allow static schedulers to weight the DAG"""
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

    def pop_read_specs(self):
        self._check_build_schedule_called()

        if len(self._ready_to_install) > 0:
            return self.build_jobs, self._ready_to_install.pop()
        else:
            return None

    def pop_all_ready_specs(self):
        for spec in self._ready_to_install:
            yield self.build_jobs, spec

        self._ready_to_install.clear()

    def install_successful(self, spec):
        for spec in self._install_successful(spec):
            self._ready_to_install.add(spec)


class BarbosaDagScheduler(DagSchedulerBase):
    """DAG scheduler that uses profiled build times to statically measure and
    minimize the makespan (the slowest build path)
    """

    # Some builds do not scale, require a minimum speedup before allowing
    # a build to use multiple cores
    FLAT_SCALING_CUTOFF = 1.15

    def __init__(self, pkg_build_times, workers=1, num_cores=get_cpu_count()):
        super.__init__()
        self.build_jobs = int(round(get_cpu_count() / workers))
        self._ready_to_install = set()
        self._pkg_build_times = pkg_build_times
        self._cores = num_cores
        self._mem_t_level = {}
        self._mem_b_level = {}

    def _init_estimators(self):
        """Makes an approximating curve and initial job counts for each spec
        node"""

        # First, create approximating curves for each profiled package
        unprofiled_nodes = set()
        for node in self.tree:
            name = node.spec.name

            # Remember the nodes that are not profiled
            if name not in self._pkg_build_times:
                unprofiled_nodes.add(node)
            else:
                e = AsymptoticTimeEstimator(
                    self._pkg_build_times[name]['make_jobs'],
                    self._pkg_build_times[name]['time'])
                node.weight_curve = e

        # When there are some profiled and some unprofiled nodes, create a mean
        # performance curve and assign it to the unprofiled
        if 0 < len(unprofiled_nodes) < len(self.tree):

            def avg(nums):
                return float(sum(nums) / max(len(nums), 1))

            job_range = list(range(1, self._cores + 1))

            time_set = []
            for node in self.tree:
                if node not in unprofiled_nodes:
                    time_set.append([node.spec.weight_curve.estimate(j) for
                                     j in job_range])

            # generate a list of mean profiled times
            mean_times = list(map(avg, (zip(*time_set))))
            mean_curve = AsymptoticTimeEstimator(job_range, mean_times)

            for node in unprofiled_nodes:
                node.weight_curve = mean_curve

            # Select number of jobs that maximize performance
            for node in self.tree:
                job_time_tup = ((j, e.estimate(j)) for j in
                                range(1, self._cores + 1))
                one_job_speed = job_time_tup[0][1]
                fast_j_count, fast_j_speed = max(job_time_tup,
                                                 key=lambda tup: tup[1])

                # When the build doesn't scale, do not allow it to use more
                # than a core
                if fast_j_speed / one_job_speed > self.FLAT_SCALING_CUTOFF:
                    node.optimal_jobs = fast_j_count
                    node.jobs = fast_j_count
                else:
                    node.optimal_jobs = 1
                    node.jobs = 1

        elif len(unprofiled_nodes) == len(self.tree):
            raise Exception('No profiling data. Not currently supported')

    def _ready_nodes(self):
        """Returns a list of nodes that have no dependencies and are ready to
        install"""
        return {node.spec for _, node in self.tree.items()
                if not len(node.dependencies)}

    def _synthetic_ready_nodes(self, completed_set):
        """Generates a set of nodes that would be ready if the nodes in the
        completed set were installed. Allows for a non-destructive DAG
        traversal"""
        c_set = set(completed_set)
        ready = []

        # TODO: do not iterate over the entire tree
        for node in [n for n in self.tree if n not in c_set]:
            if not (dep in c_set for dep in node.dependencies):
                ready.append(node)

        return ready

    def _tlevel(self, node):
        """Calculates node's top-level. top-level is defined as the longest
        path to the top of the DAG from the given node, excluding the node."""

        # # Use memoization when possible
        # if node.hash in self._mem_t_level:
        #     return self._mem_t_level[node.hash]
        # else:
        dep_tlevels = [0]
        for d in node.dependents:
            dep_tlevels.append(self._tlevel(d) + d.weight_curve(d.jobs))

        tlevel = max(dep_tlevels)
        self._mem_t_level[node.hash] = tlevel
        return tlevel

    def _blevel(self, node):
        """Calculates node's bottom-level. Bottom-level is defined as the
        path to the most distant leaf, including the node"""

        # # Use memoization
        # if node.hash in self._mem_b_level:
        #     blevel = self._mem_b_level[node.hash]
        # else:
        blevel = max([self._blevel(n) for n in node.dependencies] + [0])
        self._mem_t_level[node.hash] = blevel + node.weight_curve(node.jobs)

        return blevel

    def build_schedule(self, print_time=False):
        self._build_schedule_called = True
        self.prune_installed()

        self._mem_t_level = {}
        self._mem_b_level = {}
        self._init_estimators()

        scheduled_nodes = set()
        schedule_list = []
        ready = self._synthetic_ready_nodes(scheduled_nodes)

        while len(ready) > 0:
            # When there are more ready tasks than cores, sort the tasks by
            # t-level and select the heaviest tasks up to the core limit
            t_levels = {r: self._tlevel(r) for r in ready}
            ordered = sorted(t_levels.items(), key=lambda item: item[1],
                             reverse=True)
            ready = [v for k, v in ordered[:min(len(ready), get_cpu_count())]]

            def get_optimal_cpus():
                return sum([r.jobs for r in ready])

            # Compute the optimal number of CPUs for the tasks
            optimal_cpus = get_optimal_cpus()

            # Re-weight make-jobs while optimal CPUs exceed available
            # Do not allow jobs to decrease below 1
            while optimal_cpus > get_cpu_count():
                for t in ready:
                    t.jobs = min(1,
                                 int((get_cpu_count() / optimal_cpus) * t.jobs))
                optimal_cpus = get_optimal_cpus()

            # Compute b-level
            b_levels = {r: self._blevel(r) for r in ready}

            prev_max = set()
            while True:

                # Get nodes with min and max b-levels
                max_bl = max(b_levels.items(), key=lambda bv: bv[1])[0]
                min_bl = max(b_levels.items(), key=lambda bv: bv[1])[0]

                if min_bl in prev_max:
                    break
                else:
                    prev_max.add(max_bl)

                # Take a processor away from the faster job and give it to the
                # slower one
                min_bl.jobs -= 1
                max_bl.jobs += 1

                # Drop builds that have no jobs (i.e. it would be more
                # cost-effective to start them later)
                if min_bl.jobs == 0:
                    ready.remove(min_bl)
                    b_levels.pop(min_bl)
                else:
                    b_levels[min_bl] = self._blevel(min_bl)

                # Can't continue re-weighting if there < 2 jobs
                if len(b_levels) < 2:
                    break

            for n in ready:
                schedule_list.append(n)
                scheduled_nodes.add(n)

            # compute the set of ready tasks
            self._synthetic_ready_nodes(scheduled_nodes)
