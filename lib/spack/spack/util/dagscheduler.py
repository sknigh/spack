import copy
import time
from multiprocessing import Pool, cpu_count
import spack
from spack.spec import Spec
import llnl.util.tty as tty


class SpecNode:
    """A spec node contains a Spec, its hash, immediate dependencies and
    dependents"""

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
        # TODO: This check assumes hyperthreading is enabled
        adjusted_workers = int(max(1, min(workers, int(cpu_count() / 2))))

        self.ignore_error = ignore_error
        self.workers = adjusted_workers
        self.pool = Pool(adjusted_workers)

        if adjusted_workers != workers:
            tty.warn('ParallelConcretizer adjusted worker count from %s to %s',
                     workers, adjusted_workers)

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


class DagScheduler:
    """Curates DAG with multiple specs"""

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
        untraversed = copy.copy(spec_node.dependents)

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
