import copy
import time
from multiprocessing import Manager, cpu_count
from spack.spec import Spec
import llnl.util.tty as tty
from spack.util.web import NonDaemonPool


def install_from_queue(jobs, work_queue, installation_result_queue):
    while True:
        try:
            serialized_spec = work_queue.get(block=True)
            spec = Spec.from_yaml(serialized_spec).concretized()

            with tty.SuppressOutput(msg=True, info=True, warn=True):
                spec.package.do_install(make_jobs=jobs, install_deps=False)

            installation_result_queue.put_nowait((None, serialized_spec))
        except Exception as e:
            tty.error('Package build error!')
            tty.error(e)
            installation_result_queue.put_nowait(('ERROR', serialized_spec))


class SpecInstaller:
    """Interface for installing specs"""

    def install_dag(self, dag_scheduler):
        raise Exception('Interface function not implemented')

    @staticmethod
    def _progress_prompt_str(installing, ready, remaining):
        return '[%s|%s|%s]' % (str(installing).rjust(2), str(ready).rjust(2),
                               str(remaining).rjust(2))


class MultiProcSpecInstaller(SpecInstaller):
    """Forks multiple processes to install specs on a single node"""

    def __init__(self, max_jobs=1):
        # Assumes HT
        self.cpu_count = cpu_count() // 2
        max_jobs_bounded = min(max_jobs, self.cpu_count)

        if max_jobs_bounded != max_jobs:
            tty.warn('"max_jobs" changed from "%d" to %d' % (
                max_jobs, max_jobs_bounded))

        self.max_jobs = max_jobs_bounded
        self.parallelism = int(self.cpu_count // self.max_jobs)

    def install_dag(self, dag_scheduler):
        """Installs a list of specs"""

        start_time = time.now()

        # Initialize structures relating to the process pool
        outstanding_installs = {}
        work_queue = Manager().Queue()
        installation_result_queue = Manager().Queue()

        try:
            with NonDaemonPool(processes=self.max_jobs) as pool:
                tty.msg('Starting %d Pool Processes @ %d cores/job' % (
                    self.max_jobs, self.parallelism))

                for _ in range(self.max_jobs):
                    pool.apply_async(install_from_queue,
                                     (self.parallelism, work_queue,
                                      installation_result_queue))

                # Initialize spec structures
                    dag_scheduler_copy = copy.deepcopy(dag_scheduler)
                ready_specs = dag_scheduler_copy.ready_specs()

                def get_prompt():
                    res_qsize = installation_result_queue.qsize()
                    work_qsize = work_queue.qsize()
                    outstanding = len(outstanding_installs)
                    ready = len(ready_specs)
                    return self._progress_prompt_str(
                        outstanding - work_qsize - res_qsize,
                        ready + work_qsize,
                        dag_scheduler_copy.count())

                tty.msg(self._progress_prompt_str(
                    'Installing', 'Ready', 'Unscheduled'))

                while len(ready_specs) > 0 or len(outstanding_installs) > 0:
                    for spec in ready_specs:
                        # Note: to_json does not support all_deps
                        work_queue.put_nowait(spec.to_yaml(all_deps=True))
                        outstanding_installs[spec.full_hash()] = spec

                    ready_specs.clear()

                    # Block until something finishes
                    # TODO put a timeout and TimeoutError handler here
                    res, serialized_spec = installation_result_queue.get(True)

                    spec = Spec.from_yaml(serialized_spec)

                    if res is None:
                        ready_specs |= dag_scheduler_copy.install_successful(
                            spec)
                        outstanding_installs.pop(spec.full_hash())
                        tty.msg('%s Installed %s' % (get_prompt(), spec.name))
                    else:
                        removed_specs = dag_scheduler_copy.install_failed(spec)
                        outstanding_installs.pop(spec.full_hash())
                        removed_specs = '\n\t'.join(
                                        s.name for s in sorted(removed_specs))
                        tty.error('%s Installation of "%s" failed. Skipping %d'
                                  ' dependent packages: \n\t%s' %
                                  (get_prompt(), spec.name, len(removed_specs),
                                   removed_specs
                                   ))
                        # TODO: do something with result 'res' message
        except Exception as e:
            tty.error("Installation pool error, %s" % str(e))
        finally:
            tty.msg('Installation finished (%s)' % time.strftime(
                '%H:%M:%S', time.gmtime(time.time() - start_time)))

