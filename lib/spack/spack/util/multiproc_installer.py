import copy
import time
import os
import traceback
from multiprocessing import Manager, cpu_count
from spack.spec import Spec
import llnl.util.tty as tty
from llnl.util.lang import memoized
from spack.util.web import NonDaemonPool


@memoized
def get_cpu_count():
    return int(cpu_count()/2)


def install_from_queue(work_queue, installation_result_queue, kwargs):
    while True:
        try:
            jobs, serialized_spec = work_queue.get(block=True)
            spec = Spec.from_yaml(serialized_spec).concretized()

            if jobs:
                kwargs['make_jobs'] = jobs

            if kwargs['make_jobs'] is None:
                kwargs['make_jobs'] = get_cpu_count()

            kwargs['install_deps'] = False

            tty.msg(
                'Installing %s with %s jobs' % (spec.name, kwargs['make_jobs']))
            with tty.SuppressOutput(msg_enabled=False,
                                    warn_enabled=False,
                                    error_enabled=False):
                    spec.package.do_install(**kwargs)

            installation_result_queue.put_nowait((None, serialized_spec))
        except Exception as e:
            tty.error(e)
            traceback.print_exc()
            installation_result_queue.put_nowait(('ERROR', serialized_spec))


class SpecInstaller:
    """Interface for installing specs"""

    def install_dag(self, dag_scheduler, *kwargs):
        raise NotImplementedError()

    @staticmethod
    def _progress_prompt_str(installing, ready, remaining):
        return '[%s|%s|%s]' % (str(installing).rjust(2), str(ready).rjust(2),
                               str(remaining).rjust(2))


class MultiProcSpecInstaller(SpecInstaller):
    """Forks multiple processes to install specs on a single node"""

    def __init__(self):
        pass

    def install_dag(self, dag_scheduler, kwargs):
        """Installs a list of specs"""

        start_time = time.time()

        # Initialize structures relating to the process pool
        outstanding_installs = {}
        work_queue = Manager().Queue()
        installation_result_queue = Manager().Queue()

        try:
            with NonDaemonPool(processes=get_cpu_count()) as pool:

                # Create a process for each core, initialize it with async
                # queue for sending specs/receiving results
                for _ in range(get_cpu_count()):
                    pool.apply_async(install_from_queue,
                                     (work_queue,
                                      installation_result_queue,
                                      kwargs))

                # Initialize spec structures
                if not dag_scheduler.schedule_is_built():
                    dag_scheduler.build_schedule()
                ready_specs = set(dag_scheduler.pop_ready_specs())

                # tty.msg(self._progress_prompt_str(
                #     'Installing', 'Ready', 'Unscheduled'))

                while len(ready_specs) > 0 or len(outstanding_installs) > 0:
                    for jobs, spec in ready_specs:
                        # Note: to_json does not support all_deps
                        work_queue.put_nowait(
                            (jobs, spec.to_yaml(all_deps=True)))
                        outstanding_installs[spec.full_hash()] = spec

                    ready_specs.clear()

                    # Block until something finishes
                    # TODO put a timeout and TimeoutError handler here
                    res, serialized_spec = installation_result_queue.get(True)

                    spec = Spec.from_yaml(serialized_spec)

                    # Message indicates an error
                    if res:
                        removed_specs = list(dag_scheduler.install_failed(spec))
                        rm_specs_str = '\n\t'.join(
                                        s.name for s in sorted(removed_specs))
                        tty.error('Installation of "%s" failed. Skipping %d'
                                  ' dependent packages: \n\t%s' %
                                  (spec.name, len(removed_specs), rm_specs_str))
                    else:
                        dag_scheduler.install_successful(spec)
                        tty.msg('Installed %s' % spec.name)

                    # Greedily get all ready specs
                    for j_s in dag_scheduler.pop_ready_specs():
                        ready_specs.add(j_s)

                    outstanding_installs.pop(spec.full_hash())

        except Exception as e:
            import traceback
            tty.error("Installation pool error, %s\n" % (str(e)))
            traceback.print_exc()
        finally:
            tty.msg('Installation finished (%s)' % time.strftime(
                '%H:%M:%S', time.gmtime(time.time() - start_time)))
