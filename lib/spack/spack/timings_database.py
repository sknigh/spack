# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import sqlite3
import llnl.util.tty as tty
from multiprocessing import cpu_count


class TimingsDatabase:
    """Class for building and querying a database of timing statistics."""

    _phase_time_table = """
        CREATE TABLE IF NOT EXISTS phase_time (
         name string NOT NULL,
         phase string NOT NULL,
         jobs int NOT NULL,
         time int DEFAULT 0
        ); """

    def __init__(self, db_name='timings.sqlite3'):
        try:
            self._conn = sqlite3.connect(db_name)
        except Exception as e:
            tty.die('Timings database "%s" not found' % db_name)

        self._cur = self._conn.cursor()

        # Initialize table(s)
        self._cur.execute(self._phase_time_table)

    def __del__(self):
        self._conn.commit()
        self._conn.close()

    def add_phase_time(self, package, phase, jobs, time_seconds):
        if not jobs:
            jobs = cpu_count()

        self._cur.execute('INSERT INTO phase_time VALUES (?,?,?,?)',
                          (package, phase, jobs, time_seconds))
        self._conn.commit()

    # def phase_time_jobs(self, package, phase, jobs):
    #     """Returns average time for the specified jobs at a given phase"""
    #
    #     query = """SELECT AVG(time)
    #                FROM phase_time WHERE name=? AND phase=? AND jobs=?
    #                GROUP BY name, phase, jobs"""
    #
    #     return next(self._cur.execute(query, (package, phase, jobs)))[0]

    # def phase_time_gen(self, package, phase):
    #     """Returns tuple (jobs, avg time) generator"""
    #
    #     query = """SELECT jobs, AVG(time)
    #                FROM phase_time WHERE name=? AND phase=?
    #                GROUP BY name, phase, jobs"""
    #
    #     for c in self._cur.execute(query, (package, phase)):
    #         yield c

    # def packages_and_phases_gen(self):
    #     """Returns tuple (package, phase) generator for each unique
    #     package and phase"""
    #
    #     query = """SELECT DISTINCT name, phase FROM phase_time"""
    #
    #     for c in self._cur.execute(query):
    #         yield c

    # def phases_gen(self, package):
    #     """Returns phases for a package"""
    #
    #     query = """SELECT DISTINCT phase
    #                FROM phase_time
    #                WHERE name=?"""
    #
    #     for c in self._cur.execute(query, (package,)):
    #         yield c

    # def packages_gen(self):
    #     """Returns timed packages"""
    #
    #     query = """SELECT DISTINCT phase
    #                FROM phase_time"""
    #
    #     for c in self._cur.execute(query):
    #         yield c

    def package_timings(self, package):
        """Returns tuples of (jobs, total execution) time for a package"""

        # Average duplicate (phase, jobs, time) measurements
        averaged_times = """SELECT phase, jobs, AVG(time) as avg_time
                            FROM phase_time
                            WHERE name=?
                            GROUP BY phase, jobs"""

        # Sum every phase for a package to get total execution time
        query = """SELECT jobs, SUM(avg_time)
                   FROM (%s)
                   GROUP BY jobs""" % averaged_times

        for c in self._cur.execute(query, (package,)):
            yield c
