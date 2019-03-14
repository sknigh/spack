import numpy as np


class InstallTimeEstimatorBase:
    """Base class for fitting a curve to observed installation times."""
    def __init__(self, make_jobs=[], times=[]):
        self._jobs = make_jobs
        self._times = times
        self._estimate = None

    def _compute_estimate(self):
        """Generate a fitting curve"""
        raise NotImplementedError()

    def estimate(self, jobs):
        """Estimate installation time for a given number of make jobs"""
        if not self._estimate:
            self._compute_estimate()

        return self._estimate(jobs)

    def add_measurement(self, make_jobs, times):
        """Add a new measurement"""
        if len(make_jobs) != len(times):
            raise Exception('"make_jobs" and "times" lists are not the same '
                            'length: %s != %s' % (len(make_jobs), len(times)))
        self._jobs.extend(make_jobs)
        self._times.extend(times)
        self._estimate = None


class LogTimeEstimator(InstallTimeEstimatorBase):
    """NOTE: DEPRECATED
    Use a logarithmic third degree polynomial curve to estimate build time"""
    def __init__(self, make_jobs=[], times=[]):
        super(InstallTimeEstimatorBase, self).__init__(make_jobs, times)

    def _compute_estimate(self):
        self._estimate = np.poly1d(
            np.polyfit(np.log(self._jobs), self._times, deg=3))


class AsymptoticTimeEstimator(InstallTimeEstimatorBase):
    """Use an asymptotic first degree polynomial curve to estimate build time"""
    def __init__(self, make_jobs=[], times=[]):
        super(InstallTimeEstimatorBase, self).__init__(make_jobs, times)

    def _compute_estimate(self):
        self._estimate = np.poly1d(
            np.polyfit(1/self._jobs, self._times, deg=1))
