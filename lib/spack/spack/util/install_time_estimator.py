import numpy as np


class InstallTimeEstimatorBase:
    """Base class for fitting a curve to observed installation times."""
    def __init__(self, make_jobs=[], times=[]):
        self._jobs = np.array(make_jobs)
        self._times = np.array(times)
        self._estimate = None

    def _compute_estimate(self):
        """Generate a fitting curve"""
        raise NotImplementedError()

    def estimate(self, jobs):
        """Estimate installation time for a given number of make jobs"""
        if not self._estimate:
            self._compute_estimate()

        return self._estimate(jobs)

    def add_measurements(self, make_jobs, times):
        """Add a list of measurements"""
        if len(make_jobs) != len(times):
            raise Exception('"make_jobs" and "times" lists are not the same '
                            'length: %s != %s' % (len(make_jobs), len(times)))

        # Extend lists
        self._jobs = np.append(self._jobs, make_jobs)
        self._times = np.append(self._times, times)

        # Ensure estimate gets recalculated
        self._estimate = None

    def add_measurement(self, make_job, time):
        """Add a measurement"""
        self.add_measurements([make_job], [time])


class AsymptoticTimeEstimator(InstallTimeEstimatorBase):
    """Use an asymptotic first degree polynomial curve to estimate build time"""
    def __init__(self, make_jobs=[], times=[]):
        super().__init__(make_jobs, times)

    def _compute_estimate(self):
        self._estimate = np.poly1d(
            np.polyfit(1/self._jobs, self._times, deg=1))

    def estimate(self, jobs):
        """Estimate installation time for a given number of make jobs"""
        if not self._estimate:
            self._compute_estimate()

        return self._estimate(1/jobs)
