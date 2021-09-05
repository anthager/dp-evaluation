#!/usr/bin/python3.7

try:
    from opacus.privacy_analysis import compute_rdp, get_privacy_spent
except ImportError:
    pass
try:
    from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
except ImportError:
    pass

from dpevaluation.utils.log import log, log_elapsed_time
from sortedcontainers import SortedSet
from threading import Thread, Lock
import numpy as np
import time


class DP_Utils:
    def __init__(self,
                 lib: str,
                 target_epsilons,
                 batch_size,
                 epochs,
                 nmul_low=None,
                 nmul_high=None):
        self.lib = lib  # opacus | tf
        self.target_epsilons = target_epsilons
        self.batch_size = batch_size
        self.epochs = epochs
        """
        nmul_low and nmul_high range matters because we hit higher noise_multipliers 
        in the search for target_epsilons (which require larger opt_order). nmul_low/
        high could be set to 100 & 0.1 for extreme target_epsilons, but will require a 
        large range of opt_orders to hit the target_epsilons as close as possible, and
        thus also take long time to compute.
        """
        self.nmul_low = 20 if nmul_low is None else nmul_low
        self.nmul_high = 0.5 if nmul_high is None else nmul_high

    def calc_noise_multipliers(self, sample_size, delta, trace=False):
        """
        Optimal orders are produced based on hyperparameters (hp): epochs, batch_size, 
        sample_size, and delta.
        """
        opt_orders = SortedSet()
        """
        As a rule of thumb, noise_multipliers should be about 1.0 for good privacy.
        We compute noise_multipliers based on target_epsilons. 
        """
        noise_multipliers = []
        found_epsilons = []

        start_time = time.time()
        # Thread for each target_epsilon to speed up process
        threads = []
        lock = Lock()
        for target_eps in self.target_epsilons:
            # -1 for non-private training
            if target_eps == -1:
                found_epsilons.append(target_eps)
                noise_multipliers.append(target_eps)
            else:
                t = Thread(
                    target=self.__calc_noise_multipliers_thread,
                    args=(lock,
                          sample_size,
                          delta,
                          target_eps,
                          opt_orders,
                          noise_multipliers,
                          found_epsilons,
                          trace))
                t.daemon = True
                threads.append(t)

        if not threads:
            log("debug", "No target epsilon provided")
        else:
            log("info", "Calculating noise_multipliers for: target ε: %s (batch_size=%s, epochs=%s, sample_size=%s, δ=%.2E)" %
                (str(self.target_epsilons)[1:-1], self.batch_size, self.epochs, sample_size, delta))

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            opt_orders = list(opt_orders)

            log("info", "Found noise_multipliers for epsilons: %s" %
                str(list(np.around(found_epsilons, 4)))[1:-1])

            for i, nm in enumerate(noise_multipliers):
                log("debug", "nm=%.4f -> ε=%.4f" % (nm, found_epsilons[i]))

            if trace:
                log("debug", "opt_orders: %s" % str(opt_orders)[1:-1])

            log_elapsed_time(start_time, time.time())

        return noise_multipliers, found_epsilons, opt_orders

    def __calc_noise_multipliers_thread(self,
                                        lock,
                                        sample_size,
                                        delta,
                                        target_eps,
                                        opt_orders,
                                        noise_multipliers,
                                        found_epsilons,
                                        trace):
        """
        https://github.com/ftramer/Handcrafted-DP/blob/2a1c3a84ee54b6acceca19ebb8fc28e749c4222b/dp_utils.py#L185
        """
        nmul_low = self.nmul_low
        nmul_high = self.nmul_high
        eps_low, opt_order = self.__calc_epsilon(
            sample_size, delta, target_eps, nmul_low, opt_orders, trace)
        eps_high, opt_order = self.__calc_epsilon(
            sample_size, delta, target_eps, nmul_high, opt_orders, trace)

        assert eps_low < target_eps
        assert eps_high > target_eps

        while eps_high - eps_low > 0.01:
            nmul_mid = (nmul_high + nmul_low) / 2
            eps_mid, opt_order = self.__calc_epsilon(
                sample_size, delta, target_eps, nmul_mid, opt_orders, trace)

            if eps_mid <= target_eps:
                nmul_low = nmul_mid
                eps_low = eps_mid
            else:
                nmul_high = nmul_mid
                eps_high = eps_mid

        if trace:
            log("debug", "Found noise_multiplier=%.4f for ε=%.4f (α=%.1f)" %
                (nmul_low, eps_low, opt_order))

        lock.acquire()
        try:
            noise_multipliers.append(nmul_low)
            # we save the epsilon for the metadata
            found_epsilons.append(target_eps)
        finally:
            lock.release()

    def __calc_epsilon(self,
                       sample_size,
                       delta,
                       target_eps,
                       _noise_multiplier,
                       opt_orders,
                       trace):
        if _noise_multiplier == 0.0:
            return float("inf")
        """
        There is very little downside in expanding the list of orders for 
        which RDP is computed, according to: 
        https://github.com/tensorflow/privacy/blob/693dd666c3c05ec09dbc361f303bb045207b8163/tutorials/walkthrough/README.md
        However in order to get very small & exact epsilon values, we need 
        a lot of orders to hit all opt_orders in search for target_epsilons. 
        """
        _orders = \
            [1 + x / 10. for x in range(1, 100)] + \
            list(range(12, int(64))) + [64, 128, 512, 1024, 2056]
        # import math
        # _steps = int(math.ceil(self.epochs * sample_size / batch_size))
        _steps = self.epochs * (sample_size // self.batch_size)
        sampling_probability = self.batch_size / sample_size

        rdp = compute_rdp(
            q=sampling_probability,
            noise_multiplier=_noise_multiplier,
            steps=_steps,
            orders=_orders)

        if self.lib == "opacus":
            eps, opt_order = get_privacy_spent(_orders, rdp, delta)
        elif self.lib == "tf":
            eps, _, opt_order = \
                get_privacy_spent(_orders, rdp, None, delta)

        if trace:
            log("debug", "Target ε=%.1f: tried nm=%.1f -> (ε=%.4f, α=%.1f)" %
                (target_eps, _noise_multiplier, eps, opt_order))

        opt_orders.add(opt_order)

        return eps, opt_order
