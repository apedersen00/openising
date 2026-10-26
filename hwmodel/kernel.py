"""
kernel.py

Event-based simulator.
This is essentially just a to-do list ordered by time.
Each entry: at time t, call this function.
"""

import heapq
import itertools

ANALOG, EDGE, CONTROL = 0, 1, 2


class Handle:
    """
    Returned by schedule(). Cancelling marks the entry, which the heap then skips.
    """
    __slots__ = ('cancelled')

    def __init__(self):
        self.cancelled = False


class Simulator:
    def __init__(self):
        self.now = 0
        self.n_events = 0
        self._heap = []
        self._seq = itertools.count()
        self._stop = False

    def schedule(self, t, fn, *args, priority=EDGE):
        if t < self.now:
            raise ValueError(f'event at {t} ps is in the past (now {self.now} ps)')
        h = Handle()
        heapq.heappush(self._heap, (t, priority, next(self._seq), h, fn, args))
        return h

    def after(self, dt, fn, *args, priority=EDGE):
        return self.schedule(self.now + dt, fn, *args, priority=priority)

    @staticmethod
    def cancel(h):
        if h is not None:
            h.cancelled = True

    def peek(self):
        """
        Time of the next event (or None).
        """
        while self._heap and self._heap[0][3].cancelled:
            heapq.heappop(self._heap)
        return self._heap[0][0] if self._heap else None

    def stop(self):
        self._stop = True

    def run(self, until=None, max_events=None):
        """
        Run events with time <= until. Returns why it stopped:
            - 'empty'
            - 'until
            - 'stoppped'
            - 'max_events'
        """
        n = 0
        while True:
            t = self.peek()
            if t is None:
                return 'empty'
            if until is not None and t > until:
                self.now = until
                return 'until'
            if max_events is not None and n >= max_events:
                return 'max_events'
            t, _, _, _, fn, args = heapq.heappop(self._heap)
            self.now = t
            self.n_events += 1
            n += 1
            fn(*args)
            if self._stop:
                self._stop = False
                return 'stopped'