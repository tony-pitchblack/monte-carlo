import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.collections import PolyCollection

import random
import math

import numpy as np
import pandas as pd
# from scipy import optimize, stats

from dataclasses import dataclass
import itertools
from copy import copy

class QueueingSystemPoisson():
    def __init__(self, n, m, _lambda, _mu):
        self.n = n
        self.m = m
        self.states_count = n + m
        self._lambda = _lambda
        self._mu = _mu
        self.raffle()
    
    def __str__(self) -> str:
        s = ""
        s += "arrived: "
        for interv in self.core_intervals:
            s += "{:.2f}, ".format(interv.start)
        s += "\n"
        for state in self.states:
            s_state = ""
            for interv in state:
                s_state += "[{:.2f}, {:.2f}) ".format(interv.start, interv.end())
            s_state += "\n"
            s += s_state
        s += "left: "
        for interv in self.left_system:
            s += "{:.2f}, ".format(interv.start)
        s += "\n"
        return s

    def raffle(self, N=10):
        np.random.seed(1024)

        # Розыгрыш потока заявок
        calc_T = lambda R: -1/self._lambda*math.log(1-R)
        R = np.random.rand(N)
        T = np.array(list(map(calc_T, R)))

        # Розыгрыш потока обслуживания
        calc_tau = lambda R: -1/self._mu*math.log(1-R)
        R = np.random.rand(N)
        tau = np.array(list(map(calc_tau, R)))

        @dataclass
        class Interval:
            start: float
            delta: float
            has_left: bool

            def __post_init__(self):
                if self.has_left == None:
                    self.has_left = False
            def end(self) -> float:
                return self.start + self.delta

        arrival = np.cumsum(T)
        core_intervals = [Interval(arrival[i], tau[i], False) for i in range(N)]
        self.core_intervals = core_intervals
        states = [[] for i in range(self.states_count)] # array of arrays of intervals
        left_system = []
        for core_interval in core_intervals:
            # Найти подходящий канал или очередь
            k = -1
            for i in range(self.states_count):
                if len(states[i]) > 0:
                    if core_interval.start >= states[i][-1].end():
                        k = i
                        break
                else:
                    k = i
                    break
            if k == -1:
                core_interval.has_left = True
                left_system.append(core_interval)
                continue
            
            # Если найден свободный канал
            if k <= (self.n - 1):
                states[k].append(core_interval)
            # Иначе найдена очередь
            else:
                # Добавить интервал в ближайший свободный по времени канал
                p = np.argmin(list(map(lambda s: s[-1].end() if len(s) > 0 else 0, states[:self.n])))
                new_interv = copy(core_interval)
                if len(states[p]) > 0:
                    new_interv.start = states[p][-1].end() 
                states[p].append(new_interv)

                # Добавить интервал во все очереди каскадно
                #   Для этого сначала найти все части каскада
                get_cascade = lambda state: list(filter(lambda interval: interval.start > core_interval.end(), state))
                cascade_intervals = list(itertools.chain.from_iterable(map(get_cascade, states)))

                new_interv = copy(core_interval)
                i = k
                for casc_interv in sorted(cascade_intervals, key=lambda interv: interv.start):
                    new_interv.delta = casc_interv.start - new_interv.start
                    states[i].append(copy(new_interv))
                    if i == self.n:
                        break
                    else:
                        new_interv.start = casc_interv.start
                        i -= 1
        self.states = states
        self.left_system = left_system

    def show(self):
        DATE_START = dt.datetime(2023, 11, 15, 17, 0)
        timelines = []

        # generate arrival timeline
        date_curr = DATE_START
        timeline = []
        for interval in self.core_intervals:
            interval_start = DATE_START + dt.timedelta(minutes=interval.start)
            interval_end = DATE_START + dt.timedelta(seconds=interval.start+10)
            timeline.append((interval_start, interval_end))
        timelines.append(timeline)
        
        # generate state timelines
        for state in self.states:
            timeline_state = []
            for interval in state:
                interval_start = DATE_START + dt.timedelta(minutes=interval.start)
                interval_end = DATE_START + dt.timedelta(minutes=interval.end())
                timeline_state.append((interval_start, interval_end))
            timelines.append(timeline_state)
        
        TIMELINES_COUNT = self.n + self.m + 1
        cm = plt.cm.get_cmap('hsv', TIMELINES_COUNT) 
        colormapping = [cm(1.*i/TIMELINES_COUNT) for i in range(TIMELINES_COUNT)]

        verts = []
        colors = []
        for i, ts in enumerate(timelines):
            k = i+1
            for interval in ts:
                v =  [(mdates.date2num(interval[0]), k-.4),
                    (mdates.date2num(interval[0]), k+.4),
                    (mdates.date2num(interval[1]), k+.4),
                    (mdates.date2num(interval[1]), k-.4),
                    (mdates.date2num(interval[0]), k-.4)]
                colors.append(colormapping[i])
                verts.append(v)

        bars = PolyCollection(verts, facecolors=colors)

        fig, ax = plt.subplots()
        ax.add_collection(bars)
        ax.autoscale()
        loc = mdates.MinuteLocator(byminute=[0,15,30,45])
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

        ax.set_yticks([i for i in reversed(range(TIMELINES_COUNT))])
        labels = ["Заявки"]
        labels.extend([f"Канал {i}" if i < self.n else f"Место очереди {i - self.n}" for i in range(TIMELINES_COUNT-1)])
        ax.set_yticklabels(labels)
        plt.show()