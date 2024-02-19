import random
import datetime as dt
import matplotlib.dates as mdates
import math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
from scipy import optimize, stats
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

            def end(self) -> float:
                return self.start + self.delta

        arrival = np.cumsum(T)
        core_intervals = [Interval(arrival[i], tau[i]) for i in range(N)]
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

#     def показать(self):
#         # generate main timeline
#         start_date = np.datetime64(2023, 11, 15, 17, 0)
#         curr_date = start_date
#         timeline_main = []
#         for i, time_delta in enumerate(self.поток_заяв):
#             timeline_main.append(curr_date)
#             curr_date += np.timedelta64(time_delta, 'm')
        
#         # generate state timelines
#         timeline_state_array = []
#         state_count = self.n + self.m + 1
#         for j in range(state_count):
#             timeline_state = []
#             for i, time_delta in enumerate(self.поток_заяв):
#                 curr_date += np.timedelta64(time_delta, 'm')
#                 timeline_state.append(curr_date)
#         curr_date = start_date
#         data = []
#         for i in range(self.n + self.m + 1):
#         data.append((start, end, 'S{}'.format(i)))
#         data = [    (dt.datetime(2018, 7, 17, 0, 15), dt.datetime(2018, 7, 17, 0, 30), 'sleep'),
#                     (dt.datetime(2018, 7, 17, 0, 30), dt.datetime(2018, 7, 17, 0, 45), 'eat'),
#                     (dt.datetime(2018, 7, 17, 0, 45), dt.datetime(2018, 7, 17, 1, 0), 'work'),
#                     (dt.datetime(2018, 7, 17, 1, 0), dt.datetime(2018, 7, 17, 1, 30), 'sleep'),
#                     (dt.datetime(2018, 7, 17, 1, 15), dt.datetime(2018, 7, 17, 1, 30), 'eat'), 
#                     (dt.datetime(2018, 7, 17, 1, 30), dt.datetime(2018, 7, 17, 1, 45), 'work')
#                 ]

#         cats = {"sleep" : 1, "eat" : 2, "work" : 3}
#         colormapping = {"sleep" : "C0", "eat" : "C1", "work" : "C2"}

#         verts = []
#         colors = []
#         for d in data:
#             v =  [(mdates.date2num(d[0]), cats[d[2]]-.4),
#                 (mdates.date2num(d[0]), cats[d[2]]+.4),
#                 (mdates.date2num(d[1]), cats[d[2]]+.4),
#                 (mdates.date2num(d[1]), cats[d[2]]-.4),
#                 (mdates.date2num(d[0]), cats[d[2]]-.4)]
#             verts.append(v)
#             colors.append(colormapping[d[2]])

#         bars = PolyCollection(verts, facecolors=colors)

#         fig, ax = plt.subplots()
#         ax.add_collection(bars)
#         ax.autoscale()
#         loc = mdates.MinuteLocator(byminute=[0,15,30,45])
#         ax.xaxis.set_major_locator(loc)
#         ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

#         ax.set_yticks([1,2,3])
#         ax.set_yticklabels(["sleep", "eat", "work"])
#     plt.show()
#     cats = {'S1'}
#     print(self.поток_заяв)
#     print(self.поток_обсл)
# data = []
# first_date = np.datetime64(2023, 7, 15, 17, 0)
# for date 
# for i in range(self.n + self.m + 1):
#     data.append((start, end, 'S{}'.format(i)))
# data = [    (dt.datetime(2018, 7, 17, 0, 15), dt.datetime(2018, 7, 17, 0, 30), 'sleep'),
#             (dt.datetime(2018, 7, 17, 0, 30), dt.datetime(2018, 7, 17, 0, 45), 'eat'),
#             (dt.datetime(2018, 7, 17, 0, 45), dt.datetime(2018, 7, 17, 1, 0), 'work'),
#             (dt.datetime(2018, 7, 17, 1, 0), dt.datetime(2018, 7, 17, 1, 30), 'sleep'),
#             (dt.datetime(2018, 7, 17, 1, 15), dt.datetime(2018, 7, 17, 1, 30), 'eat'), 
#             (dt.datetime(2018, 7, 17, 1, 30), dt.datetime(2018, 7, 17, 1, 45), 'work')
#         ]

# cats = {"sleep" : 1, "eat" : 2, "work" : 3}
# colormapping = {"sleep" : "C0", "eat" : "C1", "work" : "C2"}

# verts = []
# colors = []
# for d in data:
#     v =  [(mdates.date2num(d[0]), cats[d[2]]-.4),
#           (mdates.date2num(d[0]), cats[d[2]]+.4),
#           (mdates.date2num(d[1]), cats[d[2]]+.4),
#           (mdates.date2num(d[1]), cats[d[2]]-.4),
#           (mdates.date2num(d[0]), cats[d[2]]-.4)]
#     verts.append(v)
#     colors.append(colormapping[d[2]])

# bars = PolyCollection(verts, facecolors=colors)

# fig, ax = plt.subplots()
# ax.add_collection(bars)
# ax.autoscale()
# loc = mdates.MinuteLocator(byminute=[0,15,30,45])
# ax.xaxis.set_major_locator(loc)
# ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

# ax.set_yticks([1,2,3])
# ax.set_yticklabels(["sleep", "eat", "work"])
# plt.show()