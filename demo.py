# import random
# import datetime as dt
# import matplotlib.dates as mdates
# import math
# import matplotlib.pyplot as plt
# from matplotlib.collections import PolyCollection
# import numpy as np
# import pandas as pd
# from scipy import optimize, stats
from queueing_system import QueueingSystemPoisson
qs = QueueingSystemPoisson(1, 2, 2, 1)
# TODO: fix excessive cascade queue
print(qs)