# import random
# import datetime as dt
# import matplotlib.dates as mdates
# import math
# import matplotlib.pyplot as plt
# from matplotlib.collections import PolyCollection
# import numpy as np
# import pandas as pd
# from scipy import optimize, stats

# TODO fix excessive cascade queue???
# TODO fix arrival timeline
# TODO fix y-label order
# TODO fix x-axis labels
# TODO fix colors

from queueing_system import QueueingSystemPoisson
qs = QueueingSystemPoisson(1, 2, 2, 1)
print(qs)
qs.show()