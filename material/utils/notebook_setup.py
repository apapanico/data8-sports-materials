import sys
from pathlib import Path

print('Adding datascience helper tools to path...')

ROOT = Path(__file__).parents[1]
sys.path.insert(2, str(ROOT / 'utils'))

print("Setting up Matplotlib...")

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.facecolor'] = (0.941, 0.941, 0.941, 1.0)

import pandas as pd
pd.set_option('precision', 3)
import numpy as np
np.set_printoptions(precision=3)

print("matplotlib imported as mpl")
print("matplotlib.pyplot imported as plt")
print("seaborn imported as sns")
