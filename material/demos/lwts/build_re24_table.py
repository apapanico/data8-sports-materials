import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(2, str(ROOT / 'utils'))

import pandas as pd
pd.set_option('precision', 3)
import numpy as np
np.set_printoptions(precision=3)

# custom functions that will help do some simple tasks
from datascience_utils import *
from datascience_stats import *
from datascience_topic import fast_run_expectancy, most_common_lineup_position
col_names = [
    'INN_CT', 'BAT_HOME_ID', 'EVENT_CD', 'OUTS_CT', 'BAT_ID',
    'BAT_LINEUP_ID', 'BAT_EVENT_FL', 'START_BASES_CD', 'END_BASES_CD',
    'EVENT_OUTS_CT', 'EVENT_RUNS_CT', 'FATE_RUNS_CT', 'INN_NEW_FL']

retro = pd.read_csv(
    'retrosheet_events-2017.csv.gz', usecols=col_names)

new_col_names = [
    'Inning', 'Half-Inning', 'Outs', 'Batter_ID', 'Lineup_Order',
    'Event_Type', 'PA_Flag', 'Event_Outs', 'New_Inning', 'Start_Bases',
    'End_Bases', 'Event_Runs', 'Future_Runs']
retro.columns = new_col_names
base_runner_codes = {
    0: "None on",  # No one on
    1: "1st",  # runner on 1st
    2: "2nd",  # runner on 2nd
    3: "1st and 2nd",  # runners on 1st & 2nd
    4: "3rd",  # runner on 3rd
    5: "1st and 3rd",  # runners on 1st & 3rd
    6: "2nd and 3rd",  # runners on 2nd & 3rd
    7: "Bases Loaded"  # bases loaded
}
# Replace the numeric code with a string code
retro['Start_Bases'] = replace(retro, 'Start_Bases', base_runner_codes)
retro['End_Bases'] = replace(retro, 'End_Bases', base_runner_codes)

event_codes = {
    0: 'Unknown',
    1: 'None',
    2: 'Generic out',
    3: 'K',  # Strikeout
    4: 'SB',  # Stolen Base
    5: 'Defensive indifference',
    6: 'CS',  # Caught stealing
    7: 'Pickoff error',
    8: 'Pickoff',
    9: 'Wild pitch',
    10: 'Passed ball',
    11: 'Balk',
    12: 'Other advance/out advancing',
    13: 'Foul error',
    14: 'BB',  # Walk
    15: 'IBB',  # Intentional walk
    16: 'HBP',  # Hit by pitch
    17: 'Interference',
    18: 'RBOE',  # Reached base on error
    19: 'FC',  # Fielder's choice
    20: '1B',  # Single
    21: '2B',  # Double
    22: '3B',  # Triple
    23: 'HR',  # Home run
    24: 'Missing play',
}

# Replace numeric code with string
retro['Event_Type'] = replace(retro, 'Event_Type', event_codes)

retro = retro.loc[retro['PA_Flag'] == "T"]

retro['Runs_ROI'] = retro['Future_Runs'] + retro['Event_Runs']

retro_pre9 = retro.loc[retro['Inning'] < 9]
run_expectancy = retro_pre9.groupby(['Outs', 'Start_Bases'])['Runs_ROI'].\
    mean().\
    reset_index()
run_expectancy.columns = ['Outs', 'Start_Bases', 'RE']

retro = fast_run_expectancy(retro, run_expectancy)

retro['RE24'] = retro['Run_Expectancy_Next'] - \
    retro['Run_Expectancy'] + retro['Event_Runs']

retro.to_csv('retrosheet_with_re24-2017.csv.gz', index=False, compression='gzip')
