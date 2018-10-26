import warnings
warnings.filterwarnings("ignore")

from datascience import Table as _Table
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from matplotlib.ticker import MultipleLocator as _MultipleLocator
from matplotlib.collections import PolyCollection as _PC
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from matplotlib.cm import register_cmap as _register_cmap
from IPython.display import display as _display

cmap = _plt.get_cmap('Dark2')
cmap_4thdownbot = _LinearSegmentedColormap.from_list(
    '4thdownbot_cmap', cmap.colors[:3], N=3)
_register_cmap(name='4thdownbot_cmap', cmap=cmap_4thdownbot)


def display_re_matrix(re):
    _display(re.unstack(level=0).sort_values(by=0))


def display_weights(w):
    _display(w.to_frame().transpose())


def fast_run_expectancy(retro, re):
    TABLE_FLAG = False
    if isinstance(retro, _Table):
        TABLE_FLAG = True
        retro = retro.to_df()
        re = re.to_df()

    re = re.set_index(['Outs', 'Start_Bases'])

    # Build current out-runner states
    idx = list(zip(retro['Outs'], retro['Start_Bases']))
    # Extract run potentials
    retro['Run_Expectancy'] = re.loc[idx].values

    next_outs = retro['Outs'] + retro['Event_Outs']
    # Build next out-runner states
    idx = list(zip(next_outs, retro['End_Bases']))
    # Extract run potentials
    retro['Run_Expectancy_Next'] = re.loc[idx].values

    # When the inning ends, there are 3 outs.  That is not in the run
    # expectancy matrix so inning ending plate appearances will have an NA
    # value here.  We fill those with 0.
    retro['Run_Expectancy_Next'].fillna(0, inplace=True)

    return _Table.from_df(retro) if TABLE_FLAG else retro


def most_common_lineup_position(retro):
    TABLE_FLAG = False
    if isinstance(retro, _Table):
        TABLE_FLAG = True
        retro = retro.to_df()

    # Order of operations:
    # 1. Get PA counts
    # 2. Turn Lineup_Order into a column
    # 3. Rename column to PA
    # 4. Sort on PA in descending order
    lineup_pos = retro.groupby(['Batter_ID', 'Lineup_Order'])['Inning'].\
        count().\
        reset_index(level='Lineup_Order').\
        rename(columns={'Inning': 'PA'}).\
        sort_values('PA', ascending=False)

    # Duplicates indicate other positions.  By keeping first, we keep the most
    # common due to the sorting
    most_common = ~lineup_pos.index.duplicated(keep='first')
    lineup_pos = lineup_pos.loc[most_common, ['Lineup_Order']].sort_index()

    if TABLE_FLAG:
        return _Table.from_df(lineup_pos.reset_index())
    else:
        return lineup_pos


def build_expected_value_from_hexbin(x, y, hexbin_plot):
    pc = hexbin_plot.get_children()[0]
    expected_vals = pc.get_array()
    hexbin_locs = pc.get_offsets()

    def dist(x, axis=0):
        return _np.sqrt((x**2).sum(axis=axis))

    def closest_bin(loc, bin_locs):
        return dist(bin_locs - loc, axis=1).argmin()

    hex_dist = _np.sort(dist(hexbin_locs - hexbin_locs[0], axis=1))[1]
    locs = _np.vstack([x, y]).T

    nearest_bins = []
    for loc in locs:
        d = dist(hexbin_locs - loc, axis=1).min()
        if d <= hex_dist:
            nearest_bins.append(closest_bin(loc, hexbin_locs))
        else:
            nearest_bins.append(_np.nan)
    nearest_bins = _np.array(nearest_bins)

    nan_mask = ~_np.isnan(nearest_bins)
    masked_nearest_bins = nearest_bins[nan_mask].astype(int)
    ev = _np.full(nan_mask.shape, _np.nan)
    ev[nan_mask] = expected_vals[masked_nearest_bins]
    return ev


def _ev_goforit(yrdline100, ydstogo, region, ekv, epv_model, conv_pct_model):
    # Value of failing (approximately turning over at same spot)
    conv_fail_yrdline100 = 100 - yrdline100
    conv_fail_epv = -epv_model[conv_fail_yrdline100]

    # Value of converting (approximately at the first down marker)
    if yrdline100 == ydstogo:
        conv_succ_epv = 7 - ekv
    else:
        first_down_yrdline100 = yrdline100 - ydstogo
        conv_succ_epv = epv_model[first_down_yrdline100]

    # Conversion Pct
    exp_conv_pct = conv_pct_model[region][ydstogo]
    # Overall expected value of going for it
    go_ev = exp_conv_pct * conv_succ_epv + \
        (1 - exp_conv_pct) * conv_fail_epv
    return go_ev


def _ev_punt(yrdline100, epv_model, punt_dist_model):
    # Expected next yardline
    # Model restriction
    if isinstance(punt_dist_model, _pd.Series):
        PUNT_LIM = punt_dist_model.index.min()
    else:
        PUNT_LIM = min(punt_dist_model.keys())

    if yrdline100 >= PUNT_LIM:
        exp_net_punt_dist = punt_dist_model[yrdline100]
        exp_yrdline100 = 100 - yrdline100 + exp_net_punt_dist

        # Overall expected value of punting
        punt_ev = -epv_model[exp_yrdline100]
    else:
        punt_ev = None
    return punt_ev


def _ev_fg(yrdline100, epv_model, ekv, fg_prob_model):
    # Field goal placement distance (not including the endzone)
    FG_OFFSET = 7
    # Model restriction
    if isinstance(fg_prob_model, _pd.Series):
        FG_LIM = fg_prob_model.index.max()
    else:
        FG_LIM = max(fg_prob_model.keys())

    kick_yrdline100 = yrdline100 + FG_OFFSET
    # Compute FG distance
    fg_dist = kick_yrdline100 + 10
    if fg_dist <= FG_LIM:
        # Probability of success
        exp_fg_prob = fg_prob_model[fg_dist]

        # Expected value of field success
        fg_succ_epv = 3 - ekv

        # EPV of field goal fail
        fg_fail_yrdline100 = 100 - kick_yrdline100
        fg_fail_epv = -epv_model[fg_fail_yrdline100]

        # Overall expected value kicking
        fg_ev = fg_succ_epv * exp_fg_prob + \
            fg_fail_epv * (1 - exp_fg_prob)
    else:
        fg_ev = None
    return fg_ev


def _decision_maker(yrdline100, ydstogo, ekv, epv_model, conv_pct_model,
                    punt_dist_model, fg_prob_model, print_message=False):
    YRDSTOGO_CAP = 10   # Model restriction
    if ydstogo >= YRDSTOGO_CAP:
        raise ValueError(f"Model not valid for ydstogo > {ydstogo}")

    if yrdline100 < 10:
        region = 'Inside10'
    elif yrdline100 < 20:
        region = '10to20'
    else:
        region = 'Beyond20'

    # 1. Expected value of going for it
    go_ev = _ev_goforit(yrdline100, ydstogo, region,
                        ekv, epv_model, conv_pct_model)
    # 2. Expected value of punting
    punt_ev = _ev_punt(yrdline100, epv_model, punt_dist_model)
    # 3. Expected value of kicking a field goal
    fg_ev = _ev_fg(yrdline100, epv_model, ekv, fg_prob_model)

    choices = list(filter(
        lambda choice: choice[-1] is not None,
        [('go for it', go_ev), ('punt', punt_ev), ('kick', fg_ev)]
    ))
    choices = sorted(choices, key=lambda choice: choice[1])
    decision, ev = choices[-1]
    second_choice, ev_improvement = choices[-2][0], ev - choices[-2][1]

    if print_message:
        print("Expected Values")
        print(f"Go for it: {go_ev:.2f}")
        if punt_ev is not None:
            print(f"Punt: {punt_ev:.2f}")
        else:
            print(f"Punt: TOO CLOSE TO PUNT")
        if fg_ev is not None:
            print(f"FG: {fg_ev:.2f}")
        else:
            print("FG: TOO FAR TO KICK")

        print()
        print(f"Coach, you should {decision.upper()}!")
        print()

    out = {
        'decision': decision,
        'ev': ev,
        'second_choice': second_choice,
        'ev_over_second': ev_improvement
    }
    return out


def compute_4thdownbot_data(ekv, epv_model, conv_pct_model,
                            punt_dist_model, fg_prob_model):
    yrdlines = list(range(1, 100))
    down_dist = list(range(1, 10))

    out = {}
    for yrdstogo in down_dist:
        tmp = {}
        out[yrdstogo] = tmp
        for yrdline in yrdlines:
            # Exclude impossible scenarios
            if (yrdline >= yrdstogo) and (100 - yrdline + yrdstogo >= 10):
                result = _decision_maker(
                    yrdline, yrdstogo, ekv, epv_model, conv_pct_model,
                    punt_dist_model, fg_prob_model)
            else:
                result = None
            tmp[yrdline] = result
    return out


def _prettify_4thdownbot(ax):
    ax.set_xlim(-.25, 100.25)
    ax.set_xticks(
        _np.array([-.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.5]) + .5)
    ax.set_xticklabels(['Your\nGoal', '10', '20', '30', '40',
                        '50', '40', '30', '20', '10', 'Opp\nGoal'])
    ax.tick_params(axis='both', which='major', labelsize=15)
    minorLocator = _MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.grid(b=True, which='minor', axis='x')
    ax.grid(b=False, which='major', axis='x')

    ax.set_ylim(-0.54, 8.54)
    ax.set_yticks([
        yd - .5
        for yd in range(1, 10)
    ])
    ax.set_yticks([
        yd - 1
        for yd in range(1, 10)
    ], minor=True)
    ax.set_yticklabels([
        '4th and {}'.format(yd)
        for yd in range(9, 0, -1)
    ], minor=True)
    _plt.setp(ax.get_yticklabels(), visible=False)
    _plt.setp(ax.get_yticklabels(minor=True), visible=True)


def fig_4thdown(figsize=(12, 9)):
    return _plt.subplots(figsize=figsize)


def _plot_4thdown_choice(choices, annotate=True, colorbar=False):
    data = _np.full((9, 99), _np.nan)
    data = _np.where(choices == 'go for it', 1, data)
    data = _np.where(choices == 'punt', 2, data)
    data = _np.where(choices == 'kick', 3, data)

    fig, ax = fig_4thdown()
    cax = ax.imshow(data, alpha=.6, cmap='4thdownbot_cmap', aspect='auto',
                    origin='lower', extent=(1, 100, -.5, 8.5))
    if colorbar:
        cbar = fig.colorbar(cax, ticks=[4 / 3, 2, 8 / 3])
        cbar.ax.set_yticklabels(['Go for it', 'Punt', 'Kick'], fontsize=15)
    if annotate:
        ax.text(50, 6, 'Go for it', size=25)
        ax.text(20, 1.5, 'Punt', size=25)
        ax.text(75, 1.5, 'Kick', size=25)

    _prettify_4thdownbot(ax)


def _extract_data_key(data, k):
    arr = []
    for yrdstogo in data:
        tmp = []
        arr.append(tmp)
        for yrdline in data[yrdstogo]:
            if data[yrdstogo][yrdline] is not None:
                tmp.append(data[yrdstogo][yrdline][k])
            else:
                tmp.append(_np.nan)
    arr = _np.array(arr)[::-1, ::-1]
    return arr


def plot_4thdown_decision(data):
    k = 'decision'
    arr = _extract_data_key(data, k)
    _plot_4thdown_choice(arr, annotate=True, colorbar=False)


def plot_4thdown_second_choices(data):
    k = 'second_choice'
    arr = _extract_data_key(data, k)
    _plot_4thdown_choice(arr, annotate=False, colorbar=True)


def plot_4thdown_evs(data):
    k = 'ev'
    arr = _extract_data_key(data, k)
    fig, ax = fig_4thdown()
    img = ax.imshow(arr, alpha=.8, aspect='auto', interpolation='nearest',
                    origin='lower', extent=(1, 100, -.5, 8.5))
    _plt.colorbar(img)
    _prettify_4thdownbot(ax)


def plot_4thdown_ev_over_second(data):
    k = 'ev_over_second'
    arr = _extract_data_key(data, k)
    fig, ax = fig_4thdown()
    img = ax.imshow(arr, alpha=.8, aspect='auto', interpolation='nearest',
                    origin='lower', extent=(1, 100, -.5, 8.5))
    _plt.colorbar(img)
    _prettify_4thdownbot(ax)
