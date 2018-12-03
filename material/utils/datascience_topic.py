import warnings
warnings.filterwarnings("ignore")

import random as _random
import numpy as _np
import pandas as _pd
import scipy.stats as _stats
import networkx as _nx
import matplotlib.pyplot as _plt
from sklearn.linear_model import (LogisticRegression as _LogisticRegression,
                                  LogisticRegressionCV as _LogisticRegressionCV,
                                  Ridge as _Ridge, RidgeCV as _RidgeCV)
from scipy.sparse.linalg import eigs as _eigs

from datascience import Table as _Table

from matplotlib.lines import Line2D as _Line2D
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


# Hot Hand


def shuffle(s):
    ell = list(s)
    _random.shuffle(ell)
    return ''.join(ell)


def longest_streak(seq, match='1'):
    key = '0' if match == '1' else '1'
    return max(map(len, seq.split(key)))


def longest_two_streaks(seq, match='1'):
    key = '0' if match == '1' else '1'
    return sum(sorted(map(len, seq.split(key)))[-2:])


def num_matches(s, sub):
    n = len(sub)
    ct = 0
    for i in range(len(s) - n + 1):
        if s[i:i + n] == sub:
            ct += 1
    return ct


def count_conditional(shots, conditioning_set):
    base = f'{conditioning_set}'
    conditional_hits = num_matches(shots, base + '1')
    conditional_misses = num_matches(shots, base + '0')
    return conditional_hits, conditional_misses


def compute_t_k_hit(shots, k=2, return_ct=False):
    if len(shots) <= k:
        return 0, 0
    hits, misses = count_conditional(shots, '1' * k)
    k_hits = hits + misses
    if return_ct:
        out = (hits / k_hits if k_hits > 0 else None, k_hits)
        return out
    else:
        return hits / k_hits if k_hits > 0 else None


def compute_t_k_miss(shots, k=2, return_ct=False):
    if len(shots) <= k:
        return 0, 0
    hits, misses = count_conditional(shots, '0' * k)
    k_misses = hits + misses

    if return_ct:
        out = (hits / k_misses if k_misses > 0 else None, k_misses)
        return out
    else:
        return hits / k_misses if k_misses > 0 else None


def compute_t_k(shots, k=2):
    hits_after_k_hits, H_k = compute_t_k_hit(shots, k=k, return_ct=True)
    hits_after_k_misses, M_k = compute_t_k_miss(shots, k=k, return_ct=True)
    if H_k > 0 and M_k > 0:
        t_k_hit = hits_after_k_hits / H_k if H_k > 0 else 0
        t_k_miss = hits_after_k_misses / M_k if M_k > 0 else 0
        t_k = t_k_hit - t_k_miss
    else:
        t_k_hit = t_k_miss = t_k = None
    return t_k


def compute_t_2(shots):
    return compute_t_k(shots, k=2)


def compute_t_3(shots):
    return compute_t_k(shots, k=3)


def coin_flips(n_flips, prob_heads):
    return _np.random.binomial(n_flips, prob_heads)


# RTTM


class CoinFlipper(object):

    def __init__(self, expected_heads=5, expected_tails=5, p=None):
        self.a0 = self.a = expected_heads
        self.b0 = self.b = expected_tails
        self.p = p or _np.random.beta(expected_heads, expected_tails)
        self.tosses = []
        self.recent_tosses = None

    def toss_coin(self, n=1):
        if n == 1:
            n = None
        self.recent_tosses = result = _np.random.choice(
            2, size=n, p=[1 - self.p, self.p])
        if n is None:
            self.tosses.append(result)
        else:
            self.tosses.extend(result)

    def report(self, report_belief=False):
        if self.tosses:
            n_recent = len(self.recent_tosses)
            k_recent = _np.sum(self.recent_tosses)
            print(f"Just saw {k_recent} Heads out of {n_recent} tosses")
            n = len(self.tosses)
            k = _np.sum(self.tosses)
            print(f"Overall seen {k} Heads out of {n} tosses")
            print(f"Proportion of heads: {k / n:.3f}")
            if report_belief:
                a, b = self.a, self.b
                s = f"H_old / (H_old + T_old) = {a} / ({a} + {b})"
                print(f"Previous Belief: {s} = {self.belief:.3f}")
                self.update_belief()
                s = f"H_new / (H_new + T_new) = {a} / ({a} + {b})"
                print(f"Updated Belief: {s} = {self.belief:.3f}")
        else:
            print(f"Have not tossed the coin yet")
            if report_belief:
                a, b = self.a, self.b
                s = f"H / (H + T) = {a} / ({a} + {b})"
                print(f"Current Belief: {s} = {self.belief:.3f}")

    def p_guess(self):
        return _np.mean(_np.array(self.tosses))

    def update_belief(self):
        self.a = self.a0 + int(_np.sum(_np.array(self.tosses)))
        self.b = self.b0 + int(_np.sum(1 - _np.array(self.tosses)))

    @property
    def belief(self):
        return self.a / (self.a + self.b)

    @property
    def num_tosses(self):
        return len(self.tosses)

    def plot_tosses(self, ax=None, figsize=(8, 8)):
        if ax is None:
            fig, ax = _plt.subplots(figsize=figsize)
        N = len(self.tosses)
        k = _np.sum(self.tosses)

        ax.bar([0, 1], [N - k, k], 1., align='center')
        ax.set_xticks([0, 1])
        ax.set_ylim(0, max(max([N - k, k]), 35))
        ax.set_xticklabels(['Tails', 'Heads'])
        if N == 0:
            ax.set_title(f'Prop. of Heads: N/A', fontsize=16)
        else:
            p = k / N
            ax.set_title(f'Prop. of Heads: {p:.3f}', fontsize=16)

    def plot_updated_belief(self, ax=None, figsize=(8, 8)):
        if ax is None:
            fig, ax = _plt.subplots(figsize=figsize)
        self.update_belief()
        t = _np.linspace(0, 1, 500)
        _plt.plot(t, _stats.beta.pdf(t, self.a, self.b))
        ax.set_title('Updated Belief')
        ax.set_xlim(0, 1)
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])

    def plots(self, figsize=(16, 8)):
        fig, ax = _plt.subplots(ncols=2, figsize=figsize)
        self.plot_tosses(ax=ax[0])
        self.plot_updated_belief(ax=ax[1])


def merge(t1, t2, on, how='outer', fillna=True):
    DS_FLAG = False
    if isinstance(t1, _Table):
        t1 = t1.to_df()
        DS_FLAG = True
    if isinstance(t2, _Table):
        t2 = t2.to_df()
    full_t = _pd.merge(t1, t2, how=how, left_on=on, right_on=on)
    if fillna:
        full_t.fillna(0, inplace=True)
    if DS_FLAG:
        return _Table.from_df(full_t)
    else:
        return full_t


def fit_beta_belief(data):
    m = _np.mean(data)
    v = _np.var(data)
    alpha = ((1 - m) / v - 1 / m) * m**2
    beta = alpha * (1 / m - 1)
    return alpha, beta


def plot_beta_belief(a, b):
    fig, ax = _plt.subplots()
    t = _np.linspace(0, 1, 500)
    _plt.plot(t, _stats.beta.pdf(t, a, b), 'C1')
    ax.set_title('Beta Belief')
    ax.set_xlim(0, 1)
    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])


def plot_beta_belief_and_batting_avg(a, b, batter_data):
    batter_data.hist('BA', bins=50, density=True)
    ax = _plt.gca()
    t = _np.linspace(0, 1, 500)
    ax.plot(t, _stats.beta.pdf(t, a, b), 'C1')


# Ranking

def elo_prediction(elo_rank_A, elo_rank_B, base=10, scale=400):
    q_A = base**(elo_rank_A / scale)
    q_B = base**(elo_rank_B / scale)
    e_A = q_A / (q_A + q_B)
    return e_A


def elo_update(rank_old, score, expected_score, k):
    return rank_old + k * (score - expected_score)


def all_teams_sorted(games):
    return sorted(set(games['Away Team']).union(games['Home Team']))


def elo_rank(games, initial_rating=1500, K=32, proportional_win=False):
    all_teams = all_teams_sorted(games)

    ranking = {
        team: initial_rating
        for team in all_teams
    }

    for _, row in games.iterrows():
        home_team = row['Home Team']
        home_elo_rank = ranking[home_team]
        home_score = row['Home Score']

        away_team = row['Away Team']
        away_elo_rank = ranking[away_team]
        away_score = row['Away Score']

        if proportional_win:
            home_win = home_score / (home_score + away_score)
        else:
            home_win = int(home_score > away_score)
        away_win = 1 - home_win

        home_elo_expected = elo_prediction(home_elo_rank, away_elo_rank)
        ranking[home_team] = elo_update(
            home_elo_rank, home_win, home_elo_expected, K)

        away_elo_expected = 1 - home_elo_expected
        ranking[away_team] = elo_update(
            away_elo_rank, away_win, away_elo_expected, K)

    return _pd.Series(ranking)


######################################################################
# DataFrame and Graph Building
######################################################################

def parse_score(score):
    # score_diff = score['Home Score'] - score['Away Score'] - .5
    # max_score = max(score['Home Score'], score['Away Score'])
    # margin = score_diff / max_score

    win_score = max(score['Home Score'], score['Away Score'])
    lose_score = min(score['Home Score'], score['Away Score'])
    total_score = score['Home Score'] + score['Away Score']
    if win_score == score['Home Score']:
        bunch = (score['Away Team'], score['Home Team'])
    else:
        bunch = (score['Home Team'], score['Away Team'])

    wts = {
        'win_score': win_score,
        'lose_score': lose_score,
        'total_score': total_score
    }
    return bunch, wts


def scores_to_network(df, divisions=None, drop=None):
    DG = _nx.MultiDiGraph()

    all_schools = sorted(
        set(df['Away Team']).union(df['Home Team'])
    )
    DG.add_nodes_from(all_schools)

    for _, score in df.iterrows():
        bunch, wts = parse_score(score)
        edge = bunch + (wts,)
        DG.add_weighted_edges_from([edge])

    if divisions is not None:
        team_list = []
        for div in divisions:
            team_list.extend(divisions[div])
        DG = DG.subgraph(team_list)

    if drop is not None:
        DG.remove_nodes_from(drop)
    return DG


def win_pct_rank(df):
    awaywins = df.groupby('Away Team').apply(
        lambda x: _np.sum(x['Away Score'] > x['Home Score']))
    awaygames = df.groupby('Away Team')['Away Score'].count()
    homewins = df.groupby('Home Team').apply(
        lambda x: _np.sum(x['Away Score'] < x['Home Score']))
    homegames = df.groupby('Home Team')['Home Score'].count()

    teams = set(awaywins.index).union(homewins.index)
    wins = _pd.Series({
        team: awaywins.get(team, 0) + homewins.get(team, 0)
        for team in teams
    })

    games = _pd.Series({
        team: awaygames.get(team, 0) + homegames.get(team, 0)
        for team in teams
    })
    win_pct = (wins / games)
    return win_pct


def draw_graph(graph, divisions=None, figsize=(8, 6)):
    colors = {}
    if divisions is not None:
        legend_elements = []
        for i, div in enumerate(divisions):
            div_teams = divisions[div]
            colors.update({k: f'C{i}' for k in div_teams})
            legend_elements.append(
                _Line2D([0], [0], marker='o', color=f'C{i}', label=div,
                        markerfacecolor=f'C{i}', markersize=4),
            )
        node_color = [colors[t] for t in graph.nodes()]
    else:
        node_color = ['C0' for _ in graph.nodes()]
        legend_elements = []

    fig, ax = _plt.subplots(figsize=figsize)
    pos = _nx.spectral_layout(graph, weight=None)
    _nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=4,
        cmap=_plt.cm.Blues,
        node_color=node_color,
        width=.1,
        ax=ax
    )
    ax.legend(handles=legend_elements)


######################################################################
# Random Walkers
######################################################################

class RandomWalker(object):

    def __init__(self, graph):
        self.graph = graph
        self.teams = teams = list(graph.nodes)
        self.ranking = {
            team: 0
            for team in teams
        }
        self.total_steps = 1
        self.current_team = _np.random.choice(teams)
        self.ranking[self.current_team] += 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = self.print_position()
        s += "\n"
        s += self.print_ranking()
        return s

    def sorted_ranking(self):
        ranked_teams = sorted(
            self.ranking, key=lambda k: self.ranking[k], reverse=True)
        return ranked_teams

    def print_position(self):
        s = f"Current team: {self.current_team}\n"
        s += f"Current total steps: {self.total_steps}"
        return s

    def print_ranking(self):
        s = "Current Ranking:\n"
        ranked_teams = self.sorted_ranking()
        for team in ranked_teams:
            if not _np.isinf(self.total_steps):
                r = self.ranking[team] / self.total_steps
            else:
                r = self.ranking[team]
            s += f"  {team}: {r:.3f}\n"
        return s

    def _update_state(self, new_team):
        self.current_team = new_team
        self.ranking[new_team] += 1
        self.total_steps += 1

    def current_teams_beat(self):
        in_edges = self.graph.in_edges(self.current_team)
        teams_beat = [
            team_beat
            for team_beat, _ in in_edges
        ]
        return teams_beat

    def current_teams_beaten_by(self):
        out_edges = self.graph.out_edges(self.current_team)
        teams_beaten_by = [
            team_beaten_by
            for _, team_beaten_by in out_edges
        ]
        return teams_beaten_by

    # def current_teams_beat(self):
    #     return list(self.graph.predecessors(self.current_team))

    # def current_teams_beaten_by(self):
    #     return list(self.graph.successors(self.current_team))


class PageRankWalker(RandomWalker):

    def __init__(self, graph, jump_probability=.15):
        super(PageRankWalker, self).__init__(graph)
        self.jump_probability = jump_probability

    def _walk_one_step(self, verbose=False):
        teams_beaten_by = self.current_teams_beaten_by()

        if (len(teams_beaten_by) == 0):
            jump = True
        else:
            p = self.jump_probability
            jump = _np.random.choice([True, False], p=[p, 1 - p])

        if jump:
            choices = list(self.graph.nodes)
            if verbose:
                print("JUMPING TO RANDOM TEAM")
        else:
            choices = teams_beaten_by
            if verbose:
                print(f"POTENTIAL TEAMS:  {', '.join(choices)}")

        team = _np.random.choice(choices)
        if verbose:
            print(f"NEW TEAM: {team}\n")
        return team, jump

    def walk(self, num_steps=1):
        if _np.isinf(self.total_steps):
            print("Walker has reached infinite steps...")
        else:
            print(self)
            if num_steps == 1:
                team, _ = self._walk_one_step(verbose=True)
                self._update_state(team)
            elif _np.isinf(num_steps):
                r = page_rank(
                    self.graph, jump_probability=self.jump_probability)
                self.ranking = r.to_dict()
                self.total_steps = _np.inf
                self.current_team = None
            else:
                print(f"TAKING {num_steps} STEPS...\n")
                jump_ct = 0
                for _ in range(num_steps):
                    team, jump = self._walk_one_step(verbose=False)
                    if jump:
                        jump_ct += 1
                    self._update_state(team)

                print(f"  TOTAL RANDOM JUMPS: {jump_ct}\n")

        print(self)


class MonkeyWalker(RandomWalker):

    def __init__(self, graph, winner_probability=.85):
        super(MonkeyWalker, self).__init__(graph)
        self.winner_probability = winner_probability

    def _walk_one_step(self, verbose=False):
        teams_beat = self.current_teams_beat()
        teams_beaten_by = self.current_teams_beaten_by()
        all_teams_played = teams_beat + teams_beaten_by

        new_team = _np.random.choice(all_teams_played)
        if verbose:
            print(f"POTENTIAL TEAMS:  {', '.join(all_teams_played)}")
            print(f"POTENTIAL NEW TEAM:  {new_team}")

        p = self.winner_probability
        potential_team = _np.random.choice(all_teams_played)
        if potential_team in teams_beaten_by:
            teams = [potential_team, self.current_team]
        else:
            teams = [self.current_team, potential_team]
        team = _np.random.choice(teams, p=[p, 1 - p])

        if verbose:
            if potential_team == team:
                print("JUMPING TO NEW TEAM")
                print(f"NEW TEAM: {new_team}\n")
            else:
                print("STAYING AT CURRENT TEAM")
        return team

    def walk(self, num_steps=1):
        if _np.isinf(self.total_steps):
            print("Walker has reached infinite steps...")
        else:
            print(self)
            if num_steps == 1:
                team = self._walk_one_step(verbose=True)
                self._update_state(team)
            elif _np.isinf(num_steps):
                r = monkey_rank(
                    self.graph, winner_probability=self.winner_probability)
                self.ranking = r.to_dict()
                self.total_steps = _np.inf
                self.current_team = None
            else:
                print(f"TAKING {num_steps} STEPS...\n")
                for _ in range(num_steps):
                    team = self._walk_one_step(verbose=False)
                    self._update_state(team)

        print(self)


class KeenerWalker(RandomWalker):

    def __init__(self, graph):
        super(KeenerWalker, self).__init__(graph)

    def _walk_one_step(self, verbose=False):
        teams_beat = self.current_teams_beat()
        teams_beaten_by = self.current_teams_beaten_by()
        all_teams_played = list(set(teams_beat + teams_beaten_by))

        potential_team = _np.random.choice(all_teams_played)
        if verbose:
            print(f"POTENTIAL TEAMS:  {', '.join(all_teams_played)}")
            print(f"POTENTIAL NEW TEAM:  {potential_team}")

        out_edge_data = self.graph.get_edge_data(
            self.current_team, potential_team)
        if out_edge_data is not None:
            out_edge_data = list(out_edge_data.values())
        else:
            out_edge_data = []

        in_edge_data = self.graph.get_edge_data(
            potential_team, self.current_team)
        if in_edge_data is not None:
            in_edge_data = list(in_edge_data.values())
        else:
            in_edge_data = []

        team_score = 0
        total_score = 0
        for edge in out_edge_data:
            edge_wt = edge['weight']
            team_score += edge_wt['lose_score'] + 1
            total_score += edge_wt['total_score'] + 2

        for edge in in_edge_data:
            edge_wt = edge['weight']
            team_score += edge_wt['win_score'] + 1
            total_score += edge_wt['total_score'] + 2

        stay_prob = team_score / total_score

        if verbose:
            print(out_edge_data)
            print(in_edge_data)
            print(stay_prob)

        jump = _np.random.choice([False, True], p=[stay_prob, 1 - stay_prob])

        if jump:
            if verbose:
                print("JUMPING TO NEW TEAM")
                print(f"NEW TEAM: {potential_team}\n")
            team = potential_team
        else:
            if verbose:
                print("STAYING AT CURRENT TEAM")
            team = self.current_team

        return team

    def walk(self, num_steps=1):
        if _np.isinf(self.total_steps):
            print("Walker has reached infinite steps...")
        else:
            print(self)
            if num_steps == 1:
                team = self._walk_one_step(verbose=True)
                self._update_state(team)
            elif _np.isinf(num_steps):
                r = keener_rank(self.graph, walker_model=True)
                self.ranking = r.to_dict()
                self.total_steps = _np.inf
                self.current_team = None
            else:
                print(f"TAKING {num_steps} STEPS...\n")
                for _ in range(num_steps):
                    team = self._walk_one_step(verbose=False)
                    self._update_state(team)

        print(self)

######################################################################
# Graph Rankers
######################################################################


def page_rank(graph, jump_probability=.15, weighted=False):
    if weighted:
        wt = 'weight'
    else:
        wt = None
    alpha = 1 - jump_probability
    M = _nx.google_matrix(graph, alpha, weight=wt)
    _, v = _eigs(M.T, k=1)

    r = v.flatten().real
    r /= r.sum()

    return _pd.Series(r, index=graph.nodes).sort_index()


def monkey_model(graph, alpha):
    all_teams = sorted(set(graph.nodes))
    n_teams = len(all_teams)

    index = dict(zip(all_teams, range(n_teams)))

    A = _np.full((n_teams, n_teams), 0., order=None)
    B = _np.full((n_teams, n_teams), 0., order=None)

    for u, v, attrs in graph.edges(data=True):
        i, j = index[u], index[v]

        A[i, j] += alpha
        A[j, i] += 1 - alpha

        B[i, j] += 1
        B[j, i] += 1

    N = B.sum(axis=0).reshape(-1, 1)
    C = A / N
    C = C + _np.diag(1 - C.sum(axis=1))
    return _pd.DataFrame(C, index=all_teams, columns=all_teams)


def monkey_rank(graph, winner_probability=.85):
    M = monkey_model(graph, winner_probability)
    _, v = _eigs(M.values.T, k=1)

    r = v.flatten().real
    r /= r.sum()

    return _pd.Series(r, index=M.index)


def keener_model(graph, walker_model=False):
    all_teams = sorted(set(graph.nodes))
    n_teams = len(all_teams)
    A = _np.zeros((n_teams, n_teams))
    B = _np.zeros((n_teams, n_teams))

    for lose_team, win_team, ix in graph.edges:
        edge = graph.get_edge_data(lose_team, win_team, ix)

        i = all_teams.index(lose_team)
        j = all_teams.index(win_team)

        A[i, j] += edge['weight']['lose_score'] + 1
        A[j, i] += edge['weight']['win_score'] + 1

        B[i, j] += edge['weight']['total_score'] + 2
        B[j, i] += edge['weight']['total_score'] + 2

    mask = (B != 0)
    M = A.copy()
    M[mask] = A[mask] / B[mask]

    N_opp = _np.count_nonzero(B, axis=1).reshape(-1, 1)

    if walker_model:
        M = M.T / N_opp
        M = M + _np.diag(1 - M.sum(axis=1))
    else:
        M /= N_opp

    return _pd.DataFrame(M, index=all_teams, columns=all_teams)


def keener_rank(graph, walker_model=True):
    M = keener_model(graph, walker_model=walker_model)
    if walker_model:
        lam, v = _eigs(M.values.T, k=1)
    else:
        lam, v = _eigs(M.values, k=1)

    r = v.flatten().real
    r /= r.sum()

    return _pd.Series(r, index=M.index)

######################################################################
# Matrix Rankers
######################################################################


def colley_matrix_equation(games):
    all_teams = all_teams_sorted(games)
    n_teams = len(all_teams)
    C = 2 * _np.eye(n_teams)
    b = _np.ones(n_teams)

    for _, row in games.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        i = all_teams.index(home_team)
        j = all_teams.index(away_team)

        C[i, i] += 1
        C[j, j] += 1
        C[i, j] -= 1
        C[j, i] -= 1

        home_score = row['Home Score']
        away_score = row['Away Score']

        if home_score - away_score > 0:
            b[i] += .5
            b[j] -= .5
        else:
            b[i] -= .5
            b[j] += .5

    C = _pd.DataFrame(C, index=all_teams, columns=all_teams)
    b = _pd.Series(b, index=all_teams)
    return C, b


def colley_rank(games):
    C, b = colley_matrix_equation(games)
    r = _np.linalg.solve(C, b)
    return _pd.Series(r, index=b.index)


def massey_matrix_equation(games):
    all_teams = all_teams_sorted(games)
    n_teams = len(all_teams)
    n_games = games.shape[0]
    M = _np.zeros((n_games, n_teams))
    y = _np.zeros(n_games)

    for i, (_, row) in enumerate(games.iterrows()):
        home_team = row['Home Team']
        j = all_teams.index(home_team)
        M[i, j] = 1

        away_team = row['Away Team']
        j = all_teams.index(away_team)
        M[i, j] = -1

        home_score = row['Home Score']
        away_score = row['Away Score']

        y[i] = home_score - away_score

    M = _pd.DataFrame(M, columns=all_teams)
    y = _pd.Series(y)
    return M, y


# def massey_rank(games):
#     M, y = massey_matrix_equation(games)

#     n_teams = M.shape[1]
#     M_tmp = _np.vstack([M, _np.ones(n_teams)])
#     hfa = _np.ones((M_tmp.shape[0], 1))
#     hfa[-1] = 0
#     M_tmp = _np.hstack([hfa, M_tmp])

#     y_tmp = _np.append(y, 0)
#     r, _, _, _ = _np.linalg.lstsq(M_tmp, y_tmp, rcond=None)
#     r = r[1:]

#     return _pd.Series(r, index=M.columns)


def massey_rank(games, shrinkage=1e-8):
    M, y = massey_matrix_equation(games)
    model = _Ridge(alpha=shrinkage, fit_intercept=True)
    model.fit(M, y)
    r = model.coef_.ravel()
    return _pd.Series(r, index=M.columns)


def massey_penalized_rank(games, shrinkages=_np.logspace(-2, 1)):
    M, y = massey_matrix_equation(games)
    model = _RidgeCV(alphas=shrinkages, fit_intercept=True)
    model.fit(M, y)
    print(model.alpha_)
    r = model.coef_.ravel()
    return _pd.Series(r, index=M.columns)


def bradleyterry_logistic_model(games):
    all_teams = all_teams_sorted(games)
    n_teams = len(all_teams)
    n_games = games.shape[0]
    M = _np.zeros((n_games, n_teams))
    y = _np.zeros(n_games)

    for i, (_, row) in enumerate(games.iterrows()):
        home_team = row['Home Team']
        j = all_teams.index(home_team)
        M[i, j] = 1

        away_team = row['Away Team']
        j = all_teams.index(away_team)
        M[i, j] = -1

        home_score = row['Home Score']
        away_score = row['Away Score']

        y[i] = int(home_score > away_score)

    M = _pd.DataFrame(M, columns=all_teams)
    y = _pd.Series(y)
    return M, y


def bradleyterry_rank(games, shrinkage=1e-8):
    M, y = bradleyterry_logistic_model(games)
    C = 1. / shrinkage
    model = _LogisticRegression(C=C, fit_intercept=True)
    model.fit(M, y)
    r = model.coef_.ravel()
    return _pd.Series(r, index=M.columns)


def bradleyterry_penalized_rank(games, Cs=_np.logspace(0, 3)):
    M, y = bradleyterry_logistic_model(games)
    model = _LogisticRegressionCV(Cs=Cs, fit_intercept=True)
    model.fit(M, y)
    r = model.coef_.ravel()
    return _pd.Series(r, index=M.columns)
