import warnings
warnings.filterwarnings("ignore")

import numpy as _np
import pandas as _pd

from scipy.stats import pearsonr as _pearsonr
from scipy.sparse import csr_matrix as _csr_matrix
import statsmodels.formula.api as _smf
import statsmodels.api as _sm
from sklearn import linear_model as _linear_model


from datascience import Table as _Table


def correlation(x, y):
    rho = _pearsonr(x, y)[0]
    if _np.isnan(rho):
        rho = 0.
    return rho


def correlation_matrix(data):
    if isinstance(data, _Table):
        data = data.to_df()
    return data.corr()


def linear_fit(x, y, constant=True):
    if constant:
        x = _sm.add_constant(x)
    fit = _sm.OLS(y, x).fit()
    out = (fit.params, fit.fittedvalues, fit.resid)
    return out


def curve_fit(x, y, smoothness=.5):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    results = lowess(y, x, is_sorted=True, frac=smoothness)
    return results[:, 1]


def multiple_regression(dep_var, ind_vars, data, constant=False,
                        interactions=None):

    DS_FLAG = False
    if isinstance(data, _Table):
        DS_FLAG = True
        data = data.to_df()
    if isinstance(ind_vars, tuple):
        ind_vars = list(ind_vars)
    if not isinstance(ind_vars, list):
        ind_vars = [ind_vars]
    formula = dep_var + '~' + (' + '.join(ind_vars))
    if constant:
        formula += ' + 1'
    else:
        formula += ' + 0'
    if interactions is not None:
        for (interact_1, interact_2) in interactions:
            formula += f' + {interact_1}:{interact_2}'
    results = _smf.ols(formula, data=data).fit()
    if DS_FLAG:
        params = results.params.to_dict()
        fittedvalues = results.fittedvalues.values
        resid = results.resid.values
    else:
        params = results.params
        fittedvalues = results.fittedvalues
        resid = results.resid
    return params, fittedvalues, resid


def _sdf_to_csr(sdf, dtype=_np.float64):
    cols, rows, datas = [], [], []
    for col, name in enumerate(sdf):
        s = sdf[name]
        row = s.sp_index.to_int_index().indices
        cols.append(_np.repeat(col, len(row)))
        rows.append(row)
        datas.append(s.sp_values.astype(dtype, copy=False))

    cols = _np.concatenate(cols)
    rows = _np.concatenate(rows)
    datas = _np.concatenate(datas)
    return _csr_matrix((datas, (rows, cols)), shape=sdf.shape)


def _multiple_regression_with_penalty(dep_var, ind_vars, data,
                                      weights=None, penalty=0.0,
                                      constant=False, use_sparse=True,
                                      solver='auto'):

    DS_FLAG = False
    if isinstance(data, _Table):
        DS_FLAG = True
        data = data.to_df()
    if isinstance(ind_vars, tuple):
        ind_vars = list(ind_vars)
    if not isinstance(ind_vars, list):
        ind_vars = [ind_vars]

    X = data[ind_vars].values
    y = data[dep_var].values

    if weights is not None:
        w = data[weights].values.astype(float)
        w /= w.sum()

        X, y, w = X[w > 0], y[w > 0], w[w > 0]
    else:
        w = None

        mask = ~_np.isnan(y)
        X = X[mask, :]
        y = y[mask]

    if _np.min(w) == 0.:
        raise ValueError("weight is 0")

    if use_sparse:
        X = _csr_matrix(X)

    model = _linear_model.Ridge(
        alpha=penalty, fit_intercept=constant, copy_X=False,
        solver=solver)
    model.fit(X, y, sample_weight=w)

    coefs = _pd.Series(
        {var: coef for var, coef in zip(ind_vars, model.coef_)})
    coefs['Intercept'] = model.intercept_
    if DS_FLAG:
        coefs = coefs.to_dict()
    return coefs


def multiple_regression_big(dep_var, ind_vars, data, weights=None,
                            constant=False, **kwargs):
    return _multiple_regression_with_penalty(
        dep_var, ind_vars, data, weights=weights, constant=constant, **kwargs)


def multiple_regression_big_with_penalty(dep_var, ind_vars, data, weights=None,
                                         penalty=0., constant=False, **kwargs):
    return _multiple_regression_with_penalty(
        dep_var, ind_vars, data, penalty=penalty,
        weights=weights, constant=constant, **kwargs)
