import pandas as pd
import numpy as np
from IPython.display import display
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score

def flatten_columns(df):
    n = df.columns.nlevels
    l = [list(df.columns.get_level_values(i)) for i in range(n)]
    cns = []
    for cc in zip(*l):
        ss = []
        for i in cc:
            if i:
                ss.append(i)
        cns.append('_'.join(ss))
    df.columns = cns
    return df


def display_(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df)


def find_switches(s):
    y = np.zeros(len(s))
    x = s.values.squeeze()
    y[1:] = (x[0:-1] != x[1:]).astype(int)
    y[0] = 0
    return y


def rank_by_num_switches(df):
    df['switch'] = df.groupby('pid').famInd.transform(find_switches).astype(int)
    if 'stage' in df.columns:
        df.loc[df.stage.ne('epochs'), 'switch'] = 0
    df['numSwitches'] = df.groupby('pid').switch.transform(sum)
    ranks = df.groupby('pid')[['pid','numSwitches']].head(1).reset_index()
    ranks['numSwitchesRank'] = ranks.numSwitches.rank(method='first').astype(int)
    df = df.merge(ranks[['pid', 'numSwitchesRank']], how='outer', on='pid')
    return df


def z(df, col):
    return (df[col] - df[col].mean()) / df[col].std()


def cohen_k(x, k):
    x0 = x.copy()
    xk = x.shift(k)
    return cohen_kappa_score(x0[xk.notna()], xk.dropna())


def cohen_k_crit(alpha, T, p):
    μ = -1 / T
    σ2 = (1 - (1 + 2 * np.power(p, 3).sum() - 3 * np.power(p, 2).sum()) / (1 - np.power(p, 2).sum())**2) / T
    return stats.norm.ppf(q=1-alpha, loc=μ, scale=np.sqrt(σ2))


def cramer_v(x, k):
    x0 = x.copy()
    xk = x.shift(k)
    X = pd.DataFrame({'x0': x0, 'xk': xk})
    X = X.dropna()
    ct = pd.crosstab(index=X.x0, columns=X.xk)
    v = stats.contingency.association(ct, method='cramer')
    return v


def cramer_v_crit(alpha, T, d):
    crit = stats.chi2.ppf(q=1-alpha, df=d**2)
    return np.sqrt(crit / (T * d))


def acf(x, lags, arf):
    return np.array([arf(x, l) for l in range(lags)])
