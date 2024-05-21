import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import gcf

def softmax(x, t=1.0):
    x = x - np.max(x, axis=-1)[..., np.newaxis] # Subtracting the maximum value for numerical stability
    exp_x = np.exp(x / t)
    softmax_x = exp_x / np.sum(exp_x, axis=-1)[..., np.newaxis]
    return softmax_x


def lcurve(x, b0=0.0, b1=1.0, derivative=False, zero_default=None): 
    logit = b0 + b1 * x
    if derivative:
        y = (b1 * np.exp(logit)) / (np.exp(logit) + 1)**2
        if zero_default is not None:
            y[x==0] = zero_default
        return y
    y = 1 / (1 + np.exp(-logit))
    if zero_default is not None:
        y[x==0] = zero_default
    return y


def compute_choices(options, probs):
    return [np.random.choice(options, 1, p=p)[0] for p in probs]


def gen_feedback(n_trials, b0, b1, n_subjects=1):
    arr = np.zeros([n_subjects, n_trials])
    random_numbers = np.random.rand(n_subjects, n_trials)
    c = lcurve(np.arange(n_trials), b0, b1)
    arr = random_numbers < c
    return arr.astype(int)


def gen_ind_data(n_trials, b0, b1, n_subjects=1):
    random_numbers = np.random.rand(n_subjects, n_trials)
    x = np.arange(n_trials)
    c = lcurve(x, b0, b1)
    trials = list(range(1, n_trials + 1))
    s, t, o = [], [], []
    for i in range(n_subjects):
        s += [i for _ in range(n_trials)]
        t += trials
        o += (random_numbers[i, :] < c).astype(int).tolist()
    
    df = pd.DataFrame(dict(s=s, t=t, o=o))
    return df


def as_df(x):
    N, T = x.shape
    switch = np.insert(x[:, 1:] != x[:, :-1], 0, np.zeros(N), axis=1)
    blockNum = np.cumsum(switch, axis=1)
    pid = np.repeat(np.arange(N), T)
    trials = np.tile(np.arange(T), N)
    df = pd.DataFrame(dict(pid=pid, trial=trials, choice=x.flatten(), switch=switch.flatten(), blockNum=blockNum.flatten()))
    df = df.assign(blockSize = 1)
    return df


def summarize_choices(df):
    df = df.groupby(['pid', 'blockNum'])[['blockSize']].count().reset_index()
    df = df.groupby(['pid']).agg({'blockNum': max, 'blockSize': lambda x: np.sqrt(np.mean((x-np.mean(x))**2))})
    df = df.reset_index()
    df = df.rename(columns={'blockNum': 'numSwitches', 'blockSize': 'blockSizeVar'})

    return df


def mnov(c, k, return_p = False):
    numer = c + 1
    denom = np.sum(c, axis=1) + k
    p = numer / denom[:, np.newaxis]
    if return_p:
        return -np.log(p), p
    return -np.log(p)


def rollsum(a, w):
    result = np.stack([pd.Series(c).rolling(min_periods=1, window=w).sum().values for c in a.T])
    return result.T


def sim_choices(N, T, K, tau=1., b0=0., b1=1., wlp=1.0, wpc=1.0, wnov=1.0, wrb=1.0, nu=180):
    options = list(range(K))
    subj_inds = list(range(N))
    w = np.array([wlp, wpc, wnov, wrb])
    # w = w / np.sum(w)

    # Initialize records
    history = dict()
    h_choices = np.full([N, T, K], 0)
    lp0 = np.zeros([N, K])
    nov0 = -np.log(1 / K)
    utility =  w[0] * lp0 + w[1] * nov0
    h_utility = [utility.copy()]
    h_prob = [np.zeros([N, K]) + (1 / K)]
    h_lp = [lp0]
    pc0 = lcurve(0, b0, b1)
    h_pc = [np.zeros_like(lp0) + pc0]
    h_nov = [np.zeros([N, K]) - np.log(1 / K)]
    h_rb = [np.zeros([N, K])]
    for t in range(1, T):
        # Choose
        probs = softmax(utility, tau)
        choices = compute_choices(options, probs)
        
        # Update history
        h_choices[subj_inds, t, choices] = 1

        # Count choices
        choice_counts = h_choices[:, :t+1, :].sum(axis=1)
        t0 = int(max(0, t-nu))
        choice_counts_recent = h_choices[:, t0:t+1, :].sum(axis=1)

        # Compute utility
        lp = lcurve(choice_counts, b0, b1, derivative=True) # learning progress
        pc = lcurve(choice_counts, b0, b1, derivative=False) # percent correct
        nov = mnov(choice_counts_recent, K) # avoidance cost
        rb = h_choices[:, t, :]

        # Update utility for the next trial
        utility = np.zeros_like(utility)
        utility[subj_inds, :] = w[0] * lp[subj_inds, :] + w[1] * pc[subj_inds, :]
        utility += w[2] * nov
        utility += w[3] * rb
        # print(lp,pc,utility)
        h_utility.append(utility.copy())
        h_prob.append(probs.copy())
        visited = choice_counts > 0
        h_lp.append(lp * visited)
        h_pc.append(np.where(~visited, pc0, pc*visited))
        h_nov.append(nov)
        h_rb.append(rb)


    history['utility'] = np.stack(h_utility, axis=1).squeeze()
    history['prob'] = np.stack(h_prob, axis=1).squeeze()
    history['lp'] = np.stack(h_lp, axis=1).squeeze()
    history['pc'] = np.stack(h_pc, axis=1).squeeze()
    history['nov'] = np.stack(h_nov, axis=1).squeeze()
    history['rb'] = np.stack(h_rb, axis=1).squeeze()

    return np.argmax(h_choices[:N, :, :], axis=2), history


def spatial_entropy(x):
    if len(x.shape) == 1:
        x = x[np.newaxis]
    hs = []
    for row in x:
        classes = np.unique(row)
        k = classes.size
        coords = np.arange(row.size).astype(float)
        ps = []
        edges = []
        centroids = []
        for c in classes:
            instances = row==c

            # Calculate proportions
            ps.append(np.mean(instances))

            # Count edges
            edges.append(np.sum(instances[:-1] != instances[1:],))

            # Calculate centroids
            coords_copy = coords.copy()
            coords_copy[~instances] = np.nan
            centroids.append(np.nanmean(coords_copy))

        ps, edges, centroids = [np.array(_) for _ in (ps, edges, centroids)]
        dists = []
        inds = np.arange(k)
        for i in range(k):
            dists.append(np.sum(np.abs(centroids[i] - centroids[inds != i])))
        hs.append(-np.sum((edges * dists) * ps * np.log(ps)))

    return np.array(hs)


def plot_choice_data(data, ax, color):
    # Create datarame
    df = as_df(data)

    # Plot choices
    df_ = df.query('pid < 30')
    df_ = df_.assign(numSwitches = df_.groupby('pid').switch.transform(sum))
    ranks = df_.groupby('pid')[['pid','numSwitches']].head(1).reset_index()
    ranks['numSwitchesRank'] = ranks.numSwitches.rank(method='first').astype(int)
    df_ = df_.merge(ranks[['pid', 'numSwitchesRank']], how='outer', on='pid')
    df_ = df_.sort_values(by=['numSwitchesRank'])

    sns.scatterplot(
        data = df_,
        x = 'trial', 
        y = 'numSwitchesRank', 
        hue = 'choice',
        size = 1,
        marker = 'd',
        alpha = 1.0,
        palette = 'tab10',
        ax = ax.choices
    )

    # Switching across time
    sns.lineplot(data=df.query('trial > 0'), x='trial', y='switch', color=color, alpha=.8, ax=ax.switching)

    # Entropy
    h = spatial_entropy(data)
    bins = np.linspace(0, 6000, 30)
    h = h[h < 4000]
    sns.histplot(h, ax=ax.entropy, color=color, element='step', stat='density', fill=True, alpha=.4, bins=bins)
    ax.entropy.axvline(np.mean(h), color=color, ls='--', lw=1, alpha=.5)

    # Number of switches and blocking consistency
    df = summarize_choices(df)
    df = df.query("numSwitches < 150")
    xmean = df.numSwitches.mean()
    ymean = df.blockSizeVar.mean()
    sns.scatterplot(x='numSwitches', y='blockSizeVar', data=df, ax=ax.scatter, color=color, size=2, alpha=.4)
    ax.scatter.scatter(xmean, ymean, color=color, s=40, marker="D", alpha=.8, zorder=1000)


def plot_util(T, b0, b1, wlp, wac, wrb, ax):
    w = [wlp, wac, wrb]
    w = [i / sum(w) for i in w]
    
    x = np.arange(0, T)
    lp = lcurve(x, b0, b1, derivative=True)
    lp_ = np.zeros_like(x)
    nov = mnov(np.stack([x, np.zeros_like(x)], axis=1), 2)

    # Utility
    ax.plot(x, w[0] * lp + w[1] * nov[:, 0] + w[2] * np.ones_like(x), color='k', label='U', lw=3)
    ax.plot(x, w[0] * lp_ + w[1] * nov[:, 1], color='k', label='(U)', ls='--', lw=3)
    
    # LP
    ax.plot(x, w[0] * lp, color='limegreen', label='LP')
    ax.plot(x, w[0] * lp_, color='limegreen', label='(LP)', ls='--')
    
    # Modirshanechi novelty
    ax.plot(x, w[1] * nov[:, 0], color='darkorange', label='Nov')
    ax.plot(x, w[1] * nov[:, 1], color='darkorange', label='(Nov)', ls='--')

    # Repetition bonus
    ax.plot(x, w[2] * np.ones_like(x), color='orchid', label='RB')
    ax.plot(x, w[2] * np.zeros_like(x), color='orchid', label='(RB)', ls='--')
    


def format_labels(ax):
    ax.scatter.set_ylabel('Blocking consistency')
    ax.scatter.set_xlabel('Number of switches')
    ax.scatter.get_legend().remove()

    ax.switching.set_ylabel('Proportion switching')
    ax.switching.set_xlabel('Trial')

    ax.entropy.set_ylabel('Density')
    ax.entropy.set_xlabel('Spatial entropy')
    ax.entropy.set_yticklabels([])
    ax.entropy.set_yticks([])

    # ax.lcurve.set_ylabel('Utility')
    # ax.lcurve.set_xlabel('Trial')

    ax.choices.set_yticklabels([])
    ax.choices.set_yticks([])
    ax.choices.set_ylabel('Runs by number of switches')
    ax.choices.set_xlabel('Trial')
    ax.choices.get_legend().remove()
    ax.choices.set_xlim(-1, 180)
    ax.choices.spines['top'].set_visible(False)
    ax.choices.spines['left'].set_visible(False)
    ax.choices.spines['right'].set_visible(False)

    gcf().tight_layout()