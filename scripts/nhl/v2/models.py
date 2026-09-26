"""Small statistical baselines and ML challengers; no market features."""
import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson, nbinom, binom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import TEAM_FEATURES, CORE_FEATURES

SUPPORT = np.arange(48)
TEAM_CANDIDATES = ['rate', 'opponent', 'poisson_core', 'poisson_context', 'boosting']
PLAYER_CANDIDATES = ['rate_poisson', 'opportunity_poisson', 'opportunity_nb', 'opportunity_hurdle']


def count_pmf(mean, alpha=0):
    mean = float(np.clip(mean,1e-6,18))
    values = nbinom.pmf(SUPPORT,1/alpha,1/(1+alpha*mean)) if alpha>1e-6 else poisson.pmf(SUPPORT,mean)
    if 1-values.sum() > 1e-6:
        raise ValueError('Distribution support insufficient')
    return values/values.sum()


def outcome(pmf,line,side='Over'):
    values = np.arange(len(pmf))
    win = float(pmf[values>line].sum() if side=='Over' else pmf[values<line].sum())
    push = float(pmf[values==line].sum())
    return dict(win=win,push=push,loss=max(0.,1-win-push))


class TeamModel:
    def __init__(self,kind):
        self.kind = kind
        self.features = CORE_FEATURES if kind=='poisson_core' else TEAM_FEATURES

    def fit(self,rows,games):
        X = np.array([[r[f] for f in self.features] for r in rows])
        y = np.array([r['target'] for r in rows])
        self.league = float(y.mean())
        self.home_ratio = float(np.mean([r['target'] for r in rows if r['home']])/self.league)
        extra = [g for g in games if g['extra_time']]
        # Regress overtime winner tendency to an even contest (prior 20 games).
        self.ot_home = (sum(g['home_score']>g['away_score'] for g in extra)+10)/(len(extra)+20)
        self.shootout_share = (sum(g['shootout'] for g in extra)+5)/(len(extra)+10)
        if self.kind.startswith('poisson'):
            self.estimator = make_pipeline(StandardScaler(),PoissonRegressor(alpha=.3,max_iter=300))
            self.estimator.fit(X,y)
        elif self.kind=='boosting':
            self.estimator = HistGradientBoostingRegressor(loss='poisson',max_iter=100,
                max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10,learning_rate=.05,random_state=41)
            self.estimator.fit(X,y)
        return self

    def predict(self,rows):
        if self.kind=='rate':
            return np.array([r['attack'] for r in rows])
        if self.kind=='opponent':
            return np.clip([r['attack']*r['defense']/self.league * (self.home_ratio if r['home'] else 2-self.home_ratio) for r in rows],.2,8)
        X = np.array([[r[f] for f in self.features] for r in rows])
        return np.clip(self.estimator.predict(X),.2,8)

    def joint(self,home,away):
        """Regulation scores include aggregate empty-net behavior in training outcomes.

        Every regulation tie receives exactly one OT/SO settlement goal. Shootout goals
        are added to game markets only. No marginal moneyline/puck-line fitting.
        """
        h,a = count_pmf(home),count_pmf(away)
        reg = np.outer(h,a)
        final = np.pad(reg,((0,1),(0,1)))
        for n in range(len(h)):
            tie = final[n,n]
            final[n,n] = 0
            final[n+1,n] += tie*self.ot_home
            final[n,n+1] += tie*(1-self.ot_home)
        return final


def game_outcome(joint,market,line,home_side=True,side='Over'):
    h,a = np.indices(joint.shape)
    if market=='h2h':
        value = h-a if home_side else a-h
    elif market=='spreads':
        value = (h-a if home_side else a-h)+line
    elif market=='totals':
        value = (h+a-line)*(1 if side=='Over' else -1)
    else:
        raise ValueError('Unsupported game market')
    win,push = float(joint[value>0].sum()),float(joint[value==0].sum())
    return dict(win=win,push=push,loss=max(0.,1-win-push))


class PlayerModel:
    def __init__(self,kind):
        self.kind = kind

    def fit(self,rows):
        means = np.array([r['opportunity_means'] for r in rows])
        y = np.array([r['targets'] for r in rows])
        self.alpha_shots = float(np.clip(np.sum((y[:,0]-means[:,0])**2-y[:,0])/np.sum(means[:,0]**2),0,.75))
        # Shared latent scoring intensity: goals and assists sum exactly to points.
        self.alpha_scoring = float(np.clip(np.sum((y[:,3]-means[:,3])**2-y[:,3])/np.sum(means[:,3]**2),0,.75))
        self.zero_ratio = float(np.mean(y[:,3]==0)/np.mean(np.exp(-means[:,3])))
        return self

    def pmfs(self,row):
        means = row['base_means'] if self.kind=='rate_poisson' else row['opportunity_means']
        use_nb = self.kind=='opportunity_nb'
        shots = count_pmf(means[0],self.alpha_shots if use_nb else 0)
        points = count_pmf(means[3],self.alpha_scoring if use_nb else 0)
        if self.kind=='opportunity_hurdle':
            zero = float(np.clip(np.exp(-means[3])*self.zero_ratio,.01,.99))
            positive_mean = max(0.001,means[3]/(1-zero)-1)
            points = np.r_[zero,(1-zero)*count_pmf(positive_mean)[:-1]]
            points /= points.sum()
        fraction = means[1]/max(means[3],1e-6)
        # Allocation of total points gives a coherent joint goals/assists distribution.
        # For Poisson this equals independent marginals; NB induces positive dependence.
        k,n = np.indices((len(points),len(points)))
        allocation = binom.pmf(k,n,fraction)*points[None,:]
        goals = allocation.sum(axis=1)
        assists = (binom.pmf(k,n,1-fraction)*points[None,:]).sum(axis=1)
        return [shots,goals,assists,points]

    def fast_pmfs(self,rows):
        means = np.array([r['base_means'] if self.kind=='rate_poisson' else r['opportunity_means'] for r in rows])
        out=[]
        if self.kind=='opportunity_hurdle':
            # Closed-form binomial thinning of a shifted Poisson positive component.
            z=np.clip(np.exp(-means[:,3])*self.zero_ratio,.01,.99)
            lam=np.maximum(.001,means[:,3]/(1-z)-1)
            for j in range(4):
                if j==0: p=poisson.pmf(SUPPORT[None,:],means[:,0,None])
                else:
                    frac=means[:,j]/np.maximum(means[:,3],1e-6)
                    base=poisson.pmf(SUPPORT[None,:],(lam*frac)[:,None])
                    shifted=np.pad(base[:,:-1],((0,0),(1,0)))
                    p=(1-z[:,None])*((1-frac[:,None])*base+frac[:,None]*shifted)
                    p[:,0]+=z
                out.append(p/p.sum(axis=1)[:,None])
            return out
        for j in range(4):
            alpha=(self.alpha_shots if j==0 else self.alpha_scoring) if self.kind=='opportunity_nb' else 0
            p=nbinom.pmf(SUPPORT[None,:],1/alpha,1/(1+alpha*means[:,j,None])) if alpha>1e-6 else poisson.pmf(SUPPORT[None,:],means[:,j,None])
            out.append(p/p.sum(axis=1)[:,None])
        return out
