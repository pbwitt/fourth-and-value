# NFL spread shadow, 2026

Version `nfl-spread-shadow-1`. Generated Wed, Nov 11, 2026, 7:10 AM EST.

A logged-only adjustment to passing and receiving prop projections for the team spread. It never changes published projections, probabilities or picks. See `config/nfl_spread_shadow.json`.

Log loss uses raw (uncalibrated) probabilities at the consensus line with pushes excluded; lower is better. Weeks before 8 trained the coefficients and are shown for reference only.

| Week | Sample | Props | Current | Shadow | Shadow − current | Market | Side hit (current → shadow) |
|---|---|---:|---:|---:|---:|---:|---|
| 5 | in-sample | 237 | 0.7725 | 0.7740 | +0.0015 | 0.6852 | 49.8% → 48.5% |
| 6 | in-sample | 212 | 0.7241 | 0.7242 | +0.0002 | 0.6958 | 52.8% → 56.1% |
| 7 | in-sample | 231 | 0.8136 | 0.8008 | -0.0128 | 0.6893 | 42.9% → 44.6% |
| 8 | prospective | 172 | 0.7443 | 0.7491 | +0.0048 | 0.6907 | 50.0% → 53.5% |
| 9 | prospective | 205 | 0.7365 | 0.7505 | +0.0139 | 0.6922 | 44.9% → 46.8% |

**Prospective total:** 377 props, log loss 0.7401 current vs 0.7498 shadow (+0.0098).
