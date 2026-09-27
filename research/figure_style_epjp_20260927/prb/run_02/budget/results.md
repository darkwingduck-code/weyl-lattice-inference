# Fixed total acquisition budget

The experiment fixes one total acquisition time across fields. Within a field, time is distributed equally over the same four entries and 45 frequencies. A field receiving fraction f has per-entry complex noise RMS sigma0/sqrt(f), where sigma0=1.2931669440932318e-4. Equal independent white detector noise and zero field-switching overhead are declared assumptions. Operator directions are shared across fields; affine complex gains are fitted separately in each field.

| Truth | Policy | Ua RMSE | Local Fisher SD | Approx. MC SE of RMSE |
|---|---|---:|---:|---:|
| symmetric | B0_only | 0.107959 | 0.109538 | 0.013990 |
| symmetric | Bplus_only | 0.036763 | 0.038879 | 0.003218 |
| symmetric | Bplus_25pct | 0.048077 | 0.047288 | 0.005113 |
| symmetric | Bplus_50pct | 0.035546 | 0.036399 | 0.004063 |
| symmetric | Bplus_75pct | 0.031083 | 0.032362 | 0.003589 |
| asymmetric | B0_only | 0.211026 | 0.203094 | 0.028180 |
| asymmetric | Bplus_only | 0.374835 | 0.342728 | 0.051756 |
| asymmetric | Bplus_25pct | 0.174312 | 0.181834 | 0.020108 |
| asymmetric | Bplus_50pct | 0.178089 | 0.200256 | 0.020470 |
| asymmetric | Bplus_75pct | 0.199479 | 0.236872 | 0.023433 |

The empirical best policies on this five-point grid are {'symmetric': 'Bplus_75pct', 'asymmetric': 'Bplus_25pct'}. The smallest worst-truth RMSE is obtained by Bplus_25pct. These are selections among the declared policies and two truth pairs, not an optimization over all time allocations or a uniform robustness theorem.

The local covariance is the inverse interaction Fisher matrix after projecting both affine gains and the two source directions, with both g and Ua fitted. Physical coordinates are (g/.8, Ua/8); the exact circular-complex Gaussian factor 2 is included. Local Fisher SD is not a finite-sample coverage guarantee.

Retained optimizer successes: 300/300; bound hits: 0; feasible-truth objective failures: 0; unsuccessful alternative starts: 0/1500. All estimates and every start are retained, including failures.

Error bars are one approximate Monte Carlo standard error of empirical RMSE (delta method from 30 squared errors), not parameter confidence intervals. Paired squared-error differences against both single-field baselines are retained separately. Small differences should not be assigned a robust ranking from this limited repeated sample.
