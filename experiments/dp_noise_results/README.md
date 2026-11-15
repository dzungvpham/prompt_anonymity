# Simulated-DP noise sweep: results behind the figures

Nearest neighbour on Gemini Embedding 2 (d = 3072) features, with Laplace noise added to the
**unknown-side (target) embeddings only**; the attacker's known side stays clean. The test set is
the final 25% of the timeline. Open-set, author-level DIR vs FAR.

## Files

| file | contents |
|---|---|
| `dirfar_author_conv_{swe_chat,wildchat}.csv` | DIR-FAR curves, conversation-level DP, every epsilon and every known window |
| `dirfar_author_user_{swe_chat,wildchat}.csv` | the same, user-level DP (epsilon is the user's budget, spent as epsilon / n per conversation) |
| `swe_chat_summary.csv` | one row per (unit, epsilon): AUC and DIR@10% at the 75/25 window (document level) |

CSV columns: `series` (`No defense` = epsilon infinity, otherwise `ε = 10^k` / `user ε = 10^k`),
`known_config` (`known0075` is the 75% known / 25% target split used in the paper figure),
`false_accept_rate`, `identification_rate` (DIR), `random` / `random_proportional` (chance
baselines at that FAR), `ci_low` / `ci_high` (equal to the curve unless run with `--bands`), `top1`.

## What the noise is (for overlaying a formula)

`laplace_release` in `run_experiment.py`:

* each embedding is scaled to unit L2 norm;
* L1 sensitivity is the worst case over unit vectors, `Delta1 = 2 * sqrt(d)` (any two conversations
  are neighbours), so this is pure epsilon-DP at conversation level;
* i.i.d. `Laplace(0, Delta1 / epsilon)` per coordinate;
* user level: a user with n released (unknown-side) conversations gets `epsilon / n` per conversation
  (basic composition, `noise_epsilons`).

Because the noise scale is proportional to `sqrt(d)`, the epsilon values here do not transfer to
other embedding dimensions.

## Reproducing

```
python experiments/run_experiment.py --source {swe_chat,wildchat} --feature gemini_embedding_2 \
    --attacks nearest_neighbor --data-dir data/hf --output-dir <out> \
    --unknown-noise-epsilon <EPS> --noise-unit {conversation,user} --noise-seed 0
```

(`--noise-sensitivity` defaults to `bound`, the 2*sqrt(d) calibration above; `empirical` is a
data-dependent scale and not a DP guarantee.) Epsilons run: 10^1 .. 10^5 conversation-level,
10^1 .. 10^6 user-level. The WildChat runs were launched with `--noise-seed 0`; the SWE-Chat logs do
not echo the flags, so the seed there is the code default (0) rather than something recorded.
Then draw the paper-style panels with `plot_results.py` (set `PLOT_RESULTS_DIR`, `PLOT_PLOTS_DIR`)
and the compact 75/25 figures with `experiments/plot_dp_noise.py`.

Figures: `../dp_noise_figures/lean/dp_noise_conv.pdf` (conversation level, main paper) and
`dp_noise_both.pdf` (conversation + user level, appendix). The lean figures drop epsilon = 10^1
(hidden under 10^2) and the epsilons that sit on top of epsilon = infinity (conversation 10^5,
user 10^6); `../dp_noise_figures/all/` has every epsilon.
