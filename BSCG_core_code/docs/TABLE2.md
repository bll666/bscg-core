# Prediction import and Table 2

CSV input requires columns `cohort,model,seed,sample_id,prediction`.
Optional `cow,target` are validated against the supplied input archive.
No prediction CSV is shipped. Each model/seed must cover the identical full
outer test cohort, with no duplicate or extra sample keys.

Supported cohort names: MM_LOCO, MM_TIME, MM_BOTH, CB_SEEN, CB_UNSEEN.
Required model labels for complete Table 2: BSCG, Transformer, HGB_H, HGB_HS;
MM_LOCO additionally requires LSTM and TCN. Fixed baselines Persistence and
RecentMean are calculated directly from supplied histories and targets, and
must not be imported. Other model labels may also be imported as additional rows.

For deterministic comparators use one seed value. For BSCG supply all configured
seeds. Other neural comparators should supply the same complete refit seed set;
the user remains responsible for their training and selection protocol.

Calculations:

1. Align every prediction vector to the expected cohort by sample_id.
2. Average seed predictions at the sample level.
3. Calculate each cow's MAE and RMSE separately; average cows equally.
4. Resample the complete cow error vectors, using paired draws for contrasts.
5. Export absolute metrics, paired gains, individual cow errors and seed metrics.

Absolute RMSE is the mean of cow-specific roots, not a root after averaging
all cow errors. Seed MAE means need not equal the MAE of mean predictions.

Cow bootstrap seeds are4470–4474 for the five cohorts above, with5000 draws for
MMCOWS and2000 for CowBase. MM_TIME/MM_BOTH fixed baselines share seed4471 on
their identical test cohort. Baseline equality is a consequence of identical
samples and formulas, and is not forced by rounding or copying scores.

CSV/Markdown exports contain more uncertainty detail than the compact article
table; choose the article's columns during document formatting. Paired gains
are exported separately. Results are never read from a manuscript or an old
drafting-results file.
