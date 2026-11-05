# Generic input contract

Supply one NumPy `.npz` archive per recording setting. Load uses
`allow_pickle=False`. Raw device files may be adapted to this contract by the
user. Sensor units, ordering and clock alignment must remain consistent across
training and inference. No data archive is included in this repository.

| Key | Shape | Meaning |
|---|---|---|
| x | N,T,C | Unstandardised observed sensor values; missing values are NaN |
| mask | N,T,C | Optional 0/1 observation masks; inferred from finite x if absent |
| static | N,S | Raw history/state covariates; missing state values may be NaN |
| history | N,H | Complete daily yields, latest first: d,d-1,... |
| y | N | Yield record dated d+1; optional only for checkpoint inference |
| cows | N | Cow IDs stored as non-object strings or integers |
| input_dates | N | ISO calendar dates identifying d |
| target_dates | N | Optional; if present must equal d+1 |
| sample_ids | N | Optional unique non-object strings; otherwise generated from cow/date |

Dates must use the same source-record convention for sensor windows and milk
records. They are not evidence of actual online predictor publication times.

## MMCOWS configuration

T=1440, C=10, H=3, S=5. Channels: UWB x/y/z, acceleration x/y/z,
magnetic x/y/z, binary lying. Static columns: milk(d), milk(d-1), milk(d-2),
age in years, source DIM. The model receives ten static features after adding
missing-value indicators. Binary lying is not standardised or noised.

The input day contains minute bins on d. Milk target is d+1. Missing continuous
runs of at most two interior minutes are interpolated, without crossing a day;
original observation masks remain unchanged. Remaining continuous gaps are
training-median filled. For caller datasets with missing binary posture, the
implementation uses the training majority value (median thresholded at 0.5).
The retained manuscript cohort reports no missing posture; confirm this fallback
against your final source adapter before using another cohort.

## CowBase configuration

T=12, C=1, H=7, S=15. Activity observations run from d-1 at22:00 through d at20:00
at two-hour spacing. The d22:00 observation is excluded. Require all twelve
observations. Static columns: seven daily yields latest first, mean3, mean7,
population SD7, milk(d)-milk(d-1), positive session count on d, hours since last
session at the end of d, source DIM, parity. After indicators the width is thirty.
Daily milk sums and session-state variables must be computed from source clocks
before input construction. Missing days are not zero production.

## Optional normalised-table constructor

`python -m bscg construct` accepts three caller-supplied CSV tables:

- Sensor: `cow,time,x0,...,x(C-1)`. Times already use the chosen source clock.
- Milk: `cow,date,milk`. One valid daily record per cow/date, in kg.
- State: `cow,date,age,DIM` for MMCOWS, or
  `cow,date,positive_sessions,hours_since_last_milking,DIM,parity` for CowBase.

The constructor collapses duplicate sensor timestamps by continuous means and
the last valid binary observation, builds minute bins for MMCOWS, aligns complete
histories and next-day targets, and skips ineligible windows. It does not infer
source units, reconstruct missing physical milkings, decode manufacturer files
or resolve undocumented clocks. Those source-specific choices belong in the
user's adapter. Exact local fold assignments are exported after input creation.
