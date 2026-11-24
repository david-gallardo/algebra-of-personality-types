# P0 — Spanish BFI-2 Source & Provenance Audit (`https://osf.io/kp572/`)

**BLOCKING GATE DECISION: `PASS`**  
The $N = 1{,}146$ sample analyzed in the manuscript (`BFI-2 ES Study 1`) is **100% bit-for-bit traceable** to the official public OSF archive `Data/Study 1/BFI-2 Full data (study 1).csv` at `https://osf.io/kp572/`, and independent reconstruction from the authoritative Spanish BFI-2 scoring key (`Spanish BFI-2 Form.pdf`, p. 3) reproduces `data/processed/bfi2_study1_domains.csv` **byte-for-byte** (`max_abs_diff = 0.0`, `max_abs_R_diff = 1.67e-16`). **No sample size change or downstream re-analysis is warranted.**

---

## 1. OSF Project Metadata & File Inventory

- **OSF Project:** `https://osf.io/kp572/` (*Spanish Adaptation of the BFI-2*)
- **Associated Publication:** Gallardo-Pujol, D., Rouco, V., Cortijos-Bernabeu, A., Oceja, L., Soto, C. J., \& John, O. P. (2022). Factor structure, gender invariance, measurement properties, and short forms of the Spanish adaptation of the Big Five Inventory-2. *Psychological Test Adaptation and Development*, 3(1), 44–69. DOI: `10.1027/2698-1866/a000020`
- **License:** CC-By Attribution 4.0 International (publicly downloadable via OSF Storage API v2 without authentication; 46 files total, all verified accessible).
- **Key Source Files Verified:**

| OSF Path (`kp572`) | Version | Modified (UTC) | Size (bytes) | SHA-256 | Role |
|---|---|---|---|---|---|
| `Data/Study 1/BFI-2 Full data (study 1).csv` | 1 | `2021-02-04T10:20:40Z` | 154,830 | `ec787ff7cca2a27495beee5c1ef6d37d537fc75d1a0c78c1b35818662749b68c` | **Exact raw source** of `data/raw/bfi2_study1_raw.csv` ($N = 1{,}146$) |
| `Data/Study 2/1st wave.csv` & `2nd wave.csv` | 1 | `2021-02-04T10:20:19Z` | 322,407 / 97,946 | `3f0bf9ff...` / `b9be7adb...` | Longitudinal Study 2 waves (not used here) |
| `Data/Study 3/Study 3 data.csv` | 1 | `2023-02-11T21:05:14Z` | 135,769 | `848bbe3a8f9b6446ad8b8261a317d19ea885a2a902a48670c22cf2b7d4da4fb1` | Study 3 sample ($N = 419$) |
| `Spanish BFI-2 Form.pdf` | 1 | `2021-02-04T09:51:21Z` | 66,643 | `7f7037e3af7a270c853de1d5412f60fa4553c359348671193fcd55ae31555730` | Authoritative 60-item Spanish BFI-2 form & scoring key (p. 3) |
| `Scripts/study.1.reliabs.and.descriptives.R` | 1 | `2021-02-04T10:21:41Z` | 19,854 | `abe8ca6aaed72f362a80d9a7ff1a4300a933f4837d952b3c429d8b2cd061de78` | Original R analysis script (note: line 150 contains a typo keying BFI36 as `+1` instead of `-1`, whereas line 163 and `Spanish BFI-2 Form.pdf` p. 3 correctly reverse-key `36R`) |
| `Random Intercept .../SLiD/efa_full_data.csv` | 1 | `2021-02-04T10:05:32Z` | 119,999 | `b50e9268c0ab18b762dc65ab492c1ef6dcb4041700db23a32184829b0843e12d` | Mplus EFA export (999 rows; Study 1 EFA subsample) |

---

## 2. Resolution of the $N = 1{,}000$ / $1{,}146$ / $1{,}419$ Provenance Discrepancy

No error or silent filtering occurred; the three sample sizes correspond to three distinct documented stages of the same OSF project:

1. **$N = 1{,}000$ (Published Study 1 primary analyses in Gallardo-Pujol et al., 2022):** The initial Study 1 cutoff analyzed in the 2022 publication (which explicitly notes that Study 1 data collection continued for norming purposes).
2. **$N = 1{,}146$ (Extended Study 1 OSF Archive — used in this manuscript):** The complete public OSF file `Data/Study 1/BFI-2 Full data (study 1).csv` contains **1,146 complete respondents** (0 missing cells across all 60 items `BFI1`–`BFI60`; Age 17–75, $M = 22.81$, $SD = 10.85$; Gender: 964 code `2`, 182 code `1`). Its SHA-256 checksum matches `data/raw/bfi2_study1_raw.csv` **identically**.
3. **$N = 1{,}419$ (Item Response Warehouse BFI-2 index):** Equals published Study 1 ($N = 1{,}000$) + Study 3 (`Data/Study 3/Study 3 data.csv`, verified $N = 419$).

### Row-Count Flowchart (`BFI-2 ES Study 1`, Extended OSF Archive)
- Raw OSF CSV `Data/Study 1/BFI-2 Full data (study 1).csv`: **1,146 rows** (64 columns: row index, `X1`, `Age`, `Gender`, `BFI1`–`BFI60`).
- Missingness audit across `BFI1`–`BFI60`: **0 missing cells**, **0 rows excluded**.
- Authoritative 60-item domain scoring (`Spanish BFI-2 Form.pdf`, p. 3; reverse-keyed items scored $6 - x$, 12 items/domain, unweighted mean, strict `(N, E, O, A, C)` order with `N` unreversed): **1,146 complete domain profiles**.
- Byte-for-byte comparison with `data/processed/bfi2_study1_domains.csv` via portable script `scripts/prepare_bfi2_es.py`: **`diff -q` exit code 0 (exact byte match)**.

---

## 3. Independent Reconstruction & Discrepancy Table

Full machine-readable audit saved to `docs/review_v16/01_bfi2_provenance_audit.json` and `docs/review_v16/01_bfi2_discrepancy_table.csv`.

| Metric | Manuscript / Archived | Reconstructed from OSF `kp572` | Abs. Discrepancy | Verdict |
|---|---|---|---|---|
| $N$ (complete cases) | 1,146 | 1,146 | 0 | Exact match |
| Raw file SHA-256 | `ec787ff7cca2a274...` | `ec787ff7cca2a274...` | 0 (identical) | Bit-for-bit identical |
| Max domain score diff (`N,E,O,A,C`) | 0.0 | `0.000e+00` | `0.000e+00` | Exact byte match |
| Max pairwise $R_{ij}$ diff | 0.0 | `1.665e-16` | `1.665e-16` | Machine epsilon |
| $\lambda_1$ | 1.792 | 1.792328 | `0.000328` | Matches Table 1 |
| $\lambda_2$ | 1.043 | 1.043053 | `0.000053` | Matches Table 1 |
| $\lambda_2 / \lambda_1$ | 0.582 | 0.581954 | `0.000046` | Matches Table 1 |
| $\det(\mathbf{R})$ | 0.662 | 0.661648 | `0.000352` | Matches Table 1 |
| $|\mathbf{R}|^{-1/2}$ | 1.229 | 1.229381 | `0.000381` | Matches Table 1 ($< 1.25$) |

---

## 4. Authoritative Data Availability Paragraph (for `v16`)

> Spanish BFI-2 Study 1 item-level data (`BFI-2 Full data (study 1).csv`, $N = 1,146$ extended Study 1 norming archive; CC-By Attribution 4.0) and the authoritative scoring key (`Spanish BFI-2 Form.pdf`) are publicly archived on OSF at \url{https://osf.io/kp572/} (Gallardo-Pujol et al., 2022, DOI: 10.1027/2698-1866/a000020); portable domain-scoring code (`scripts/prepare_bfi2_es.py`) reproduces the five standardized domain scores (`N, E, O, A, C`) directly from the raw OSF file without local path dependencies or copyrighted item text.
