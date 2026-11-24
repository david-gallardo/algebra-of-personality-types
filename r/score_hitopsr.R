#!/usr/bin/env Rscript
# Official R scoring script for KU HiTOP-SR (hitop::ku_hitopsr) using pinned hitop package.
# Outputs scored 76-scale individual file to data/processed/hitop_empirical/ (gitignored)
# and aggregate item-level reliability/QC summary to docs/hitop_empirical/ku_item_reliability.csv.

suppressPackageStartupMessages({
  library(hitop)
})

args <- commandArgs(trailingOnly = TRUE)
out_scores <- if (length(args) >= 1) args[1] else "data/processed/hitop_empirical/ku_hitopsr_scored.csv"
out_rel <- if (length(args) >= 2) args[2] else "docs/hitop_empirical/ku_item_reliability.csv"

dir.create(dirname(out_scores), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(out_rel), recursive = TRUE, showWarnings = FALSE)

pkg_ver <- as.character(packageVersion("hitop"))
cat(sprintf("Scoring KU HiTOP-SR with hitop version %s (R %s)\n", pkg_ver, R.version.string))

data("ku_hitopsr", package = "hitop")
items <- sprintf("hsr_%03d", 1:405)
stopifnot(all(items %in% names(ku_hitopsr)))
stopifnot(nrow(ku_hitopsr) == 411L)

# Verify no duplicate participant IDs
stopifnot(length(unique(ku_hitopsr$participant)) == nrow(ku_hitopsr))

# Official scoring via score_hitopsr()
scores <- score_hitopsr(ku_hitopsr, items = items, append = FALSE)
stopifnot(nrow(scores) == nrow(ku_hitopsr))
stopifnot(ncol(scores) == 76L)

# Export participant ID, biosex, and 76 official scales (gitignored local file)
out_df <- cbind(
  data.frame(
    participant = ku_hitopsr$participant,
    biosex = as.character(ku_hitopsr$biosex),
    stringsAsFactors = FALSE
  ),
  as.data.frame(scores)
)
write.csv(out_df, out_scores, row.names = FALSE)
cat(sprintf("Wrote KU scored scales: %s (%d x %d)\n", out_scores, nrow(out_df), ncol(out_df)))

# Compute official item-level scale reliability (alpha & omega) using hitop::reliability_hitopsr()
rel_df <- tryCatch(
  {
    r <- reliability_hitopsr(ku_hitopsr, items = items)
    as.data.frame(r)
  },
  error = function(e) {
    message("reliability_hitopsr() note: ", conditionMessage(e))
    # Fallback: per-scale Cronbach's alpha via calc_alpha
    sc_tbl <- hitop::hitopsr_scales
    res <- lapply(seq_len(nrow(sc_tbl)), function(i) {
      sc_name <- sc_tbl$camelCase[i]
      item_nums <- sc_tbl$itemNumbers[[i]]
      item_cols <- sprintf("hsr_%03d", item_nums)
      sub <- ku_hitopsr[, item_cols, drop = FALSE]
      a <- tryCatch(calc_alpha(sub), error = function(e2) NA_real_)
      data.frame(scale = paste0("hsr_", sc_name), n_items = length(item_cols), alpha = as.numeric(a))
    })
    do.call(rbind, res)
  }
)
write.csv(rel_df, out_rel, row.names = FALSE)
cat(sprintf("Wrote KU item-level reliability table: %s (%d rows)\n", out_rel, nrow(rel_df)))
