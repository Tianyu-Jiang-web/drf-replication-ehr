library(readr)
library(tidyverse)


df <- read.csv(
  "data/master_clean_final.csv"
)

glimpse(df)


# scripts/00_prepare_data.R
# Step 0: prepare X/Y from master table

library(dplyr)
library(readr)

# ---- 0) paths ----
in_path  <- "data/master_clean_final.csv"
out_dir  <- "data/processed"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1) read ----
df <- read_csv(in_path, show_col_types = FALSE)

# ---- 2) define columns ----
# IDs (not used as features)
id_cols <- c("subject_id", "hadm_id", "stay_id")

# choose ONE primary label for this project:
y_col <- "readmit_1m"

# ---- 3) basic checks ----
stopifnot(y_col %in% names(df))
if (!all(df[[y_col]] %in% c(0, 1))) {
  stop(paste0("Outcome ", y_col, " is not binary 0/1. Please check coding."))
}

# ---- 4) build Y ----
Y <- df[[y_col]]

# ---- 5) build X ----
# drop IDs + the outcome column to avoid leakage
drop_cols <- intersect(c(id_cols, y_col), names(df))

X <- df %>%
  select(-all_of(drop_cols))

# ---- 6) minimal cleaning for DRF compatibility ----
# convert character -> factor
X <- X %>%
  mutate(across(where(is.character), as.factor))

# remove zero-variance columns
nzv <- sapply(X, function(v) {
  v2 <- v[!is.na(v)]
  length(unique(v2)) <= 1
})
if (any(nzv)) {
  X <- X[, !nzv, drop = FALSE]
}

# ---- 7) save ----
saveRDS(X, file.path(out_dir, "X.rds"))
saveRDS(Y, file.path(out_dir, "Y.rds"))

# ---- 8) print summary ----
cat("Step 0 done. Saved:\n")
cat("- ", file.path(out_dir, "X.rds"),
    " (n=", nrow(X), ", p=", ncol(X), ")\n", sep = "")
cat("- ", file.path(out_dir, "Y.rds"),
    " (n=", length(Y), ", positives=", sum(Y == 1), ")\n", sep = "")
