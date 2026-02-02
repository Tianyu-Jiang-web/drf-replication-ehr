# scripts/05_paper_plots.R
# Make "paper-style" figures + cross-model comparison table
# Models: DRF, GLMNET (ridge logistic), RF (ranger), XGB (xgboost)

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# metrics
library(pROC)     # AUROC
library(PRROC)    # AUPRC

# ---- 0) dirs ----
dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)

# ---- 1) load predictions ----
# DRF outputs from Step 3
drf_df <- read_csv("results/step3_predictions_uncertainty.csv", show_col_types = FALSE)

# Baselines outputs from Step 2b
base_df <- read_csv("results/pred_baselines.csv", show_col_types = FALSE)

# ---- 2) standardize columns ----
# Expect DRF columns like: idx, y_true (or y_test), p_hat, entropy, ess ...
# We'll robustly map y + p
if (!("idx" %in% names(drf_df))) stop("DRF file must contain column: idx")
y_col_drf <- if ("y_true" %in% names(drf_df)) "y_true" else if ("y_test" %in% names(drf_df)) "y_test" else stop("DRF file must contain y_true or y_test")
p_col_drf <- if ("p_hat" %in% names(drf_df)) "p_hat" else if ("p" %in% names(drf_df)) "p" else stop("DRF file must contain p_hat (or p)")
ent_col_drf <- if ("entropy" %in% names(drf_df)) "entropy" else NA

drf_std <- drf_df %>%
  transmute(
    idx = idx,
    y = as.integer(.data[[y_col_drf]]),
    p = as.numeric(.data[[p_col_drf]]),
    entropy = if (!is.na(ent_col_drf)) as.numeric(.data[[ent_col_drf]]) else NA_real_
  )

# baseline predictions: idx, y_test, p_glmnet, p_rf, p_xgb
stopifnot(all(c("idx","y_test","p_glmnet","p_rf","p_xgb") %in% names(base_df)))

base_long <- base_df %>%
  transmute(
    idx = idx,
    y = as.integer(y_test),
    glmnet = as.numeric(p_glmnet),
    rf = as.numeric(p_rf),
    xgb = as.numeric(p_xgb)
  ) %>%
  pivot_longer(cols = c(glmnet, rf, xgb), names_to = "model", values_to = "p") %>%
  mutate(
    model = recode(model,
                   glmnet = "Logistic (glmnet)",
                   rf     = "Random Forest",
                   xgb    = "XGBoost")
  )

# DRF as long format
drf_long <- drf_std %>%
  mutate(model = "DRF") %>%
  select(idx, y, model, p, entropy)

# Combine all models
pred_all <- bind_rows(
  drf_long,
  base_long %>% mutate(entropy = NA_real_) # fill later
)

# ---- 3) add entropy for ALL models (so we can compare coverage–risk, etc.) ----
# entropy(p) = -p log p - (1-p) log(1-p)
safe_entropy <- function(p) {
  eps <- 1e-15
  p <- pmin(pmax(p, eps), 1 - eps)
  -(p * log(p) + (1 - p) * log(1 - p))
}

pred_all <- pred_all %>%
  mutate(
    entropy = ifelse(is.na(entropy), safe_entropy(p), entropy)
  )

# ---- 4) helper metrics ----
logloss <- function(y, p) {
  eps <- 1e-15
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

brier <- function(y, p) mean((y - p)^2)

ece_10bin <- function(y, p, n_bins = 10) {
  # equal-width bins on [0,1]
  brks <- seq(0, 1, length.out = n_bins + 1)
  bin <- cut(p, breaks = brks, include.lowest = TRUE)
  df <- data.frame(y = y, p = p, bin = bin) |>
    dplyr::group_by(bin) |>
    dplyr::summarise(
      n = dplyr::n(),
      acc = mean(y),
      conf = mean(p),
      .groups = "drop"
    )
  sum((df$n / sum(df$n)) * abs(df$acc - df$conf))
}

auprc <- function(y, p) {
  # PRROC expects scores for positives and negatives separately
  pr <- PRROC::pr.curve(scores.class0 = p[y == 1], scores.class1 = p[y == 0], curve = FALSE)
  pr$auc.integral
}

auroc <- function(y, p) {
  as.numeric(pROC::auc(pROC::roc(response = y, predictor = p, quiet = TRUE)))
}

# ---- 5) compute cross-model metrics table ----
metrics_tbl <- pred_all %>%
  group_by(model) %>%
  summarise(
    n = n(),
    pos_rate = mean(y),
    AUROC = auroc(y, p),
    AUPRC = auprc(y, p),
    LogLoss = logloss(y, p),
    Brier = brier(y, p),
    ECE = ece_10bin(y, p, n_bins = 10),
    .groups = "drop"
  ) %>%
  arrange(desc(AUROC))

write_csv(metrics_tbl, "results/metrics_models.csv")

# ---- 6) Fig A: Calibration curves (all models) ----
calib_df <- pred_all %>%
  group_by(model) %>%
  mutate(bin = ntile(p, 10)) %>%   # equal-count bins (often nicer)
  group_by(model, bin) %>%
  summarise(
    p_mean = mean(p),
    y_mean = mean(y),
    n = n(),
    .groups = "drop"
  )

figA <- ggplot(calib_df, aes(x = p_mean, y = y_mean, color = model)) +
  geom_line() +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(x = "Mean predicted risk", y = "Observed event rate", title = "Calibration (10 bins)") +
  theme_minimal()

ggsave("results/figures/FigA_calibration.png", figA, width = 7, height = 5, dpi = 200)

# ---- 7) Fig B: Coverage–Risk curves (entropy as uncertainty for all models) ----
# coverage = fraction kept; abstain top-uncertainty
coverage_grid <- seq(0.1, 1.0, by = 0.05)

covrisk <- pred_all %>%
  group_by(model) %>%
  group_modify(~{
    df <- .x %>% arrange(entropy)  # low uncertainty first
    n <- nrow(df)
    out <- lapply(coverage_grid, function(cov){
      k <- max(1, floor(n * cov))
      kept <- df[1:k, , drop = FALSE]
      tibble(
        coverage = cov,
        k = k,
        risk_error = mean((kept$p >= 0.5) != kept$y),
        risk_logloss = logloss(kept$y, kept$p)
      )
    })
    bind_rows(out)
  }) %>%
  ungroup()

figB <- ggplot(covrisk, aes(x = coverage, y = risk_error, color = model)) +
  geom_line() +
  geom_point(size = 1) +
  scale_x_continuous(limits = c(0.1, 1.0)) +
  labs(x = "Coverage (keep least-uncertain)", y = "Error rate", title = "Coverage–Risk (entropy)") +
  theme_minimal()

ggsave("results/figures/FigB_coverage_risk.png", figB, width = 7, height = 5, dpi = 200)

# ---- 8) Fig C: Uncertainty vs error (deciles) ----
unc_df <- pred_all %>%
  group_by(model) %>%
  mutate(decile = ntile(entropy, 10)) %>%
  group_by(model, decile) %>%
  summarise(
    mean_entropy = mean(entropy),
    error_rate = mean((p >= 0.5) != y),
    mean_logloss = logloss(y, p),
    n = n(),
    .groups = "drop"
  )

figC <- ggplot(unc_df, aes(x = mean_entropy, y = mean_logloss, color = model)) +
  geom_line() +
  geom_point() +
  labs(x = "Mean entropy (by decile)", y = "Mean log loss", title = "Uncertainty vs error (deciles)") +
  theme_minimal()

ggsave("results/figures/FigC_uncertainty_vs_error.png", figC, width = 7, height = 5, dpi = 200)

# ---- 9) Fig D: DRF top neighbors explanation (from your existing CSV) ----
# Expect file results/top_neighbors_example.csv with columns like:
# train_row, weight, y_train, cum_weight
if (file.exists("results/top_neighbors_example.csv")) {
  neigh <- read_csv("results/top_neighbors_example.csv", show_col_types = FALSE)
  
  # show top 20 weights
  neigh_plot <- neigh %>%
    slice_head(n = 20) %>%
    mutate(train_row = factor(train_row, levels = rev(train_row))) # for readable bars
  
  figD <- ggplot(neigh_plot, aes(x = train_row, y = weight)) +
    geom_col() +
    coord_flip() +
    labs(x = "Training row (neighbor)", y = "DRF weight", title = "DRF explanation: top neighbors (weights)") +
    theme_minimal()
  
  ggsave("results/figures/FigD_top_neighbors.png", figD, width = 7, height = 5, dpi = 200)
}

cat("Done.\nSaved:\n",
    "- results/metrics_models.csv\n",
    "- results/figures/FigA_calibration.png\n",
    "- results/figures/FigB_coverage_risk.png\n",
    "- results/figures/FigC_uncertainty_vs_error.png\n",
    "- results/figures/FigD_top_neighbors.png (if results/top_neighbors_example.csv exists)\n", sep = "")
