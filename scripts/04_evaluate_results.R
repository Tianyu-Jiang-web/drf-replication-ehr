# scripts/04_evaluate_results.R

library(dplyr)
library(Matrix)
library(pROC)
library(PRROC)
library(readr)

# ---- paths ----
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ---- load ----
obj <- readRDS("models/drf_step2_fit.rds")
fit_drf  <- obj$fit
idx_test <- obj$idx_test
idx_train <- obj$idx_train

X <- readRDS("data/processed/X.rds")
Y <- readRDS("data/processed/Y.rds")

X_test <- X[idx_test, , drop = FALSE]
y_test <- as.numeric(Y[idx_test])

# ---- predict weights ----
pred <- predict(fit_drf, newdata = X_test)
W <- pred$weights
y_train <- as.numeric(pred$y)

# ---- step3 quantities (recompute here for convenience) ----
p_hat <- as.numeric(W %*% y_train)
eps <- 1e-12
p_clip <- pmin(pmax(p_hat, eps), 1 - eps)

entropy <- -(p_clip * log(p_clip) + (1 - p_clip) * log(1 - p_clip))
ess <- 1 / Matrix::rowSums(W^2)

# =========================
# 1) Performance metrics
# =========================
auc <- as.numeric(pROC::auc(pROC::roc(y_test, p_hat, quiet = TRUE)))

# AUPRC (PRROC expects scores for positive + negative)
pr <- PRROC::pr.curve(
  scores.class0 = p_hat[y_test == 1],
  scores.class1 = p_hat[y_test == 0],
  curve = FALSE
)
auprc <- pr$auc.integral

logloss <- -mean(y_test * log(p_clip) + (1 - y_test) * log(1 - p_clip))
brier <- mean((p_hat - y_test)^2)

metrics <- tibble(
  AUROC = auc,
  AUPRC = auprc,
  LogLoss = logloss,
  Brier = brier,
  n_test = length(y_test),
  pos_rate = mean(y_test == 1)
)

write_csv(metrics, "results/metrics.csv")

# =========================
# 2) Calibration (10 bins)
# =========================
calib <- tibble(p_hat = p_hat, y = y_test) %>%
  mutate(bin = ntile(p_hat, 10)) %>%
  group_by(bin) %>%
  summarise(
    p_mean = mean(p_hat),
    y_rate = mean(y),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(abs_gap = abs(p_mean - y_rate))

# ECE (expected calibration error)
ece <- sum((calib$n / sum(calib$n)) * calib$abs_gap)
write_csv(calib, "results/calibration.csv")
write_csv(tibble(ECE = ece), "results/ece.csv")

# =========================
# 3) Selective prediction: coverage-risk
# =========================
# uncertainty score: larger = more uncertain
u <- entropy  # or use -ess / (1/ess), but entropy is simplest

ord <- order(u)  # from most certain to most uncertain
y_ord <- y_test[ord]
p_ord <- p_hat[ord]

# define coverage levels (keep top k most certain)
cover_grid <- seq(0.1, 1.0, by = 0.05)

covrisk <- lapply(cover_grid, function(cov) {
  k <- max(1, floor(cov * length(y_ord)))
  y_k <- y_ord[1:k]
  p_k <- p_ord[1:k]
  # risk = error rate (0/1 using 0.5 threshold)
  pred_k <- as.integer(p_k >= 0.5)
  risk <- mean(pred_k != y_k)
  tibble(coverage = cov, k = k, risk = risk)
}) %>% bind_rows()

write_csv(covrisk, "results/coverage_risk.csv")

# =========================
# 4) Neighbourhood explanation (one example)
# =========================
# pick one test sample (e.g., the first in idx_test)
t <- 1
w_t <- as.numeric(W[t, ])  # weights over training set
top_idx <- order(w_t, decreasing = TRUE)[1:20]

neighbors <- tibble(
  train_row = top_idx,
  weight = w_t[top_idx],
  y_train = y_train[top_idx]
) %>%
  mutate(cum_weight = cumsum(weight))

write_csv(neighbors, "results/top_neighbors_example.csv")

cat("Step 4 done. Files saved in results/:\n")
cat("- metrics.csv, ece.csv\n")
cat("- calibration.csv\n")
cat("- coverage_risk.csv\n")
cat("- top_neighbors_example.csv\n")
