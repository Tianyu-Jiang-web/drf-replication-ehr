# scripts/03_compute_uncertainty.R
library(Matrix)
library(dplyr)

# 0) load model object saved in step 2
obj <- readRDS("models/drf_step2_fit.rds")
fit_drf   <- obj$fit
idx_test  <- obj$idx_test

# 1) load processed data (from step 0)
X <- readRDS("data/processed/X.rds")
Y <- readRDS("data/processed/Y.rds")

X_test <- X[idx_test, , drop = FALSE]

# 2) predict -> get weights for test set
pred <- predict(fit_drf, newdata = X_test)
W    <- pred$weights   # dgCMatrix: (n_test x n_train)

# 3) point prediction p_hat(x) = sum_i w_i(x) * y_i
# ensure Y is numeric vector 0/1
y_train <- as.numeric(pred$y)
p_hat <- as.numeric(W %*% y_train)

# 4) predictive entropy (binary)
eps <- 1e-12
p_clip <- pmin(pmax(p_hat, eps), 1 - eps)
entropy <- -(p_clip * log(p_clip) + (1 - p_clip) * log(1 - p_clip))

# 5) ESS = 1 / sum_i w_i^2  (row-wise)
# W is sparse: do rowSums(W^2) efficiently
w2_sum <- Matrix::rowSums(W^2)
ess <- 1 / w2_sum

# 6) (optional) Beta credible interval using n_eff = ESS
alpha0 <- 1
beta0  <- 1
a_post <- alpha0 + p_hat * ess
b_post <- beta0  + (1 - p_hat) * ess
ci_low  <- qbeta(0.025, a_post, b_post)
ci_high <- qbeta(0.975, a_post, b_post)

# 7) assemble outputs
y_true <- as.integer(Y[idx_test])

res <- tibble(
  idx = idx_test,
  y_true = y_true,
  p_hat = p_hat,
  entropy = entropy,
  ess = ess,
  ci_low = ci_low,
  ci_high = ci_high
)

# 8) save
dir.create("results", showWarnings = FALSE, recursive = TRUE)
saveRDS(res, "results/step3_predictions_uncertainty.rds")
write.csv(res, "results/step3_predictions_uncertainty.csv", row.names = FALSE)

cat("Step 3 done. Saved:\n",
    "- results/step3_predictions_uncertainty.rds\n",
    "- results/step3_predictions_uncertainty.csv\n")
