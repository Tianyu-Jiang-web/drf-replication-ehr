# scripts/01_sanity_check_toy.R
# Step 1: sanity check - DRF toy example

library(drf)

set.seed(20260130)

# ---- toy data ----
n <- 800
p <- 10

X <- matrix(runif(n * p, -1, 1), nrow = n, ncol = p)

# binary outcome with nonlinear signal
eta <- 1.2 * (X[, 1] > 0) - 0.8 * X[, 2] + 0.5 * sin(3 * X[, 3])
prob <- 1 / (1 + exp(-eta))
Y <- rbinom(n, size = 1, prob = prob)

# ---- fit ----
fit <- drf(
  X = X,
  Y = Y,
  num.trees = 300,
  splitting.rule = "FourierMMD"
)

# ---- predict on a few points ----
X_test <- matrix(runif(20 * p, -1, 1), nrow = 20, ncol = p)

pred <- predict(fit, newdata = X_test)

str(pred, max.level = 2)

cat("\nStep 1 OK: DRF toy fit + predict finished.\n")
