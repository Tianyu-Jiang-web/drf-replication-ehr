# ------------------------------------------------------------
# 02_fit_drf.R
#
# Step 2: Fit Distributional Random Forest (DRF)
# - Input: processed X / Y from Step 0
# - Output: fitted DRF model and test-set weights
#
# This script performs ONE DRF fit with a fixed, reproducible
# configuration (no hyperparameter tuning).
# ------------------------------------------------------------

library(drf)
library(Matrix)

# ---- load processed data ----
X <- readRDS("data/processed/X.rds")
Y <- readRDS("data/processed/Y.rds")

stopifnot(nrow(X) == length(Y))

set.seed(20260130)

n <- nrow(X)
idx_train <- sample(seq_len(n), size = floor(0.8 * n))
idx_test  <- setdiff(seq_len(n), idx_train)

X_train <- X[idx_train, ]
Y_train <- Y[idx_train]

X_test  <- X[idx_test, ]
Y_test  <- Y[idx_test]

set.seed(20260130)

fit_drf <- drf(
  X = X_train,
  Y = Y_train,
  splitting.rule = "FourierMMD",  # 论文主推
  num.trees = 1000,               # 复现友好 & 稳定
  min.node.size = 15              # 防止权重过尖
)

pred_test <- predict(fit_drf, newdata = X_test)

W_test <- pred_test$weights   # 稀疏权重矩阵
y_train <- pred_test$y        # 训练标签（顺序对齐）

dim(W_test)
# [ n_test , n_train ]

summary(Matrix::rowSums(W_test != 0))

saveRDS(
  list(
    fit = fit_drf,
    W_test = W_test,
    y_train = y_train,
    idx_train = idx_train,
    idx_test = idx_test
  ),
  file = "models/drf_step2_fit.rds"
)
