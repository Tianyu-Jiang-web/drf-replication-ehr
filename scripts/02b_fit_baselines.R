# scripts/02b_fit_baselines.R
# Step 2b: fit baselines on the same split as DRF and save test predictions

library(readr)
library(dplyr)

# baselines
library(glmnet)
library(ranger)
library(xgboost)

set.seed(20260130)

# ---- 0) dirs ----
dir.create("models",  showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ---- 1) load split from Step 2 (must exist) ----
obj <- readRDS("models/drf_step2_fit.rds")
idx_train <- obj$idx_train
idx_test  <- obj$idx_test

# ---- 2) load data from Step 0 ----
X <- readRDS("data/processed/X.rds")
Y <- readRDS("data/processed/Y.rds")

X_train <- X[idx_train, , drop = FALSE]
X_test  <- X[idx_test,  , drop = FALSE]
y_train <- as.integer(Y[idx_train])
y_test  <- as.integer(Y[idx_test])

# Safety: ensure 0/1
stopifnot(all(y_train %in% c(0,1)))
stopifnot(all(y_test  %in% c(0,1)))

# ---- 2b) simple imputation for models that can't handle NA (glmnet/xgboost) ----
impute_train_test <- function(train_df, test_df) {
  train_df <- as.data.frame(train_df)
  test_df  <- as.data.frame(test_df)
  
  for (nm in names(train_df)) {
    if (is.numeric(train_df[[nm]]) || is.integer(train_df[[nm]])) {
      med <- median(train_df[[nm]], na.rm = TRUE)
      if (is.na(med)) med <- 0  # if a column is all-NA
      train_df[[nm]][is.na(train_df[[nm]])] <- med
      test_df[[nm]][is.na(test_df[[nm]])]   <- med
    } else {
      # character/factor -> add explicit MISSING level
      train_df[[nm]] <- as.character(train_df[[nm]])
      test_df[[nm]]  <- as.character(test_df[[nm]])
      train_df[[nm]][is.na(train_df[[nm]])] <- "MISSING"
      test_df[[nm]][is.na(test_df[[nm]])]   <- "MISSING"
      train_df[[nm]] <- factor(train_df[[nm]])
      # align test levels to train levels; unseen -> MISSING
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
      test_df[[nm]] <- addNA(test_df[[nm]])  # just in case
      test_df[[nm]][is.na(test_df[[nm]])] <- "MISSING"
      test_df[[nm]] <- factor(test_df[[nm]], levels = levels(train_df[[nm]]))
    }
  }
  list(train = train_df, test = test_df)
}

imp <- impute_train_test(X_train, X_test)
X_train_imp <- imp$train
X_test_imp  <- imp$test

# ---- 3) Logistic regression baseline (glmnet) ----
# model.matrix: factors -> one-hot, drop intercept
mm_train <- model.matrix(~ . - 1, data = X_train_imp, na.action = na.pass)
mm_test  <- model.matrix(~ . - 1, data = X_test_imp,  na.action = na.pass)

mm_train[is.na(mm_train)] <- 0
mm_test[is.na(mm_test)] <- 0

cvfit <- cv.glmnet(
  x = mm_train, y = y_train,
  family = "binomial",
  alpha = 0,                  # ridge (stable baseline)
  nfolds = 5,
  type.measure = "deviance"
)

p_glmnet <- as.numeric(predict(cvfit, newx = mm_test, s = "lambda.1se", type = "response"))

# ---- 4) Random Forest baseline (ranger) ----
# ranger can use factors directly
rf_fit <- ranger(
  dependent.variable.name = "y",
  data = data.frame(y = factor(y_train), X_train),
  probability = TRUE,
  num.trees = 2000,
  min.node.size = 20
)

rf_pred <- predict(rf_fit, data = X_test)$predictions
# probability of class "1" (factor levels are "0","1")
p_rf <- as.numeric(rf_pred[, "1"])

# ---- 5) XGBoost baseline ----
dtrain <- xgb.DMatrix(data = mm_train, label = y_train)
dtest  <- xgb.DMatrix(data = mm_test,  label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8
)

evals <- list(train = dtrain, test = dtest)

xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  evals = evals,
  verbose = 0
)

p_xgb <- as.numeric(predict(xgb_fit, dtest))

# ---- 6) save predictions for Step 4 ----
pred_df <- tibble(
  idx = idx_test,
  y_test = y_test,
  p_glmnet = p_glmnet,
  p_rf = p_rf,
  p_xgb = p_xgb
)

write_csv(pred_df, "results/pred_baselines.csv")

# optional: save fitted model objects
saveRDS(
  list(
    glmnet = cvfit,
    rf = rf_fit,
    xgb = xgb_fit
  ),
  "models/baselines_step2b.rds"
)

cat("Step 2b done.\nSaved:\n- results/pred_baselines.csv\n- models/baselines_step2b.rds\n")
