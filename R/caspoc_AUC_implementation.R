
#' Deflation of data
#'
#' This is an internal function used to deflate data across train, tune and test sets using training loadings. Only used if ncomp > 1
#'
#' @importFrom magrittr %>%
#' @export
deflate_sPLS_data <- function(splsModel, ncomp, trainX, trainY, tuneX, tuneY, testX, testY) {
  # This function takes the sPLS model and manually calculates the deflation of the input data for the 2nd and subsequent components by using the training loadings
  # Initialize some variables

  trainLoadingsX <- vector("list", ncomp)
  trainLoadingsY <- vector("list", ncomp)

  trainScoresX <- vector("list", ncomp)
  trainScoresY <- vector("list", ncomp)

  tuneScoresX <- vector("list", ncomp)
  tuneScoresY <- vector("list", ncomp)

  testScoresX <- vector("list", ncomp)
  testScoresY <- vector("list", ncomp)

  splsModel = splsModel
  ncomp = ncomp
  trainX = trainX
  trainY = trainY
  tuneX = tuneX
  tuneY = tuneY
  testX = testX
  testY = testY

  dfl_train <- vector("list", ncomp)
  dfl_tune <- vector("list", ncomp)
  dfl_test <- vector("list", ncomp)

  # save training loadings to be used later for calculating scores
  for (z in 1:ncomp){
    trainLoadingsX[[z]] <- splsModel$loadings$X[,z]
    trainLoadingsY[[z]] <- splsModel$loadings$Y[,z]
  }

  # our first data matrix when calculating component 1 is just our centered scaled data:
  dfl_train[[1]] <- list(X = trainX, Y = trainY)
  dfl_tune[[1]] <- list(X = tuneX, Y = tuneY)
  dfl_test[[1]] <- list(X = testX, Y = testY)

  # similarly, the first component SCORES are calculated from the centered scaled data, therefore we calculate them here
  trainScoresX[[1]] <- as.matrix(dfl_train[[1]]$X) %*% trainLoadingsX[[1]]
  trainScoresY[[1]] <- as.matrix(dfl_train[[1]]$Y) %*% trainLoadingsY[[1]]

  tuneScoresX[[1]] <- as.matrix(dfl_tune[[1]]$X) %*% trainLoadingsX[[1]]
  tuneScoresY[[1]] <- as.matrix(dfl_tune[[1]]$Y) %*% trainLoadingsY[[1]]

  testScoresX[[1]] <- as.matrix(dfl_test[[1]]$X) %*% trainLoadingsX[[1]]
  testScoresY[[1]] <- as.matrix(dfl_test[[1]]$Y) %*% trainLoadingsY[[1]]
  # subsequent scores will be calculated on deflated data, therefore we must first calculate the deflation!

  # If there is more than one component
  if (ncomp != 1) {
    # start deflation of the subsequent components (for tuning and testing)
    for (j in 2:ncomp){

      # deflate training data
      t <- trainScoresX[[j-1]]
      u <- trainScoresY[[j-1]]
      c <- crossprod(dfl_train[[j-1]]$X, t)/sum(t^2)
      d <- crossprod(dfl_train[[j-1]]$Y, t)/sum(t^2)
      Xdfl_train <- dfl_train[[j-1]]$X - tcrossprod(t, c)
      Ydfl_train <- dfl_train[[j-1]]$Y - tcrossprod(t, d)

      dfl_train[[j]] <- list( # store deflated
        X = Xdfl_train,
        Y = Ydfl_train
      )

      # calculate training scores
      trainScoresX[[j]] <- as.matrix(dfl_train[[j]]$X) %*% as.matrix(trainLoadingsX[[j]])
      trainScoresY[[j]] <- as.matrix(dfl_train[[j]]$Y) %*% as.matrix(trainLoadingsY[[j]])

      # deflate tuning data
      tt <- tuneScoresX[[j-1]]
      ut <- tuneScoresY[[j-1]]
      ct <- crossprod(dfl_tune[[j-1]]$X, tt)/sum(tt^2)
      dt <- crossprod(dfl_tune[[j-1]]$Y, tt)/sum(tt^2)
      Xdfl_tune <- dfl_tune[[j-1]]$X - tcrossprod(tt, ct)
      Ydfl_tune <- dfl_tune[[j-1]]$Y - tcrossprod(tt, dt)

      dfl_tune[[j]] <- list(
        X = Xdfl_tune,
        Y = Ydfl_tune
      )

      # calculate predicted tune scores!
      tuneScoresX[[j]] <- as.matrix(dfl_tune[[j]]$X) %*% as.matrix(trainLoadingsX[[j]])
      tuneScoresY[[j]] <- as.matrix(dfl_tune[[j]]$Y) %*% as.matrix(trainLoadingsY[[j]])

      # deflate testing data
      tt2 <- testScoresX[[j-1]]
      ut2 <- testScoresY[[j-1]]
      ct2 <- crossprod(dfl_test[[j-1]]$X, tt2)/sum(tt2^2)
      dt2 <- crossprod(dfl_test[[j-1]]$Y, tt2)/sum(tt2^2)
      Xdfl_test <- dfl_test[[j-1]]$X - tcrossprod(tt2, ct2)
      Ydfl_test <- dfl_test[[j-1]]$Y - tcrossprod(tt2, dt2)

      dfl_test[[j]] <- list(
        X = Xdfl_test,
        Y = Ydfl_test
      )

      # calculate predicted tune scores!
      testScoresX[[j]] <- as.matrix(dfl_test[[j]]$X) %*% as.matrix(trainLoadingsX[[j]])
      testScoresY[[j]] <- as.matrix(dfl_test[[j]]$Y) %*% as.matrix(trainLoadingsY[[j]])
    }
  }

  return(list(trainLoadingsX = trainLoadingsX,
              trainLoadingsY = trainLoadingsY,
              trainScoresX = trainScoresX,
              trainScoresY = trainScoresY,
              tuneScoresX = tuneScoresX,
              tuneScoresY = tuneScoresY,
              testScoresX = testScoresX,
              testScoresY = testScoresY))
}

#' CASPOC
#'
#' This is the main function in the caspoc package. It takes two matrices \(or vectors\), performs repeated K-fold cross-validation, and returns tune and test sets with all samples separately
#'
#' @param X A matrix or vector
#' @param Y A matrix or vector
#' @param ncomp Number of components in the model
#' @param numRepeats Number of repeats for the cross-validation
#' @param numFolds Number of folds for the cross-validation
#' @param keepX_options Tune grid options for the keepX parameter - the number of included variables from X
#' @param keepY_options Tune grid options for the keepY parameter - the number of included variables from Y
#' @param fixX A vector of keepX values for each component; if you want to fix eg. keepX = 10 for comp1 but want to do a grid search on comp2. Leave as NULL for grid search on all components
#' @param fixY A vector of keepY values for each component; if you want to fix eg. keepY = 10 for comp1 but want to do a grid search on comp2. Leave as NULL for grid search on all components
#' @param base_seed Random seed for reproducibility. Use instead of 'set.seed()', since the function internally updates the seed between repeats.
#' @param manual_folds Manually supply folds. Should be a list of lists. Outer list should be of length numRepeats. Inner list should be of length numFolds and contain integer vectors supplying row indices for each fold.
#' @param sign_flipping A boolean option for automatic alignment of signs in the output, attempting to resolve sign ambiguity from the sPLS using a PCA method. This will only be done for the significant associations between X and Y. A flip summary and log will be returned. Default is TRUE.
#' @param binary A boolean option for continuous univariate or multivariate analysis (FALSE) or binary univarite analysis (TRUE). Default is FALSE.
#' @return A list containing several elements:
#' \describe{
#'   \item{results_tune_df}{A data.frame with correlation results for each repeat and hyperparameter combination from the tuning folds}
#'   \item{results_test_df}{A data.frame with correlation results for each repeat and hyperparameter combination from the testing folds}
#'   \item{full_train_loadingsX}{A data.frame with X loadings for each variable in each fold, repeat and hyperparameter combination based on training folds}
#'   \item{full_train_loadingsY}{A data.frame with Y loadings for each variable in each fold, repeat and hyperparameter combination based on training folds}
#'   \item{full_tuneX}{A data.frame with all X component scores for each repeat and hyperparameter combination from the tuning folds}
#'   \item{full_tuneY}{A data.frame with all Y component scores for each repeat and hyperparameter combination from the tuning folds}
#'   \item{full_testX}{A data.frame with all X component scores for each repeat and hyperparameter combination from the testing folds}
#'   \item{full_testY}{A data.frame with all Y component scores for each repeat and hyperparameter combination from the testing folds}
#'   \item{folds}{Return the list of folds for CV}
#'   \item{flip_summary_df}{Return summary data.frame indicating whether sign flipping was detected for each significant tuning combination}
#'   \item{flip_log_df}{Return detailed log showing which repeat, fold, keepX/keepY combination, and component was flipped to resolve sign ambiguity}
#'   }
#' @examples
#' library(mixOmics)
#' data("breast.TCGA")
#' mydata_X <- breast.TCGA$data.train$mrna
#' mydata_Y <- breast.TCGA$data.train$protein
#' dim(mydata_X)
#' dim(mydata_Y)
#' #my_analysis <- CASPOC(mydata_X, mydata_Y, numRepeats = 5, numFolds = 10,
#' #   keepX_options = c(50, 100, 200), keepY_options = c(50, 100, 142),
#' #   ncomp = 1, base_seed = 42)
#' @importFrom magrittr %>%
#' @export
CASPOC <- function (X, Y, ncomp = 1, numRepeats = 11, numFolds = 10, keepX_options = NULL, keepY_options = NULL, fixX = NULL, fixY = NULL, base_seed = 1, manual_folds = NULL, sign_flipping = TRUE, binary = FALSE) {
  if(!requireNamespace("dplyr", quietly = TRUE))
    stop("dplyr package required")
  if(!requireNamespace("tibble", quietly = TRUE))
    stop("tibble package required")
  if(!requireNamespace("mixOmics", quietly = TRUE))
    stop("mixOmics package required")
  if(!requireNamespace("caret", quietly = TRUE))
    stop("caret package required")
  if (!requireNamespace("PRROC", quietly = TRUE)) {
    stop("PRROC package required")
  }
  # if(!requireNamespace("PRROC", quietly = TRUE))
  #   stop("abind package required")

  # Utility functions
  arr3d_to_df <- function(x, comp_prefix = "comp") {
    d <- dim(x)
    stopifnot(length(d) == 3)

    out <- lapply(seq_len(d[3]), function(k) {
      df <- as.data.frame(x[, , k, drop = FALSE][, , 1])
      if(!is.null(colnames(Y))){
        colnames(df) <- colnames(Y)
      } else {
        colnames(df) <- paste0("Y", sprintf(paste0("%0", nchar(ncol(Y)), "d"), 1:ncol(Y)))
      }
      df$Component <- paste0(comp_prefix, k)
      df
    })

    do.call(rbind, out)
  }
  
  # RMSE and R2 calculation function for continuous outcome (univariate and multivariate)
  calc_prediction_metrics_by_component <- function(yhat_list, obsY_list) {

    yhat_df <- do.call(rbind, yhat_list)
    names(yhat_df)[names(yhat_df) == "fold"] <- "Fold"
    
    meta_cols <- c("Component", "Fold", "SampleID", "SampleIndex")
    y_cols <- setdiff(colnames(yhat_df), meta_cols)
    
    obs_colnames <- colnames(as.data.frame(obsY_list[[1]]))
    y_cols <- intersect(y_cols, obs_colnames)
    
    if (length(y_cols) == 0) {
      stop("No matching Y outcome columns found between yhat_list and obsY_list.")
    }
    
    out <- list()
    counter <- 1
    
    for (comp_name in unique(yhat_df$Component)) {
      
      comp_pred <- yhat_df[yhat_df$Component == comp_name, , drop = FALSE]
      comp_num <- as.integer(gsub("comp", "", comp_name))
      
      obs_matrix_all <- NULL
      pred_matrix_all <- NULL
      
      for (fold_id in seq_along(obsY_list)) {
        
        pred_fold <- comp_pred[comp_pred$Fold == fold_id, y_cols, drop = FALSE]
        obs_fold <- as.data.frame(obsY_list[[fold_id]])[, y_cols, drop = FALSE]
        
        pred_fold <- as.matrix(pred_fold)
        obs_fold <- as.matrix(obs_fold)
        
        storage.mode(pred_fold) <- "numeric"
        storage.mode(obs_fold) <- "numeric"
        
        if (!all(dim(pred_fold) == dim(obs_fold))) {
          stop(
            paste0(
              "Prediction and observation dimension mismatch for ",
              comp_name, ", fold ", fold_id,
              ": dim(pred_fold) = ", paste(dim(pred_fold), collapse = " x "),
              ", dim(obs_fold) = ", paste(dim(obs_fold), collapse = " x ")
            )
          )
        }
        
        obs_matrix_all <- rbind(obs_matrix_all, obs_fold)
        pred_matrix_all <- rbind(pred_matrix_all, pred_fold)
      }
      
      ss_res <- sum((obs_matrix_all - pred_matrix_all)^2, na.rm = TRUE)
      
      obs_centered <- sweep(
        obs_matrix_all,
        2,
        colMeans(obs_matrix_all, na.rm = TRUE),
        FUN = "-"
      )
      
      ss_tot <- sum(obs_centered^2, na.rm = TRUE)
      
      out[[counter]] <- data.frame(
        Component = comp_num,
        YPred_RMSE = sqrt(mean((obs_matrix_all - pred_matrix_all)^2, na.rm = TRUE)),
        YPred_R2 = 1 - (ss_res / ss_tot)
      )
      
      counter <- counter + 1
    }
    
    do.call(rbind, out)
  }
  
  # AUC calculation function for binary outcome
  # AUC calculation function for binary outcome
  calc_binary_metrics_by_component <- function(yhat_list, obsY_list, positive_class = 1) {
    
    yhat_df <- do.call(rbind, yhat_list)
    names(yhat_df)[names(yhat_df) == "fold"] <- "Fold"
    
    meta_cols <- c("Component", "Fold", "SampleID", "SampleIndex")
    y_cols <- setdiff(colnames(yhat_df), meta_cols)
    
    obs_colnames <- colnames(as.data.frame(obsY_list[[1]]))
    y_cols <- intersect(y_cols, obs_colnames)
    
    if (length(y_cols) == 0) {
      stop("No matching Y outcome columns found between yhat_list and obsY_list.")
    }
    
    out <- list()
    counter <- 1
    
    for (comp_name in unique(yhat_df$Component)) {
      
      comp_pred <- yhat_df[yhat_df$Component == comp_name, , drop = FALSE]
      comp_num <- as.integer(gsub("comp", "", comp_name))
      
      auc_values <- numeric(0)
      auprc_values <- numeric(0)
      
      for (y_col in y_cols) {
        
        obs_all <- numeric(0)
        score_all <- numeric(0)
        
        for (fold_id in seq_along(obsY_list)) {
          
          score_fold <- comp_pred[comp_pred$Fold == fold_id, y_col, drop = TRUE]
          obs_fold <- as.data.frame(obsY_list[[fold_id]])[, y_col, drop = TRUE]
          
          score_fold <- as.numeric(score_fold)
          obs_fold <- as.numeric(obs_fold)
          
          if (length(score_fold) != length(obs_fold)) {
            stop(
              paste0(
                "Prediction and observation length mismatch for ",
                comp_name, ", ", y_col, ", fold ", fold_id,
                ": length(score_fold) = ", length(score_fold),
                ", length(obs_fold) = ", length(obs_fold)
              )
            )
          }
          
          obs_all <- c(obs_all, obs_fold)
          score_all <- c(score_all, score_fold)
        }
        
        keep <- stats::complete.cases(obs_all, score_all)
        obs_all <- obs_all[keep]
        score_all <- score_all[keep]
        
        if (length(unique(obs_all)) != 2) {
          auc_val <- NA_real_
        } else {
          
          obs_binary <- ifelse(obs_all == positive_class, 1, 0)
          
          if (length(unique(obs_binary)) != 2) {
            auc_val <- NA_real_
          } else {
            roc_obj <- pROC::roc(
              response = obs_binary,
              predictor = score_all,
              levels = c(0, 1),
              direction = "<",
              quiet = TRUE
            )
            
            auc_val <- as.numeric(pROC::auc(roc_obj))
          }
          if (length(unique(obs_binary)) != 2) {
            auprc_val <- NA_real_
          } else {
            pr_obj <- PRROC::pr.curve(
              scores.class0 = score_all[obs_binary == 1],
              scores.class1 = score_all[obs_binary == 0],
              curve = FALSE
            )
            
            auprc_val <- pr_obj$auc.integral
          }
        }
        
        auc_values <- c(auc_values, auc_val)
        auprc_values <- c(auprc_values, auprc_val)
      }
      
      auc_val_final <- mean(auc_values, na.rm = TRUE)
      auprc_val_final <- mean(auprc_values, na.rm = TRUE)
      
      if (is.nan(auc_val_final)) {
        auc_val_final <- NA_real_
      }
      
      if (is.nan(auprc_val_final)) {
        auprc_val_final <- NA_real_
      }
      
      out[[counter]] <- data.frame(
        Component = comp_num,
        Ypred_AUC = auc_val_final,
        Ypred_AUPRC = auprc_val_final
      )
      
      counter <- counter + 1
    }
    
    do.call(rbind, out)
  }

  # Some safeguard error messages
  if(missing(X)) {
    stop("Error: X data not provided.")
  }
  if(missing(Y)) {
    stop("Error: Y data not provided.")
  }
  if(dim(X)[1] != dim(Y)[1]) {
    stop("Error: X & Y must have the same number of samples.")
  }
  if(!is.numeric(numRepeats) || numRepeats %% 2 == 0) {
    stop("Error: 'numRepeats' must be an odd number.")
  }
  if(!is.numeric(numFolds)) {
    stop("Error: 'numFolds' must be numerical.")
  }
  if(!is.numeric(ncomp)) {
    stop("Error: 'ncomp' must be numerical.")
  }
  if (!is.logical(sign_flipping) || length(sign_flipping) != 1L || is.na(sign_flipping)) {
    stop("Error: 'sign_flipping' must be TRUE/FALSE.")
  }
  if (!is.logical(binary) || length(binary) != 1L || is.na(binary)) {
    stop("Error: 'binary' must be TRUE/FALSE.")
  }
  if (binary == TRUE && !requireNamespace("pROC", quietly = TRUE)) {
    stop("pROC package required when binary = TRUE.")
  }
  if (binary == TRUE && !requireNamespace("PRROC", quietly = TRUE)) {
    stop("PRROC package required when binary = TRUE.")
  }
  if (binary == TRUE) {
    
    y_vals <- unique(as.vector(Y))
    y_vals <- y_vals[!is.na(y_vals)]
    
    if (!all(y_vals %in% c(0, 1))) {
      stop("Error: when binary = TRUE, Y must be coded only as 0 and 1.")
    }
    
    if (length(y_vals) != 2) {
      stop("Error: when binary = TRUE, Y must contain both classes: 0 and 1.")
    }
  }

  cat("Dimensions of data:\n")
  cat(sprintf("  X dimensions: %d x %d\n", dim(X)[1], dim(X)[2]))
  cat(sprintf("  Y dimensions: %d x %d\n", dim(Y)[1], dim(Y)[2]))

  cat("\nHyperparameters:\n")
  cat(sprintf("  ncomp       = %d\n", ncomp))
  cat(sprintf("  numRepeats  = %d\n", numRepeats))
  cat(sprintf("  numFolds    = %d\n", numFolds))

  if(!is.null(fixX)) {
    cat(sprintf("  fixX    = %d\n", fixX))
  }
  if(!is.null(fixY)) {
    cat(sprintf("  fixY    = %d\n", fixY))
  }

  if (is.null(keepX_options)) {
    p <- dim(X)[2]  # Number of features in X

    if (p <= 10) {
      keepX_options <- seq(1, p, by = 1)
    } else {
      step_size <- ceiling(p / 10)  # Get whole-number step size
      steps <- seq(0, p, by = step_size)  # start from 0
      steps[1] <- 1  # Replace 0 with 1 for initial step
      if (utils::tail(steps, 1) != p) steps <- c(steps, p)  # ensure p is included
      keepX_options <- unique(steps)
    }

    cat("keepX_options was not supplied. Using automatically generated sequence based on dimensions of X:\n")
  }

  cat("  keepX_options =", paste(keepX_options, collapse = ", "), "\n")

  if (is.null(keepY_options)) {
    p <- dim(Y)[2]  # Number of features in X

    if (p <= 10) {
      keepY_options <- seq(1, p, by = 1)
    } else {
      step_size <- ceiling(p / 10)  # Get whole-number step size
      steps <- seq(0, p, by = step_size)  # start from 0
      steps[1] <- 1  # Replace 0 with 1 for initial step
      if (utils::tail(steps, 1) != p) steps <- c(steps, p)  # ensure p is included
      keepY_options <- unique(steps)
    }

    cat("keepY_options was not supplied. Using automatically generated sequence based on dimensions of X:\n")
  }

  cat("  keepY_options =", paste(keepY_options, collapse = ", "), "\n")
  
  cat(sprintf("  binary      = %s\n\n", binary))
  
  # So we don't lose the sample ID
  sample_id <- rownames(X)
  sample_index <- seq_len(nrow(X))
  
  if (is.null(sample_id)) {
    sample_id <- paste0("sample_", sample_index)
  }

  # Ensure data are matrices
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  # Ensure Y has column names.
  # This is important because yhat and observed Y are matched by column name.
  if (is.null(colnames(Y))) {
    colnames(Y) <- paste0("Y", sprintf(paste0("%0", nchar(ncol(Y)), "d"), seq_len(ncol(Y))))
  }

  # Register the input arguments
  ncomp = ncomp
  numRepeats = numRepeats
  numFolds = numFolds

  # Initialize folds
  if(is.null(manual_folds)){
    folds <- vector("list", length = numRepeats)
  }
  if(!is.null(manual_folds)){
    if(!is.list(manual_folds) | !is.list(manual_folds[[1]])){
      stop("manual_folds should be a list of lists")
    }
    if(length(manual_folds) != numRepeats){
      stop("Length of outer list in manual_folds should be equal to numRepeats")
    }
    if(length(manual_folds[[1]]) != numFolds){
      stop("Length of outer list in manual_folds should be equal to numFolds")
    }
    folds <- manual_folds
  }

  # Notification of sign flipping on/off
  if(sign_flipping == TRUE) {
    cat(sprintf("Output will be checked for sign flipping ambiguity, and will attempt to automatically align the signs of the output using PCA.\n\n"))
  } else {
    cat(sprintf("Note: Output will not be checked for sign flipping ambiguity. Please verify manually if sign flipping occured by viewing the loadings.\n\n"))
  }
  
  # Initialize dataframe to store results
  results_tune_df <- data.frame()
  results_test_df <- data.frame()
  full_tuneX <- data.frame()
  full_tuneY <- data.frame()
  full_testX <- data.frame()
  full_testY <- data.frame()
  full_train_loadingsX <- data.frame()
  full_train_loadingsY <- data.frame()
  yhat_tune <- list()
  yhat_test <- list()
  full_yhat_tune <- data.frame() # array(numeric(), dim = c(0, ncol(Y), ncomp))
  full_yhat_test <- data.frame() # array(numeric(), dim = c(0, ncol(Y), ncomp))
  # full_exp_var_tuneX <- data.frame()
  # full_exp_var_tuneY <- data.frame()
  # full_variates_trainX <- data.frame()
  # full_variates_trainY <- data.frame()

  cat("Performing CASPOC\n")

  # Start of algorithm
  for (rep in 1:numRepeats) {
    # Set seed for stochastic variation and reproducibility for each repeat
    set.seed(base_seed + rep)

    if(is.null(manual_folds)){
      # Split the data into random folds within this repeat
      folds[[rep]] <- caret::createFolds(seq_len(nrow(X)), k = numFolds, list = TRUE)
    }

    # Running through every combination of keepX and keepY
    for (x in keepX_options) {
      for (y in keepY_options) {
        
        # Temporary variables for storing predicted Y
        yhat_tune <- vector("list", numFolds)
        yhat_test <- vector("list", numFolds)

        # Temporary variables for storing data
        concatenated_tuneX <- vector("list", numFolds)
        concatenated_tuneY <- vector("list", numFolds)
        concatenated_testX <- vector("list", numFolds)
        concatenated_testY <- vector("list", numFolds)
        concatenated_trainX_loadings <- vector("list", numFolds)
        concatenated_trainY_loadings <- vector("list", numFolds)
        # concatenated_exp_varX <- vector("list", numFolds)
        # concatenated_exp_varY <- vector("list", numFolds)
        # concatenated_train_variatesX <- vector("list", numFolds)
        # concatenated_train_variatesY <- vector("list", numFolds)
        observed_tuneY <- vector("list", numFolds)
        observed_testY <- vector("list", numFolds)

        

        # Beginning of cross-validation
        for (i in 1:numFolds) {
          # Define indices for training, tuning, and testing
          tuneIdx <- folds[[rep]][[i]]
          testIdx <- folds[[rep]][[(i %% numFolds) + 1]]  # Ensure the index cycles correctly
          trainIdx <- unlist(folds[[rep]][-c(i, (i %% numFolds) + 1)])  # Exclude tuning and testing fold
          
          # Index IDs for train/tune/test
          trainSampleID <- sample_id[trainIdx]
          tuneSampleID <- sample_id[tuneIdx]
          testSampleID <- sample_id[testIdx]
          
          trainSampleIndex <- sample_index[trainIdx]
          tuneSampleIndex <- sample_index[tuneIdx]
          testSampleIndex <- sample_index[testIdx]

          # Split data
          trainX <- X[trainIdx, ]
          trainY <- Y[trainIdx, ]


          # Scale to store preprocess parameters
          trainX <- scale(trainX, center = T)
          trainY <- scale(trainY, center = T)

          # Apply the training and tuning preprocess parameters to tuning and testing
          tuneX <- scale(X[tuneIdx, ], center=attr(trainX, "scaled:center"), scale=attr(trainX, "scaled:scale"))
          tuneY <- scale(Y[tuneIdx, ], center=attr(trainY, "scaled:center"), scale=attr(trainY, "scaled:scale"))
          testX <- scale(X[testIdx, ], center=attr(trainX, "scaled:center"), scale=attr(trainX, "scaled:scale"))
          testY <- scale(Y[testIdx, ], center=attr(trainY, "scaled:center"), scale=attr(trainY, "scaled:scale"))

          # tuneX <- X[tuneIdx, ]
          # tuneY <- Y[tuneIdx, ]
          # testX <- X[testIdx, ]
          # testY <- Y[testIdx, ]

          # initialize lists to hold
          # - X and Y loadings (from training)
          # - X and Y scores (variates) from training as well as tuning
          # - (deflated) data (training and tuning)

          # Construct keepX vector for current combination of x
          KeepX <- if (!is.null(fixX)) {
            c(fixX, rep(x, ncomp - length(fixX)))
          } else {
            rep(x, ncomp)
          }

          # Construct keepY vector for current combination of y
          KeepY <- if (!is.null(fixY)) {
            c(fixY, rep(y, ncomp - length(fixY)))
          } else {
            rep(y, ncomp)
          }

          # Run model
          splsModel <- mixOmics::spls(trainX, trainY, mode = "regression", keepX = KeepX, keepY = KeepY, ncomp = ncomp, all.outputs = T, scale = T)

          # Deflate
          sPLS_scores <- deflate_sPLS_data(splsModel, ncomp, trainX, trainY, tuneX, tuneY, testX, testY)

          # Retrieve the output from deflation
          trainScoresX <- do.call(cbind, sPLS_scores$trainScoresX)
          trainScoresY <- do.call(cbind, sPLS_scores$trainScoresY)
          tuneScoresX <- as.data.frame(do.call(cbind, sPLS_scores$tuneScoresX))
          tuneScoresY <- as.data.frame(do.call(cbind, sPLS_scores$tuneScoresY))
          testScoresX <- as.data.frame(do.call(cbind, sPLS_scores$testScoresX))
          testScoresY <- as.data.frame(do.call(cbind, sPLS_scores$testScoresY))
          trainLoadingsX <- as.data.frame(do.call(cbind, sPLS_scores$trainLoadingsX))
          trainLoadingsY <- as.data.frame(do.call(cbind, sPLS_scores$trainLoadingsY))

          # Store variates for sanity
          # variates_train_X <- as.data.frame(splsModel$variates$X)
          # variates_train_Y <- as.data.frame(splsModel$variates$Y)

          # tuneScoresX <- as.data.frame(tuneScoresX)
          # tuneScoresY <- as.data.frame(tuneScoresY)
          # testScoresX <- as.data.frame(testScoresX)
          # testScoresY <- as.data.frame(testScoresY)
          # trainLoadingsX <- as.data.frame(trainLoadingsX)
          # trainLoadingsY <- as.data.frame(trainLoadingsY)

          # Rename columns
          names(tuneScoresX) <- c(paste0("comp", 1:ncomp))
          names(tuneScoresY) <- c(paste0("comp", 1:ncomp))
          names(testScoresX) <- c(paste0("comp", 1:ncomp))
          names(testScoresY) <- c(paste0("comp", 1:ncomp))
          names(trainLoadingsX) <- c(paste0("comp", 1:ncomp))
          names(trainLoadingsY) <- c(paste0("comp", 1:ncomp))

          # The following lines of code documents where everything is coming from (i.e., repeat, combinations of keepX and KeepY, fold, and the variables for training)

          tuneScoresX$Repeat <- rep
          tuneScoresY$Repeat <- rep
          testScoresX$Repeat <- rep
          testScoresY$Repeat <- rep
          trainLoadingsX$Repeat <- rep
          trainLoadingsY$Repeat <- rep
          # exp_var_tuneX$Repeat <- rep
          # exp_var_tuneY$Repeat <- rep
          # variates_train_X$Repeat <- rep
          # variates_train_Y$Repeat <- rep

          tuneScoresX$KeepX <- x
          tuneScoresX$KeepY <- y
          tuneScoresY$KeepX <- x
          tuneScoresY$KeepY <- y
          testScoresX$KeepX <- x
          testScoresX$KeepY <- y
          testScoresY$KeepX <- x
          testScoresY$KeepY <- y
          trainLoadingsX$KeepX <- x
          trainLoadingsX$KeepY <- y
          trainLoadingsY$KeepX <- x
          trainLoadingsY$KeepY <- y
          # exp_var_tuneX$KeepX <- x
          # exp_var_tuneX$KeepY <- y
          # exp_var_tuneY$KeepX <- x
          # exp_var_tuneY$KeepY <- y
          # variates_train_X$KeepX <- x
          # variates_train_X$KeepY <- y
          # variates_train_Y$KeepX <- x
          # variates_train_Y$KeepY <- y

          tuneScoresX$Fold <- i
          tuneScoresY$Fold <- i
          testScoresX$Fold <- i
          testScoresY$Fold <- i
          trainLoadingsX$Fold <- i
          trainLoadingsY$Fold <- i
          # exp_var_tuneX$Fold <- i
          # exp_var_tuneY$Fold <- i
          # variates_train_X$Fold <- i
          # variates_train_Y$Fold <- i
          
          # Add sample IDs to tuning/testing scores
          tuneScoresX$SampleID <- tuneSampleID
          tuneScoresY$SampleID <- tuneSampleID
          testScoresX$SampleID <- testSampleID
          testScoresY$SampleID <- testSampleID
          tuneScoresX$SampleIndex <- tuneSampleIndex
          tuneScoresY$SampleIndex <- tuneSampleIndex
          testScoresX$SampleIndex <- testSampleIndex
          testScoresY$SampleIndex <- testSampleIndex

          trainLoadingsX$Variable <- (trainLoadingsX %>% tibble::rownames_to_column("Variable") %>% dplyr::select(Variable))$Variable
          trainLoadingsY$Variable <- (trainLoadingsY %>% tibble::rownames_to_column("Variable") %>% dplyr::select(Variable))$Variable
          # exp_var_tuneX$Variable <- (exp_var_tuneX %>% rownames_to_column("comp") %>% dplyr::select(comp))$comp
          # exp_var_tuneY$Variable <- (exp_var_tuneY %>% rownames_to_column("comp") %>% dplyr::select(comp))$comp



          # Store predictions for each fold
          concatenated_tuneX[[i]] <- tuneScoresX
          concatenated_tuneY[[i]] <- tuneScoresY
          concatenated_testX[[i]] <- testScoresX
          concatenated_testY[[i]] <- testScoresY

          concatenated_trainX_loadings[[i]] <- trainLoadingsX
          concatenated_trainY_loadings[[i]] <- trainLoadingsY
          
          if (binary == TRUE) {
            observed_tuneY[[i]] <- as.matrix(Y[tuneIdx, , drop = FALSE])
            observed_testY[[i]] <- as.matrix(Y[testIdx, , drop = FALSE])
          } else {
            observed_tuneY[[i]] <- tuneY
            observed_testY[[i]] <- testY
          }

          # concatenated_exp_varX[[i]] <- exp_var_tuneX
          # concatenated_exp_varY[[i]] <- exp_var_tuneY
          #
          # concatenated_train_variatesX[[i]] <- variates_train_X
          # concatenated_train_variatesY[[i]] <- variates_train_Y

          # Store yhat for each fold
          predict_mixOmics_pls <- getS3method("predict", "mixo_spls")
          yhat_tune[[i]] <- arr3d_to_df(predict_mixOmics_pls(splsModel, tuneX)$predict) %>%
            dplyr::group_by(Component) %>%
            dplyr::mutate(
              SampleID = tuneSampleID,
              SampleIndex = tuneSampleIndex
            ) %>%
            dplyr::ungroup() %>%
            dplyr::mutate(Fold = i)
          
          yhat_test[[i]] <- arr3d_to_df(predict_mixOmics_pls(splsModel, testX)$predict) %>%
            dplyr::group_by(Component) %>%
            dplyr::mutate(
              SampleID = testSampleID,
              SampleIndex = testSampleIndex
            ) %>%
            dplyr::ungroup() %>%
            dplyr::mutate(Fold = i)

        }

        # Concatenate results across all folds for the current iteration of keepX/keepY per current repeat
        full_tuneX <- rbind(full_tuneX, do.call(rbind, concatenated_tuneX))
        full_tuneY <- rbind(full_tuneY, do.call(rbind, concatenated_tuneY))

        full_testX <- rbind(full_testX, do.call(rbind, concatenated_testX))
        full_testY <- rbind(full_testY, do.call(rbind, concatenated_testY))

        full_train_loadingsX <- rbind(full_train_loadingsX, do.call(rbind, concatenated_trainX_loadings))
        full_train_loadingsY <- rbind(full_train_loadingsY, do.call(rbind, concatenated_trainY_loadings))


        # # array version
        # full_yhat_tune <- abind(full_yhat_tune, do.call(abind, list(yhat_tune, along = 1)), along = 1)
        # full_yhat_test <- abind(full_yhat_test, do.call(abind, list(yhat_test, along = 1)), along = 1)
        # df version
        full_yhat_tune <- rbind(full_yhat_tune, do.call(rbind, yhat_tune) %>% mutate(KeepX = x, KeepY = y, Repeat = rep))
        full_yhat_test <- rbind(full_yhat_test, do.call(rbind, yhat_test) %>% mutate(KeepX = x, KeepY = y, Repeat = rep))
        
        if (binary == FALSE) {
          
          metrics_tune_df <- calc_prediction_metrics_by_component(
            yhat_list = yhat_tune,
            obsY_list = observed_tuneY
          )
          
          metrics_test_df <- calc_prediction_metrics_by_component(
            yhat_list = yhat_test,
            obsY_list = observed_testY
          )
          
        } else {
          
          metrics_tune_df <- calc_binary_metrics_by_component(
            yhat_list = yhat_tune,
            obsY_list = observed_tuneY,
            positive_class = 1
          )
          
          metrics_test_df <- calc_binary_metrics_by_component(
            yhat_list = yhat_test,
            obsY_list = observed_testY,
            positive_class = 1
          )
        }
        # full_exp_var_tuneX <- rbind(full_exp_var_tuneX, do.call(rbind, concatenated_exp_varX))
        # full_exp_var_tuneY <- rbind(full_exp_var_tuneY, do.call(rbind, concatenated_exp_varY))
        #
        # full_variates_trainX <- rbind(full_variates_trainX, do.call(rbind, concatenated_train_variatesX))
        # full_variates_trainY <- rbind(full_variates_trainY, do.call(rbind, concatenated_train_variatesY))

        # Calculate the correlations between the scores for X and Y
        for (comp in 1:ncomp) {
          
          tuneX_all <- do.call(rbind, concatenated_tuneX)
          tuneY_all <- do.call(rbind, concatenated_tuneY)
          
          correlation <- stats::cor.test(
            tuneX_all[, comp],
            tuneY_all[, comp],
            method = "spearman",
            exact = FALSE
          )
          
          if (binary == FALSE) {
            
            results_tune_df <- rbind(results_tune_df, data.frame(
              Repeat = rep,
              KeepX = x,
              KeepY = y,
              Component = comp,
              Score_cor = correlation$estimate[[1]],
              Score_cor_p = correlation$p.value,
              Ypred_RMSE = metrics_tune_df$YPred_RMSE[
                metrics_tune_df$Component == comp
              ],
              Ypred_R2 = metrics_tune_df$YPred_R2[
                metrics_tune_df$Component == comp
              ],
              n = length(tuneX_all[, comp])
            ))
            
          } else {
            
            results_tune_df <- rbind(results_tune_df, data.frame(
              Repeat = rep,
              KeepX = x,
              KeepY = y,
              Component = comp,
              Score_cor = correlation$estimate[[1]],
              Score_cor_p = correlation$p.value,
              Ypred_AUC = metrics_tune_df$Ypred_AUC[
                metrics_tune_df$Component == comp
              ],
              Ypred_AUPRC = metrics_tune_df$Ypred_AUPRC[
                metrics_tune_df$Component == comp
              ],
              n = length(tuneX_all[, comp])
            ))
          }
        }
        
        for (comp in 1:ncomp) {
          
          testX_all <- do.call(rbind, concatenated_testX)
          testY_all <- do.call(rbind, concatenated_testY)
          
          correlation <- stats::cor.test(
            testX_all[, comp],
            testY_all[, comp],
            method = "spearman",
            exact = FALSE
          )
          
          if (binary == FALSE) {
            
            results_test_df <- rbind(results_test_df, data.frame(
              Repeat = rep,
              KeepX = x,
              KeepY = y,
              Component = comp,
              Score_cor = correlation$estimate[[1]],
              Score_cor_p = correlation$p.value,
              Ypred_RMSE = metrics_test_df$YPred_RMSE[
                metrics_test_df$Component == comp
              ],
              Ypred_R2 = metrics_test_df$YPred_R2[
                metrics_test_df$Component == comp
              ],
              n = length(testX_all[, comp])
            ))
            
          } else {
            
            results_test_df <- rbind(results_test_df, data.frame(
              Repeat = rep,
              KeepX = x,
              KeepY = y,
              Component = comp,
              Score_cor = correlation$estimate[[1]],
              Score_cor_p = correlation$p.value,
              Ypred_AUC = metrics_test_df$Ypred_AUC[
                metrics_test_df$Component == comp
              ],
              Ypred_AUPRC = metrics_test_df$Ypred_AUPRC[
                metrics_test_df$Component == comp
              ],
              n = length(testX_all[, comp])
            ))
          }
        }

      }
    }
    print(paste0("Repeat ", rep,": ", rep, "/", numRepeats, " repeats complete!"))
  }
  
  sig_combos <- tibble::tibble()
  flip_log_df <- tibble::tibble()
  flip_summary_df <- tibble::tibble()
  
  if (sign_flipping == TRUE) {
    cat("\nChecking for sign ambiguity and resolving output.\n")
    
    # Significant combinations to inspect for sign flipping
    sig_combos <- results_tune_df %>%
      group_by(KeepX, KeepY, Component) %>%
      mutate(corr_median = median(Score_cor, na.rm = TRUE)) %>%
      filter(abs(Score_cor - corr_median) < 1e-10) %>%
      filter(Score_cor_p <= 0.05, Score_cor > 0) %>%
      ungroup() %>%
      distinct(KeepX, KeepY, Component)
    
    # Detailed log: one row per flipped Repeat/Fold case
    flip_log_list <- vector("list", nrow(sig_combos))
    
    # Summary: one row per sig combo
    flip_summary_list <- vector("list", nrow(sig_combos))
    
    for (i in seq_len(nrow(sig_combos))) {
      
      keepx_i <- sig_combos$KeepX[i]
      keepy_i <- sig_combos$KeepY[i]
      comp_i  <- sig_combos$Component[i]
      comp_col_name <- paste0("comp", comp_i)
      
      # Combined X and Y for robust PCA orientation
      full_train_loadings <- rbind(
        full_train_loadingsX %>%
          filter(KeepX == keepx_i, KeepY == keepy_i),
        full_train_loadingsY %>%
          filter(KeepX == keepx_i, KeepY == keepy_i)
      )
      
      pca_input <- full_train_loadings %>%
        dplyr::select(Repeat, Fold, Variable, all_of(comp_col_name)) %>%
        tidyr::pivot_wider(
          id_cols = c(Repeat, Fold),
          names_from = Variable,
          values_from = all_of(comp_col_name)
        ) %>%
        arrange(Repeat, Fold)
      
      pc1 <- pca_input %>%
        dplyr::select(Repeat, Fold)
      
      pca <- pca_input %>%
        dplyr::select(-Repeat, -Fold) %>%
        prcomp(scale. = FALSE, center = FALSE)
      
      pc1$PC1 <- pca$x[, 1]
      
      # Flipped rows for this significant combo
      flip_index <- pc1 %>%
        filter(PC1 < 0) %>%
        mutate(
          KeepX = keepx_i,
          KeepY = keepy_i,
          Component = comp_i
        ) %>%
        dplyr::select(Repeat, Fold, KeepX, KeepY, Component)
      
      flip_log_list[[i]] <- flip_index
      
      flip_summary_list[[i]] <- tibble::tibble(
        KeepX = keepx_i,
        KeepY = keepy_i,
        Component = comp_i,
        n_flips = nrow(flip_index),
        flipped_any = nrow(flip_index) > 0
      )
      
      # Keys to flip
      flip_keys <- paste(flip_index$Repeat, flip_index$Fold)
      
      if (length(flip_keys) > 0) {
        
        idx <- with(
          full_train_loadingsX,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_train_loadingsX[idx, comp_col_name] <-
          -full_train_loadingsX[idx, comp_col_name]
        
        idx <- with(
          full_train_loadingsY,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_train_loadingsY[idx, comp_col_name] <-
          -full_train_loadingsY[idx, comp_col_name]
        
        idx <- with(
          full_tuneX,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_tuneX[idx, comp_col_name] <-
          -full_tuneX[idx, comp_col_name]
        
        idx <- with(
          full_tuneY,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_tuneY[idx, comp_col_name] <-
          -full_tuneY[idx, comp_col_name]
        
        idx <- with(
          full_testX,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_testX[idx, comp_col_name] <-
          -full_testX[idx, comp_col_name]
        
        idx <- with(
          full_testY,
          KeepX == keepx_i &
            KeepY == keepy_i &
            paste(Repeat, Fold) %in% flip_keys
        )
        full_testY[idx, comp_col_name] <-
          -full_testY[idx, comp_col_name]
      }
    }
    
    # Final logs
    flip_log_df <- dplyr::bind_rows(flip_log_list)
    flip_summary_df <- dplyr::bind_rows(flip_summary_list)
    
    # Counts
    n_sig_combos <- nrow(sig_combos)
    n_sig_combos_flipped <- sum(flip_summary_df$flipped_any)
    total_n_flips <- nrow(flip_log_df)
    
    cat(sprintf("Number of significant combinations checked: %d.\n", n_sig_combos))
    cat(sprintf("Total number of flipped Repeat/Fold cases: %d.\n", total_n_flips))
    
    if (n_sig_combos_flipped > 0) {
      cat("\nSign flipping has been detected and has adjusted the output of the significant combinations accordingly.\nA detailed sign flip summary and log has been outputed in flip_summary_df and flip_log_df.")
    } else {
      cat("\nNo sign flips detected.")
    }
  }

  return(list(results_tune_df = results_tune_df,
              results_test_df = results_test_df,
              full_train_loadingsX = full_train_loadingsX,
              full_train_loadingsY = full_train_loadingsY,
              full_tuneX = full_tuneX,
              full_tuneY = full_tuneY,
              full_testX = full_testX,
              full_testY = full_testY,
              folds = folds,
              full_yhat_tune = full_yhat_tune %>%
                dplyr::select(Repeat, Fold, KeepX, KeepY, Component, SampleID, SampleIndex, everything()),
              full_yhat_test = full_yhat_test %>%
                dplyr::select(Repeat, Fold, KeepX, KeepY, Component, SampleID, SampleIndex, everything()),
              flip_summary_df = flip_summary_df,
              flip_log_df = flip_log_df
              )
  )
}
