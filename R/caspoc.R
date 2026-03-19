
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
CASPOC <- function (X, Y, ncomp = 1, numRepeats = 11, numFolds = 10, keepX_options = NULL, keepY_options = NULL, fixX = NULL, fixY = NULL, base_seed = 1, manual_folds = NULL) {
  if(!requireNamespace("dplyr", quietly = TRUE))
    stop("dplyr package required")
  if(!requireNamespace("tibble", quietly = TRUE))
    stop("tibble package required")
  if(!requireNamespace("mixOmics", quietly = TRUE))
    stop("mixOmics package required")
  if(!requireNamespace("caret", quietly = TRUE))
    stop("caret package required")
  # if(!requireNamespace("abind", quietly = TRUE))
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
      df$component <- paste0(comp_prefix, k)
      df
    })

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

  cat("Performing CRISS-CROSS\n")
  cat(sprintf("X dimensions: %d x %d\n", dim(X)[1], dim(X)[2]))
  cat(sprintf("Y dimensions: %d x %d\n", dim(Y)[1], dim(Y)[2]))

  cat("\nHyperparameters:\n")
  cat(sprintf("  ncomp       = %d\n", ncomp))
  cat(sprintf("  numRepeats  = %d\n", numRepeats))
  cat(sprintf("  numFolds    = %d\n\n", numFolds))

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

  cat("keepX_options =", paste(keepX_options, collapse = ", "), "\n")

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

  cat("keepY_options =", paste(keepY_options, collapse = ", "), "\n\n")



  # Ensure data are matrices
  X <- as.matrix(X)
  Y <- as.matrix(Y)

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

        # Beginning of cross-validation
        for (i in 1:numFolds) {
          # Define indices for training, tuning, and testing
          tuneIdx <- folds[[rep]][[i]]
          testIdx <- folds[[rep]][[(i %% numFolds) + 1]]  # Ensure the index cycles correctly
          trainIdx <- unlist(folds[[rep]][-c(i, (i %% numFolds) + 1)])  # Exclude tuning and testing fold

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
          keepX <- if (!is.null(fixX)) {
            c(fixX, rep(x, ncomp - length(fixX)))
          } else {
            rep(x, ncomp)
          }

          # Construct keepY vector for current combination of y
          keepY <- if (!is.null(fixY)) {
            c(fixY, rep(y, ncomp - length(fixY)))
          } else {
            rep(y, ncomp)
          }

          # Run model
          splsModel <- mixOmics::spls(trainX, trainY, mode = "regression", keepX = keepX, keepY = keepY, ncomp = ncomp, all.outputs = T, scale = T)

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

          # The following lines of code documents where everything is coming from (i.e., repeat, combinations of keepX and keepY, fold, and the variables for training)

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

          tuneScoresX$keepX <- x
          tuneScoresX$keepY <- y
          tuneScoresY$keepX <- x
          tuneScoresY$keepY <- y
          testScoresX$keepX <- x
          testScoresX$keepY <- y
          testScoresY$keepX <- x
          testScoresY$keepY <- y
          trainLoadingsX$keepX <- x
          trainLoadingsX$keepY <- y
          trainLoadingsY$keepX <- x
          trainLoadingsY$keepY <- y
          # exp_var_tuneX$keepX <- x
          # exp_var_tuneX$keepY <- y
          # exp_var_tuneY$keepX <- x
          # exp_var_tuneY$keepY <- y
          # variates_train_X$keepX <- x
          # variates_train_X$keepY <- y
          # variates_train_Y$keepX <- x
          # variates_train_Y$keepY <- y

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

          # concatenated_exp_varX[[i]] <- exp_var_tuneX
          # concatenated_exp_varY[[i]] <- exp_var_tuneY
          #
          # concatenated_train_variatesX[[i]] <- variates_train_X
          # concatenated_train_variatesY[[i]] <- variates_train_Y

          # Store yhat for each fold
          predict_mixOmics_pls <- getS3method("predict", "mixo_spls")
          yhat_tune[[i]] <- arr3d_to_df(predict_mixOmics_pls(splsModel, tuneX)$predict)
          yhat_test[[i]] <- arr3d_to_df(predict_mixOmics_pls(splsModel, testX)$predict)

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
        full_yhat_tune <- rbind(full_yhat_tune, do.call(rbind, yhat_tune) %>% mutate(keepX = x, keepY = y, Repeat = rep))
        full_yhat_test <- rbind(full_yhat_test, do.call(rbind, yhat_test) %>% mutate(keepX = x, keepY = y, Repeat = rep))

        # full_exp_var_tuneX <- rbind(full_exp_var_tuneX, do.call(rbind, concatenated_exp_varX))
        # full_exp_var_tuneY <- rbind(full_exp_var_tuneY, do.call(rbind, concatenated_exp_varY))
        #
        # full_variates_trainX <- rbind(full_variates_trainX, do.call(rbind, concatenated_train_variatesX))
        # full_variates_trainY <- rbind(full_variates_trainY, do.call(rbind, concatenated_train_variatesY))

        # Calculate the correlations between the scores for X and Y
        for (comp in 1:ncomp) {
          correlation <- stats::cor.test(do.call(rbind, concatenated_tuneX)[, comp], do.call(rbind, concatenated_tuneY)[, comp], method = "spearman", exact = FALSE)
          # Append results to the dataframe
          results_tune_df <- rbind(results_tune_df, data.frame(
            Repeat = rep,
            KeepX = x,
            KeepY = y,
            Component = comp,
            Correlation = correlation$estimate[[1]],
            Pvalue = correlation$p.value,
            n = do.call(rbind, concatenated_tuneX)[, comp] %>% length))
        }
        for (comp in 1:ncomp) {
          correlation <- stats::cor.test(do.call(rbind, concatenated_testX)[, comp], do.call(rbind, concatenated_testY)[, comp], method = "spearman", exact = FALSE)
          # Append results to the dataframe
          results_test_df <- rbind(results_test_df, data.frame(
            Repeat = rep,
            KeepX = x,
            KeepY = y,
            Component = comp,
            Correlation = correlation$estimate[[1]],
            Pvalue = correlation$p.value,
            n = do.call(rbind, concatenated_testX)[, comp] %>% length))
        }

      }
    }
    print(paste0("Repeat ", rep,": ", rep, "/", numRepeats, " repeats complete!"))
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
              full_yhat_tune = full_yhat_tune,
              full_yhat_test = full_yhat_test))
}
