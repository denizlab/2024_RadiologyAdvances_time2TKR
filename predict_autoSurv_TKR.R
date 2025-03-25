#' Discrete-time survival prediction models 
#' 
#' @description 
#' Compute the predicted survival probabilities for trained discrete-time models
#'
#' @section Authors:
#' Krithika Suresh (\email{krihtika.suresh@@cuanschutz.edu})
#'
#' @param object autoSurv() object that contains trained models 
#' @param newdata data set on which you want to obtain survival predictions 
#' @param times vector of time points at which survival probability is predicted
#' @param timeVar string corresponding to the variable name of the time-to-event outcome. Used to compute performance metrics
#' @param statusVar string corresponding to the variable name of the status indicator. Used to compute performance metrics
#'
#' @return A list with predicted probabilities and performance metrics
#' @export

predict_autoSurv_TKR <- function(object, newdata, times, timeVar, statusVar) {
    # create survival formula for censoring
    .form_cens <- as.formula(paste("Surv(", timeVar,",", statusVar, ") ~ 1"))
    #list of models that were fit to the original data
    trainModels <- names(object$models)
    #store predictions for assessment
    preds <- list()
    
    if("rsf" %in% trainModels) {
        preds_rsf <-  matrix(NA, nrow(newdata), length(times))
        for (i in 1:length(times)){
            preds_rsf[,i] <- predictRSF(object$models$rsf, newdata=newdata, times=times[i],
                                        importance = TRUE)
        }
        preds[["rsf"]] <- preds_rsf
    }
    
    
    temp_Score <- riskRegression::Score(preds,
                                        formula=.form_cens,
                                        data=newdata,
                                        times=times,
                                        summary="ibs")
    
    scoreAUC <- temp_Score$AUC$score
    scoreBrier <- temp_Score$Brier$score
    scoreBrier$R2 <- 1-scoreBrier$Brier/scoreBrier$Brier[which(scoreBrier$model=="Null model")]
    scoreBrier$R2_IBS <- 1-scoreBrier$IBS/scoreBrier$IBS[which(scoreBrier$model=="Null model")]
    
    scoreALL <- data.frame(model = as.character(scoreAUC$model),
                           AUC = scoreAUC$AUC,
                           Brier = scoreBrier$Brier[-which(scoreBrier$model=="Null model")],
                           R2 = scoreBrier$R2[-which(scoreBrier$model=="Null model")],
                           IBS = scoreBrier$IBS[-which(scoreBrier$model=="Null model")],
                           R2_IBS = scoreBrier$R2_IBS[-which(scoreBrier$model=="Null model")])
    
    list("auc" = scoreAUC,
         "brier" = scoreBrier,
         "metrics" = scoreALL,
         "pred_probabilities" = lapply(preds, function(x) 1-x),
         "models" = object$models)
}
