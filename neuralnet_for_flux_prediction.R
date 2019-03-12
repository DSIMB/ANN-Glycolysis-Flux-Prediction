# Load libraries ----------------------------------------------------------
library(neuralnet)
library(caret)

# Set seed for reproducible resutls ---------------------------------------
set.seed(1)

# Read input --------------------------------------------------------------
input_data <- read.csv("ann_input_for_flux_prediction.csv", sep=',', header = T)
n <- nrow(input_data)


# Normalisation of data ---------------------------------------------------
normalize <- function(x) {
  x <- as.numeric(x)
  return((x - min(x)) / (max(x) - min(x)))
}

normalisedData <- as.data.frame(apply(input_data, 2,  normalize))

# Number of fold (if k = n, leave one out cross validation) ---------------
k <- n 
folds <- createFolds(1:n, k)

predValue <- NULL
nn_result <- NULL

for (fold in folds){
  train_data <- normalisedData[-fold, ]
  test_data  <- normalisedData[fold, c("PGI","PFK","FBA","TPI")]
  form <- as.formula("Jobs~PGI+PFK+FBA+TPI")
  NeuralNet <- neuralnet(
    form, train_data, hidden = 13, act.fct = "logistic",threshold = 0.01,        
    stepmax = 1e+05, rep = 1, startweights = NULL, 
    learningrate.limit = NULL, 
    learningrate.factor = list(minus = 0.5, plus = 1.2), 
    learningrate=NULL, lifesign = "none", 
    lifesign.step = 1000, algorithm = "rprop+", 
    err.fct = "sse", linear.output = TRUE, exclude = NULL, 
    constant.weights = NULL, likelihood = FALSE 
  )
  
  # In case of tanh activation function hidden = 6, act.fct = "tanh"
  nn_result <- compute(NeuralNet, test_data) # Predicting flux for new input
  predValue  <- c(predValue, nn_result$net.result)
}

normalisedResults = data.frame(normalisedData[unlist(folds),], predValue)
colnames(normalisedResults) <- c ('PGI', 'PFK', 'FBA', 'TPI', 'Jobs', 'NormalisedPredFlux')

# Denormalisation of data and predicted flux ------------------------------
denormalizeFlux <- normalisedResults$NormalisedPredFlux * (max(input_data$Jobs) - min(input_data$Jobs)) + min(input_data$Jobs)
denormalizedData <- cbind(input_data[unlist(folds), ], denormalizeFlux)
colnames(denormalizedData) <- c("PGI", "PFK", "FBA", "TPI", "Jexp", "Jann")

# Writing output ----------------------------------------------------------
write.csv(denormalizedData, file = "output.csv", row.names = F)
