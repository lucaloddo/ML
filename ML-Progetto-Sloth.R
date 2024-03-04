###  Machine Learning - Progetto
###  Sloths Species Prediction

## Membri del gruppo: - Loddo Luca 844529
##                    - Rondena Matteo 847381 

# installazione delle librerie utilizzate e loro caricamento

install.packages("FactoMineR")
install.packages("factoextra")
install.packages("caret")
install.packages("rpart")
install.packages("e1071")
install.packages("rattle")
install.packages("rpart.plot")
install.packages("RColorBrewer")
install.packages("corrplot")
install.packages("C50")
install.packages("ROCR")
install.packages("pROC")

library(FactoMineR)
library(factoextra)
library(caret)
library(rpart)
library(e1071)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(corrplot)
library(C50)
library(ROCR)
library(pROC)

# link al download del dataset
# https://www.kaggle.com/datasets/bertiemackie/sloth-species

# working directory
setwd("./")
working_dir <- getwd()
working_dir

# caricamento del dataset da file .csv
dataset = read.csv("sloth_data.csv")

dim(dataset) # dimensioni del dataset
head(dataset) # visualizzazione delle prime istanze del dataset
summary(dataset) # struttura del dataframe

# il dataset presenta valori minori di 0 (zero), perciò è necessario eliminare questi dati
colSums(dataset < 0)

# eliminazione delle istanze con dati negativi
dataset = dataset[dataset$tail_length_cm >= 0, ]

# il dataset non presenta valori NA
round(colMeans(is.na(dataset))*100, 2)

# conversione in factor della colonna target
dataset$specie = factor(dataset$specie)

# riduzione del dataset
# togliamo le colonne "X", "endangered" e "sub_specie" perchè inutili ai fini
# della predizione
dataset = subset(
  dataset, 
  select = c(
    "claw_length_cm", 
    "size_cm", 
    "tail_length_cm", 
    "weight_kg", 
    "specie"
  )
)

# distribuzione della variabile target "specie"
barplot(
  table(dataset$specie),
  col = c(4,6),
  ylim = c(0,3000),
  main = "Barplot distribuzione variabile target"
)

# numero di istanze suddivise per assegnazione della variabile target
table(dataset$specie)

# pie chart della variabile target
pie.specie = table(dataset$specie)
pie_percent = round(100*pie.specie/sum(pie.specie), 0)
pie(
  pie.specie,
  labels = pie_percent,
  main = "Distribuzione variabile target (Specie)",
  col = c(4,6)
)
legend(
  "topright",
  legend = sort(unique(dataset$specie)),
  cex = 0.8,
  fill = c(4,6)
)

# istogrammi distribuzione features

# l'attributo "claw_length_cm" presenta una distribuzione normale
hist(
  dataset$claw_length_cm, 
  col = "red", 
  main = "Istogramma claw_length_cm", 
  ylab = "Frequenza", 
  xlab = "centimetri"
)

# l'attributo "size_cm" presenta una distribuzione asimmetrica
hist(
  dataset$size_cm, 
  col = "green", 
  main = "Istogramma size_cm",
  ylab = "Frequenza", 
  xlab = "centimetri"
)

# l'attributo "tail_length_cm" presenta una distribuzione asimmetrica
hist(
  dataset$tail_length_cm, 
  col = "lightblue", 
  main = "Istogramma tail_length_cm", 
  ylab = "Frequenza", 
  xlab = "centimetri"
)

# l'attributo "weight_kg" presenta una distribuzione normale
hist(
  dataset$weight_kg, 
  col = "yellow", 
  main = "Istogramma weight_kg", 
  ylab = "Frequenza", 
  xlab = "chilogrammi"
)

# boxplot delle features in relazione alla variabile target

# la colonna "claw_length_cm" non è significativa nella previsione della specie 
# poichè le distribuzioni delle osservazioni si sovrappongono
boxplot(
  claw_length_cm ~ specie, 
  main = "Boxplot claw_length_cm", 
  data = dataset, 
  col = c(4,6)
)

# la colonna "size_cm", probabilmente, è significativa nella previsione della specie
# in quanto le distribuzioni non si sovrappongono; la maggior parte degli esemplari 
# più grandi di 60 cm, infatti, risultano appartenenti alla specie "two_toed"
boxplot(
  size_cm ~ specie, 
  main = "Boxplot size_cm",
  data = dataset, 
  col = c(4,6)
)

# la colonna "tail_length_cm", probabilmente, è significativa nella previsione
# della specie in quanto le distribuzioni non si sovrappongono; la maggior parte degli
# esemplari con la coda più piccola di 3 cm, infatti, risultano appartenenti alla
# specie "two_toed"
boxplot(
  tail_length_cm ~ specie, 
  main = "Boxplot tail_length_cm",
  data = dataset, 
  col = c(4,6)
)

# la colonna "weight_kg", probabilmente, è significativa nella previsione
# della specie in quanto le distribuzioni si sovrappongo in piccola parte;
# una parte consistente degli esemplari che pesano più di 5 kg, infatti, risultano
# appartenenti alla specie "two_toed"
boxplot(
  weight_kg ~ specie, 
  main = "Boxplot weight_kg",
  data = dataset, 
  col = c(4,6)
)

# osservando il featurePlot, le classi "three_toed" e "two_toed" vengono
# rappresentante molto bene dall'attributo "tail_length_cm" e, in buona parte,
# da "size_cm". L'attributo "weight_kg" presenta, invece, una parziale sovrapposizione,
# mentre con l'attributo "claw_length_cm" la sovrapposizione è totale.
featurePlot(
  dataset[,1:4], 
  dataset[,5], 
  plot = "density",
  main = "Feature plot",
  scales = list(
    x = list(relation = "free"),
    y = list(relation = "free")
  ), 
  auto.key = list(columns = 2)
)

# osservando lo scatter plot, notiamo che gli attributi "size_cm" e "tail_length_cm"
# presentano una distribuzione abbastanza buona che ci permette di scegliere
# SVM come modello di ML
featurePlot(
  x = dataset[,1:4], 
  y = dataset[,5],
  main = "Scatter plot",
  plot = "pairs", 
  auto.key = list(columns = 2)
)

# splitting del dataset in trainset e testset (70% e 30%)
set.seed(123)
ind = sample(2, nrow(dataset), replace = TRUE, prob = c(0.7, 0.3))
trainset = dataset[ind == 1, ]
testset = dataset[ind == 2, ]

# lancio modello decision tree
decisionTree = rpart(
  specie ~ .,
  data = trainset,
  method = "class"
)

# plot decision tree
fancyRpartPlot(decisionTree)

# controllo misura di accuratezza
testset$Prediction = predict(decisionTree, testset, type = "class")
confusion.matrix = table(testset$specie, testset$Prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

# plot della tabella dei parametri di complessità
plotcp(decisionTree)

# cut dell'albero a cp = 0.02
prunedDecisionTree = prune(decisionTree, cp=.02)

# plot decision tree
fancyRpartPlot(prunedDecisionTree)

# previsione sul testset
testset$Prediction = predict(prunedDecisionTree, testset, type = "class")

# calcolo veloce dell'accuratezza
confusion.matrix = table(testset$specie, testset$Prediction)
confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix)

# previsione sul testset con calcolo delle annesse probabilità per disegnare la curva ROC del modello
tree_prob = predict(prunedDecisionTree, testset, type = "prob")

# calcolo matrice di confusione complessiva per la classe "positiva" (three_toed)
tree_cm_positive = confusionMatrix(
  testset$Prediction, 
  testset$specie, 
  mode = "prec_recall"
)
tree_cm_positive

# calcolo matrice di confusione complessiva per la classe "negativa" (two_toed)
tree_cm_negative = confusionMatrix(
  testset$Prediction,
  testset$specie,
  mode = "prec_recall",
  positive = "two_toed"
)
tree_cm_negative

# calcolo curva ROC
dt.ROC = roc(
  response = testset$specie, 
  predictor = tree_prob[,1],
  levels = levels(testset$specie)
)

# plot curva ROC per decision tree
plot(dt.ROC, col="green", print.auc = TRUE, ylab = "Sensitivity (%)", xlab = "Specificity (%)")

# tuning del modello SVM attraverso una serie di iperparametri tra cui costo, gamma e funzione kernel
# il costo verrà scelto tra una serie di valori, quali: 0.1, 1, 10, 100, 1000
# gamma verrà scelto tra una serie di valori, quali: 0.5, 1, 2, 3, 4
# la funzione kernel scelta sarà lineare o radiale

cost_range = c(0.1, 1, 10, 100, 1000)
gamma_range = c(0.5, 1, 2, 3, 4)
kernel_range = c("linear", "radial")

tuned = tune(
  svm,
  specie ~ .,
  data = trainset,
  ranges = list(
    cost = cost_range, 
    gamma = gamma_range, 
    kernel = kernel_range
  )
)

summary(tuned)

# SVM con i parametri migliori calcolati nel tuning

best_params = tuned$best.parameters
cost = best_params$cost
gamma = best_params$gamma
kernel = best_params$kernel

subset = subset(trainset, select = c("tail_length_cm", "size_cm", "specie"))

# lancio SVM con solo le feature "tail_length_cm" e "size_cm" per visualizzare
# graficamente l'iperpiano
final_svm = svm(
  specie ~ tail_length_cm + size_cm,
  data = subset, 
  kernel = kernel, 
  cost = cost, 
  gamma = gamma
)

plot(final_svm, subset, col = c(4,6))

# lancio modello SVM
final_svm = svm(
  specie ~ .,
  data = trainset, 
  kernel = kernel, 
  cost = cost, 
  gamma = gamma,
  prob = TRUE
)

# previsione sul testset
testset$Prediction = predict(final_svm, testset, prob = TRUE)

# calcolo matrice di confusione per la classe "positiva" (three_toed)
svm_cm_positive = confusionMatrix(
  testset$Prediction, 
  testset$specie, 
  mode = "prec_recall"
)
svm_cm_positive

# calcolo matrice di confusione per la classe "negativa" (two_toed)
svm_cm_negative = confusionMatrix(
  testset$Prediction, 
  testset$specie, 
  mode = "prec_recall",
  positive = "two_toed"
)
svm_cm_negative

# calcolo curva ROC
svm.ROC = roc(
  response = testset$specie, 
  predictor = attr(testset$Prediction, "probabilities")[,1],
  levels = levels(testset$specie)
)

# plot curva ROC per SVM
plot(svm.ROC, col = "blue", print.auc = TRUE, ylab = "Sensityvity (%)", xlab = "Specificity (%)")

# plot curva ROC sovrapposte
plot(dt.ROC, col="green")
plot(svm.ROC, add = TRUE, col = "blue", print.auc = TRUE)
