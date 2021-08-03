alldfo_nn = read.csv("alldfo_nn.csv", header=TRUE, sep=",")
glmfit = glm(disagreement_truth_sentise ~ shannon + shannon_polar_domain + shannon_polar_overall + AP_HEAD_FINAL_Shannon+AP_HEAD_INITIAL_Shannon+AP_HEAD_MEDIAL_Shannon+AP_ANY_Shannon+VP_ANY_Shannon+VP_HEAD_FINAL_Shannon, data = alldfo_nn, family = binomial())
summary(glmfit)
anova(glmfit, test="Chisq")
# https://stats.stackexchange.com/questions/11676/pseudo-r-squared-formula-for-glms/194961
rsquared = 1 - glmfit$deviance/glmfit$null.deviance
library(car)
Anova(glmfit, type=2)