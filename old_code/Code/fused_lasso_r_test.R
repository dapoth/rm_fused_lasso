### Research Modul Statistic
### Clara, Christoph and David

###########################################################################################################
###########################################################################################################
### 1.Data generating process
###########################################################################################################
###########################################################################################################
require("MASS")
require("penalized")
source("generate_blocks.R")

set.seed(1)

n = 50
p = 300
block_length = 10    # per active Block of betas != 0
amplitude_a = 2
amplitude_b = 5

#Generate data sample to multiply with oracle beta
means = matrix( 0, p, 1)
covs = diag(p)
data_fused = mvrnorm ( n, means, covs)

error = rnorm( n, 0, 1)
################################################################################
# 1.1 Generierung von Unterschiedlichen oracle betas und den dazugeörigen Datensätze.
################################################################################

beta_a1 <- generate_blocks( p, 1, block_length, amplitude_a, FALSE)
beta_b1 <- generate_blocks( p, 1, block_length, amplitude_b, FALSE)
beta_a2 <- generate_blocks( p, 5, block_length, amplitude_a, FALSE)
beta_b2 <- generate_blocks( p, 5, block_length, amplitude_b, FALSE)
beta_a3 <- generate_blocks( p, 5, block_length, amplitude_a, TRUE)
beta_b3 <- generate_blocks( p, 5, block_length, amplitude_b, TRUE)

Y_a1  = data_fused %*% beta_a1 + error
Y_b1  = data_fused %*% beta_b1 + error
Y_a2  = data_fused %*% beta_a2 + error
Y_b2  = data_fused %*% beta_b2 + error
Y_a3  = data_fused %*% beta_a3 + error
Y_b3  = data_fused %*% beta_b3 + error

df_a1 <-as.data.frame(cbind(Y_a1, data_fused))
df_b1 <-as.data.frame(cbind(Y_b1, data_fused))
df_a2 <-as.data.frame(cbind(Y_a2, data_fused))
df_b2 <-as.data.frame(cbind(Y_b2, data_fused))
df_a3 <-as.data.frame(cbind(Y_a3, data_fused))
df_b3 <-as.data.frame(cbind(Y_b3, data_fused))

plot(beta_b3, main ="Different Oracle betas", ylab="Amplitude", xlab="p")
points(beta_b2, col="green", pch = 0)
points(beta_b1, col= "blue", pch = 2)

plot(beta_a3, main ="Different Oracle betas", ylab="Amplitude", xlab="p")
points(beta_a2, col= "green", pch = 0)
points(beta_a1, col="blue", pch = 2)

###########################################################################################################################
###########################################################################################################################
### 2. Application of Estimators
###########################################################################################################################
###########################################################################################################################
par(mfrow=c(3,2))
settings <-c("a1", "a2", "a3", "b1", "b2", "b3")

for(setting in settings){
  
  opt_setting <-optL1(Y_setting, df_setting[1:p+1])
  opt_setting$lambda
  
  fit_fused_1   <-penalized( response = Y_setting, penalized = df_setting[1:p+1], unpenalized = ~0, lambdsetting = opt_setting$lambda, lambda2 = 1, data = df_setting, fusedl = TRUE, model="linear")
  fit_fused_5   <-penalized( response = Y_setting, penalized = df_setting[1:p+1], unpenalized = ~0, lambdsetting = opt_setting$lambda, lambda2 = 5, data = df_setting, fusedl = TRUE, model="linear")
  fit_fused_20  <-penalized( response = Y_setting, penalized = df_setting[1:p+1], unpenalized = ~0, lambdsetting = opt_setting$lambda, lambda2 = 20, data = df_setting, fusedl = TRUE, model="linear")
  fit_fused_100 <-penalized( response = Y_setting, penalized = df_setting[1:p+1], unpenalized = ~0, lambdsetting = opt_setting$lambda, lambda2 = 100, data = df_setting, fusedl = TRUE, model="linear")
  fit_lasso_1   <-penalized( response = Y_setting, penalized = df_setting[1:p+1], unpenalized = ~0, lambdsetting = opt_setting$lambda, data = df_setting, fusedl = FALSE, model="linear")
  
  #Plotten Unterschiedlicher Spezifikationen zum Vergleich!
  
  plot(beta_setting, main = "Lasso and Fused Lasso with one Block", sub ="Amplitude = 2", ylab ="Estimates of Coefficients", xlab= "p")
  lines(coefficients(fit_fused_1,  "all")[-1], type = "s", col = "blue")
  lines(coefficients(fit_fused_5,  "all")[-1], type = "s", col = "green")
  lines(coefficients(fit_fused_20, "all")[-1], type = "s", col = "yellow")
  lines(coefficients(fit_fused_100,"all")[-1], type = "s", col = "orange")
  lines(coefficients(fit_lasso_1,  "all")[-1], type = "s", col = "red")
  
}


###########################################################################################################################
# 2.1 Single Block
###########################################################################################################################
#  Single Block with Amplitude a
#########################################################################

opt_a1 <-optL1(Y_a1, df_a1[1:p+1])
opt_a1$lambda

fit_fused_1   <-penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_a1$lambda, lambda2 = 1, data = df_a1, fusedl = TRUE, model="linear")
fit_fused_5   <-penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_a1$lambda, lambda2 = 5, data = df_a1, fusedl = TRUE, model="linear")
fit_fused_20  <-penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_a1$lambda, lambda2 = 20, data = df_a1, fusedl = TRUE, model="linear")
fit_fused_100 <-penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_a1$lambda, lambda2 = 100, data = df_a1, fusedl = TRUE, model="linear")
fit_lasso_1   <-penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_a1$lambda, data = df_a1, fusedl = FALSE, model="linear")

#Plotten Unterschiedlicher Spezifikationen zum Vergleich!

plot(beta_a1, main = "Lasso and Fused Lasso with one Block", sub ="Amplitude = 2", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_fused_1,  "all")[-1], type = "s", col = "blue")
lines(coefficients(fit_fused_5,  "all")[-1], type = "s", col = "green")
lines(coefficients(fit_fused_20, "all")[-1], type = "s", col = "yellow")
lines(coefficients(fit_fused_100,"all")[-1], type = "s", col = "orange")
lines(coefficients(fit_lasso_1,  "all")[-1], type = "s", col = "red")

#########################################################################
#########################################################################
#  Single Block with Amplitude b
#########################################################################

opt_b1 <-optL1(Y_b1, df_b1[1:p+1])
opt_b1$lambda

fit_fused_1   <-penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_b1$lambda, lambda2 = 1, data = df_b1, fusedl = TRUE, model="linear")
fit_fused_5   <-penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_b1$lambda, lambda2 = 5, data = df_b1, fusedl = TRUE, model="linear")
fit_fused_20  <-penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_b1$lambda, lambda2 = 20, data = df_b1, fusedl = TRUE, model="linear")
fit_fused_100 <-penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_b1$lambda, lambda2 = 100, data = df_b1, fusedl = TRUE, model="linear")
fit_lasso_1   <-penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_b1$lambda, data = df_b1, fusedl = FALSE, model="linear")

#Plotten Unterschiedlicher Spezifikationen zum Vergleich!

plot(beta_a1, main = "Lasso and Fused Lasso with one Block", sub ="Amplitude = 2", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_fused_1,"all")[-1], type = "s", col = "blue")
lines(coefficients(fit_fused_5,"all")[-1], type = "s", col = "green")
lines(coefficients(fit_fused_20,"all")[-1], type = "s", col = "yellow")
lines(coefficients(fit_fused_100,"all")[-1], type = "s", col = "orange")
lines(coefficients(fit_lasso_1,"all")[-1], type = "s", col = "red")


fit_high1_1 <- profL1( Y_b1, df_a1[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
plotpath(fit_high1_1$fullfit, log="x")
fit_high1_2 <- profL2( Y_b1, df_a1[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
plotpath(fit_high1_2$fullfit, log="x")

opt_high1_1 <- optL1( Y_b1, df_a1[1:p+1], fold= fit_high1_1$fold)
opt_high1_1$lambda
opt_high1_2 <- optL2( Y_b1, df_a1[1:p+1], fold= fit_high1_2$fold)
opt_high1_2$lambda

########################################################################
#Plotten Unterschiedlicher Spezifikationen zum Vergleich!
#Manuell Konfigurierter Fused funktioniert besser als fused_cv

fit_high_fused_cv_1 <- penalized( response = Y_b1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt_high1_1$lambda, lambda2 = opt_high1_2$lambda, data = df_a1, fusedl = TRUE, model="linear")
fit_high_fused_1 <- penalized( response = Y_b1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_a1, fusedl = TRUE, model="linear")
fit_high_lasso_1 <- penalized( response = Y_b1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a1, fusedl = FALSE, model="linear")
plot(beta_b1, main = "Lasso and Fused Lasso with one Block", sub ="Amplitude = 2", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_1,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_high_fused_1,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_high_lasso_1,"all")[-1],main = "fusedlasso", type = "s", col ="red")

###########################################################################################################################
# 2.2 Five Blocks
###########################################################################################################################

#  Five Blocks with Amplitude of 2
#CV
#########################################################################
fit2_1 <- profL1( Y_a2, df_a2[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
plotpath(fit2_1$fullfit, log="x")
fit2_2 <- profL2( Y_a2, df_a2[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
plotpath(fit2_2$fullfit, log="x")

opt2_1 <- optL1( Y_a2, df_a2[1:p+1], fold= fit2_1$fold)
opt2_1$lambda
opt2_2 <- optL2( Y_a2, df_a2[1:p+1], fold= fit2_2$fold)
opt2_2$lambda

fit_fused_cv_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = opt2_1$lambda, lambda2 = opt2_2$lambda, data = df_a2, fusedl = TRUE, model="linear")
fit_fused_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 15, data = df_a2, fusedl = TRUE, model="linear")
fit_lasso_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a2, fusedl = FALSE, model="linear")
plot(beta_a2, main = "Lasso and Fused Lasso with five equal Blocks", sub ="Amplitude = 2", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_fused_cv_2,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_fused_2,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_lasso_2,"all")[-1],main = "fusedlasso", type = "s", col ="red")


#########################################################################
#########################################################################
#  Five Blocks with Amplitude of 5
#CV
#########################################################################
fit_high2_1 <- profL1( Y_b2, df_a2[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
plotpath(fit_high2_1$fullfit, log="x")
fit_high2_2 <- profL2( Y_b2, df_a2[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
plotpath(fit_high2_2$fullfit, log="x")

opt_high2_1 <- optL1( Y_b2, df_a2[1:p+1], fold= fit_high2_1$fold)
opt_high2_1$lambda
opt_high2_2 <- optL2( Y_b2, df_a2[1:p+1], fold= fit_high2_2$fold)
opt_high2_2$lambda

########################################################################
#Plotten Unterschiedlicher Spezifikationen zum Vergleich!
#Manuell Konfigurierter Fused funktioniert besser als fused_cv

fit_high_fused_cv_2 <- penalized( response = Y_b2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = opt_high2_1$lambda, lambda2 = opt_high2_2$lambda, data = df_a2, fusedl = TRUE, model="linear")
fit_high_fused_2 <- penalized( response = Y_b2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_a2, fusedl = TRUE, model="linear")
fit_high_lasso_2 <- penalized( response = Y_b2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a2, fusedl = FALSE, model="linear")
plot(beta_b2, main = "Lasso and Fused Lasso with five Blocks", sub ="Amplitude = 5", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_2,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_high_fused_2,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_high_lasso_2,"all")[-1],main = "fusedlasso", type = "s", col ="red")

###########################################################################################################################
# 2.3 Five Blocks with different amplitudes
###########################################################################################################################
#  Five Blocks with Amplitude of 2
#CV
#########################################################################

fit1 <- profL1( Y_a3, df_a3[1:p+1], minl = 0.01, maxl= 100, plot =TRUE)
plotpath(fit1$fullfit, log="x")
opt1 <- optL1( Y_a3, df_a3[1:p+1], fold= fit1$fold)
opt1$lambda
fit2 <- profL2( Y_a3, df_a3[1:p+1], minl = 0.01, maxl = 100, plot = TRUE)
plotpath(fit2$fullfit, log="x")

opt2 <- optL2( Y_a3, df_a3[1:p+1], fold= fit2$fold)
opt2$lambda

fit_cv_3 <- penalized( response = Y_a3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = opt1$lambda, lambda2 = 5, data = df_a3, fusedl = TRUE, model="linear")
plot(coefficients(fit_cv_3,"all")[-1],main = "fusedlasso")

#  Five Blocks with Amplitude of 5
#CV
#########################################################################
fit_high3_1 <- profL1( Y_b3, df_a3[1:p+1], minl = 0.01, maxl= 100, plot =TRUE)
plotpath(fit_high3_1$fullfit, log="x")
fit_high3_2 <- profL2( Y_b3, df_a3[1:p+1], minl = 0.01, maxl = 100, plot = TRUE)
plotpath(fit_high3_2$fullfit, log="x")

opt_high3_1 <- optL1( Y_b3, df_a3[1:p+1], fold= fit_high3_1$fold)
opt_high3_1$lambda
opt_high3_2 <- optL2( Y_b3, df_a3[1:p+1], fold= fit_high3_2$fold)
opt_high3_2$lambda

fit_high_fused_cv_3 <- penalized( response = Y_b3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = opt_high3_1$lambda, lambda2 = opt_high3_2$lambda, data = df_a3, fusedl = TRUE, model="linear")
fit_high_fused_3 <- penalized( response = Y_b3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 15, data = df_a3, fusedl = TRUE, model="linear")
fit_high_lasso_3 <- penalized( response = Y_b3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a3, fusedl = FALSE, model="linear")
plot(beta_b3, main = "Lasso and Fused Lasso with five Blocks", sub ="Amplitude either 5 or 10", ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_3,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_high_fused_3,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_high_lasso_3,"all")[-1],main = "fusedlasso", type = "s", col ="red")







#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
##########################################    REAL DATA APPLICATION    ##############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

# Import and transform data

cgh <- read.delim("cgh.txt", header = FALSE)
cgh_matrix <-matrix(unlist(cgh))
X_cgh <- diag(length(cgh_matrix))
cgh_df <-as.data.frame(cbind(cgh_matrix,X_cgh))

# Exploring data, to maybe provide warm start of crossvalidation

cgh_lm <- lm(cgh_matrix ~ diag(length(cgh_matrix)), data = cgh_df )
plot(fit1_cgh$coefficients, type= "l")
fit_cgh <- penalized( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], lambda1 = 0.5, lambda2 = 3, fusedl = TRUE, model = "linear" )
plot(coefficients(fit_cgh,"all"), type = "l", main = "fusedlasso cgh", col= "blue", ylim=range(-2,5))
par(new =TRUE)
plot(fit1_cgh$coefficients, type= "p", col= "grey", axes= FALSE)

# Apply Crossvalidation to Penalty Terms

fit1_cgh <- profL1( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], minlambda1 =  1, maxlambda1 = 6, fold = 50, plot =TRUE)
plotpath(fit1_cgh$fullfit, log="x")
fit2_cgh <- profL2( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], minlambda2 = 0.01, maxlambda2 = 1000, fold = 50, plot = TRUE)
plotpath(fit2_cgh$fullfit, log="x")

opt1_cgh <- optL1( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold= fit1_cgh$fold)
opt1_cgh$lambda
opt2_cgh <- optL2( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold= fit2_cgh$fold)
opt2_cgh$lambda

# Fit Fused-Lasso with Crossvalidation

fit_cgh <- penalized( cgh_matrix, cgh_df[2:length(cgh_matrix)+1], lambda1 = opt1_cgh$lambda, lambda2 = opt2_cgh$lambda, fusedl = TRUE, model = "linear" )
plot(coefficients(fit_cgh,"all"), type = "l", main = "fusedlasso cgh")
