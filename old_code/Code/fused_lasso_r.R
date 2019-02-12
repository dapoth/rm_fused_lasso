### Research Modul Econometrics and Statistics
### Clara, Christoph and David

###########################################################################################################
###########################################################################################################
### 1.Data generating process
###########################################################################################################
###########################################################################################################
require("MASS")
require("penalized")
require("HDPenReg")
source("generate_blocks.R")

set.seed(1)

n = 20
p = 100
block_length = 10    # per active Block of betas unequal to 0
block_number_a = 3
block_number_b = 10
amplitude_a = 5
amplitude_b = 0.3

#Generate data sample to multiply with oracle beta
means = matrix( 0, p, 1)
covs = diag(p)
data_fused = mvrnorm ( n, means, covs)

error = rnorm( n, 0, 0.75)
################################################################################
# 1.1 Generating oracle betas and the resulting datasets.
################################################################################

beta_a1 <- generate_blocks( p, block_number_a, block_length, amplitude_a, FALSE)
beta_b1 <- generate_blocks( p, block_number_a, block_length, amplitude_b, FALSE)
beta_a2 <- generate_blocks( p, block_number_b, block_length, amplitude_a, FALSE)
beta_b2 <- generate_blocks( p, block_number_b, block_length, amplitude_b, FALSE)
beta_a3 <- generate_blocks( p, block_number_b, block_length, amplitude_a, TRUE)
beta_b3 <- generate_blocks( p, block_number_b, block_length, amplitude_b, TRUE)


Y_a1  = data_fused %*% beta_a1 + error
Y_b1  = data_fused %*% beta_b1 + error
Y_a2  = data_fused %*% beta_a2 + error
Y_b2  = data_fused %*% beta_b2 + error
Y_a3  = data_fused %*% beta_a3 + error
Y_b3  = data_fused %*% beta_b3 + error

df_a1 <-as.data.frame( cbind( Y_a1, data_fused))
df_b1 <-as.data.frame( cbind( Y_b1, data_fused))
df_a2 <-as.data.frame( cbind( Y_a2, data_fused))
df_b2 <-as.data.frame( cbind( Y_b2, data_fused))
df_a3 <-as.data.frame( cbind( Y_a3, data_fused))
df_b3 <-as.data.frame( cbind( Y_b3, data_fused))

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
#par(mfrow=c(3,2))
lambdas <- matrix( nrow = 6, ncol = 2)

###########################################################################################################################
# 2.1 Small Block Number
###########################################################################################################################
#  Single Block with Amplitude a
#CV
#########################################################################

# fit1_1 <- profL1( Y_a1, df_a1[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
# plotpath(fit1_1$fullfit, log="x")
# fit1_2 <- profL2( Y_a1, df_a1[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
# plotpath(fit1_2$fullfit, log="x")
# 
# opt1_1 <- optL1( Y_a1, df_a1[1:p+1], fold= fit1_1$fold, fusedl = TRUE)
# opt1_1$lambda
# opt1_2 <- optL2( Y_a1, df_a1[1:p+1], fold= fit1_2$fold, fusedl = TRUE)
# opt1_2$lambda

opt1_1 <- optL1( Y_a1, df_a1[1:p+1], fusedl = TRUE, fold = 4, minlambda1 = 1, maxlambda1 = 20)
opt1_1$lambda
#### Plot solution path and cv likelihood
test1_1 <- profL1(Y_a1, penalized = df_a1[1:p+1],fusedl = TRUE, fold = opt1_1$fold, steps=10)
plot(test1_1$lambda,test1_1$cvl, type="l")
plotpath(test1_1$fullfit)
#####
lambdas[1,1] <-opt1_1$lambda

opt1_2 <- optL2( Y_a1, df_a1[1:p+1], fusedl = TRUE, fold = 4, minlambda2 = 5, maxlambda2 = 50, lambda1 = opt1_1$lambda )
opt1_2$lambda
#### Plot solution path and cv likelihood
test1_2 <- profL1(Y_a1, penalized = df_a1[1:p+1],fusedl = TRUE, fold = opt1_1$fold, steps=10)
plot(test1_2$lambda,test1_2$cvl, type="l")
plotpath(test1_2$fullfit)
#####
lambdas[1,2] <- opt1_2$lambda

########################################################################
#Plotten Unterschiedlicher Spezifikationen zum Vergleich!
#Manuell Konfigurierter Fused funktioniert besser als fused_cv

fit_fused_cv_1  <- penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = opt1_1$lambda, lambda2 = opt1_2$lambda, data = df_a1, fusedl = TRUE, model="linear")
fit_fused_1     <- penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_a1, fusedl = TRUE, model="linear")
fit_lasso_1     <- penalized( response = Y_a1, penalized = df_a1[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a1, fusedl = FALSE, model="linear")

plot(beta_a1, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_a), sub = sprintf( "Amplitude = %d", amplitude_a), ylab ="Estimates of Coefficients", xlab= "p")
lines( coefficients( fit_fused_cv_1,"all")[-1], type = "s", col = "green")
lines( coefficients( fit_fused_1,"all")[-1], type = "s", col ="blue")
lines( coefficients( fit_lasso_1,"all")[-1], type = "s", col ="red")


#########################################################################
#########################################################################
#  Single Block with Amplitude b
#CV
#########################################################################
# fit_high1_1 <- profL1( Y_b1, df_b1[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
# plotpath(fit_high1_1$fullfit, log="x")
# fit_high1_2 <- profL2( Y_b1, df_b1[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
# plotpath(fit_high1_2$fullfit, log="x")
# 
# opt_high1_1 <- optL1( Y_b1, df_b1[1:p+1], fold= fit_high1_1$fold)
# opt_high1_1$lambda
# opt_high1_2 <- optL2( Y_b1, df_b1[1:p+1], fold= fit_high1_2$fold)
# opt_high1_2$lambda
opt_high1_1 <- optL1( Y_b1, df_b1[1:p+1], fusedl = TRUE, fold = n/2)
opt_high1_1$lambda
lambdas[2,1] <- opt_high1_1$lambda
opt_high1_2 <- optL2( Y_b1, df_b1[1:p+1], fusedl = TRUE, fold = n/2)
opt_high1_2$lambda
lambdas[2,2] <- opt_high1_2$lambda

########################################################################
#Plotten Unterschiedlicher Spezifikationen zum Vergleich!
#Manuell Konfigurierter Fused funktioniert besser als fused_cv

fit_high_fused_cv_1 <- penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = opt_high1_1$lambda, lambda2 = opt_high1_2$lambda, data = df_b1, fusedl = TRUE, model="linear")
fit_high_fused_1 <- penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_b1, fusedl = TRUE, model="linear")
fit_high_lasso_1 <- penalized( response = Y_b1, penalized = df_b1[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_b1, fusedl = FALSE, model="linear")
plot(beta_b1, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_a) , sub = sprintf( "Amplitude = %d", amplitude_b), ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_1,"all")[-1], type = "s", col = "green")
lines(coefficients(fit_high_fused_1,"all")[-1], type = "s", col ="blue")
lines(coefficients(fit_high_lasso_1,"all")[-1], type = "s", col ="red")

###########################################################################################################################
# 2.2 High Block Number
###########################################################################################################################

#  Five Blocks with Amplitude a
#CV
#########################################################################
# fit2_1 <- profL1( Y_a2, df_a2[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
# plotpath(fit2_1$fullfit, log="x")
# fit2_2 <- profL2( Y_a2, df_a2[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
# plotpath(fit2_2$fullfit, log="x")
# opt2_1 <- optL1( Y_a2, df_a2[1:p+1], fold= fit2_1$fold)
# opt2_1$lambda
# opt2_2 <- optL2( Y_a2, df_a2[1:p+1], fold= fit2_2$fold)
# opt2_2$lambda

opt2_1 <- optL1( Y_a2, df_a2[1:p+1], fusedl = TRUE, fold= n/2)
opt2_1$lambda
lambdas[3,1] <- opt2_1$lambda
opt2_2 <- optL2( Y_a2, df_a2[1:p+1], fusedl = TRUE, fold= n/2)
opt2_2$lambda
lambdas[3,2] <- opt2_2$lambda

fit_fused_cv_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = opt2_1$lambda, lambda2 = opt2_2$lambda, data = df_a2, fusedl = TRUE, model="linear")
fit_fused_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_a2, fusedl = TRUE, model="linear")
fit_lasso_2 <- penalized( response = Y_a2, penalized = df_a2[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a2, fusedl = FALSE, model="linear")
plot(beta_a2, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_b), sub = sprintf( "Amplitude = %d", amplitude_a), ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_fused_cv_2,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_fused_2,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_lasso_2,"all")[-1],main = "fusedlasso", type = "s", col ="red")


#########################################################################
#########################################################################
#  Five Blocks with Amplitude b
#CV
#########################################################################
# fit_high2_1 <- profL1( Y_b2, df_a2[1:p+1], minl = 0.001, maxl= 100, plot =TRUE)
# plotpath(fit_high2_1$fullfit, log="x")
# fit_high2_2 <- profL2( Y_b2, df_a2[1:p+1], minl = 0.001, maxl = 100, plot = TRUE)
# plotpath(fit_high2_2$fullfit, log="x")
# opt_high2_1 <- optL1( Y_b2, df_a2[1:p+1], fold= fit_high2_1$fold)
# opt_high2_1$lambda
# opt_high2_2 <- optL2( Y_b2, df_a2[1:p+1], fold= fit_high2_2$fold)
# opt_high2_2$lambda

opt_high2_1 <- optL1( Y_b2, df_b2[1:p+1], fold= n/2, fusedl = TRUE)
opt_high2_1$lambda
lambdas[4,1] <- opt_high2_1$lambda
opt_high2_2 <- optL2( Y_b2, df_b2[1:p+1], fold= n/2, fusedl = TRUE)
opt_high2_2$lambda
lambdas[4,2] <- opt_high2_2$lambda

########################################################################
#Plotten Unterschiedlicher Spezifikationen zum Vergleich!
#Manuell Konfigurierter Fused funktioniert besser als fused_cv

fit_high_fused_cv_2 <- penalized( response = Y_b2, penalized = df_b2[1:p+1], unpenalized = ~0, lambda1 = opt_high2_1$lambda, lambda2 = opt_high2_2$lambda, data = df_b2, fusedl = TRUE, model="linear")
fit_high_fused_2 <- penalized( response = Y_b2, penalized = df_b2[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_b2, fusedl = TRUE, model="linear")
fit_high_lasso_2 <- penalized( response = Y_b2, penalized = df_b2[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_b2, fusedl = FALSE, model="linear")
plot(beta_b2, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_b), sub = sprintf( "Amplitude = %d", amplitude_b), ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_2,"all")[-1],main = "fusedlasso", type = "s", col = "green")
lines(coefficients(fit_high_fused_2,"all")[-1],main = "fusedlasso", type = "s", col ="blue")
lines(coefficients(fit_high_lasso_2,"all")[-1],main = "fusedlasso", type = "s", col ="red")

###########################################################################################################################
# 2.3 High Number of Blocks with different amplitudes
###########################################################################################################################
#  High number of Blocks with Amplitude a and two levels of amplitude
#CV
#########################################################################

# fit1 <- profL1( Y_a3, df_a3[1:p+1], minl = 0.01, maxl= 100, plot =TRUE)
# plotpath(fit1$fullfit, log="x")
# fit2 <- profL2( Y_a3, df_a3[1:p+1], minl = 0.01, maxl = 100, plot = TRUE)
# plotpath(fit2$fullfit, log="x")
# opt1 <- optL1( Y_a3, df_a3[1:p+1], fold= fit1$fold)
# opt1$lambda
# opt2 <- optL2( Y_a3, df_a3[1:p+1], fold= fit2$fold)
# opt2$lambda

opt3_1 <- optL1( Y_a3, df_a3[1:p+1], fusedl = TRUE, fold= n/2)
opt3_1$lambda
lambdas[5,1] <- opt3_1$lambda
opt3_2 <- optL2( Y_a3, df_a3[1:p+1], fusedl = TRUE, fold= n/2)
opt3_2$lambda
lambdas[5,2] <- opt3_2$lambda

fit_fused_cv_3 <- penalized( response = Y_a3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = opt_high2_1$lambda, lambda2 = opt_high2_2$lambda, data = df_a3, fusedl = TRUE, model="linear")
fit_fused_3 <- penalized( response = Y_a3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_a3, fusedl = TRUE, model="linear")
fit_lasso_3 <- penalized( response = Y_a3, penalized = df_a3[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_a3, fusedl = FALSE, model="linear")
plot(beta_a3, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_b), sub = sprintf( "Amplitude = %d and %d", amplitude_a, amplitude_a*2), ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_fused_cv_3,"all")[-1], col = "green")
lines(coefficients(fit_fused_3,"all")[-1], col ="blue")
lines(coefficients(fit_lasso_3,"all")[-1], col ="red")

#  Five Blocks with Amplitude of b and two levels of amplitude
#CV
#########################################################################
# fit_high3_1 <- profL1( Y_b3, df_a3[1:p+1], minl = 0.01, maxl= 100, plot =TRUE)
# plotpath(fit_high3_1$fullfit, log="x")
# fit_high3_2 <- profL2( Y_b3, df_a3[1:p+1], minl = 0.01, maxl = 100, plot = TRUE)
# plotpath(fit_high3_2$fullfit, log="x")
# opt_high3_1 <- optL1( Y_b3, df_a3[1:p+1], fold= fit_high3_1$fold)
# opt_high3_1$lambda
# lambdas[5,1] <- opt_high3_1$lambda
# opt_high3_2 <- optL2( Y_b3, df_a3[1:p+1], fold= fit_high3_2$fold)
# opt_high3_2$lambda
# lambdas[5,1] <- opt_high3_2$lambda

opt_high3_1 <- optL1( Y_b3, df_b3[1:p+1], fold= n/2, fusedl = TRUE)
opt_high3_1$lambda
lambdas[6,1] <- opt_high3_1$lambda
opt_high3_2 <- optL2( Y_b3, df_b3[1:p+1], fold= n/2, fusedl = TRUE)
opt_high3_2$lambda
lambdas[6,2] <- opt_high3_2$lambda

fit_high_fused_cv_3 <- penalized( response = Y_b3, penalized = df_b3[1:p+1], unpenalized = ~0, lambda1 = opt_high3_1$lambda, lambda2 = opt_high3_2$lambda, data = df_b3, fusedl = TRUE, model="linear")
fit_high_fused_3 <- penalized( response = Y_b3, penalized = df_b3[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 10, data = df_b3, fusedl = TRUE, model="linear")
fit_high_lasso_3 <- penalized( response = Y_b3, penalized = df_b3[1:p+1], unpenalized = ~0, lambda1 = 2, data = df_b3, fusedl = FALSE, model="linear")
plot(beta_b3, main = sprintf("Lasso and Fused Lasso with %d Block",block_number_b), sub = sprintf( "Amplitude = %d and %d", amplitude_b, amplitude_b*2), ylab ="Estimates of Coefficients", xlab= "p")
lines(coefficients(fit_high_fused_cv_3,"all")[-1], col = "green")
lines(coefficients(fit_high_fused_3,"all")[-1], col ="blue")
lines(coefficients(fit_high_lasso_3,"all")[-1], col ="red")






#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#######################################   4. REAL DATA APPLICATION     ##############################################################
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

# fit1_cgh <- profL1( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], minlambda1 =  1, maxlambda1 = 6, fold = 50, plot =TRUE)
# plotpath(fit1_cgh$fullfit, log="x")
# fit2_cgh <- profL2( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], minlambda2 = 0.01, maxlambda2 = 1000, fold = 50, plot = TRUE)
# plotpath(fit2_cgh$fullfit, log="x")
# opt1_cgh <- optL1( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold = fit1_cgh$fold, fusedl = TRUE)
# opt1_cgh$lambda
# opt2_cgh <- optL2( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold = fit1_cgh$fold, fusedl = TRUE)
# opt2_cgh$lambda

opt1_cgh <- optL1( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold = 30, fusedl = TRUE)
opt1_cgh$lambda
opt2_cgh <- optL2( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], fold = 30, fusedl = TRUE)
opt2_cgh$lambda

# Fit Fused-Lasso with Crossvalidation

fit_cgh <- penalized( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], lambda1 = opt1_cgh$lambda, lambda2 = opt2_cgh$lambda, fusedl = TRUE, model = "linear" )
fit_cgh_control <- penalized( cgh_matrix, cgh_df[1:length(cgh_matrix)+1], lambda1 = .5, lambda2 = 3, fusedl = TRUE, model = "linear" )
plot(cgh_matrix, main ="Signal Approximator on CGH data", ylab = "Signal", xlab= "p")
lines(coefficients(fit_cgh_control,"all")[-1], col = "green")
lines(coefficients(fit_cgh,"all")[-1], col = "red")


