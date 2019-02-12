### Research Modul Statistic
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

n = 60
p = 300
block_length = 10    # per active Block of betas != 0
block_number_a = 5
#block_number_b = 10
amplitude_a = 1
#amplitude_b = 0.3

#Generate data sample to multiply with oracle beta
means = matrix( 0, p, 1)
covs = diag(p)
data_fused = mvrnorm ( n, means, covs)

error = rnorm( n, 0, 0.75)

beta_pres <- generate_blocks( p, block_number_a, block_length, amplitude_a, TRUE)
Y_pres = data_fused %*% beta_pres + error
df_pres <-as.data.frame( cbind( Y_pres, data_fused))


################################################################################
# (beta try: For presentation).
################################################################################

fit_l2small <- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 1.5 , data = df_pres, fusedl = TRUE, model="linear")
fit_l2high  <- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 200, data = df_pres, fusedl = TRUE, model="linear")
fit_l2opt   <- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = 2, lambda2 = 20, data = df_pres, fusedl = TRUE, model="linear")
fit_lasso   <- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = 60, data = df_pres, fusedl = FALSE, model="linear")

pdf("oracleBetas.pdf",height = 5, width = 12)
plot(beta_pres, ylab ="Coefficients", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
dev.off()


pdf("LassoOnFusedData.pdf",height = 5, width = 12)
plot(beta_pres, ylab ="Estimates of Coefficients", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
lines(coefficients(fit_lasso,"all")[-1], type = "s", col ="green")
dev.off()
 
#opt_lasso <- optL1( Y_pres, df_pres[1:p+1], fusedl = FALSE, fold = n/2)
#opt_lasso
#fit_optlasso <- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = opt_lasso$lambda, data = df_pres, fusedl = FALSE, model="linear")
#fit_lasso2<- penalized( response = Y_pres, penalized = df_pres[1:p+1], unpenalized = ~0, lambda1 = 120, data = df_pres, fusedl = FALSE, model="linear")
#
#
#plot(beta_pres, ylab ="Estimates of Coefficients", xlab= "p", ylim=c(-1,3))
#lines(coefficients(fit_lasso,"all")[-1], type = "s", col ="green")
#lines(coefficients(fit_lasso2,"all")[-1], type = "s", col ="blue")
#lines(coefficients(fit_optlasso,"all")[-1], type = "s", col ="red")

pdf("FusionPenalty.pdf", height = 5, width = 12)
par(mfrow=c(1,2))
plot(beta_pres, main = sprintf("Low Fused Lasso Penalty"), ylab ="Estimates of Coefficients", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
#lines(coefficients(fit_ols,"all")[-1], type = "s", col = "red")
lines(coefficients(fit_l2small,"all")[-1], type = "s", col ="green")
plot(beta_pres, main = sprintf("High Fused Lasso Penalty"), ylab ="Estimates of Coefficients", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
lines(coefficients(fit_l2high,"all")[-1], type = "s", col ="blue")
#plot(beta_pres, main = sprintf("Adequate Fused Lasso Penalty"), ylab ="Estimates of Coefficients", xlab= "p")
#lines(coefficients(fit_l2opt,"all")[-1], type = "s", col ="orange")
dev.off()

pdf("LassoandFusedOnFusedData.pdf",height = 5, width = 12)
plot(beta_pres, ylab ="Estimates of Coefficients", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
lines(coefficients(fit_lasso,"all")[-1], type = "s", col ="green")
lines(coefficients(fit_l2opt,"all")[-1], type = "s", col ="orange")
dev.off()

pdf("FusedOnFusedData.pdf",height = 5, width = 12)
plot(beta_pres, ylab ="Estimates of Coefficients", type = "p", xlab= "P", ylim=c(-1,3), cex.lab=1.5)
lines(coefficients(fit_l2opt,"all")[-1], type = "s", col ="orange")
dev.off()

