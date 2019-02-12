### Research Modul Statistic
### Clara, Christoph and David

###########################################################################################################
###########################################################################################################
### 1.Data generating process
###########################################################################################################
###########################################################################################################
require("MASS")
require("penalized")
source("generate_blocks.r")

set.seed(1)

n = 50
p = 2000
p_lasso = 100
block_length_fused = 50    # per active Block of betas != 0
block_length_lasso = 1
block_number = 20
amplitude = 1
beta_presentation_fused <- generate_blocks(p, block_number, block_length_fused, amplitude, TRUE)
beta_presentation_lasso <- generate_blocks(p_lasso, block_number, block_length_lasso, amplitude, TRUE)

pdf("coefficient_structure_fused.pdf", height = 4, width = 12)
plot(beta_presentation_fused, main = "Structure example of underlying Coefficients for Fused LASSO", type = "l", ylab = "Beta Structure", xlab = "Variables")
dev.off()

pdf("coefficient_structure_lasso.pdf", height = 4, width = 12)
plot(beta_presentation_lasso, main = "Structure example of underlying Coefficients for LASSO", type = "l", ylab = "Beta Structure", xlab = "Variables")
dev.off()