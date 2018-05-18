# install R.matlab
# install scrat
# install scater # defines the mnnCorrect function
# you may need to install other dependencies like pracma

library(R.matlab)
library(scran)
library(scater)
infile = readMat("in.mat")
# B = readMat("B.mat")
A = infile$A
B = infile$B
mnn.out<-mnnCorrect(t(A),t(B),k=20, sigma=0.1,cos.norm.in=TRUE, cos.norm.out=TRUE, var.adj=TRUE,compute.angle=TRUE)
A_corr = mnn.out$corrected[[1]] 
B_corr = mnn.out$corrected[[2]]
writeMat("out.mat",Acorr=t(A_corr),B_corr=t(B_corr))

