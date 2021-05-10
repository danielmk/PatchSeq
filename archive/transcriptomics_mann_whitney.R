library(dplyr)
library(psych)
library(glmnet)
library(car)
library(rankMANOVA)
library(npmv)

total <- read.csv("C:\\Users\\Daniel\\repos\\PatchSeq\\trans_df.csv", row.names = 1)
#ephys <- read.csv("C:\\Users\\Daniel\\repos\\PatchSeq\\ephys_df.csv", row.names = 1)
#total <- merge(trans, ephys[row.names(trans),], by=0)
#ephys_bool <- trans$ephys == "True"
#seq_bool <- trans$sequencing == "True"
#trans <- trans[ephys_bool & seq_bool, ]
#trans <- trans[trans$]
#in_names <- row.names(ephys[ephys$PC.vs.IN.Cluster == "IN",])
#trans <- trans[in_names,]
#trans <- trans[ephys_bool,]

wilcox.test(Max..Freq...Hz. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Slow.AHP..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Rheobase..pA. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(I.at.Max..Freq...pA. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Adaptation.ratio ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Avg.Spike.Time..s. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Sag.Amplitude..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Input.R..MOhm. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Capacitance..pF. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(Resting..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.AHP.Amp...mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.Max..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.Min..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.Peak..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.Half.Width..ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(RS.Threshold..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.AHP.Amp...mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.Max..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.Min..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.Peak..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.Half.Width..ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
#wilcox.test(FS.Threshold..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.AHP.Amp...mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.Max..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.Min..Slope..mV.ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.Peak..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.Half.Width..ms. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)
wilcox.test(LS.Threshold..mV. ~ SST...Slc17a8.Coloc, data = total, exact=FALSE)

describeBy(total[,23:50], total$SST...Slc17a8.Coloc)


