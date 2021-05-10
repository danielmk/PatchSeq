library(dplyr)
library(psych)
library(glmnet)

ephys <- read.csv("C:\\Users\\Daniel\\repos\\PatchSeq\\ephys_df.csv", stringsAsFactors = TRUE)
ephys <- ephys[ephys$label == 'SST-EYFP' | ephys$label == 'VGlut3-EYFP', ]
ephys$label <- as.numeric(ephys$label)
ephys$label[ephys$label==1] <- 0
ephys$label[ephys$label==3] <- 1


model_norm <- glm(label ~ Max..Freq...Hz. +
                     Slow.AHP..mV. +
               Rheobase..pA. + 
               I.at.Max..Freq...pA. +
               Adaptation.ratio +
               Avg.Spike.Time..s.+
               Sag.Amplitude..mV.
,
               family=binomial("logit"),
             maxit=100,
             data = ephys)

x <- as.matrix(ephys[,2:17])
y <- ephys$label

model <- glmnet(x, y, family=gaussian, alpha=0.5)

#Resting..mV.
#Input.R..MOhm.+
#Capacitance..pF.+

res.aov = aov(label ~ Max..Freq...Hz. +
                Slow.AHP..mV. +
                Rheobase..pA. + 
                I.at.Max..Freq...pA. +
                Adaptation.ratio +
                Avg.Spike.Time..s.+
                Sag.Amplitude..mV.+
                Input.R..MOhm.+
                Capacitance..pF.+
                Resting..mV.+
                RS.AHP.Amp...mV.+
                RS.Max..Slope..mV.ms.+
                RS.Min..Slope..mV.ms.+
                RS.Peak..mV.+
                RS.Half.Width..ms.+
                RS.Threshold..mV.+
                FS.AHP.Amp...mV.+
                FS.Max..Slope..mV.ms.+
                FS.Min..Slope..mV.ms.+
                FS.Peak..mV.+
                FS.Half.Width..ms.+
                FS.Threshold..mV.+
                LS.AHP.Amp...mV.+
                LS.Max..Slope..mV.ms.+
                LS.Min..Slope..mV.ms.+
                LS.Peak..mV.+
                LS.Half.Width..ms.+
                LS.Threshold..mV., data=ephys)
summary(res.aov)
describe.by()

describeBy(ephys, ephys$label)




