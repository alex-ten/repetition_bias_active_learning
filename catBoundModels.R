library(tidyverse)
library(lme4)
library(rstatix)

df <- read_csv("data/catbounds.csv")
df <- mutate(df,
    pid = as.factor(pid),
    condition = as.factor(condition)
)

aovm <- aov(correct ~ condition * catBoundDist, data = df)
aovm2 <- anova_test(
    data = df,
    dv = correct,
    wid = pid,
    between = condition,
    within = catBoundDist)
get_anova_table(aovm2)

df2 <- read_csv("data/catboundstrials.csv")
df2 <- mutate(df2,
    pid = as.factor(pid),
    condition = as.factor(condition),
    correctNum = correct == "True"
)

m <- glmer(correct ~ condition * catBoundDist + (catBoundDist|pid), data=df, family=binomial)
m2 <- glm(correct ~ condition * catBoundDist, data=df, family='binomial')



m %>% summary
