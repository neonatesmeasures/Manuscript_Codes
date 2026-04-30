#packages
# install.packages("data.table")
# install.packages("tidyverse")
# could not install nloptr package, used: packageurl<-"https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_1.2.1.tar.gz"
# install.packages("lme4")
# could not directly install tidyverse, used conda install -y -c r r-tidyverse
# then it raises another issue with string.i you should install it separately 
# also needs vctrs to be installed separately. 


library(data.table) 
library(tidyverse)
library(lme4)
library(lmerTest)
library(Hmisc)
library(dplyr)




df_measurements_6mnt_outl_rmv <- read.csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/babies_measurements_unitsClean_outlierRmv_WBCagg_6months_pivot.csv")

baby_GA <- read.csv("/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/Babies_GA.csv")
tmp <- data.frame(merge(df_measurements_6mnt_outl_rmv,baby_GA, by ='person_id' ))
tmp$GA_cat <- cut(tmp$Total_GA_Days,
              breaks=c(0,258,400),
              labels=c('Pre-term', 'Term'))

columns_to_excl <- c("birth_DATETIME", "measurement_DATE")
variables_to_incl <- c("age_at_meas", "Total_GA_Days", "person_id",'GA_cat')
tmp_excl <- data.frame(tmp[ , !(names(tmp) %in% columns_to_excl)])
vars <- names(data.frame(tmp_excl[, !(names(tmp_excl) %in% variables_to_incl)]))
tmp_excl_scaled <- tmp_excl %>% mutate(across(all_of(vars), scale))


store_res <- data.frame(matrix(ncol=2, nrow=0))
names_tmp <- c("Variables", "p_value")
colnames(store_res) <- names_tmp
number_of_vars <- ncol(data.frame(tmp_excl[, !(names(tmp_excl) %in% variables_to_incl)]))
for ( i in 1:ncol(tmp_excl_scaled)) 
{
    if (!colnames(tmp_excl_scaled)[i] %in% (variables_to_incl)) 
    {
        print(sprintf("variable number: %i of %i", i, number_of_vars))
        print(sprintf("variable name is : %s", colnames(tmp_excl)[i]))
        models <- NULL
        var <- colnames(tmp_excl_scaled)[i] 
        df_ <- tmp_excl[ ,c(colnames(tmp_excl_scaled)[i], "age_at_meas", "person_id")]
        
        try(models <- lapply(var, function(x) {lmer(substitute(i ~1+age_at_meas*GA_cat + (1+age_at_meas|person_id), list(i=as.name(x))), REML =FALSE,data=tmp_excl_scaled, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))}))
        if (!is.null(models))
        {
            p_value <- anova(models[[1]])$"Pr(>F)"
            new_row <- c(var,p_value[3])
            store_res[nrow(store_res)+1, ] <- new_row
        }
        if (is.null(models))
        {
            p_value <- "error"
            new_row <- c(var,p_value)
            store_res[nrow(store_res)+1, ] <- new_row
        }
          
    }
}

write.csv(store_res,"/home/npayrov/my_project_dir/Neonatal_Age_Project/scripts/Neonatal_measurements_final/Data/linear_mixed_eff_GA.csv", row.names = FALSE)


colnames(tmp_excl_scaled)[colnames(tmp_excl_scaled) == 'Neutrophils.100.leukocytes.in.Blood'] = 'neut'
colnames(tmp_excl_scaled)[colnames(tmp_excl_scaled) == 'Basophils.100.leukocytes.in.Blood'] = 'bas'
colnames(tmp_excl_scaled)[colnames(tmp_excl_scaled) == 'Monocytes.100.leukocytes.in.Blood'] = 'mon'
colnames(tmp_excl_scaled)[colnames(tmp_excl_scaled) == 'Lymphocytes.100.leukocytes.in.Blood'] = 'lymph'
colnames(tmp_excl_scaled)[colnames(tmp_excl_scaled) == 'Eosinophils.100.leukocytes.in.Blood'] = 'eosi'

# effect of GA (across time)

randomeff_neut <- lmer (neut ~ 1+age_at_meas*GA_cat+(1+age_at_meas|person_id), data=tmp_excl_scaled, REML = FALSE, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))
randomeff_bas <- lmer (bas ~ 1+age_at_meas*GA_cat+(1+age_at_meas|person_id), data=tmp_excl_scaled, REML = FALSE, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))
randomeff_mon <- lmer (mon ~ 1+age_at_meas*GA_cat+(1+age_at_meas|person_id), data=tmp_excl_scaled, REML = FALSE, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))
randomeff_lymph <- lmer (lymph ~ 1+age_at_meas*GA_cat+(1+age_at_meas|person_id), data=tmp_excl_scaled, REML = FALSE, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))
randomeff_eosi <- lmer (eosi ~ 1+age_at_meas*GA_cat+(1+age_at_meas|person_id), data=tmp_excl_scaled, REML = FALSE, control=lmerControl(optimizer="bobyqa", optCtrl =list(maxfun=5e5)))


print(anova(randomeff_neut))
print(anova(randomeff_bas))
print(anova(randomeff_mon))
print(anova(randomeff_lymph))
print(anova(randomeff_eosi))
