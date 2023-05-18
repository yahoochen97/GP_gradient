### Code to generate synthetic data
### from a GP with ARD
# Synthetic Data with Latitude and Longitude

library(MASS)
library(RcppArmadillo)
library(inline)
library(tidyverse)
set.seed(234)

## Helper funs ####
## inputs: 
## - X_in: Numeric matrix, profiles in columns!
## - lrho_in: log of ARD parameters
K_comp <- cxxfunction(sig = signature(X_in = "numeric",
                                      lrho_in = "numeric"),
                      plugin = "Rcpp",
                      body = '
  Rcpp::NumericMatrix X = as<Rcpp::NumericMatrix>(X_in);
  Rcpp::NumericVector lrho = as<Rcpp::NumericVector>(lrho_in);
  int N = X.ncol();
  Rcpp::NumericMatrix K_res(N, N);
  for(int i = 0; i < (N-1); ++i){
    K_res(i, i) = 1.0 + 0.1;
    for(int j = i + 1; j < N; ++j){
      K_res(i, j) = k_rbf(X(_,i), X(_,j), lrho);
      K_res(j, i) = K_res(i, j);
    }
  }
  K_res(N-1, N-1) = 1.0 + 0.1;
  return(wrap(K_res));
  ', 
  includes = '
double k_rbf(const NumericMatrix::Column x1,
             const NumericMatrix::Column x2,
             const NumericVector lrho) 
{ 
  return(exp(-0.5 * sum(pow((x1 - x2) / exp(lrho), 2.0))));
}
')

## Create demographic profiles
state <- c("Alaska", "Arizona", "Arkansas", "California")
latitude_u <- runif(length(state), 30, 50)
names(latitude_u) <- state
longitude_u <- runif(length(state), -120, -70)
names(longitude_u) <- state
gender <- c("Male","Female", "Non-binary")
race <- c("White","Black","Hispanic","Asian","Other")
year <- seq(1990:2000)
df_group <- expand.grid(state=state,
                        gender=gender,
                        race=race, 
                        year=year)

df_group = df_group %>% 
  mutate(latitude = scale(latitude_u[state]),
         longitude = scale(longitude_u[state]),
         year = scale(year),
         n = sample(5:100, nrow(df_group), replace=TRUE)) #nr. of indivs per profile

# number of profiles
N <- nrow(df_group)
# number of items
#J <- 5
# item difficulties
#alpha <- -2:2
# item discriminations
#beta <- rep(1.5, J)

# add unit-level variation to lat/long
df_group["latitude"] = df_group["latitude"] + rnorm(N, 0, 0.1)
df_group["longitude"] = df_group["longitude"] + rnorm(N, 0, 0.1)

## Create design matrix without omitted levels
cont.list <- lapply(df_group[,2:3], contrasts, contrasts=FALSE)
X_mat <- model.matrix(~latitude + longitude + gender + race + year -1,
                      data = df_group,
                      contrasts.arg = cont.list)
## Compute kernel
rho <- c(.5,.5,rep(2.0,8),.5)
K <- K_comp(t(X_mat), log(rho))

## Sample theta from GP
theta <- mvrnorm(n = 1,
                 mu = rep(0, N),
                 Sigma = K)

## Create item responses
#Y <- rep(0, N)
#for(j in 1:N){
#   Y[j] <- rbinom(1, df_group$n[j], plogis(theta[j])) 
#}

## Add ideal points per profile and grouped binomial outcomes
df_group <- df_group %>%
  mutate(theta = theta,
         Y = rbinom(N, df_group$n, plogis(theta)))

# Create test/ traindataset
train_ind <- sample(1:nrow(df_group), floor(nrow(df_group)*.7))
train_df <- df_group[train_ind,]
test_df <- df_group[-train_ind,]

rho = data.frame(t(rho))
colnames(rho)=colnames(X_mat)
write.csv(df_group,"data.csv")
write.csv(train_df,"train_data.csv")
write.csv(test_df,"test_data.csv")
write.csv(rho, "rho.csv")

