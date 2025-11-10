data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> rr;
  array[N] real ds;
  array[N] int<lower=0,upper=1> cond;
  int<lower=1> J;
  array[N] int<lower=1,upper=J> id;
}

parameters {
  vector[4] beta;                    // fixed-effects parameters
  vector<lower=0,upper=1>[J] lambda; // lapse rate
  vector<lower=0>[4] sigma_u;        // random effects standard deviations
  cholesky_factor_corr[4] L_u;       // L_u is the Choleski factor of the correlation matrix
  matrix[4,J] z_u;                   // random effect matrix
}

transformed parameters {
  matrix[4,J] u;
  u = diag_pre_multiply(sigma_u, L_u) * z_u; // use Cholesky to set correlation
}

model {
  real mu; // linear predictor

  //priors
  L_u ~ lkj_corr_cholesky(2);   // LKJ prior for the correlation matrix
  to_vector(z_u) ~ normal(0,1); // before Cholesky, random effects are normal variates with SD=1
  sigma_u ~ cauchy(0, 1);       // SD of random effects (vectorized)
  beta ~ normal(0, 2);          // prior 
  lambda ~ beta(1, 7);          // lapse rate ~50% prior that <0.1, pbeta(0.1, 1, 7)

  //likelihood
  for (i in 1:N){
    
    mu = beta[1] + u[1,id[i]] + 
      cond[i]*(beta[2]+u[2,id[i]]) + 
      (beta[3] + u[3,id[i]] + cond[i]*(beta[4] + u[4,id[i]] ))*ds[i];
    
    rr[i] ~ bernoulli((1-lambda[id[i]])*Phi(mu)+lambda[id[i]]/2);
  }
}
