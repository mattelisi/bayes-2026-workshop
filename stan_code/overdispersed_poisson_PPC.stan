data {
  int<lower=1> N;          // total # of (eth, precinct) data points
  int<lower=1> n_eth;      // number of ethnicity categories, e.g. 3
  int<lower=1> n_prec;     // number of precincts
  
  // int<lower=0> y[N];    // (old array syntax)  
  array[N] int<lower=0> y; // outcome counts y_{ep}
  vector[N] past_arrests;  // baseline/reference rate
  
  array[N] int<lower=1, upper=n_eth> eth;       // ethnicity ID for each row
  array[N] int<lower=1, upper=n_prec> precinct; // precinct ID for each row
}

parameters {
  // Ethnicity effects (fixed)
  vector[n_eth] alpha; 
  
  // Random intercepts for precinct
  real<lower=0> sigma_beta; 
  vector[n_prec] beta_raw;   // non-centered param for precinct

  // Overdispersion
  real<lower=0> sigma_eps;
  vector[N] eps_raw;         // non-centered param for each (e,p) observation
}

transformed parameters {
  vector[n_prec] beta; 
  beta = sigma_beta * beta_raw;

  vector[N] eps;
  eps = sigma_eps * eps_raw;
}

model {
  // Priors
  alpha ~ normal(0, 5);           
  sigma_beta ~ exponential(1);    
  beta_raw ~ normal(0, 1);

  sigma_eps ~ exponential(1);     
  eps_raw ~ normal(0, 1);
  
  // Likelihood
  for (i in 1:N) {
    y[i] ~ poisson(past_arrests[i] * (15.0 / 12.0) * exp(alpha[eth[i]] + beta[precinct[i]] + eps[i]));
  }
}

generated quantities {
  vector[N] log_lik; // Log-likelihood for each observation (for LOO-CV)
  real eps_rep; // random effects observation-level overdipersion
  array[N] int y_rep;

  for (i in 1:N) {

    // Compute log-likelihood for observed data    
    log_lik[i] = poisson_lpmf(y[i] | past_arrests[i] * (15.0 / 12.0) * exp(alpha[eth[i]] + beta[precinct[i]] + eps[i]));

    // Simulate replicated data from posterior predictive distribution
    eps_rep = sigma_eps * normal_rng(0, 1);
    y_rep[i] = poisson_rng(past_arrests[i] * (15.0 / 12.0) * exp(alpha[eth[i]] + beta[precinct[i]] + eps_rep));
  }
}
