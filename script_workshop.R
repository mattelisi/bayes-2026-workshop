rm(list=ls())
library(tidyverse)

library(rstan)
rstan_options(auto_write = TRUE)               # save compiled model
options(mc.cores = parallel::detectCores()-1)  # make all cores-1 available for running

d <- read.csv("data/hubble.csv")
str(d)

plot(d$distance, d$velocity,
     xlab="Distance [Mpc]", ylab="Velocity [km/s]",
     pch=19)


stan_data <- list(
  N = nrow(d),
  distance = d$distance,  # Mpc
  velocity = d$velocity   # km/s
)
str(stan_data)


fit <- stan(
  file = "stan_code/hubble_model.stan",
  data = stan_data,
  iter = 2000,
  chains = 4)


library(tidybayes)

posterior_samples <- fit %>%
  spread_draws(beta) 

head(posterior_samples)

traceplot(fit, pars = c("beta","sigma"))

print(fit, pars = c("beta","sigma"), probs = c(.025,.5,.975))



fit %>%
  spread_draws(beta, sigma)

fit %>%
  gather_draws(beta, sigma)


mpc_km <- 3.09e19 # km per Megaparsec
sec_per_year <- 60^2 * 24 * 365

# transform in Km
hubble.const <- posterior_samples$beta/ mpc_km 

# invert to get age in seconds
age <- 1/hubble.const

# transform age in billion years
age <- (age/sec_per_year) / 10^9

# point estimate
mean(age)

#
d <- read_delim("data/police_stops.txt")
str(d)

d <- d %>%
  group_by(precinct, eth) %>%
  summarise(stops = sum(stops),
            past.arrests = sum(past.arrests),
            pop = unique(pop))


stan_data <- list(
  N         = nrow(d),
  n_eth     = length(unique(d$eth)),
  n_prec    = length(unique(d$precinct)),
  y         = d$stops,
  eth       = d$eth,
  past_arrests  = d$past.arrests,
  precinct  = d$precinct
)

fit <- stan(
  file = "stan_code/overdispersed_poisson.stan",
  data = stan_data,
  iter = 2000,
  chains = 4,
  cores = 4)

print(fit, pars = c("alpha", "sigma_beta", "sigma_eps"))


library(tidybayes)

posterior <- fit %>% 
  spread_draws(alpha[e]) %>%
  mutate(rate = exp(alpha))  # exponentiate to interpret as a multiplicative factor

# Summarize the rate by ethnicity
posterior %>%
  group_by(e) %>%
  mean_hdi(rate, .width = 0.95)



posterior_ratios <- posterior %>%
  pivot_wider(
    id_cols = c(.chain, .iteration, .draw),  # these 3 columns identify each draw
    names_from = e,                          # which column to pivot on
    values_from = rate,                      # which column holds the values
    names_prefix = "eth_") %>%
  mutate(
    black_vs_white = eth_1 / eth_3,
    hisp_vs_white  = eth_2 / eth_3) %>%
  # Pivot longer so each ratio is in its own row
  pivot_longer(
    cols = c(black_vs_white, hisp_vs_white),
    names_to = "contrast",
    values_to = "ratio"
  ) %>%
  # Summarize with mean_hdi -> returns columns .mean, .lower, .upper
  group_by(contrast) %>%
  mean_hdi(ratio, .width = 0.95)

posterior_ratios


ggplot(posterior_ratios, aes(x = contrast, y = ratio)) +
  geom_point(size=3) +
  geom_hline(yintercept=1, lty=2)+
  geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0.1) +
  labs(
    x = NULL,
    y = "Ratio relative to White",
    title = "Ethnicity vs. White Stop Ratios",
    caption = "Points = posterior means; bars = 95% HDI"
  )

