// data {
//     int<lower=0> N_x;  // Number of observed data points
//     int<lower=0> N_z;  // Number of predictions
//     int<lower=0> N_m;  // Number of missing data points
//     int<lower=0> N_b;  // Number of blank data points
//     int<lower=0> N_e;  // Number of expressed data points
//     array[N_x] int<lower=1, upper=N_x + N_z + N_m> x_idx;  // Training index
//     array[N_z] int<lower=1, upper=N_x + N_z + N_m> z_idx;  // Testing index
//     array[N_m] int<lower=1, upper=N_x + N_z + N_m> m_idx;  // Missing index
//     array[N_b] int<lower=1, upper=N_x + N_z + N_m> b_idx;  // Blank index
//     array[N_e] int<lower=1, upper=N_x + N_z + N_m> e_idx;  // Expressed index
//     vector[N_x + N_z + N_m] IC;
//     vector[N_x] X;
//     vector[N_z] Z;
//     vector[N_x + N_z + N_m] t;
//     real mu0;
// }

//
// Latent logistic function with delay in beginning growth modelled with multiplicative noise
//
data {
  int<lower=1> N_obs;  // number of obs
  int<lower=0> N_miss;  // number of missing obs
  int<lower=0> N_pred;  // number of predictions
  int<lower=0> N_c;  // number of blanks
  int<lower=0> N_exp;  // number of exps (not blanks)
  array[N_c] int<lower=0> c_idx;
  array[N_exp] int<lower=0> exp_idx;
  array[N_obs] int<lower=1, upper=N_obs+N_miss+N_pred> y_train_idx;
  array[N_miss] int<lower=1, upper=N_obs+N_miss+N_pred> y_missing_idx;
  array[N_pred] int<lower=1, upper=N_obs+N_miss+N_pred> y_test_idx;
  vector[N_obs+N_miss+N_pred] IC;
  vector[N_obs] y_obs_train;
  vector[N_pred] y_obs_test;
  vector[N_obs+N_miss+N_pred] time;
  real mu_0;
  int<lower=0, upper=1> include_likelihood;
}

transformed data {
  int<lower=0> N = N_obs+N_miss+N_pred;
  int<lower=0> N_log_lik = N*include_likelihood;
  int<lower=0> N_miss_l = N_miss*include_likelihood;
  int<lower=0> N_pred_l = N_pred*include_likelihood;
  real<lower=0> sol_volume = 200;
  vector[N] IC_scaled = IC ./ sol_volume;
}

parameters {
    real<lower=0> tau;
    real<lower=log10(max(IC_scaled))> L;
    real<lower=log10(min(IC_scaled))> delta_tilde;
    real<lower=0> beta;
    real<lower=0> basal;
    real<lower=0> sigma_meas;
    vector<lower=0>[N_miss_l] y_missing;
    vector<lower=0>[N_pred_l] y_test;
}

transformed parameters {
    vector[N_exp] f;
    real<lower=max(IC_scaled)> K = pow(10, L);
    real<lower=min(IC_scaled)> delta = pow(10, delta_tilde);
    vector[N] y;
    y[y_train_idx] = y_obs_train;
    if (include_likelihood) {
        y[y_missing_idx] = y_missing;
        y[y_test_idx] = y_test;
    }
    for (i in 1:N_exp) {
        f[i] = K/(1 + ((K - IC_scaled[exp_idx[i]])/IC_scaled[exp_idx[i]])*exp(-(beta)*fmax(time[exp_idx[i]] - tau, 0.0)));
    }
}

model {
    tau ~ gamma(5, 1);
    sigma_meas ~ normal(0, 0.5);
    L ~ normal(9, 2);
    beta ~ std_normal();
    basal ~ lognormal(log(mu_0), 1);
    delta_tilde ~ cauchy(log10(max(IC_scaled)), 1);

    if (include_likelihood) {
        y[exp_idx] ~ lognormal(log(basal + f/delta), sigma_meas);
        y[c_idx] ~ lognormal(log(basal), sigma_meas);
    }
}

generated quantities {
    array[N] real y_tot;
    array[N_obs] real y_rep;
    array[N_pred] real y_pred;
    vector[N_log_lik] log_lik;
    vector[N] y_obs;
    y_obs[y_train_idx] = y_obs_train;
    y_obs[y_missing_idx] = y[y_missing_idx];
    y_obs[y_test_idx] = y_obs_test;

    y_tot[exp_idx] = lognormal_rng(log(basal + f/delta), sigma_meas);
    for (i in c_idx)
    y_tot[i] = lognormal_rng(log(basal), sigma_meas);
    if (include_likelihood) {
    for (i in c_idx)
        log_lik[i] = lognormal_lpdf(y_obs[i] | log(basal), sigma_meas);
    for (j in 1:N_exp)
        log_lik[exp_idx[j]] = lognormal_lpdf(y_obs[exp_idx[j]] | log(basal + f[j]/delta), sigma_meas);
    }
    y_rep = y_tot[y_train_idx];
    y_pred = y_tot[y_test_idx];
}
