functions {
    vector logistic(real t, vector y, real beta, real K) {
        vector[1] dydt;

        dydt[1] = beta*y[1]*(1 - y[1]/K);

        return dydt;
    }
}

data {
    int<lower=0> N;  // Number of data points
    int<lower=0> T;  // Number of time points
    vector[N] y;  // Input data
    array[T] real time;
    array[N] int<lower=1, upper=T> time_idx;  // Index of time vector in data
    int<lower=0, upper=1> include_likelihood;
}

parameters {
    real<lower=0> K;  // Carrying capacity
    real<lower=0> beta;  // Growth rate
    real<lower=0> y0;  // Initial condition
    real<lower=0> sigma;  // Noise scale
}

transformed parameters {
    // Solve ODE
    array[T - 1] vector[1] mu_hat = ode_rk45(logistic, to_vector({y0}), time[1], time[2:T], beta, K);
    array[T] vector[1] mu;

    mu[1, 1] = y0;
    mu[2:T, 1] = mu_hat[, 1];
}

model {
    K ~ normal(10, 1);
    beta ~ std_normal();
    sigma ~ std_normal();
    y0 ~ std_normal();

    if (include_likelihood) y ~ normal(mu[time_idx, 1], sigma);
}

generated quantities {
   array[N] real f_reps = mu[time_idx, 1];  // Posterior samples of ODE solution
   array[N] real y_reps = normal_rng(mu[time_idx, 1], sigma);  // Posterior predictive
}
