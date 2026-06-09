###############################################################################
# prng_validation.R
# -----------------------------------------------------------------------------
# GNA/GVA part of the practical: generate a stream of pseudo-random numbers
# with our own generator and submit it to a battery of >= 5 statistical tests.
#
# Generator : Linear Congruential Generator (LCG)
#             X(n+1) = (a * X(n) + c) mod m     U = X / m  in [0,1)
#             -- IDENTICAL parameters to the LCG in queue_sim.py, so the R
#                validation and the Python simulation share one generator.
#
# Battery (8 tests, well above the required 5):
#   1. Frequency / Monobit      (randtoolbox::freq.test)
#   2. Serial                   (randtoolbox::serial.test)
#   3. Gap                      (randtoolbox::gap.test)
#   4. Poker                    (randtoolbox::poker.test)
#   5. Order                    (randtoolbox::order.test)
#   6. Chi-square uniformity    (base R, manual)
#   7. Kolmogorov-Smirnov       (base R, ks.test against Uniform(0,1))
#   8. Runs test (up/down)      (manual, independence of successive values)
#   + ACF plot                  (graphical independence check)
#
# Decision rule: at significance level alpha = 0.05, a p-value < 0.05 is
# evidence AGAINST randomness for that test.
###############################################################################

# ---- 1. Environment ---------------------------------------------------------
if (!require("randtoolbox")) install.packages("randtoolbox", repos = "https://cloud.r-project.org")
library(randtoolbox)

set.seed(42)  # only affects base-R helpers; our LCG seed is explicit below

# ---- 2. Linear Congruential Generator ---------------------------------------
m    <- 2^31
a    <- 1103515245
c    <- 12345
seed <- 42

generate_lcg <- function(n, seed, m, a, c) {
  rng_values <- numeric(n)
  current <- seed
  for (i in 1:n) {
    current <- (a * current + c) %% m
    rng_values[i] <- current / m
  }
  rng_values
}

N <- 10000
u <- generate_lcg(N, seed, m, a, c)

cat(sprintf("Generated %d LCG values. mean=%.4f (expect ~0.5), var=%.4f (expect ~0.0833)\n",
            N, mean(u), var(u)))

# ---- 3. Robust interpreter --------------------------------------------------
# randtoolbox tests sometimes return p.value as a vector; reduce to the most
# conservative (smallest) p-value so a single failing dimension is not hidden.
interpret <- function(result, test_name) {
  p_raw <- tryCatch(result$p.value, error = function(e) NA)
  p_val <- suppressWarnings(min(as.numeric(p_raw), na.rm = TRUE))

  cat("\n======================================\n")
  cat(" TEST:", test_name, "\n")
  cat("======================================\n")
  cat(sprintf(" p-value (min over dimensions): %.5f\n", p_val))

  if (is.finite(p_val) && p_val < 0.05) {
    cat(" VERDICT: NOT random  (p < 0.05 -> significant pattern)\n")
    return(FALSE)
  } else {
    cat(" VERDICT: appears random  (p >= 0.05 -> no evidence against)\n")
    return(TRUE)
  }
}

results <- logical(0)

# ---- 4. The test battery ----------------------------------------------------

# 1. Frequency / Monobit: are values spread uniformly over [0,1]?
results["Frequency"] <- interpret(freq.test(u), "Frequency (Monobit)")

# 2. Serial: are consecutive PAIRS uniform over the unit square?
results["Serial"]    <- interpret(serial.test(u, d = 8), "Serial")

# 3. Gap: are gaps between values falling in [0,0.5] geometric?
results["Gap"]       <- interpret(gap.test(u, lower = 0, upper = 0.5), "Gap")

# 4. Poker: treating digits as 'cards', do hand patterns match expectation?
results["Poker"]     <- interpret(poker.test(u), "Poker")

# 5. Order: do permutations of d consecutive values occur equally often?
#    order.test needs length divisible by d.
d_ord <- 5
u_ord <- u[1:(floor(N / d_ord) * d_ord)]
results["Order"]     <- interpret(order.test(u_ord, d = d_ord), "Order")

# 6. Chi-square uniformity (manual): bin into k cells, compare to uniform.
k <- 20
bins <- cut(u, breaks = seq(0, 1, length.out = k + 1), include.lowest = TRUE)
chi  <- chisq.test(table(bins))
results["ChiSquare"] <- interpret(list(p.value = chi$p.value), "Chi-square uniformity")

# 7. Kolmogorov-Smirnov against the theoretical Uniform(0,1) CDF.
ks <- ks.test(u, "punif")
results["KS"]        <- interpret(list(p.value = ks$p.value), "Kolmogorov-Smirnov")

# 8. Runs test (up/down) -- independence of the SEQUENCE order.
runs_test <- function(x) {
  s <- sign(diff(x)); s <- s[s != 0]
  runs <- 1 + sum(s[-1] != s[-length(s)])
  n <- length(s) + 1
  mu_r <- (2 * n - 1) / 3
  var_r <- (16 * n - 29) / 90
  z <- (runs - mu_r) / sqrt(var_r)
  list(p.value = 2 * (1 - pnorm(abs(z))), z = z, runs = runs)
}
results["Runs"]      <- interpret(runs_test(u), "Runs (up/down)")

# ---- 5. Graphical complement: autocorrelation -------------------------------
# An ideal generator shows no significant autocorrelation at any non-zero lag.
acf(u, lag.max = 40, main = "ACF of LCG output (independence check)")

# ---- 6. Summary -------------------------------------------------------------
cat("\n\n##################### BATTERY SUMMARY #####################\n")
for (nm in names(results)) {
  cat(sprintf("  %-12s : %s\n", nm, ifelse(results[nm], "PASS", "FAIL")))
}
cat(sprintf("  ----------------------------------------\n"))
cat(sprintf("  PASSED %d / %d tests at alpha = 0.05\n",
            sum(results), length(results)))
cat("###########################################################\n")

# Note: this textbook LCG (a=1103515245) is known to have weak low-order bits
# and can FAIL serial/order-type tests in higher dimensions. That is the point
# of the exercise -- the battery is what tells you whether a generator is fit
# for a simulation. Swapping in a better generator (e.g. Mersenne-Twister via
# runif) and re-running the same battery is the recommended follow-up.
