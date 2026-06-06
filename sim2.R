# ==============================================================================
# PR-01: VALIDATION MECHANISM (GNA/GVA)
# ==============================================================================

# 1. Environment Setup and Library Loading [cite: 31, 32]
if (!require("randtoolbox")) install.packages("randtoolbox")
library(randtoolbox) # [cite: 33]

# 2. Linear Congruential Generator (LCG) Implementation 
# X(n+1) = (aXn + c) mod m 
m <- 2^31 # [cite: 36]
a <- 1103515245 # [cite: 37]
c <- 12345 # [cite: 38]
seed <- 42 # [cite: 39]

generate_lcg <- function(n, seed, m, a, c) { # 
  rng_values <- numeric(n)
  current <- seed
  for (i in 1:n) {
    current <- (a * current + c) %% m
    rng_values[i] <- current / m
  }
  return(rng_values)
}

# Generate a sequence of 10,000 numbers [cite: 49, 50]
numbers_custom <- generate_lcg(10000, seed, m, a, c)

# 3. Helper Function for Automatic Interpretation [cite: 51, 53]
interpret <- function(result, test_name) {
  p_val <- result$p.value
  cat("\n======================================\n")
  cat(" TEST:", test_name, "\n")
  cat("======================================\n")
  print(result)
  
  if (p_val < 0.05) { # [cite: 59]
    cat("\nVERDICT: THE SEQUENCE DOES NOT APPEAR RANDOM.\n") # [cite: 60]
  } else {
    cat("\nVERDICT: THE SEQUENCE APPEARS RANDOM.\n") # [cite: 63]
  }
}


# 4. Execution of the Statistical Test Battery (At least 5 tests)
# A. Gap Test: controlla la distribuzione degli intervalli
interpret(gap.test(numbers_custom), "Gap Test")

# B. Frequency (Monobit) Test: controlla l'uniformità globale
interpret(freq.test(numbers_custom), "Frequency Test")

# C. Order Test: controlla le permutazioni in dimensioni multiple
interpret(order.test(numbers_custom, d = 4), "Order Test")

# D. Poker Test: analizza i pattern (come le mani a poker)
interpret(poker.test(numbers_custom), "Poker Test")

# E. Serial Test: verifica l'indipendenza di coppie o n-uple consecutive
interpret(serial.test(numbers_custom), "Serial Test")