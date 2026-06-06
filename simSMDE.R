# B. Gap Test (Test degli Intervalli) CORRETTO PER DATI DISCRETI
gap.test <- function(x, lower = 0.0, upper = 0.5) {
  in_interval <- (x >= lower) & (x <= upper)
  occurrences <- which(in_interval)
  gaps <- diff(occurrences) - 1
  p_prob <- upper - lower
  
  # Raggruppiamo i salti (0, 1, 2, 3, 4, e "5 o più")
  max_gap <- 5
  gaps_capped <- ifelse(gaps > max_gap, max_gap, gaps)
  
  # Contiamo quante volte si è verificato ogni gap
  observed <- as.numeric(table(factor(gaps_capped, levels = 0:max_gap)))
  
  # Calcoliamo le probabilità teoriche usando la Distribuzione Geometrica
  expected_probs <- dgeom(0:(max_gap-1), prob = p_prob)
  expected_probs <- c(expected_probs, (1-p_prob)^max_gap)
  
  # Calcoliamo i conteggi attesi
  expected_counts <- length(gaps) * expected_probs
  
  # Applichiamo il test del Chi-Quadrato per dati discreti
  chisq_stat <- sum((observed - expected_counts)^2 / expected_counts)
  p_val <- 1 - pchisq(chisq_stat, df = max_gap)
  
  return(list(p.value = p_val, statistic = chisq_stat, method = "Gap Test (Chi-Square)"))
}

# Rilanciamo il test
interpret(gap.test(numbers_custom), "Gap Test")