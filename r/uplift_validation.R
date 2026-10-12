# Supplementary R validation: uplift decile lift and Qini-style check.
#
# This script is a companion to the Python pipeline (it is not executed by
# run_all.py). To use it, export the scored marketing sample first:
#
#   python -c "
#   import sys; sys.path.insert(0, 'src')
#   from generate_data import generate_marketing
#   from marketing_models import train_uplift_two_model
#   df = generate_marketing(); scored, _ = train_uplift_two_model(df)
#   scored.to_csv('outputs/marketing_scored.csv', index=False)"
#
# Then run:  Rscript r/uplift_validation.R

scored <- read.csv("outputs/marketing_scored.csv")

# Rank by uplift score, split into deciles (1 = highest uplift)
scored$decile <- as.integer(cut(
  rank(scored$uplift_score, ties.method = "first"),
  breaks = quantile(rank(scored$uplift_score, ties.method = "first"),
                    probs = seq(0, 1, 0.1)),
  labels = FALSE, include.lowest = TRUE
))
scored$decile <- 11L - scored$decile  # decile 1 = top

overall_rate <- mean(scored$response)
treated <- scored[scored$treatment == 1, ]

decile_tbl <- aggregate(response ~ decile, data = treated, FUN = mean)
names(decile_tbl)[2] <- "treated_response_rate"
decile_tbl$lift <- decile_tbl$treated_response_rate / overall_rate - 1
print(decile_tbl[order(decile_tbl$decile), ])

top3_rate <- mean(treated$response[treated$decile <= 3])
cat(sprintf(
  "\nTop-3-decile lift: %.1f%% (treated top-3 rate %.1f%% vs overall %.1f%%)\n",
  100 * (top3_rate / overall_rate - 1), 100 * top3_rate, 100 * overall_rate
))

# Qini-style curve: cumulative incremental responses vs random targeting
ord <- scored[order(-scored$uplift_score), ]
ord$cum_treated_resp <- cumsum(ord$treatment * ord$response)
ord$cum_control_resp <- cumsum((1 - ord$treatment) * ord$response)
n_t <- sum(ord$treatment); n_c <- sum(ord$treatment == 0)
ord$qini <- ord$cum_treated_resp - ord$cum_control_resp * n_t / n_c
png("docs/assets/qini_curve_r.png", width = 900, height = 500)
plot(seq_len(nrow(ord)) / nrow(ord), ord$qini, type = "l", lwd = 2,
     xlab = "Share of population targeted (by uplift rank)",
     ylab = "Cumulative incremental responses",
     main = "Qini curve (R validation)")
abline(h = tail(ord$qini, 1), lty = 2, col = "grey40")
dev.off()
cat("Qini curve written to docs/assets/qini_curve_r.png\n")
