# Reference values from R for the nomibench tests (synthetic data in tests/data).
#   Rscript tests/reference/make_reference.R
# Firth estimates: brglm2 (Kosmidis & Firth 2021), type "AS_mean", which for logistic regression is Firth's
#   bias-reduced estimator (identical to Jeffreys-prior penalised ML).
# Conditional logistic: survival::clogit(method = "exact"), unpenalised.
# Mantel-Haenszel: stats::mantelhaen.test (common OR; CI with the Robins-Breslow-Greenland variance).
# Fisher: stats::fisher.test (conditional MLE OR, exact CI, two-sided P).
suppressMessages({library(brglm2); library(survival)})
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
dat <- file.path(here, "..", "data")
num <- function(x) sprintf("%.17g", x)
out <- c("{", sprintf('"_source": "R %s; brglm2 %s; survival %s",', paste(R.version$major, R.version$minor, sep = "."),
                      packageVersion("brglm2"), packageVersion("survival")))
sets <- list(moderate = c("x1", "x2", "x3"), quasi_separation = c("b", "c"), rare_events = c("e", "z", "z2"),
             separation = c("u", "v"))
for (nm in names(sets)) {
  d <- read.csv(file.path(dat, paste0("firth_", nm, ".csv")))
  f <- glm(reformulate(sets[[nm]], "y"), family = binomial, data = d, method = "brglmFit", type = "AS_mean",
           control = brglmControl(epsilon = 1e-12, maxit = 1000))
  out <- c(out, sprintf('"brglm2_%s": [%s],', nm, paste(num(coef(f)), collapse = ", ")))
}
g <- read.csv(file.path(dat, "g1_design.csv"))      # G1's actual design [1, E, s(logit PS)], no extra intercept
fg <- glm(y ~ 0 + c0 + c1 + c2 + c3 + c4, family = binomial, data = g, method = "brglmFit", type = "AS_mean",
          control = brglmControl(epsilon = 1e-12, maxit = 1000))
out <- c(out, sprintf('"brglm2_g1_design": [%s],', paste(num(coef(fg)), collapse = ", ")))
cd <- read.csv(file.path(dat, "conditional.csv"))
cf <- clogit(y ~ x1 + x2 + strata(stratum), data = cd, method = "exact")
out <- c(out, sprintf('"clogit_coef": [%s],', paste(num(coef(cf)), collapse = ", ")),
         sprintf('"clogit_loglik": %s,', num(cf$loglik[2])))
mh <- read.csv(file.path(dat, "mh_tables.csv"))
arr <- array(0, dim = c(2, 2, nrow(mh)))
for (k in seq_len(nrow(mh))) arr[, , k] <- matrix(c(mh$a[k], mh$c[k], mh$b[k], mh$d[k]), 2, 2)
m <- mantelhaen.test(arr, exact = FALSE)
out <- c(out, sprintf('"mh_or": %s,', num(m$estimate)), sprintf('"mh_ci": [%s, %s],', num(m$conf.int[1]), num(m$conf.int[2])))
tabs <- list(c(12, 500, 170, 10400), c(3, 250, 150, 9000), c(0, 30, 12, 200), c(20, 300, 150, 9800))
fis <- sapply(tabs, function(t) { r <- fisher.test(matrix(t, 2, 2, byrow = TRUE));
  sprintf("[%s, %s, %s, %s]", num(r$estimate), num(r$conf.int[1]), num(r$conf.int[2]), num(r$p.value)) })
out <- c(out, sprintf('"fisher_tables": [%s],', paste(sapply(tabs, function(t) sprintf("[%s]", paste(t, collapse = ", "))), collapse = ", ")),
         sprintf('"fisher": [%s]', paste(fis, collapse = ", ")), "}")
writeLines(out, file.path(here, "r_reference.json"))
cat(out, sep = "\n")
