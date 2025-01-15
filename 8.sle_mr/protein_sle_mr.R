pacman::p_load(stringi, tidyverse, TwoSampleMR)
source("/Volumes/data_files/SLE/liftover/f.R")

exposure_files <- list.files("/Volumes/data_files/SLE/ppp", pattern = "\\.gz$", full.names = TRUE)
outcome_file <- '/Volumes/data_files/SLE/SLE_GWAS_results_with_hg38.txt'

pattern <- c("chr", "pos", "beta", "se", "p")
replacement <- c("CHR", "POS", "BETA", "SE", "P")

results <- data.frame(
  Outcome = character(),
  Exposure = character(),
  Method = character(),
  Nsnp = numeric(),
  Beta = numeric(),
  Se = numeric(),
  P = numeric(),
  Pfdr = numeric(),
  Q = numeric(),
  Pheterogeneity = numeric(),
  Eggerintercept = numeric(),
  Ppleiotropy = numeric(),
  stringsAsFactors = FALSE
)

dat.Y.raw <- read.table(outcome_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
names(dat.Y.raw) <- toupper(names(dat.Y.raw))
dat.Y.raw <- dat.Y.raw %>%
  rename(CHR_OLD = CHR)

dat.Y.raw <- dat.Y.raw %>%
  rename(
    SNP = RSID,
    CHR = CHR_HG38,
    POS = START_HG38,
    EA = EFFECT_ALLELE,
    NEA = OTHER_ALLELE,
    BETA = BETA,
    SE = SE,
    P = PVAL
  ) %>%
  mutate(N = NA)

for (file in exposure_files) {
  exposure_name <- tools::file_path_sans_ext(basename(file))

  dat.X.raw <- read.table(file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  names(dat.X.raw) <- stri_replace_all_regex(
    toupper(names(dat.X.raw)),
    pattern = toupper(pattern),
    replacement = replacement,
    vectorize_all = FALSE
  )

  dat.X.sig <- dat.X.raw %>%
    filter(P <= 5e-8) %>%
    mutate(mb = ceiling(POS / 1e5))

  dat.X.iv <- dat.X.sig %>%
    group_by(mb) %>%
    slice(which.min(P)) %>%
    ungroup() %>%
    select(SNP)

  dat.X <- dat.X.raw %>%
    merge(dat.X.iv, by = "SNP") %>%
    format_data(
      type = "exposure",
      snp_col = "SNP",
      chr_col = "CHR",
      pos_col = "POS",
      effect_allele_col = "EA",
      other_allele_col = "NEA",
      samplesize_col = "N",
      beta_col = "BETA",
      se_col = "SE",
      pval_col = "P"
    )

  dat.Y <- dat.Y.raw %>%
    merge(dat.X.iv, by = "SNP") %>%
    format_data(
      type = "outcome",
      snp_col = "SNP",
      chr_col = "CHR",
      pos_col = "POS",
      effect_allele_col = "EA",
      other_allele_col = "NEA",
      samplesize_col = "N",
      beta_col = "BETA",
      se_col = "SE",
      pval_col = "P"
    )

  dat <- harmonise_data(dat.X, dat.Y, action = 1)

  mr_results <- mr(dat)

  heterogeneity <- mr_heterogeneity(dat)
  pleiotropy <- mr_pleiotropy_test(dat)

  for (i in 1:nrow(mr_results)) {
    results <- rbind(
      results,
      data.frame(
        Outcome = "SLE",
        Exposure = exposure_name,
        Method = mr_results$method[i],
        Nsnp = mr_results$nsnp[i],
        Beta = mr_results$b[i],
        Se = mr_results$se[i],
        P = mr_results$pval[i],
        Pfdr = p.adjust(mr_results$pval[i], method = "fdr"),
        Q = heterogeneity$Q[i],
        Pheterogeneity = heterogeneity$Q_pval[i],
        Eggerintercept = ifelse(mr_results$method[i] == "MR Egger", pleiotropy$egger_intercept, NA),
        Ppleiotropy = ifelse(mr_results$method[i] == "MR Egger", pleiotropy$pval, NA)
      )
    )
  }
}
write.table(results, "/Volumes/data_files/SLE/MR_results.tsv", sep = "\t", row.names = FALSE, quote = FALSE)


library(ggplot2)
results <- read_tsv("/Volumes/data_files/SLE/MR_results.tsv")
ivw_results <- results[results$Method == "Inverse variance weighted", ]
ivw_results <- ivw_results[order(ivw_results$Beta), ]

ggplot(ivw_results, aes(x = reorder(Exposure, Beta), y = Beta, ymin = Beta - 1.96 * Se, ymax = Beta + 1.96 * Se)) +
  geom_pointrange(aes(color = Exposure), size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  coord_flip() +
  labs(
    title = "Forest Plot of IVW Method Results (Sorted by Beta)",
    x = "Exposure",
    y = "Beta (95% CI)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )
