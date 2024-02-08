# Import libraries
library(ScottKnottESD)
library(readr)
library(ggplot2)

POINTS <- 1000
DIR <- "data/ecsa2023ext/"

create_plot <- function(in_file, metric, out_file, y_lim) {
      # apply ScottKnottESD and prepare a ScottKnottESD dataframe
      dataset <- read_csv(in_file)
      sk_results <- sk_esd(dataset)
      sk_ranks <- data.frame(model = names(sk_results$groups),
                  rank = paste0('Rank-', sk_results$groups))

      # prepare a dataframe for generating a visualisation
      plot_data <- melt(dataset)
      plot_data <- merge(plot_data, sk_ranks, by.x = 'variable', by.y = 'model')

      # generate a visualisation
      g <- ggplot(data = plot_data, aes(x = variable, y = value, fill = rank)) +
      geom_boxplot() +
      ylim(y_lim) +
      facet_grid(~rank, scales = 'free_x') +
      scale_fill_brewer(direction = -1) +
      ylab(metric) + xlab('Model') + ggtitle('') + theme_bw() +
      theme(text = element_text(size = 16),
            legend.position = 'none')
      #pdf(out_file)
      #print(g)
      #dev.off()
      ggsave(out_file, width = 5, height = 5)
}


create_plot(
      paste(DIR, "auc_dataset", POINTS, ".csv", sep = ""),
      "AUC", paste(DIR, "plots/rank_cls", POINTS, ".pdf", sep = ""),
      c(0.5, 1)
)
create_plot(
      paste(DIR, "nmse_dataset", POINTS, ".csv", sep = ""),
      "NMSE", paste(DIR, "plots/rank_regr", POINTS, ".pdf", sep = ""),
      c(-0.1, 0)
)
