suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(ggrepel)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grepl("^--file=", args)]
script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[[1]]) else "scripts/figures_r/make_nn_figures.R"
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE)
source(file.path(repo_root, "scripts", "figures_r", "theme_nn.R"), chdir = TRUE)

fig_dir <- file.path(repo_root, "paper", "figs")
review_dir <- file.path(repo_root, "results", "figure_review_r")
source_dir <- file.path(review_dir, "source_data_used")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(review_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

final_metrics_path <- file.path(repo_root, "results", "neural_networks_submission", "csv", "final_metrics.csv")
gate_diag_path <- file.path(repo_root, "results", "neural_networks_submission", "csv", "gate_diagnostics.csv")
per_eval_path <- file.path(repo_root, "results", "neural_networks_submission", "csv", "per_eval_metrics.csv")
frontier_path <- file.path(repo_root, "results", "neural_networks_submission", "reviewer_analysis", "integrated_frontier.csv")
aer_path <- file.path(repo_root, "results", "neural_networks_submission", "reviewer_analysis", "aer_summary.csv")

display_colors <- setNames(unname(method_colors), label_method(names(method_colors)))
display_shapes <- setNames(unname(method_shapes), label_method(names(method_shapes)))
method_linetypes <- c(
  er = "solid", derpp = "longdash", gate_loss = "solid", gate_spr = "dashed",
  gate_conf = "dotdash", gate_predstab = "twodash", gate_reprstab = "dotted",
  gate_teacher = "longdash", gate_coteach = "dashed", ewc = "dotdash",
  si = "dotted", oracle = "solid"
)
display_linetypes <- setNames(unname(method_linetypes), label_method(names(method_linetypes)))

read_source <- function(path) {
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

write_source <- function(name, data) {
  out <- file.path(source_dir, paste0(name, ".csv"))
  readr::write_csv(data, out)
  message("source data: ", normalizePath(out, winslash = "/", mustWork = FALSE))
  print(utils::head(data, 12))
}

save_plot <- function(plot, name, width, height) {
  pdf_path <- file.path(fig_dir, paste0(name, ".pdf"))
  png_path <- file.path(review_dir, paste0(name, ".png"))
  ggplot2::ggsave(pdf_path, plot, width = width, height = height, units = "in",
                  device = grDevices::cairo_pdf)
  ggplot2::ggsave(png_path, plot, width = width, height = height, units = "in",
                  dpi = 300, bg = "white")
  message("wrote: ", normalizePath(pdf_path, winslash = "/", mustWork = FALSE))
  message("preview: ", normalizePath(png_path, winslash = "/", mustWork = FALSE))
}

mean_sd <- function(data, groups, value) {
  data %>%
    group_by(across(all_of(groups))) %>%
    summarise(
      mean = mean(.data[[value]], na.rm = TRUE),
      sd = sd(.data[[value]], na.rm = TRUE),
      n = sum(!is.na(.data[[value]])),
      .groups = "drop"
    ) %>%
    mutate(sd = if_else(is.na(sd), 0, sd))
}

noise_label <- function(x) paste0(round(100 * x), "%")

condition_label <- c(
  sym20 = "Symmetric 20%",
  sym60 = "Symmetric 60%",
  asym40 = "Asymmetric 40%",
  c10n_aggre = "Aggregate",
  c10n_worse = "Worst"
)

fig4_buffer_gate <- function(final_metrics) {
  selected <- c("er", "gate_loss", "gate_spr", "gate_conf", "gate_predstab",
                "gate_reprstab", "gate_teacher", "gate_coteach", "oracle")
  base <- final_metrics %>%
    filter(benchmark == "split_mnist", noise_type == "symmetric", method %in% selected) %>%
    mutate(method_label = order_methods(method))

  purity <- mean_sd(base, c("method", "method_label", "noise_rate"), "buffer_purity") %>%
    filter(!is.nan(mean)) %>%
    mutate(panel = "(a)", metric = "Buffer purity")
  corr <- mean_sd(base, c("method", "method_label", "noise_rate"), "gate_corr") %>%
    filter(!is.nan(mean)) %>%
    mutate(panel = "(b)", metric = "Gate-correctness correlation")
  sep <- mean_sd(base, c("method", "method_label", "noise_rate"), "gate_separation") %>%
    filter(!is.nan(mean)) %>%
    mutate(panel = "(c)", metric = "Gate separation")
  src <- bind_rows(purity, corr, sep)
  write_source("fig4_buffer_purity_gate_alignment", src)

  make_panel <- function(data, tag, ylab, ylim, shade = FALSE, zero = FALSE) {
    p <- ggplot(data, aes(noise_rate, mean, color = method_label, shape = method_label,
                          linetype = method_label, group = method_label))
    if (shade) {
      p <- p + annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
                        fill = "#C44E52", alpha = 0.08)
    }
    if (zero) {
      p <- p + geom_hline(yintercept = 0, color = "#333333", linewidth = 0.35)
    }
    tag_x <- 0.03
    tag_y <- ylim[2] - 0.065 * diff(ylim)
    p +
      annotate("text", x = tag_x, y = tag_y, label = tag,
               hjust = 0, vjust = 1, fontface = "bold", size = 3.0) +
      geom_line(linewidth = 0.6) +
      geom_point(size = 1.9, stroke = 0.7) +
      scale_x_continuous("Noise rate", limits = c(0, 0.8), breaks = seq(0, 0.8, by = 0.2),
                         labels = noise_label) +
      scale_y_continuous(ylab, limits = ylim, breaks = pretty_breaks(5)) +
      scale_color_manual(values = display_colors) +
      scale_shape_manual(values = display_shapes) +
      scale_linetype_manual(values = display_linetypes) +
      guides(color = guide_legend(nrow = 3), shape = "none", linetype = "none") +
      theme_nn(8.5)
  }

  legend_methods <- selected[selected %in% unique(base$method)]
  legend_df <- tibble(
    method = legend_methods,
    method_label = order_methods(legend_methods),
    col = rep(1:3, length.out = length(legend_methods)),
    row = rep(3:1, each = 3, length.out = length(legend_methods))
  )
  legend_plot <- ggplot(legend_df, aes(color = method_label, shape = method_label,
                                       linetype = method_label)) +
    geom_segment(aes(x = col - 0.42, xend = col - 0.25, y = row, yend = row),
                 linewidth = 0.55) +
    geom_point(aes(x = col - 0.335, y = row), size = 1.8, stroke = 0.7) +
    geom_text(aes(x = col - 0.18, y = row, label = method_label),
              hjust = 0, size = 2.45, color = "#202020") +
    scale_color_manual(values = display_colors) +
    scale_shape_manual(values = display_shapes) +
    scale_linetype_manual(values = display_linetypes) +
    coord_cartesian(xlim = c(0.55, 3.7), ylim = c(0.55, 3.45), clip = "off") +
    theme_void(base_size = 8.5) +
    theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))

  panels <- make_panel(purity, "(a)", "Buffer purity", c(0, 1), FALSE, FALSE) +
    make_panel(corr, "(b)", "Gate-correctness correlation", c(-1, 1), FALSE, TRUE) +
    make_panel(sep, "(c)", "Gate separation", c(-1, 1), TRUE, TRUE) +
    plot_layout(nrow = 1) &
    theme(legend.position = "none")
  plot <- panels / legend_plot + plot_layout(heights = c(1, 0.22))
  save_plot(plot, "fig4_buffer_purity_gate_alignment", 7.2, 2.75)
}

fig5_memorization <- function(gate_diag) {
  src <- gate_diag %>%
    filter(
      method == "gate_loss",
      (benchmark == "split_mnist" & noise_type == "symmetric" & abs(noise_rate - 0.6) < 1e-9) |
        (benchmark == "permuted_mnist" & noise_type == "symmetric" & abs(noise_rate - 0.4) < 1e-9)
    ) %>%
    mutate(
      condition = case_when(
        benchmark == "split_mnist" ~ "(a) Split-MNIST, 60% noise",
        TRUE ~ "(b) Permuted-MNIST, 40% noise"
      )
    ) %>%
    group_by(condition, benchmark, task) %>%
    summarise(
      clean_label = mean(gate_on_correct, na.rm = TRUE),
      mislabeled = mean(gate_on_wrong, na.rm = TRUE),
      gate_separation = mean(gate_separation, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(condition = factor(condition, levels = c("(a) Split-MNIST, 60% noise", "(b) Permuted-MNIST, 40% noise")))
  write_source("fig5_memorization_inversion", src)

  long <- src %>%
    pivot_longer(c(clean_label, mislabeled), names_to = "label_type", values_to = "mean_gate") %>%
    mutate(label_type = recode(label_type, clean_label = "Clean label", mislabeled = "Mislabeled"))
  shade <- src %>% filter(mislabeled > clean_label)

  p <- ggplot(long, aes(task + 1, mean_gate, color = label_type, linetype = label_type, group = label_type)) +
    geom_rect(data = shade, aes(xmin = task + 0.5, xmax = task + 1.5, ymin = -Inf, ymax = Inf),
              inherit.aes = FALSE, fill = "#C44E52", alpha = 0.08) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~ condition, nrow = 1) +
    scale_x_continuous("Task index", breaks = pretty_breaks()) +
    scale_y_continuous("Mean gate value", limits = c(0, 1), breaks = seq(0, 1, by = 0.25)) +
    scale_color_manual(values = c("Clean label" = "#0072B2", "Mislabeled" = "#D55E00")) +
    scale_linetype_manual(values = c("Clean label" = "solid", "Mislabeled" = "dashed")) +
    theme_nn(9) +
    guides(color = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1))
  p <- p + theme(strip.text = element_text(size = 8.8))
  save_plot(p, "fig5_memorization_inversion", 7.0, 2.6)
}

fig_integrated_frontier <- function() {
  central_methods <- c(
    "gate_coteach", "gate_spr", "oracle", "gate_loss", "derpp", "ewc",
    "gate_teacher", "gate_reprstab", "gate_conf", "gate_predstab", "si"
  )
  src <- read_source(frontier_path) %>%
    mutate(
      method_label = label_method(method),
      is_central = method %in% central_methods,
      family = case_when(
        method == "oracle" ~ "Oracle clean-selector",
        method %in% c("derpp", "ewc", "si") ~ "Baseline",
        !is_central ~ "Secondary control",
        method %in% c("gate_conf", "gate_predstab", "gate_reprstab") ~ "Label-free/stability",
        method == "gate_spr" ~ "SPR gate",
        method == "gate_teacher" ~ "Slow teacher",
        TRUE ~ "Gated replay"
      ),
      point_size = case_when(
        method == "oracle" ~ "Oracle",
        is_central ~ "Central",
        TRUE ~ "Secondary"
      )
    )
  write_source("fig_integrated_power_risk_frontier", src)

  x_lim <- range(src$power_mean_gain, 0, na.rm = TRUE)
  y_lim <- range(src$risk_worst_penalty, 0, na.rm = TRUE)
  x_pad <- diff(x_lim) * 0.12
  y_pad <- diff(y_lim) * 0.10
  x_limits <- c(x_lim[1] - x_pad, x_lim[2] + x_pad)
  y_limits <- c(-0.006, y_lim[2] + y_pad)
  label_data <- src %>% filter(is_central)

  p <- ggplot(src, aes(power_mean_gain, risk_worst_penalty)) +
    annotate("rect", xmin = 0, xmax = Inf, ymin = 0, ymax = 0.05,
             fill = "#009E73", alpha = 0.10) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#333333", linewidth = 0.35) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#333333", linewidth = 0.35) +
    geom_point(
      data = src %>% filter(!is_central),
      shape = 16, size = 1.8, color = "#9A9A9A", alpha = 0.35
    ) +
    geom_point(
      data = src %>% filter(is_central, method != "oracle"),
      aes(color = method_label, shape = family),
      size = 2.15, stroke = 0.75
    ) +
    geom_point(
      data = src %>% filter(method == "oracle"),
      shape = 23, size = 3.4, stroke = 1.05, color = "#111111", fill = "white"
    ) +
    geom_text_repel(
      data = label_data,
      aes(label = method_label, color = method_label),
      size = 2.35, box.padding = 0.55, point.padding = 0.24,
      min.segment.length = 0, max.overlaps = Inf, segment.size = 0.20,
      force = 18, force_pull = 0.35, max.time = 2, max.iter = 12000,
      nudge_x = if_else(label_data$method %in% c("derpp", "gate_teacher", "gate_loss", "ewc"), 0.012, 0),
      nudge_y = if_else(label_data$method == "oracle", -0.012, 0),
      show.legend = FALSE
    ) +
    annotate("text", x = 0.185, y = 0.025,
             label = "Favorable region:\ngain with low penalty", hjust = 1, vjust = 0.5,
             size = 2.55, color = "#007A52", lineheight = 0.9) +
    annotate("text", x = 0.012, y = 0.009,
             label = "Lower penalty = safer", hjust = 0, vjust = 0,
             size = 2.45, color = "#555555") +
    scale_x_continuous("Mean accuracy gain over ER", labels = label_percent(accuracy = 1),
                       limits = x_limits) +
    scale_y_continuous("Worst penalty versus ER (lower is safer)", labels = label_percent(accuracy = 1),
                       limits = y_limits) +
    scale_color_manual(values = display_colors, guide = "none") +
    scale_shape_manual(values = c(
      "Baseline" = 15,
      "Gated replay" = 16,
      "Label-free/stability" = 0,
      "SPR gate" = 18,
      "Slow teacher" = 8,
      "Oracle clean-selector" = 23
    )) +
    theme_nn(9) +
    theme(legend.position = "none")
  save_plot(p, "fig_integrated_power_risk_frontier", 5.9, 3.8)
}

fig3_main_cifar <- function(final_metrics) {
  methods <- c("er", "derpp", "gate_loss", "gate_conf", "oracle")
  src <- final_metrics %>%
    filter(benchmark %in% c("split_cifar10", "seq_cifar10"),
           noise_type == "symmetric", method %in% methods) %>%
    mutate(
      method_label = order_methods(method),
      benchmark_label = recode(benchmark, split_cifar10 = "Split-CIFAR-10", seq_cifar10 = "Seq-CIFAR-10")
    ) %>%
    group_by(benchmark, benchmark_label, method, method_label, noise_rate) %>%
    summarise(
      average_accuracy_mean = mean(average_accuracy, na.rm = TRUE),
      average_accuracy_sd = sd(average_accuracy, na.rm = TRUE),
      mean_forgetting_mean = mean(mean_forgetting, na.rm = TRUE),
      mean_forgetting_sd = sd(mean_forgetting, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(across(ends_with("_sd"), ~ if_else(is.na(.x), 0, .x)))
  write_source("fig3_main_cifar_benchmark", src)

  make_metric <- function(data, bench, mean_col, sd_col, tag, ylab) {
    d <- data %>% filter(benchmark == bench)
    ggplot(d, aes(noise_rate, .data[[mean_col]], color = method_label, shape = method_label,
                  linetype = method_label, group = method_label)) +
      geom_errorbar(aes(ymin = .data[[mean_col]] - .data[[sd_col]],
                        ymax = .data[[mean_col]] + .data[[sd_col]]),
                    width = 0.018, linewidth = 0.35, alpha = 0.8) +
      geom_line(linewidth = 0.58) +
      geom_point(size = 1.8, stroke = 0.7) +
      scale_x_continuous("Symmetric noise rate", breaks = c(0.2, 0.4, 0.6), labels = noise_label) +
      scale_y_continuous(ylab, labels = label_percent(accuracy = 1)) +
      scale_color_manual(values = display_colors) +
      scale_shape_manual(values = display_shapes) +
      scale_linetype_manual(values = display_linetypes) +
      guides(color = guide_legend(nrow = 1), shape = "none", linetype = "none") +
      theme_nn(8.5)
  }

  plot <- (make_metric(src, "split_cifar10", "average_accuracy_mean", "average_accuracy_sd", "(a)", "Final average accuracy") +
             make_metric(src, "split_cifar10", "mean_forgetting_mean", "mean_forgetting_sd", "(b)", "Mean forgetting")) /
    (make_metric(src, "seq_cifar10", "average_accuracy_mean", "average_accuracy_sd", "(c)", "Final average accuracy") +
       make_metric(src, "seq_cifar10", "mean_forgetting_mean", "mean_forgetting_sd", "(d)", "Mean forgetting")) +
    plot_layout(guides = "collect") +
    plot_annotation(tag_levels = list(c("(a)", "(b)", "(c)", "(d)"))) &
    theme(legend.position = "bottom",
          plot.tag = element_text(face = "bold", size = 9),
          plot.tag.position = c(0.01, 0.98))
  save_plot(plot, "fig3_main_cifar_benchmark", 7.2, 4.8)
}

figA_c10n_frontier <- function(final_metrics) {
  methods <- c("gate_loss", "gate_conf", "gate_coteach", "oracle")
  er <- final_metrics %>%
    filter(benchmark == "cifar10n", method == "er") %>%
    group_by(condition) %>%
    summarise(er_acc = mean(average_accuracy, na.rm = TRUE), .groups = "drop")
  src <- final_metrics %>%
    filter(benchmark == "cifar10n", method %in% methods) %>%
    group_by(condition, method) %>%
    summarise(
      gate_separation = mean(gate_separation, na.rm = TRUE),
      average_accuracy = mean(average_accuracy, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    left_join(er, by = "condition") %>%
    mutate(
      accuracy_gain = average_accuracy - er_acc,
      method_label = order_methods(method),
      label_set = recode(condition, !!!condition_label)
    ) %>%
    filter(!is.nan(gate_separation))
  write_source("figA_c10n_frontier", src)

  p <- ggplot(src, aes(gate_separation, accuracy_gain)) +
    annotate("rect", xmin = -Inf, xmax = 0, ymin = -Inf, ymax = Inf,
             fill = "#C44E52", alpha = 0.08) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#333333", linewidth = 0.35) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#333333", linewidth = 0.35) +
    geom_point(aes(color = method_label, shape = label_set), size = 2.7, stroke = 0.85) +
    geom_text_repel(aes(label = method_label, color = method_label),
                    size = 2.75, box.padding = 0.25, point.padding = 0.12,
                    max.overlaps = Inf, segment.size = 0.25, show.legend = FALSE) +
    annotate("text", x = -0.082, y = -0.035,
             label = "Inversion zone", hjust = 0, vjust = 0.5, size = 2.7, color = "#7A3030") +
    scale_x_continuous("Mean gate separation", limits = c(-0.10, 1.08), breaks = seq(0, 1, by = 0.25)) +
    scale_y_continuous("Accuracy gain over ER", limits = c(-0.045, 0.010),
                       labels = label_percent(accuracy = 0.1)) +
    scale_color_manual(values = display_colors, guide = "none") +
    scale_shape_manual(values = c("Aggregate" = 16, "Worst" = 17), name = "Label set") +
    theme_nn(9) +
    theme(legend.position = "bottom")
  save_plot(p, "figA_c10n_frontier", 5.7, 3.6)
}

fig7_c10n_alignment_purity <- function(final_metrics) {
  label_sets <- c(c10n_aggre = "Aggregate", c10n_worse = "Worst")
  sep_methods <- c("gate_loss", "gate_conf", "gate_coteach")
  purity_methods <- c("er", "derpp", "gate_loss", "gate_conf", "gate_coteach", "oracle")

  sep <- final_metrics %>%
    filter(benchmark == "cifar10n", method %in% sep_methods) %>%
    group_by(condition, method) %>%
    summarise(mean = mean(gate_separation, na.rm = TRUE), sd = sd(gate_separation, na.rm = TRUE), n = n(), .groups = "drop") %>%
    mutate(metric = "Gate separation", method_label = order_methods(method),
           label_set = recode(condition, !!!label_sets), sd = if_else(is.na(sd), 0, sd))
  purity <- final_metrics %>%
    filter(benchmark == "cifar10n", method %in% purity_methods) %>%
    group_by(condition, method) %>%
    summarise(mean = mean(buffer_purity, na.rm = TRUE), sd = sd(buffer_purity, na.rm = TRUE), n = n(), .groups = "drop") %>%
    mutate(metric = "Buffer purity", method_label = order_methods(method),
           label_set = recode(condition, !!!label_sets), sd = if_else(is.na(sd), 0, sd))
  write_source("fig7_c10n_alignment_purity", bind_rows(sep, purity))

  fill_values <- c("Aggregate" = "#8EC7C2", "Worst" = "#DDBB69")
  p_sep <- ggplot(sep, aes(method_label, mean, fill = label_set)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0,
             fill = "#C44E52", alpha = 0.08) +
    geom_hline(yintercept = 0, color = "#333333", linewidth = 0.35) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62, color = "#333333", linewidth = 0.25) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), position = position_dodge(width = 0.72),
                  width = 0.16, linewidth = 0.3) +
    scale_y_continuous("Gate separation", limits = c(-0.06, 0.72), breaks = pretty_breaks(5)) +
    scale_x_discrete(NULL) +
    scale_fill_manual(values = fill_values) +
    theme_nn(8.5) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  p_purity <- ggplot(purity, aes(method_label, mean, fill = label_set)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62, color = "#333333", linewidth = 0.25) +
    geom_errorbar(aes(ymin = pmax(0, mean - sd), ymax = pmin(1, mean + sd)),
                  position = position_dodge(width = 0.72), width = 0.16, linewidth = 0.3) +
    scale_y_continuous("Buffer purity", limits = c(0, 1), labels = label_percent(accuracy = 1)) +
    scale_x_discrete(NULL) +
    scale_fill_manual(values = fill_values) +
    theme_nn(8.5) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  plot <- p_sep + p_purity + plot_layout(guides = "collect", widths = c(1, 1.35)) &
    theme(legend.position = "bottom")
  plot <- plot + plot_annotation(tag_levels = list(c("(a)", "(b)"))) &
    theme(plot.tag = element_text(face = "bold", size = 9),
          plot.tag.position = c(0.01, 0.98))
  save_plot(plot, "fig7_c10n_alignment_purity", 7.2, 3.2)

  save_plot(p_sep + theme(legend.position = "bottom"), "figB_c10n_separation", 4.8, 3.1)
  save_plot(p_purity + theme(legend.position = "bottom"), "figC_c10n_purity", 5.6, 3.1)
}

figS1_taskwise <- function(per_eval) {
  methods <- c("er", "derpp", "gate_loss", "gate_conf", "oracle")
  src <- per_eval %>%
    filter(benchmark %in% c("split_cifar10", "seq_cifar10"),
           noise_type == "symmetric", noise_rate %in% c(0.2, 0.6), method %in% methods) %>%
    mutate(
      method_label = order_methods(method),
      task_learned = eval_after_task + 1,
      panel = case_when(
        benchmark == "split_cifar10" & noise_rate == 0.2 ~ "(a) Split-CIFAR-10 20%",
        benchmark == "split_cifar10" & noise_rate == 0.6 ~ "(b) Split-CIFAR-10 60%",
        benchmark == "seq_cifar10" & noise_rate == 0.2 ~ "(c) Seq-CIFAR-10 20%",
        TRUE ~ "(d) Seq-CIFAR-10 60%"
      ),
      panel = factor(panel, levels = c("(a) Split-CIFAR-10 20%", "(b) Split-CIFAR-10 60%",
                                       "(c) Seq-CIFAR-10 20%", "(d) Seq-CIFAR-10 60%"))
    ) %>%
    group_by(panel, benchmark, noise_rate, method, method_label, task_learned) %>%
    summarise(
      mean = mean(avg_acc_so_far, na.rm = TRUE),
      sd = sd(avg_acc_so_far, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(sd = if_else(is.na(sd), 0, sd))
  write_source("figS1_taskwise_cifar_trajectories", src)

  p <- ggplot(src, aes(task_learned, mean, color = method_label, linetype = method_label,
                       shape = method_label, group = method_label)) +
    geom_ribbon(aes(ymin = pmax(0, mean - sd), ymax = pmin(1, mean + sd), fill = method_label),
                alpha = 0.10, color = NA, show.legend = FALSE) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.45, stroke = 0.6) +
    facet_wrap(~ panel, nrow = 2) +
    scale_x_continuous("Tasks learned", breaks = pretty_breaks(5)) +
    scale_y_continuous("Average accuracy so far", labels = label_percent(accuracy = 1), limits = c(0, 1)) +
    scale_color_manual(values = display_colors) +
    scale_fill_manual(values = display_colors) +
    scale_shape_manual(values = display_shapes) +
    scale_linetype_manual(values = display_linetypes) +
    guides(color = guide_legend(nrow = 1), shape = "none", linetype = "none") +
    theme_nn(8.2) +
    theme(strip.text = element_text(size = 8.2))
  save_plot(p, "figS1_taskwise_cifar_trajectories", 7.2, 4.55)
}

figS2_aer <- function() {
  aer <- read_source(aer_path)
  aer_labeled <- aer %>%
    mutate(
      model_label = if_else(model == "er", "ER bridge", "AER"),
      condition_label = recode(condition, sym20 = "Symmetric 20%", sym60 = "Symmetric 60%", asym40 = "Asymmetric 40%")
    )
  acc_src <- bind_rows(
    aer_labeled %>%
      transmute(model, model_label, condition, condition_label, n,
                metric = "Class-IL", mean = class_il_mean, sd = class_il_std),
    aer_labeled %>%
      transmute(model, model_label, condition, condition_label, n,
                metric = "Masked Task-IL", mean = task_il_mean, sd = task_il_std)
  ) %>%
    filter(model != "er" | condition == "sym60") %>%
    mutate(
      condition_label = factor(condition_label, levels = c("Symmetric 20%", "Symmetric 60%", "Asymmetric 40%")),
      model_label = factor(model_label, levels = c("ER bridge", "AER")),
      series = case_when(
        model_label == "AER" & metric == "Class-IL" ~ "AER Class-IL",
        model_label == "AER" & metric == "Masked Task-IL" ~ "AER masked Task-IL",
        model_label == "ER bridge" & metric == "Class-IL" ~ "ER bridge Class-IL",
        TRUE ~ "ER bridge masked Task-IL"
      ),
      series = factor(series, levels = c(
        "AER Class-IL", "AER masked Task-IL",
        "ER bridge Class-IL", "ER bridge masked Task-IL"
      )),
      metric = factor(metric, levels = c("Class-IL", "Masked Task-IL"))
    )
  runtime_src <- aer %>%
    filter((model == "er" & condition == "sym60") | (model == "er_ace_aer_abs" & condition == "sym60")) %>%
    mutate(model_label = if_else(model == "er", "ER bridge", "AER"),
           model_label = factor(model_label, levels = c("ER bridge", "AER")),
           runtime_min = runtime_s_mean / 60) %>%
    arrange(model_label)
  write_source("figS2_aer_external_baseline", list(acc_src = acc_src, runtime_src = runtime_src) %>% bind_rows(.id = "source_table"))

  aer_bars <- acc_src %>% filter(model_label == "AER")
  er_points <- acc_src %>% filter(model_label == "ER bridge")
  dodge_acc <- position_dodge(width = 0.62)
  p_acc <- ggplot() +
    geom_col(data = aer_bars,
             aes(condition_label, mean, fill = series, group = metric),
             position = dodge_acc, width = 0.54, color = "#333333", linewidth = 0.25) +
    geom_errorbar(data = aer_bars,
                  aes(condition_label, ymin = mean - sd, ymax = mean + sd, group = metric),
                  position = dodge_acc, width = 0.11, linewidth = 0.3) +
    geom_point(data = er_points,
               aes(condition_label, mean, shape = series, color = series, group = metric),
               position = dodge_acc, size = 2.6, stroke = 0.9, fill = "white") +
    geom_errorbar(data = er_points,
                  aes(condition_label, ymin = mean - sd, ymax = mean + sd, group = metric),
                  position = dodge_acc, width = 0.11, linewidth = 0.3, color = "#333333") +
    scale_y_continuous("Accuracy (%)", limits = c(0, 95), breaks = pretty_breaks(5)) +
    scale_x_discrete(NULL) +
    scale_fill_manual(
      values = c("AER Class-IL" = "#5BA8A0", "AER masked Task-IL" = "#D9B84F"),
      breaks = c("AER Class-IL", "AER masked Task-IL")
    ) +
    scale_color_manual(
      values = c("ER bridge Class-IL" = "#111111", "ER bridge masked Task-IL" = "#555555"),
      breaks = c("ER bridge Class-IL", "ER bridge masked Task-IL")
    ) +
    scale_shape_manual(
      values = c("ER bridge Class-IL" = 21, "ER bridge masked Task-IL" = 24),
      breaks = c("ER bridge Class-IL", "ER bridge masked Task-IL")
    ) +
    guides(fill = guide_legend(nrow = 1), color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) +
    theme_nn(8.5) +
    theme(axis.text.x = element_text(angle = 15, hjust = 1))

  p_rt <- ggplot(runtime_src, aes(model_label, runtime_min, fill = model_label)) +
    geom_col(width = 0.56, color = "#333333", linewidth = 0.25, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.1f", runtime_min)), vjust = -0.35, size = 2.7) +
    annotate("text", x = 2, y = max(runtime_src$runtime_min) * 1.16, label = "+31%",
             size = 3.0, fontface = "bold", color = "#333333") +
    scale_y_continuous("Runtime (min)", limits = c(0, max(runtime_src$runtime_min) * 1.28)) +
    scale_x_discrete(NULL) +
    scale_fill_manual(values = c("ER bridge" = "#999999", "AER" = "#6AAED6")) +
    guides(fill = "none") +
    theme_nn(8.5) +
    theme(legend.position = "none")
  plot <- p_acc + p_rt + plot_layout(widths = c(1.55, 0.9), guides = "collect") &
    theme(legend.position = "bottom")
  plot <- plot + plot_annotation(tag_levels = list(c("(a)", "(b)"))) &
    theme(plot.tag = element_text(face = "bold", size = 9),
          plot.tag.position = c(0.01, 0.98))
  save_plot(plot, "figS2_aer_external_baseline", 7.2, 3.2)
}

main <- function(selected = NULL) {
  final_metrics <- read_source(final_metrics_path)
  gate_diag <- read_source(gate_diag_path)
  per_eval <- read_source(per_eval_path)

  should_run <- function(name) is.null(selected) || name %in% selected

  if (should_run("fig4")) fig4_buffer_gate(final_metrics)
  if (should_run("fig5")) fig5_memorization(gate_diag)
  if (should_run("frontier")) fig_integrated_frontier()
  if (should_run("fig3")) fig3_main_cifar(final_metrics)
  if (should_run("figA")) figA_c10n_frontier(final_metrics)
  if (should_run("fig7")) fig7_c10n_alignment_purity(final_metrics)
  if (should_run("figS1")) figS1_taskwise(per_eval)
  if (should_run("figS2")) figS2_aer()
}

cli_args <- commandArgs(trailingOnly = TRUE)
fig_arg <- cli_args[grepl("^--figs=", cli_args)]
selected_figs <- NULL
if (length(fig_arg)) {
  selected_figs <- strsplit(sub("^--figs=", "", fig_arg[[1]]), ",", fixed = TRUE)[[1]]
}

main(selected_figs)
