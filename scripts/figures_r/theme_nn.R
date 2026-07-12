library(ggplot2)
library(scales)

method_order <- c(
  "er", "derpp", "gate_loss", "gate_spr", "gate_conf", "gate_predstab",
  "gate_reprstab", "gate_teacher", "gate_coteach", "ewc", "si", "oracle"
)

method_labels <- c(
  er = "ER",
  derpp = "DER++",
  gate_loss = "Small-loss",
  gate_spr = "SPR gate",
  gate_conf = "Confidence",
  gate_predstab = "Prediction stability",
  gate_reprstab = "Representation stability",
  gate_teacher = "Slow teacher",
  gate_coteach = "Co-teach",
  ewc = "EWC",
  si = "SI",
  oracle = "Oracle",
  adam = "Adam",
  mas = "MAS",
  ewc_conf = "EWC + confidence",
  ewc_loss = "EWC + small-loss",
  replay_adaptive = "Adaptive replay",
  replay_agree = "Agreement replay",
  replay_early = "Early replay",
  replay_early5 = "Early replay 5",
  replay_teacher_m95 = "Teacher replay"
)

method_colors <- c(
  er = "#6B6B6B",
  derpp = "#D55E00",
  gate_loss = "#009E73",
  gate_spr = "#CC79A7",
  gate_conf = "#0072B2",
  gate_predstab = "#56B4E9",
  gate_reprstab = "#E69F00",
  gate_teacher = "#8A60A8",
  gate_coteach = "#117733",
  ewc = "#882255",
  si = "#999933",
  oracle = "#111111",
  adam = "#777777",
  mas = "#AA4499",
  ewc_conf = "#44AA99",
  ewc_loss = "#DDCC77",
  replay_adaptive = "#332288",
  replay_agree = "#88CCEE",
  replay_early = "#CC6677",
  replay_early5 = "#AA6655",
  replay_teacher_m95 = "#6699CC"
)

method_shapes <- c(
  er = 15,
  derpp = 17,
  gate_loss = 16,
  gate_spr = 18,
  gate_conf = 0,
  gate_predstab = 2,
  gate_reprstab = 5,
  gate_teacher = 8,
  gate_coteach = 7,
  ewc = 3,
  si = 4,
  oracle = 23,
  adam = 1,
  mas = 9,
  ewc_conf = 10,
  ewc_loss = 11,
  replay_adaptive = 12,
  replay_agree = 13,
  replay_early = 14,
  replay_early5 = 6,
  replay_teacher_m95 = 25
)

label_method <- function(x) {
  out <- unname(method_labels[x])
  out[is.na(out)] <- x[is.na(out)]
  out
}

order_methods <- function(x) {
  extras <- sort(setdiff(unique(x), method_order))
  factor(x, levels = c(method_order, extras), labels = label_method(c(method_order, extras)))
}

theme_nn <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      panel.grid.major = element_line(color = "#E5E5E5", linewidth = 0.25),
      panel.grid.minor = element_line(color = "#F1F1F1", linewidth = 0.18),
      axis.title = element_text(color = "#202020", size = rel(1.0)),
      axis.text = element_text(color = "#303030", size = rel(0.92)),
      strip.text = element_text(face = "bold", color = "#202020", size = rel(1.0)),
      strip.background = element_rect(fill = "#F7F7F7", color = NA),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = rel(0.86)),
      legend.key.width = unit(0.9, "lines"),
      legend.key.height = unit(0.65, "lines"),
      plot.margin = margin(5.5, 5.5, 5.5, 5.5),
      panel.spacing = unit(0.7, "lines")
    )
}

panel_tag <- function(tag) {
  labs(tag = tag) +
    theme(
      plot.tag = element_text(face = "bold", size = rel(1.05)),
      plot.tag.position = c(0.015, 0.985)
    )
}
