# ============================================================
# Train a Lightweight Temporal Self-Attention Encoder
# ============================================================

# Load required libraries
library(sits)
library(ggplot2)
library(torch)
library(luz)
library(stringr)

# Define the parameters: These are user-defined variables
time_series_name  <- "TS-tiles_012014-012015-013014-013015_1y_2024-08-01_2025-07-31_all-samples-new-pol-avg-false_2026-02-24_20h01m.rds"

# Extract the tiles and date of the string separated by "_"
tiles      <- str_split(str_extract(time_series_name, "(?<=tiles_)[^_]+"), "-")[[1]]
start_date <- stringr::str_split_i(time_series_name, "_", 4)
end_date   <- stringr::str_split_i(time_series_name, "_", 5)

# Function to read class names and their colors::IMPORTANT
read_class_config <- function(config_file = "class_config.txt") {
  
  if (!file.exists(config_file)) {
    stop(paste("Configuration file not found:", config_file))
  }
  
  lines <- readLines(config_file, encoding = "UTF-8", warn = FALSE)
  
  # Remove empty lines and comments
  lines <- trimws(lines)
  lines <- lines[nchar(lines) > 0 & !startsWith(lines, "#")]
  
  # Identify sections and populate lists
  current_section  <- NULL
  class_trans_list <- list()
  colors_list      <- list()
  
  for (line in lines) {
    if (startsWith(line, "[") && endsWith(line, "]")) {
      current_section <- gsub("\\[|\\]", "", line)
      next
    }
    
    if (!is.null(current_section) && grepl("=", line)) {
      parts <- strsplit(line, "=", fixed = TRUE)[[1]]
      key   <- trimws(parts[1])
      value <- trimws(paste(parts[-1], collapse = "=")) # preserves '=' in hex codes
      
      if (current_section == "CLASS_TRANSLATION") {
        class_trans_list[[key]] <- value
      } else if (current_section == "COLORS") {
        colors_list[[key]] <- value
      }
    }
  }
  
  class_translation <- unlist(class_trans_list)
  my_colors         <- unlist(colors_list)
  
  message(sprintf("Config loaded: %d class translations | %d colors",
                  length(class_translation), length(my_colors)))
  
  return(list(
    class_translation = class_translation,
    my_colors         = my_colors
  ))
}

# Date and time of the start of processing
date_process    <- format(Sys.Date(), "%Y-%m-%d_")
time_process    <- format(Sys.time(), "%Hh%Mm", tz = "America/Sao_Paulo")
process_version <- paste0(date_process, time_process)

# File and folder paths
time_series_path  <- file.path("data/rds/time_series/", time_series_name)
rds_path          <- "data/rds/"
plots_path        <- "data/plots/"
config_dir        <- ".."

# Identifier to distinguish this model run from previous versions
var <- stringr::str_split_i(time_series_name, "_", 6)

# Plots organized by var
plots_dir <- file.path(plots_path, var)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. Define and Load Data Cubes
# ============================================================

# Step 1.1 -- Create a training cube from a collection
cube <- sits_cube(
  source      = "BDC",
  collection  = "SENTINEL-2-16D",
  bands       = c('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'NBR', 'EVI', 'CLOUD'),
  tiles       = tiles,
  start_date  = start_date,
  end_date    = end_date,
  progress    = TRUE)

# Step 1.2 -- Calculate the number of years in the training cube
no.years <- paste0(floor(lubridate::year(end_date) - lubridate::year(start_date)), "y")
tiles_train <- paste(sort(tiles), collapse = "-")
no.cubes <- paste0(length(cube$tile), "t")

# ============================================================
# 2. Cross-validation of training data
# ============================================================

# Step 2.1 -- Reading training samples
train_samples <- readRDS(time_series_path)

# Step 2.2 -- Load color palette from external config file
config     <- read_class_config(file.path(config_dir, "class_config.txt"))
my_colors  <- config$my_colors
my_colors  <- my_colors[names(my_colors) %in% unique(train_samples$label)]

# Step 2.3 -- Using k-fold validation
sits_kfold_validate_start <- Sys.time()
rfor_validate <- sits_kfold_validate(
  samples = train_samples,
  folds = 5, # how many times to split the data (default = 5)
  ml_method = sits_rfor(),
  multicores = 28,
  progress = TRUE) # adapt to your computer CPU core availability
sits_kfold_validate_end <- Sys.time()
sits_kfold_validate_time <- as.numeric(sits_kfold_validate_end - sits_kfold_validate_start, units = "secs")
sprintf("SITS kfold_validate process duration (HH:MM): %02d:%02d", 
        as.integer(sits_kfold_validate_time / 3600), 
        as.integer((sits_kfold_validate_time %% 3600) / 60))

# Step 2.3.1 -- Plot the confusion matrix
plot(rfor_validate, type = "confusion_matrix")

# Step 2.3.2 -- Plot the metrics by class
plot(rfor_validate, type = "metrics")

# Step 2.4 -- Save confusion matrix plot
g_cm <- plot(rfor_validate, type = "confusion_matrix")
ggplot2::ggsave(
  filename = file.path(
    plots_dir,
    paste0(
      "Kfold-confusion-matrix_",
      tiles_train, "_",
      start_date, "_", end_date, "_",
      var, "_",
      format(Sys.Date(), "%Y-%m-%d"),
      ".png"
    )
  ),
  plot = g_cm,
  width = 1600,
  height = 1000,
  units = "px",
  dpi = 200
)

# Step 2.4.1 -- Save metrics plot
g_metrics <- plot(rfor_validate, type = "metrics")
ggplot2::ggsave(
  filename = file.path(
    plots_dir,
    paste0(
      "Kfold-metrics_",
      tiles_train, "_",
      start_date, "_", end_date, "_",
      var, "_",
      format(Sys.Date(), "%Y-%m-%d"),
      ".png"
    )
  ),
  plot = g_metrics,
  width = 1600,
  height = 1000,
  units = "px",
  dpi = 200
)

# ============================================================
# 3. Training and saving model
# ============================================================

# Step 3.1 -- Train LTAE-based model using training samples
tuned_ltae <- sits_lighttae(
  samples = train_samples,
  epochs = 150L,
  batch_size = 128L,
  validation_split = 0.2,
  optimizer = torch::optim_adamw,
  opt_hparams = list(lr = 5e-04, eps = 1e-08, weight_decay = 7e-04),
  lr_decay_epochs = 50L,
  lr_decay_rate = 1,
  patience = 20L,
  min_delta = 0.01,
  seed = NULL,
  verbose = FALSE,
  multicores = 24,    # parallel execution
  progress = TRUE     # display progress
)

# Step 3.2 -- Save tuned model object to disk (RDS format)
saveRDS(
  tuned_ltae,
  paste0(
    rds_path, "model/ltae/", "LTAE-model_",
    length(cube$tile), "-tiles-", tiles_train, "_",
    no.years, "-period-",
    cube_dates[1], "_", cube_dates[length(cube_dates)],
    "_", var, "_", process_version, ".rds"
  )
)
print("Model trained successfully!")
