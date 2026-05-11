# ============================================================
# Linear Spectral Mixture Model (LSMM) for Feature Extraction
# ============================================================

# Load Required Libraries
library(sits)
library(tibble)

# Paths for files and folders
mixture_path <- "data/raw/mixture_model"

# ============================================================
# 1. Define and Load Raster Data Cubes from a collection
# ============================================================

cube <- sits_cube(
  source     = "BDC",
  collection = "SENTINEL-2-16D",
  bands      = c('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'CLOUD'),
  tiles      = c('020023', '020024'),
  start_date = '2022-08-01',
  end_date   = '2025-07-31',
  progress   = TRUE)

# ============================================================
# 2. Create Fraction Image Features from the Mixture Model Cube
# ============================================================

# Step 2.1 -- Define endmember values
endmembers <- tibble::tribble(
  ~class,    ~B02, ~B03, ~B04, ~B05, ~B06, ~B07, ~B08, ~B8A, ~B11, ~B12,
  "SOIL",    1799, 2154, 3028, 3303, 3472, 3656, 3566, 3686, 5097, 4736,
  "VEG",      827,  892,  410, 1070, 4206, 5646, 5495, 6236, 2101,  775,
  "WATER",    946,  739,  280,  208,  180,  167,  135,  129,   26,   14,
  "SOILO",    1556, 2291, 5485, 6236, 6889, 7323, 7176, 7530, 10252, 8745,
  "VEGO",      909,  969,  447, 1126, 4762, 6323, 6193, 6629, 1731,  712,
)

# Step 2.2 -- Generate the mixture model cube and calculate the process duration
sits_mixture_model_start <- Sys.time()
mm_cube <- sits_mixture_model(
  data       = cube,
  endmembers = endmembers,
  multicores = 28,  # adapt to your computer CPU core availability
  memsize    = 180, # adapt to your computer memory availability
  output_dir = mixture_path
)
sits_mixture_model_end  <- Sys.time()
sits_mixture_model_time <- as.numeric(sits_mixture_model_end - sits_mixture_model_start, units = "secs")
sprintf("SITS LSMM process duration (HH:MM): %02d:%02d", as.integer(sits_mixture_model_time / 3600), as.integer((sits_mixture_model_time %% 3600) / 60))
print("Linear Spectral Mixture Model created successfully.")
