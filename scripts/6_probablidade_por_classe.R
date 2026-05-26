# =============================================================================
# Análise de Incerteza por Classe — SITS  (inputs em .gpkg)
# =============================================================================
# O script:
#  1. Lê o raster de entropia e os dois .gpkg (classificação + probabilidades)
#  2. Amostra uma fração dos segmentos ANTES de qualquer extração (mais rápido)
#  3. Extrai o valor de entropia no centróide de cada segmento amostrado
#  4. Une as probabilidades por índice de linha (os dois .gpkg têm os mesmos
#     polígonos — mesma ordem — então não precisa de join espacial)
#  5. Calcula estatísticas de incerteza por classe e gera 4 gráficos
#  6. Imprime ranking de prioridade para coleta de novas amostras
# =============================================================================

library(terra)
library(sf)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(scales)

# =============================================================================
# 1. CONFIGURAÇÕES — ajuste os caminhos
# =============================================================================

RASTER_INCERTEZA   <- "data/class/020024_test/020024/original_class/SENTINEL-2_MSI_020024_2022-08-13_2024-07-27_entropy_rf-2y-second-samples.tif"
BANDA_INCERTEZA    <- "entropy"

GPKG_CLASSIF       <- "data/class/020024_test/020024/original_class/SENTINEL-2_MSI_020024_2022-08-13_2024-07-27_class_first-samples-2026-05-15-15h39m.gpkg"
GPKG_PROBS         <- "data/class/020024_test/020024/original_class/SENTINEL-2_MSI_020024_2022-08-13_2024-07-27_probs_first-samples-2026-05-15-15h39m.gpkg"

COLUNA_CLASSE      <- NULL    # NULL = detecta automaticamente ("class", "label" etc.)
FRACAO_AMOSTRA     <- 0.10    # 10 % dos segmentos — suficiente para diagnóstico
PERCENTIL_DESTAQUE <- 0.90    # destaca os 10 % mais incertos nos gráficos
DIR_SAIDA          <- "analise_incerteza"

# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

# Detecta automaticamente qual coluna guarda o nome da classe
detectar_coluna_classe <- function(sf_obj) {
  candidatas <- c("label", "class", "predicted", "classe",
                  "classname", "class_name")
  cols <- tolower(names(sf_obj))
  hit  <- candidatas[candidatas %in% cols]
  if (length(hit) == 0)
    stop("Coluna de classe não detectada. Disponíveis: ",
         paste(names(sf_obj), collapse = ", "))
  names(sf_obj)[match(hit[1], cols)]
}

# Detecta colunas numéricas que representam probabilidades por classe
# (exclui colunas de metadados comuns do SITS)
detectar_colunas_prob <- function(sf_obj) {
  excluir <- tolower(c("geom", "geometry", "fid", "id", "label", "class",
                       "predicted", "classe", "start_date", "end_date",
                       "cube", "tile", "version", "cloud_cover",
                       "pixel_count", "supercells", "x", "y", "pol_id"))
  numericas <- names(sf_obj)[sapply(sf::st_drop_geometry(sf_obj), is.numeric)]
  cols_prob  <- numericas[!tolower(numericas) %in% excluir]
  if (length(cols_prob) == 0)
    stop("Nenhuma coluna de probabilidade encontrada.")
  cols_prob
}

# Reprojetar somente se o CRS for diferente do raster de referência
reprojetar_se_necessario <- function(sf_obj, proj4_alvo, nome = "") {
  proj4_sf <- sf::st_crs(sf_obj)$proj4string
  if (is.null(proj4_sf) || proj4_sf != proj4_alvo) {
    message("  Reprojetando '", nome, "' para o CRS do raster...")
    sf_obj <- sf::st_transform(sf_obj, crs = sf::st_crs(proj4_alvo))
  }
  sf_obj
}

# =============================================================================
# 3. LEITURA
# =============================================================================

message("\n=== Análise de Incerteza por Classe ===\n")
message("[1/6] Lendo arquivos de entrada...")
dir.create(DIR_SAIDA, showWarnings = FALSE, recursive = TRUE)

# --- Raster de entropia ---
r_inc <- terra::rast(RASTER_INCERTEZA)
idx   <- grep(BANDA_INCERTEZA, names(r_inc), ignore.case = TRUE)
if (length(idx) == 0)
  stop("Banda '", BANDA_INCERTEZA, "' não encontrada. Disponíveis: ",
       paste(names(r_inc), collapse = ", "))
r_inc        <- r_inc[[idx[1]]]
names(r_inc) <- "incerteza"
message("  Banda selecionada: ", BANDA_INCERTEZA)

# --- Classificação ---
sf_cls <- sf::st_read(GPKG_CLASSIF, quiet = TRUE)
col_cls <- if (!is.null(COLUNA_CLASSE)) COLUNA_CLASSE else detectar_coluna_classe(sf_cls)
message("  Coluna de classe: '", col_cls, "'")

sf_cls <- sf_cls |>
  rename(classe = all_of(col_cls)) |>
  mutate(classe = as.character(classe))

message("  Classes: ", paste(sort(unique(sf_cls$classe)), collapse = ", "))
message("  Total de segmentos: ", nrow(sf_cls))

# --- Probabilidades ---
sf_prob   <- sf::st_read(GPKG_PROBS, quiet = TRUE)
cols_prob <- detectar_colunas_prob(sf_prob)
message("  Colunas de probabilidade: ", paste(cols_prob, collapse = ", "))

# Verifica se os dois gpkg têm o mesmo número de feições
# (saem do mesmo pipeline SITS, então devem ter)
if (nrow(sf_cls) != nrow(sf_prob))
  warning("Número de feições diferente entre classificação (", nrow(sf_cls),
          ") e probabilidades (", nrow(sf_prob), "). ",
          "O join será feito por índice — verifique se a ordem está correta.")

# =============================================================================
# 4. REPROJEÇÃO E AMOSTRAGEM
# =============================================================================

message("[2/6] Alinhando CRS e amostrando (", FRACAO_AMOSTRA * 100, "%)...")

proj4_inc <- terra::crs(r_inc, proj = TRUE)
sf_cls    <- reprojetar_se_necessario(sf_cls,  proj4_inc, "classificação")
sf_prob   <- reprojetar_se_necessario(sf_prob, proj4_inc, "probabilidades")

# Amostragem ANTES da extração pontual — muito mais eficiente
set.seed(42)
n_amostra <- min(nrow(sf_cls), max(10000, round(nrow(sf_cls) * FRACAO_AMOSTRA)))
idx_amostra <- sample(nrow(sf_cls), n_amostra)

sf_cls_am  <- sf_cls[idx_amostra, ]
# Usa os mesmos índices nas probabilidades (mesma ordem de linhas)
sf_prob_am <- sf::st_drop_geometry(sf_prob)[idx_amostra, cols_prob, drop = FALSE]

message("  Segmentos amostrados: ", nrow(sf_cls_am))

# =============================================================================
# 5. EXTRAÇÃO DA ENTROPIA (centróide do polígono → valor do raster)
# =============================================================================

message("[3/6] Extraindo entropia nos centróides...")

# Centróide é mais rápido e suficiente para capturar o valor central do segmento
# O aviso "st_centroid assumes attributes are constant" é esperado — pode ignorar
suppressWarnings(
  sf_pts <- sf::st_centroid(sf_cls_am)
)

inc_vals         <- terra::extract(r_inc, terra::vect(sf_pts))
sf_cls_am$incerteza <- inc_vals$incerteza

# Remove segmentos fora da extensão do raster (NA)
validos    <- !is.na(sf_cls_am$incerteza)
sf_cls_am  <- sf_cls_am[validos, ]
sf_prob_am <- sf_prob_am[validos, ]

message("  Segmentos com incerteza válida: ", nrow(sf_cls_am))

# =============================================================================
# 6. MONTAGEM DO DATA FRAME FINAL
# =============================================================================

message("[4/6] Montando tabela de análise...")

# ── CORREÇÃO DO ERRO ANTERIOR ──────────────────────────────────────────────
# O erro ocorreu porque st_drop_geometry() foi chamado ANTES do st_join(),
# tornando sf_prob_clean um data.frame comum (sem geometria).
# st_join() exige que AMBOS os objetos sejam sf.
#
# A solução correta é mais simples: como os dois .gpkg saem do mesmo pipeline
# SITS e têm os mesmos segmentos na mesma ordem, basta um bind_cols() por
# índice — sem join espacial, sem risco de erro, e muito mais rápido.
# ──────────────────────────────────────────────────────────────────────────

pts <- bind_cols(
  sf::st_drop_geometry(sf_cls_am) |> select(classe, incerteza),
  sf_prob_am
)

pts$classe <- as.factor(pts$classe)
message("  Registros finais para análise: ", nrow(pts))

# =============================================================================
# 7. ESTATÍSTICAS POR CLASSE
# =============================================================================

message("[5/6] Calculando estatísticas...")

limiar_alto <- quantile(pts$incerteza, PERCENTIL_DESTAQUE, na.rm = TRUE)

stats_classe <- pts |>
  group_by(classe) |>
  summarise(
    n_segs          = n(),
    inc_media       = mean(incerteza,     na.rm = TRUE),
    inc_mediana     = median(incerteza,   na.rm = TRUE),
    inc_dp          = sd(incerteza,       na.rm = TRUE),
    inc_q25         = quantile(incerteza, 0.25, na.rm = TRUE),
    inc_q75         = quantile(incerteza, 0.75, na.rm = TRUE),
    inc_max         = max(incerteza,      na.rm = TRUE),
    pct_alta_incert = mean(incerteza >= limiar_alto, na.rm = TRUE) * 100,
    .groups = "drop"
  ) |>
  arrange(desc(inc_media))

cat("\n── Ranking de Incerteza por Classe ───────────────────────────────\n")
print(
  stats_classe |>
    mutate(across(where(is.numeric), ~ round(.x, 4))) |>
    rename(Classe = classe, Segmentos = n_segs,
           Média = inc_media, Mediana = inc_mediana, DP = inc_dp,
           Q25 = inc_q25, Q75 = inc_q75, Máx = inc_max,
           `% Alta Incert.` = pct_alta_incert),
  n = Inf
)
cat("──────────────────────────────────────────────────────────────────\n\n")

write.csv(stats_classe,
          file.path(DIR_SAIDA, "stats_incerteza_por_classe.csv"),
          row.names = FALSE)

# =============================================================================
# 8. GRÁFICOS
# =============================================================================

message("[6/6] Gerando gráficos...")

n_cls <- nlevels(pts$classe)
cores <- scales::hue_pal()(n_cls)

# ── 8.1  Violin + Boxplot — distribuição de entropia por classe ──────────────
p_box <- pts |>
  mutate(classe = fct_reorder(classe, incerteza, .fun = median, .desc = TRUE)) |>
  ggplot(aes(x = classe, y = incerteza, fill = classe)) +
  geom_violin(alpha = 0.35, color = NA) +
  geom_boxplot(width = 0.3, outlier.size = 0.4, outlier.alpha = 0.3) +
  geom_hline(yintercept = limiar_alto, linetype = "dashed",
             color = "red", linewidth = 0.8) +
  annotate("text", x = Inf, y = limiar_alto, hjust = 1.05, vjust = -0.5,
           label = paste0("P", PERCENTIL_DESTAQUE * 100),
           color = "red", size = 3.2) +
  scale_fill_manual(values = cores, guide = "none") +
  labs(title    = "Distribuição de Entropia por Classe",
       subtitle = "Linha vermelha = limiar de alta incerteza",
       x = NULL,
       y = "Entropia") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1),
        plot.title  = element_text(face = "bold"))

ggsave(file.path(DIR_SAIDA, "boxplot_incerteza_por_classe.png"),
       p_box, width = 12, height = 6, dpi = 150, bg = "white")
plot(p_box)

# ── 8.2  Barras — % de segmentos em alta incerteza por classe ────────────────
p_pct <- stats_classe |>
  mutate(classe = fct_reorder(classe, pct_alta_incert)) |>
  ggplot(aes(x = classe, y = pct_alta_incert, fill = pct_alta_incert)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = paste0(round(pct_alta_incert, 1), "%")),
            hjust = -0.1, size = 3.2) +
  scale_fill_gradient(low = "#fee08b", high = "#d73027", guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  coord_flip() +
  labs(title    = "% de Segmentos com Alta Incerteza por Classe",
       subtitle = paste0("Acima do percentil ", PERCENTIL_DESTAQUE * 100, "%"),
       x = NULL, y = "% dos segmentos") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(DIR_SAIDA, "pct_alta_incerteza_por_classe.png"),
       p_pct, width = 10, height = 6, dpi = 150, bg = "white")
plot(p_pct)

# ── 8.3  Scatter — tamanho da classe × incerteza média ───────────────────────
p_sct <- stats_classe |>
  ggplot(aes(x = n_segs, y = inc_media, label = classe, color = inc_media)) +
  geom_point(aes(size = pct_alta_incert), alpha = 0.8) +
  scale_color_gradient(low = "#1a9850", high = "#d73027",
                       name = "Entropia\nmédia") +
  scale_size_continuous(name = "% alta\nincerteza", range = c(3, 12)) +
  scale_x_log10(labels = scales::comma) +
  labs(title    = "Tamanho da Classe vs Entropia Média",
       subtitle = "Ponto maior = mais segmentos em alta incerteza",
       x = "Nº de segmentos (escala log)", y = "Entropia média") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

if (requireNamespace("ggrepel", quietly = TRUE)) {
  p_sct <- p_sct +
    ggrepel::geom_text_repel(size = 3.2, show.legend = FALSE, max.overlaps = 20)
} else {
  p_sct <- p_sct + geom_text(size = 3, hjust = -0.2)
  message("  Dica: install.packages('ggrepel') para rótulos sem sobreposição.")
}

ggsave(file.path(DIR_SAIDA, "scatter_tamanho_vs_incerteza.png"),
       p_sct, width = 10, height = 6, dpi = 150, bg = "white")
plot(p_sct)

# ── 8.4  Heatmap — probabilidade média: classe predita × candidata ───────────
# Mostra onde o modelo "hesita": valores altos FORA da diagonal indicam
# confusão entre pares de classes (ex.: Savana_Arborizada ↔ Campo_Sujo_seco)
if (length(cols_prob) > 1) {
  message("  Gerando heatmap de probabilidades...")
  
  heat_df <- pts |>
    group_by(classe) |>
    summarise(across(all_of(cols_prob), ~ mean(.x, na.rm = TRUE)),
              .groups = "drop") |>
    pivot_longer(-classe, names_to = "prob_classe", values_to = "prob_media")
  
  p_heat <- heat_df |>
    ggplot(aes(x = prob_classe, y = classe, fill = prob_media)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = round(prob_media, 2)), size = 2.5) +
    scale_fill_gradient2(
      low = "#2166ac", mid = "#f7f7f7", high = "#d73027",
      midpoint = 1 / length(cols_prob),
      name = "Prob.\nmédia"
    ) +
    labs(title    = "Probabilidade Média: Classe Predita × Classe Candidata",
         subtitle = "Diagonal = confiança do modelo  |  Fora da diagonal = confusão entre classes",
         x = "Classe candidata", y = "Classe predita") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 40, hjust = 1),
          plot.title  = element_text(face = "bold"))
  
  w <- max(10, length(cols_prob) * 1.1)
  h <- max(7,  n_cls * 0.65)
  ggsave(file.path(DIR_SAIDA, "heatmap_probabilidades.png"),
         p_heat, width = w, height = h, dpi = 150, bg = "white")
}
plot(p_heat)
# =============================================================================
# 9. RECOMENDAÇÃO FINAL
# =============================================================================

rec <- stats_classe |>
  mutate(
    prioridade = case_when(
      pct_alta_incert >= 30 ~ "CRITICA",
      pct_alta_incert >= 15 ~ "ALTA",
      pct_alta_incert >= 5  ~ "MODERADA",
      TRUE                  ~ "BAIXA"
    )
  ) |>
  select(classe, inc_media, pct_alta_incert, prioridade)

cat("\n── Prioridade para Coleta de Novas Amostras ──────────────────────\n")
print(rec, n = Inf)
cat("──────────────────────────────────────────────────────────────────\n\n")

write.csv(rec,
          file.path(DIR_SAIDA, "recomendacao_coleta.csv"),
          row.names = FALSE)

# ── Gráfico de recomendação final ────────────────────────────────────────────
cores_prior <- c(
  "CRITICA"  = "#d73027",
  "ALTA"     = "#fc8d59",
  "MODERADA" = "#fee08b",
  "BAIXA"    = "#91cf60"
)

p_rec <- rec |>
  mutate(
    classe     = fct_reorder(classe, pct_alta_incert),
    prioridade = factor(prioridade, levels = c("CRITICA", "ALTA", "MODERADA", "BAIXA"))
  ) |>
  ggplot(aes(x = classe, y = pct_alta_incert, fill = prioridade)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = paste0(round(pct_alta_incert, 1), "%")),
    hjust = -0.1, size = 3.2
  ) +
  # Linha de referência dos limiares
  geom_hline(yintercept = c(5, 15, 30), linetype = "dashed",
             color = "grey40", linewidth = 0.4) +
  annotate("text", x = 0.4, y = c(5, 15, 30) + 0.8,
           label = c("5% — Moderada", "15% — Alta", "30% — Crítica"),
           hjust = 0, size = 2.8, color = "grey30") +
  scale_fill_manual(values = cores_prior, name = "Prioridade",
                    drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  coord_flip() +
  labs(
    title    = "Prioridade de Coleta de Novas Amostras por Classe",
    subtitle = paste0("% de segmentos acima do percentil ",
                      PERCENTIL_DESTAQUE * 100, "% de entropia"),
    x = NULL, y = "% dos segmentos em alta incerteza"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold"),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(DIR_SAIDA, "recomendacao_coleta.png"),
       p_rec, width = 11, height = 7, dpi = 150, bg = "white")
message("  recomendacao_coleta.png")
plot(p_rec)



message("Concluído! Resultados em: ", normalizePath(DIR_SAIDA))
message("\nArquivos gerados:")
message("  boxplot_incerteza_por_classe.png")
message("  pct_alta_incerteza_por_classe.png")
message("  scatter_tamanho_vs_incerteza.png")
if (length(cols_prob) > 1) message("  heatmap_probabilidades.png")
message("  stats_incerteza_por_classe.csv")
message("  recomendacao_coleta.csv")