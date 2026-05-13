PRODES-Pantanal experiments and pipelines developed using the SITS package
================

<img src="./inst/extdata/sticker/biomasbr_logo.jpeg" alt="BiomasBR logo" align="right" height="150" width="150"/>

This repository brings together reproducible experiments and processing pipelines
from the PRODES-Pantanal project, developed using the SITS package. Its purpose
is to clearly and systematically document the adopted workflows,
providing references for experimentation, validation, and methodological
improvements, while supporting reproducibility and the continuous
evolution of project's analyses.

# Getting started

To use the scripts in this repository, clone the project to
your local machine using the command below:

```sh
git clone https://github.com/BiomasBR/prodes-pantanal
```

After cloning, open the `prodes-pantanal` directory in RStudio and install the
package using the command:

```r
devtools::install(".")
```

# Documentation

Project documentation and methodological references will be progressively
made available in this repository.

# Repository structure

- `data/`: Datasets used and generated throughout the analyses
- `inst/extdata/`: Supplementary resources required for the workflows
- `scripts/`: Processing and experimentation routines

# Main workflows

- Satellite image preprocessing
- Geospatial data organization
- Segmentation routines
- Classification workflows
- Temporal analysis
- Environmental monitoring pipelines
- GIS integration and automation

# Technologies

- R
- SITS
- Terra
- sf
- GDAL
- QGIS

# License

The data and results available in this repository are licensed under the
terms described in the LICENSE file.

# Support

For questions, suggestions, or issues, please use the **Issues** section or
contact the repository maintainers.
