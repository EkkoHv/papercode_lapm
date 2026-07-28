# Data access and provenance

This repository does not redistribute the two measured soil data sets. The
random fields are generated locally and are not third-party data.

## German iSOIL case

The German measurements originate from the iSOIL project data dissemination
platform of the European Soil Data Centre (ESDAC). The manuscript uses the Bad
Lauchstädt gamma-spectrometry measurements.

- Official data page:  
  https://esdac.jrc.ec.europa.eu/content/isoil-project-interactions-between-soil-related-sciences
- Access condition: complete the request form on the ESDAC page and follow the
  download instructions supplied by ESDAC.
- Variables used by the released analysis: spatial coordinates, dose rate as
  the prediction target, and potassium and thorium as covariates.
- Unit handling: inspect the downloaded column headers. The source files use a
  storage-scale annotation, so apply any stated scale factor before reporting
  physical units.

Relevant references:

- Panagos, P., Van Liedekerke, M., Jones, A., Montanarella, L. (2012).
  European Soil Data Centre: Response to European policy support and public
  data requirements. *Land Use Policy*, 29, 329-338.  
  https://doi.org/10.1016/j.landusepol.2011.07.003
- Van Egmond, F. M., Dietrich, P., Werban, U., Sauer, U. (2009). iSOIL:
  exploring the soil as the basis for quality crop production and food
  security. *Quality Assurance and Safety of Crops & Foods*, 1, 117-120.  
  https://doi.org/10.1111/j.1757-837X.2009.00019.x

## Swiss Jura case

The Swiss Jura data are distributed with the `gstat` R package. The public
package objects contain 359 measured locations and an ancillary grid. The
analysis predicts Pb and uses land use and rock type as covariates.

- Official documentation:  
  https://r-spatial.github.io/gstat/reference/jura.html
- Source repository:  
  https://github.com/r-spatial/gstat
- Pinned file used by `workflows/download_swiss_jura.py`:  
  https://raw.githubusercontent.com/r-spatial/gstat/2a578765502dd29520dcc3b40af42c953237faa3/data/jura.rda
- Expected SHA-256:  
  `f1d9e8a7e6686aa7473a1ce452fffe74f0c45a3e2ba464e35a9163c0d447d845`

Reference:

- Atteia, O., Dubois, J.-P., Webster, R. (1994). Geostatistical analysis
  of soil contamination in the Swiss Jura. *Environmental Pollution*, 86,
  315-327.  
  https://doi.org/10.1016/0269-7491(94)90172-4

## Local input format

Case data should be converted locally to a CSV file. The generic prediction
script accepts:

- two coordinate columns, `x` and `y` by default;
- one numeric target column;
- one or more numeric covariate columns.

The data files remain ignored by Git through `data/external/`.
