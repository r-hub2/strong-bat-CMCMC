# CMCMC example scripts

This directory contains **one script per GPU kernel**. Each script:

- runs a small synthetic example,
- saves all iterations via `saved_iterations = 0` (so traces can be plotted),
- prints posterior means computed from the last few iterations, and
- writes a PDF trace plot for the first 5 parameters.

Run scripts with `Rscript`, optionally passing `key=value` arguments (e.g. `sampler=inca`).
Outputs are written under `inst/scripts/outputs/`.

## Kernels

- `mvnorm_example.R` (`GPUkernel = "MVNorm"`)
- `logistic_example.R` (`GPUkernel = "Logistic"`)
- `hier_beta_binom_joint_example.R` (`GPUkernel = "HierBetaBinomJoint"`)
- `nne_way_normal_example.R` (`GPUkernel = "NneWayNormal"`)
- `nne_way_normal_nc_example.R` (`GPUkernel = "NneWayNormalNC"`)

## Legacy

Older smoke tests and ad-hoc scripts are kept in `legacy/`.
