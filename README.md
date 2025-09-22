# POLARIS - Core

Code to easily train, manage and use neural networks. 
The code also allows: 
- The prediction of direct observations (.fits).
- Compute emission maps from molecules.
- Load, use and group datacube simulations (Todo AMR).

This is the code used in POLARIS software.

Todo:
- Go from list of list to dict in Datasets
- Dataset setting: score_fct, make score_parameters instead of add score_offset
- Dataset split settings
- Multiple image sizes in datasets
- Batch actually have two definitions, this makes code understanding hard.
- Adapt the code for multiple targets and inputs.
- Serializer of networks, to make import and export custom networks and keep using this code.