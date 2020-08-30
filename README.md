# Setup

Install poetry from here https://python-poetry.org/docs/#installation. Then, after cloning the hft-abm-smc-abc repository, in the terminal run 'poetry install'. This automatically installs all the libraries required to run the code in the hft-abm-smc-abc repository.

## Preis-Golke-Paul-Schneider Agent Based Model
To run the PGPS Model (Preis et al., 2006) the Python script files preisOrderBookSeed.py and preisSeed.py along with the config.py are required to initialise the limit order book and execute PGPS Model given the parameters specified in the config.py file. 

## SMC ABC using pyABC
To calibrate the PGPS Model (Preis et al., 2006) using pyABC to implement  SMC ABC simply run the Python script SMC_ABC.py. This uses the SMC ABC configurations supplied in SMC_ABC_init.py. Note that this requires a Linux operating system to run since the parallelisation scheme employed in pyABC is not setup for Windows.

## Visualising Outputs
To extract the SMC ABC results and visualise them, refer to the Python scriptopenSMCABChistory.py. Additionally, to plot the stylised facts of financial time series make use of the script stylised_facts.py.

## ARMA Calibration
To reproduce the ARMA calibration refer to the Python script file ARMACalibrationExample.py.
