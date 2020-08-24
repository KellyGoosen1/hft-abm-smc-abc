# ABC_ABM_HFT
Calibrating an Agent Based Model for High Frequency Trading using Approximate Bayesian Computation

### PreisSeed
Builds agent based model. Requires PreisOrderBookSeed.

### PreisOrderBookSeed
Builds order book.

### SMCABC_init
Initialises all parameters for the Preis model and SMC ABC calibration. Requires PreisSeed.

### SMC_ABC
Main file to run calibration of Preis et. al model using SMC ABC. Requires SMCABC_init.
