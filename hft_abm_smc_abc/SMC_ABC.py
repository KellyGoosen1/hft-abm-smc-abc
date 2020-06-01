"""Purpose of this script is to run SMC ABC given abc object and model details"""
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\My Work Documents\\repo\\hft_abm_smc_abc', 'C:/My Work Documents/repo/hft_abm_smc_abc',
                 '/home/gsnkel001/hft_abm_smc_abc'])


from pyabc import sge
from hft_abm_smc_abc.SMC_ABC_init import abc, db
from hft_abm_smc_abc.config import temp_output_folder, version_number, smcabc_minimum_epsilon, \
    smcabc_max_nr_populations, smcabc_min_acceptance_rate
import pickle
import os
import random

random.seed(4)

print(sge.nr_cores_available())
print(version_number)

if __name__ == '__main__':

    # Run SMCABC
    history = abc.run(minimum_epsilon=smcabc_minimum_epsilon,
                      max_nr_populations=smcabc_max_nr_populations,
                      min_acceptance_rate=smcabc_min_acceptance_rate)

    # Return True if is ABC history class
    print(history is abc.history)
    print(history.all_runs())

    # Save history object
    with open(os.path.join(temp_output_folder, 'SMCABC_history_V' + version_number + '.class'), 'wb') as history_file:
        # Step 3
        pickle.dump(history, history_file)

    with open(os.path.join(temp_output_folder, "db.txt"), "w") as text_file:
        print(db, file=text_file)



