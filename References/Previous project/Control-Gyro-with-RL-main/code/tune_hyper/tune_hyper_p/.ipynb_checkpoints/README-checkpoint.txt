run 3 iterations of hyperparameter search

after iter2 (last iter) we choose iter2_gamma02 as final parameters and train iter2_final

the epoch is changed from 2000 to 5000 to guarantee convergence

all models are test on teset set 'states10k.csv', the final model is also tested on 'states10k_validation.csv'