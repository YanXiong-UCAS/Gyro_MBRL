Test reward functions on GyroscopeEnvV1

Power function and exponential works best

See details in ipynb files

Use PE = P+E functions for future study

A,N,Q,S,P are trained with 500 epochs. E and PE are trained with 2000 epochs

Small p leads to slow convergence with a gamma close to 1
Large e leads to gredient loss

A short conclusion
Q leads to SS -> use A -> better solution P and E (at the cost of slower convergence)
e20 is best (for now). PE brings improvement for large E.
