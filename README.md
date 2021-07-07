**COVID-Variants**

# Scripts for "Modeling the emergence of vaccine-resistant variants with Gaussian convolution"



This repository includes the code for reproducing the results in the manuscript "Modeling the emergence of vaccine-resistant variants with Gaussian convolution". The citation for the manuscript is:

> **Modeling the emergence of vaccine-resistant variants with Gaussian convolution** 
Christian Bongiorno and John Cagnol. 
medRxiv 2021.xxx (to be added)
doi: https://doi.org/10.xxx (to be added)




## Files

The repository contains these files:

**`simulation.py`**
This script performs the numerical simulation of dynamical system (2) of the paper. It produces the graph representing the evolution of pandemic. It can also produce the graphs representing the distribution of the variants at each time step.

To create the videos:
```ffmpeg -framerate 50 -pattern_type glob -i 'Variants/*.png' -c:v libx264 -pix_fmt yuv420p variants.mp4```


**`control-pontryagin.py`**
This script uses the Pontryagin principle to derive the optimal control.

**`control-simanneal.py`**
This script uses the simulated annealing algorithm to derive the optimal control.


**`gaussian-convolution.py`** and **`EiffelTower.png`**
The script checks the computation of 4th-order tensor P is consistent with the multiplication of the Toeplitz matrices. It also verifies the computation of P..I can be achieved with Gaussian_blur (with proper options).
The blurred image appearing in Appendix B was created with this script. The original picture is EiffelTower.png



## Runs

We used:
* Python 3.7.6 with matplotlib 3.1.3, numpy 1.18.1, scipy 1.4.1 and simanneal 0.5.0 on a cluster running CentOS Linux release 7.7.1908.
* Python 3.9.5 with matplotlib 3.4.2, numpy 1.21.0, scipy 1.7.0 and simanneal 0.5.0 on a Mac running Mac OS version 11.4.

The movies were created with ffmpeg version 4.4.



## Improvements and bug report

The intent of the repository is to provide the code for reproducing the results in the article. We have not tried to make them portable or to optimize them.

Any suggestions to improve upon the scripts are welcome.

We have been careful when creating the scripts but a mistake is always possible.
Please let us know if you find any bug.

