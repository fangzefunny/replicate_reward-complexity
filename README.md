# Estimate the mutual information across set sizes

## Introduction

This repo is created to replicate the some of the results shown in:
[Gershman and Lai 2021, The reward-complexity trade-off in schizophrenia](https://www.biorxiv.org/content/10.1101/2020.11.16.385013v2.full.pdf)

Specifically, This repo reproduce the Fig2 an Fig 5.

## How to use the code 

With any python terminal, run:

    python plot_figures.py -f=fig2
   
or 

    python plot_figures.py -f=fig5
    
## Do I replicate the results?

Yes and no. Yes means I perfectly replicate the Fig.2, and no means I do not know whether I am correct with Fig.5

If you visit Gershman and Lai 2021 Fig.2, you may find [my replication](https://github.com/fangzefunny/replicate_reward-complexity/blob/main/figures/Gershman21_fig2.png) is exactly the same with that in Gershman's paper. 

However, Fig.5 is created using the simulated data, and, thus, the figures generated in this way may vary. Since I have no access to the origin code for Fig.5, I am not sure if I made the correct guess. One positive evidence implying I am correct is the ["Pi complexity vs set size"](https://github.com/fangzefunny/replicate_reward-complexity/blob/main/figures/Gershman21_fig5.png) is the same scale with Gershman's Fig.5. 

The code is messy, I will be back and clear them up later.

  
## Thanks

Thanks Collins and Frank for making their data set avilable;

Thanks Lai and Gershman for posting the [data set and code](http://github.com/lucylai96/plm/)


