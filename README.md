# Estimate the mutual information across set sizes

## Introduction

This repo is created to replicate some of the results shown in:
[Gershman and Lai 2021, The reward-complexity trade-off in schizophrenia](https://www.biorxiv.org/content/10.1101/2020.11.16.385013v2.full.pdf)

Specifically, This repo reproduces Fig.2 and Fig.5.

## How to use the code 

With any python terminal, run:

    python plot_figures.py -f=fig2
   
or 

    python plot_figures.py -f=fig5
    
## Where I have no access to the code and need to guess

The original [Github repo] (http://github.com/lucylai96/plm/) includes codes to develop Fig.2 but not Fig.5 that includes a process model. This repo is developed on 04/28/21, the code of Gershman and Lai (2021) has not been posted. To replicate Fig.5, I create the code of this section by guessing.

To get the clues of how to include the process model in the analysis. Here we cite some paragraphs from Gershman and Lai, 2021 paper:
    
   (Page 7, after equation 14)
    
   "We fit four free parameters to each individual’s choice behavior using maximum likelihood estimation. To assess the match to the data, we then simulated the fitted model for each
    participant, using the same stimuli presented to the human subjects..."
    
   (Page 10, the second paragraph of section *Modeling*)

   " We fit the actor-critic model to the choice data using maximum likelihood estimation, and then simulated the fitted model on the task. Applying the same analyses to these simulations (Figures
    5 and 6) verified that this model achieved a reasonably good match with the experimental data..."

These paragraphs suggest that I need to

    * fit the process model to each individual's data
    * run the model with fitted parameters to generate simulation data
    * Do the **SAME** analysis on the simulation data

I am not sure what does the word "same" means and how similar the analysis is. In this repo, I use literarily the same analysis. 
    
    
## Do I replicate the results?

Yes and no. Yes means I perfectly replicate Fig.2, and no means I do not know whether I am correct with Fig.5

If you visit Gershman and Lai 2021 Fig.2, you may find [my replication](https://github.com/fangzefunny/replicate_reward-complexity/blob/main/figures/Gershman21_fig2.png) is the same as that in Gershman's paper. 

However, Fig.5 is created using the simulated data, and, thus, the figures generated in this way may vary. Since I have no access to the origin code for Fig.5, I am not sure if I made the correct guess. One positive evidence implying I am correct is the ["Pi complexity vs set size"](https://github.com/fangzefunny/replicate_reward-complexity/blob/main/figures/Gershman21_fig5.png) is the same scale with Gershman's Fig.5. 

The code is messy, I will be back and clear them up later.

  
## Thanks

Thanks to Collins and Frank for making their data set available;

Thanks to Lai and Gershman for posting the [data set and code](http://github.com/lucylai96/plm/)


