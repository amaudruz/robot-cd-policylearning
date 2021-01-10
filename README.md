### Acknowledgments

This ode is developed based on [CrowdNav](https://github.com/vita-epfl/CrowdNav) and [Social-NCE](https://github.com/vita-epfl/social-nce)

Behavioral Cloning (Vanilla)
  ```
  python imitate.py --gpu
  python utils/tests.py --model_path data/output/imitate-baseline-data-0.5-traj
  ```
* Social-NCE 
  ```
 python imitate.py --auxiliary_task contrastive --contrast_weight 2.0 --contrast_sampling event --gpu
 python utils/tests.py --model_path data/output/imitate-event-data-0.5-weight-2.0-horizon-4-temperature-0.20-nboundary-0-traj
  ```
* TrajPred (proposed) 
  ```
  python imitate.py --auxiliary_task traj --traj_weight 2.5 --traj_length 4 --gpu
  python utils/tests.py --model_path data/output/imitate-trajpred-2.50-weight-1to4-length-traj
  ```
  

