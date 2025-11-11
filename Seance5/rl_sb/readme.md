
mes hypermarametre de chaque fintuning : 

GridWorldStatic-v0:
  n_timesteps: 100000
  policy: "MlpPolicy"
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  gamma: 0.99
  learning_rate: 3.0e-4
  ent_coef: 0.0
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

GridWorldMoving-v0:
  n_timesteps: 50000          # fine-tuning, moins de pas
  policy: "MlpPolicy"
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  gamma: 0.99
  learning_rate: 1.0e-4       # LR plus petit pour le fine-tune
  ent_coef: 0.0
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5


NB : j ai dfintuner GridWorldMoving sur GridWorldStatic  4X4