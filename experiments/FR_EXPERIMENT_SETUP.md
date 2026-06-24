algorithm: Fruchterman-Reingold
state: full
action space: pure_fr_multiscale
reward: aesthetic_delta
layout_scale: 1.0
max_macro_steps: 5
iterations_per_step: 20
training timesteps: 50,000 or 100,000
training seeds: 2026, 2027, 2028
evaluation splits: val, test_seen, test_unseen_size, test_unseen_family
baselines: no_change, random, fixed large_decrease_k