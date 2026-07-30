# LightRL Architecture

LightRL separates the agentic RL stack into three explicit extension axes:

- Harness: how the rollout agent acts in the environment.
- Model: which policy/tokenizer/checkpoint family backs training.
- Algorithm: how rollout, reward, exploration, world-model, and optimization are configured.

`agentic_rl/` contains the framework-specific layer, `slime/` remains the training backend,
and `Megatron-LM/` remains the vendorized low-level model backend.
