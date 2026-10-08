.PHONE: train play

# VENV_ACTIVATE = .env_isaaclab/bin/activate
TASK ?= Twowheelbalance-Twb-Twb
PHYSICS ?= isaacsim_physx


train:
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs 4096 physics=$(PHYSICS)

play:
	isaaclab play --rl_library rsl_rl --task $(TASK) --checkpoint latest --viz kit physics=$(PHYSICS)

train_verbose:
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs 4096 physics=$(PHYSICS) --verbose
