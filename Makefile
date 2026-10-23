.PHONY: train train_verbose train_video play play_video tensorboard logs

# VENV_ACTIVATE = .env_isaaclab/bin/activate
TASK ?= Twowheelbalance-Twb-Twb
PHYSICS ?= isaacsim_physx
NUM_GPUS ?= 4
NUM_ENVS ?= 4096
ITERATIONS ?=4000

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


train: ## Train the RL agent
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs $(NUM_ENVS) physics=$(PHYSICS) --max_iterations $(ITERATIONS)

play:  ## Play/visualize the trained agent
	isaaclab play --rl_library rsl_rl --task $(TASK) --checkpoint latest --viz kit physics=$(PHYSICS)

train_verbose:  ## Train with verbose output
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs $(NUM_ENVS) physics=$(PHYSICS) --verbose --max_iterations $(ITERATIONS)

train_video:  ## Train and record a video
	HEADLESS=1 ENABLE_CAMERAS=1 isaaclab train --rl_library rsl_rl --task Twowheelbalance-Twb-Twb --viz kit --num_envs $(NUM_ENVS) physics=isaacsim_physx --video --video_length 1000 --video_interval 2000

train_multi:  ## Train on several GPUs of one node
	python -m torch.distributed.run --nnodes=1 --nproc_per_node $(NUM_GPUS) --local_ranks_filter 0 --no-python \
		isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs $(NUM_ENVS) --max_iterations $(ITERATIONS) physics=$(PHYSICS) --distributed	

play_video:  ## Play and record a video
	HEADLESS=1 ENABLE_CAMERAS=1 isaaclab play --rl_library rsl_rl --task Twowheelbalance-Twb-Twb --viz kit --num_envs 128 physics=isaacsim_physx --video --video_length 10000 --video_interval 2000

video_arrow: ## Add indication arrows to recording
	HEADLESS=1 ENABLE_CAMERAS=1 isaaclab play --rl_library rsl_rl --task Twowheelbalance-Twb-Twb --viz kit --num_envs 32 physics=isaacsim_physx --video --video_length 1200 env.commands.base_velocity.debug_vis=True  env.commands.base_velocity.heading_command=False  env.commands.base_velocity.rel_standing_envs=0.0 'env.commands.base_velocity.ranges.lin_vel_x=[1.0,1.0]' 'env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]' env.observations.policy.enable_corruption=False

reset_logs:  ## Clear previous training
	rm -rf ~/RL-Balancing-System/logs/

tensorboard:	## Start forwarded tensorboard live
	tensorboard --logdir logs --port 6006

logs: ## Save logs on pc lab
	scp -r logs mlucca@129.132.39.171:.
