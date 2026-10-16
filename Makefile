.PHONE: traini train_verbose train_video play play_video tensorboard

# VENV_ACTIVATE = .env_isaaclab/bin/activate
TASK ?= Twowheelbalance-Twb-Twb
PHYSICS ?= isaacsim_physx

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


train: ## Train the RL agent
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs 4096 physics=$(PHYSICS)

play:  ## Play/visualize the trained agent
	isaaclab play --rl_library rsl_rl --task $(TASK) --checkpoint latest --viz kit physics=$(PHYSICS)

train_verbose:  ## Train with verbose output
	isaaclab train --rl_library rsl_rl --task $(TASK) --viz none --num_envs 4096 physics=$(PHYSICS) --verbose

train_video:  ## Train and record a video
	HEADLESS=1 ENABLE_CAMERAS=1 isaaclab train --rl_library rsl_rl --task Twowheelbalance-Twb-Twb --viz kit --num_envs 4096 physics=isaacsim_physx --video --video_length 1000 --video_interval 2000

play_video:  ## Play and record a video
	HEADLESS=1 ENABLE_CAMERAS=1 isaaclab play --rl_library rsl_rl --task Twowheelbalance-Twb-Twb --viz kit --num_envs 4096 physics=isaacsim_physx --video --video_length 1000 --video_interval 2000

reset_logs:  ## Clear previous training
	rm -rf ~/RL-Balancing-System/logs/

tensorboard:	## Start forwarded tensorboard live
	tensorboard --logdir logs --port 6006
