#!/bin/sh
algo="maddpg"
seed_max=1

# scenarios=( "lbforaging:Foraging-15x15-4p-3f-v3" "lbforaging:Foraging-15x15-3p-4f-v3" "lbforaging:Foraging-10x10-3p-3f-v3" )
total_times=( 5_000_000 )
# scenarios=( "lbforaging:Foraging-15x15-3p-4f-v3" )
scenarios=( "lbforaging:Foraging-15x15-4p-3f-v3" "lbforaging:Foraging-15x15-3p-4f-v3" "lbforaging:Foraging-10x10-3p-3f-v3" "custom_lbf:Custom-Foraging-8x8-2p-4f-coop-v3" "custom_lbf:Custom-Foraging-10x10-2p-3f-coop-v3" "custom_lbf:Custom-Foraging-15x15-3p-4f-coop-v3" )

n_exp=${#scenarios[@]}

for seed in `seq ${seed_max}`; do
    for ((i=0; i<n_exp; i++)); do
        scenario="${scenarios[$i]}"
        total_ts=5_000_000
        python src/main.py --config=${algo} --env-config=gymma with env_args.time_limit=50 \
                env_args.key="${scenario}" t_max=${total_ts} common_reward=False \
                standardise_rewards=True use_rnn=False hidden_dim=64 lr=0.0003 \
                target_update_interval_or_tau=200
    done
done