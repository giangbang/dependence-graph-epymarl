#!/bin/sh
algo="ippo_graph"
seed_max=1
thread=10
clip_epses=( 0.1 )
k=2
scenarios=( "lbforaging:Foraging-15x15-4p-3f-v3" "lbforaging:Foraging-15x15-3p-4f-v3" "lbforaging:Foraging-10x10-3p-3f-v3" )
# scenarios=( "lbforaging:Foraging-10x10-2p-3f-coop-v3" "lbforaging:Foraging-15x15-3p-4f-coop-v3" "lbforaging:Foraging-8x8-2p-4f-coop-v3" )
total_times=( 5_000_000 )
scenarios=( "lbforaging:Foraging-15x15-4p-3f-v3" "lbforaging:Foraging-15x15-3p-4f-v3" "lbforaging:Foraging-10x10-3p-3f-v3" "custom_lbf:Custom-Foraging-8x8-2p-4f-coop-v3" "custom_lbf:Custom-Foraging-10x10-2p-3f-coop-v3" "custom_lbf:Custom-Foraging-15x15-3p-4f-coop-v3" )
# scenarios=( "lbforaging:Foraging-15x15-3p-4f-v3" "custom_lbf:Custom-Foraging-15x15-3p-4f-coop-v3" "custom_lbf:Custom-Foraging-10x10-2p-3f-coop-v3" )  
# scenarios=( "custom_lbf:Custom-Foraging-8x8-2p-4f-coop-v3" )

n_exp=${#scenarios[@]}

for seed in `seq ${seed_max}`;
    do
    for ((i=0; i<n_exp; i++)); do
        scenario="${scenarios[$i]}"
        total_ts=5_000_000
        clip_eps=0.2
        python src/main.py --config=${algo} --env-config=gymma with env_args.time_limit=50 \
                env_args.key="${scenario}" t_max=${total_ts} common_reward=False \
                standardise_rewards=False use_rnn=False hidden_dim=128 eps_clip=${clip_eps} lr=0.0005 \
                buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
                use_full_path=True test_deterministic=False
    done
done