#!/bin/sh
algo="mappo_graph"
seed_max=1
thread=10
# clip_epses=( 0.2 0.2  0.05 0.1  0.1 0.05 )
k=3
scenarios=( "MMM"        "MMM2"     "corridor"  "bane_vs_bane"  "25m_vs_30m" "3s5z_vs_3s6z")
total_times=( 1_000_000  2_000_000  3_000_000   2_000_000       2_000_000     3_000_000 )



n_exp=${#scenarios[@]}

for seed in `seq ${seed_max}`; do
    for ((i=0; i<n_exp; i++)); do
        scenario="${scenarios[$i]}"
        clip_eps=0.2
        total_ts="${total_times[$i]}"
        python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
                env_args.map_name="custom-smaclite/${scenario}" t_max=${total_ts} common_reward=False \
                standardise_rewards=False use_rnn=True hidden_dim=64 eps_clip=${clip_eps} lr=0.0005 \
                buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
                entropy_coef=0.001
    done
done
