#!/bin/sh
algo="mappo"
seed_max=3
thread=20


for seed in `seq ${seed_max}`;
do
    python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
            env_args.map_name="custom-smaclite/27m_vs_30m" t_max=1_020_000 common_reward=False \
            standardise_rewards=False use_rnn=False hidden_dim=128 eps_clip=0.05 lr=0.0003 \
            buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} save_model=True save_model_interval=200_000 
done

for seed in `seq ${seed_max}`;
do
    python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
            env_args.map_name="custom-smaclite/27m_vs_30m" t_max=1_020_000 common_reward=True \
            standardise_rewards=False use_rnn=False hidden_dim=128 eps_clip=0.05 lr=0.0003 \
            buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} save_model=True save_model_interval=200_000 
done


for seed in `seq ${seed_max}`;
do
    python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
            env_args.map_name="custom-smaclite/3s5z_vs_3s6z" t_max=3_020_000 common_reward=True \
            standardise_rewards=False use_rnn=True hidden_dim=64 eps_clip=0.05 lr=0.0003 \
            buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} save_model=True save_model_interval=200_000 
done

for seed in `seq ${seed_max}`;
do
    python src/main.py --config=${algo} --env-config=smaclite with env_args.time_limit=150 env="smaclite" \
            env_args.map_name="custom-smaclite/3s5z_vs_3s6z" t_max=3_020_000 common_reward=False \
            standardise_rewards=False use_rnn=True hidden_dim=64 eps_clip=0.05 lr=0.0003 \
            buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} save_model=True save_model_interval=200_000 
done

