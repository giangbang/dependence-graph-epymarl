#!/bin/sh

thread=10
N_agents=( 5 7 10 20 )
N_agents=( 5 )
# N=10
total_ts=1000000
seed="$RANDOM"
for N in "${N_agents[@]}"
do
    # ---------------------------------------------------------------------------
    # (1) MAPPO-local  
    # ---------------------------------------------------------------------------
    python src/main.py --config=mappo --env-config=star_spread_mpe with \
        env_args.n_agents=${N} env_args.time_limit=50 \
        t_max=${total_ts} seed=${seed} \
        common_reward=False standardise_rewards=True \
        use_rnn=False hidden_dim=64 eps_clip=0.2 lr=0.0005 \
        buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
        entropy_coef=0.01 use_gae=True test_deterministic=False \
        use_cuda=True norm_adv=True \
        exp_name="star-spread-v2-local-N${N}"

    # ---------------------------------------------------------------------------
    # (2) IPPO-DG oracle 
    # ---------------------------------------------------------------------------
    python src/main.py --config=ippo_graph --env-config=star_spread_mpe with \
        env_args.n_agents=${N} env_args.time_limit=50 \
        t_max=${total_ts} seed=${seed} \
        common_reward=False standardise_rewards=True \
        use_rnn=False hidden_dim=64 eps_clip=0.2 lr=0.0005 \
        buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
        entropy_coef=0.01 test_deterministic=False \
        use_knn_graph=False revs_model_hidden_dim=128 \
        use_cuda=True norm_adv=True \
        exp_name="star-spread-v2-dg-N${N}" rev_model_multi_head=False use_encoder=False

    # ---------------------------------------------------------------------------
    # (2) MAPPO-DG oracle 
    # ---------------------------------------------------------------------------
    python src/main.py --config=mappo_graph --env-config=star_spread_mpe with \
        env_args.n_agents=${N} env_args.time_limit=50 \
        t_max=${total_ts} seed=${seed} \
        common_reward=False standardise_rewards=True \
        use_rnn=False hidden_dim=64 eps_clip=0.2 lr=0.0005 \
        buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
        entropy_coef=0.01 test_deterministic=False \
        use_knn_graph=False critic_type="mult_head_cv_critic" \
        use_cuda=True norm_adv=True revs_model_hidden_dim=128 \
        exp_name="star-spread-v2-dg-N${N}" use_encoder=False rev_model_multi_head=False

    # ---------------------------------------------------------------------------
    # (3) MAPPO-global 
    # ---------------------------------------------------------------------------
    python src/main.py --config=mappo --env-config=star_spread_mpe with \
        env_args.n_agents=${N} env_args.time_limit=50 \
        t_max=${total_ts} seed=${seed} \
        common_reward=True standardise_rewards=True \
        use_rnn=False hidden_dim=64 eps_clip=0.2 lr=0.0005 \
        buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
        entropy_coef=0.01 use_gae=True test_deterministic=False \
        use_cuda=True norm_adv=True \
        exp_name="star-spread-v2-global-N${N}"


    # ---------------------------------------------------------------------------
    # (1) IPPO-local  
    # ---------------------------------------------------------------------------
    python src/main.py --config=ippo --env-config=star_spread_mpe with \
        env_args.n_agents=${N} env_args.time_limit=50 \
        t_max=${total_ts} seed=${seed} \
        common_reward=False standardise_rewards=True \
        use_rnn=False hidden_dim=64 eps_clip=0.2 lr=0.0005 \
        buffer_size=${thread} batch_size_run=${thread} batch_size=${thread} \
        entropy_coef=0.01 use_gae=True test_deterministic=False \
        use_cuda=True norm_adv=True \
        exp_name="star-spread-v2-local-N${N}"

    

done
