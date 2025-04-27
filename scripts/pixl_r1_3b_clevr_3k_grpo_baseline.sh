set -x
export WANDB_API_KEY=a3642c8b11ba4d06e93d04f615579a18d8a19e07

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

mkdir -p logs

FORMAT_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

DATA_FILE="hunarbatra/clevr-cogent-baseline-3k"

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=r1v \
    trainer.experiment_name=pixl_r1_clevr_3k_grpo_v5_baseline \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=ckpts/pixl_r1_clevr_3k_grpo_v5_baseline \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.training_steps=50 \
    worker.rollout.n=8 \
    trainer.total_episodes=50 \
    data.format_prompt="${FORMAT_PROMPT}" \
    > logs/v5_pixl_r1_baseline_clevr_3k_grpo.log 2>&1
    # worker.rollout.tensor_parallel_size=1 \
    