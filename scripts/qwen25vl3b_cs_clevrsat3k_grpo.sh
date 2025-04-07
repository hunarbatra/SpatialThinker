set -x

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=/home/hunar/LLaMA-Factory/output/qwen25_vl_lora_sft
# MODEL_PATH=/gpfs/scratch/ehpc80/PIXL-R1-Project/LLaMA-Factory/output/qwen25_vl_lora_sft/

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""

/home/hunar/miniforge3/envs/pixl-r1-env2/bin/python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files=hunarbatra/Clevr_SAT_3k@train \
    data.val_files=hunarbatra/Clevr_SAT_3k@val \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen25vl3b_cs_clevr_sat_3k \
    trainer.n_gpus_per_node=4
