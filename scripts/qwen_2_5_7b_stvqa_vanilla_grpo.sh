set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
# MODEL_PATH=/gpfs/scratch/ehpc80/hf_cache_hb/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/

FORMAT_PROMPT="""<image> You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags, and only return the final choice including the correct option and answer within the answer tags, e.g., <answer> ({correct_option}) {correct_answer} </answer>.
 
Q. """

DATA_FILE = "hunarbatra/STVQA-7K"
# DATA_FILE="/gpfs/scratch/ehpc80/hf_cache_hb/huggingface/hub/datasets--hunarbatra--spatialthinker_vqa_10k_filtered/snapshots/c43e6d9272e395d79b2bee20bd62d1c4a529d636/data/"


python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=r1v \
    trainer.experiment_name=qwen2_5_vl_stvqa_baseline_7B \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=ckpts/qwen2_5_vl_stvqa_baseline_7B \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=8 \
    trainer.max_steps=75 \
    trainer.total_episodes=75 \
    data.prompt_key="question_with_options" \
    data.answer_key="answer_option_text_only" \
    data.image_key="images" \
    data.format_prompt="${FORMAT_PROMPT}" 
