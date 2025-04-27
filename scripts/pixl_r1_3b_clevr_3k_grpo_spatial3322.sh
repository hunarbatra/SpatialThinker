set -x
export WANDB_API_KEY=a3642c8b11ba4d06e93d04f615579a18d8a19e07

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

mkdir -p logs

# FORMAT_PROMPT="""## Task Description
# You are a highly intelligent vision-language assistant. You are shown an image depicting a 3D scene with multiple objects. Your task is to:
# 1. Visually identify the objects present in the scene.
# 2. Estimate their 3D spatial coordinates (x, y, z) based on the layout and camera viewpoint constraints.
# 3. Use this spatial understanding to answer the question.

# ## Scene Information
# - Room Dimensions: {{'x': 6.0, 'y': 6.0, 'z': 1.0}}
# - Coordinate Range:
#   - x and y values range from -3.0 to 3.0, defining the ground plane:
#     - x represents left (-) to right (+)
#     - y represents near (-) to far (+) from the camera
#   - z values represent object height above ground, generally within the range 0.0 to 1.0
# - Coordinate Origin: (0, 0, 0) is the center of the room floor.

# ## Output Format
# Begin with your reasoning in the <think> tags. Then list all detected objects and their positions from left to right (sorted by x-coordinate) inside a <scene> block as a JSON array. Use 'object_type_{{n}}' naming format, incrementing n only for duplicate objects. Finally, place your final answer in the <answer> tag.

# Example output:
# <think>
# {thinking process here}
# </think>
# <scene>[{{"id": 1, "object_name": "cube_1", "x": -2.4, "y": 0.8, "z": 0.7}}, {{"id": 2, "object_name": "sphere_1", "x": 0.0, "y": -1.5, "z": 0.4}}, {{"id": 3, "object_name": "cube_2", "x": 2.1, "y": 1.1, "z": 0.7}}]</scene>
# <answer>{final answer here}</answer>

# ## Question
# """

DATA_FILE="hunarbatra/clevr-cogent-pixl-r1-3k"

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=spatial3322 \
    trainer.experiment_name=pixl_r1_clevr_3k_grpo_v5_spatial3322 \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=ckpts/pixl_r1_clevr_3k_grpo_v5_spatial3322 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.training_steps=50 \
    worker.rollout.n=8 \
    trainer.total_episodes=50 \
    > logs/v5_pixl_r1_spatial3322_clevr_3k_grpo.log 2>&1
    # worker.rollout.tensor_parallel_size=1 \
    # data.format_prompt="'${FORMAT_PROMPT}'" \
    