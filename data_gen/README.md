# SpatialThinker Data Generation Pipeline

This module implements the data generation pipeline for constructing the **STVQA (SpatialThinker Visual Question Answering)** dataset, a spatial reasoning VQA dataset fully grounded in human-annotated scene graphs from Visual Genome.

![Dataset Examples and Distribution](../assets/data_dist_eg.jpg)

## Overview

The pipeline synthetically generates spatial visual question-answer pairs using scene graph data from Visual Genome. It produces multiple-choice questions across **9 distinct spatial reasoning categories**, enabling comprehensive coverage of both 2D and 3D spatial understanding.

> **Paper Dataset**: The **STVQA-7K** dataset was generated using **Claude Sonnet 4** (`claude-sonnet-4-20250514`) for question synthesis, followed by rating-based filtering (56K → 10K) and **GPT-4o** consistency validation (10K → 7K).

### Spatial Reasoning Categories

| Category | Description | Example Question |
|----------|-------------|------------------|
| **Relation** | Spatial predicates (above, behind, near, on top of, etc.) | "Where is the man with respect to the bench?" |
| **Reach** | Physical interaction (holding, touching, carrying) | "What is the woman doing with the bottle?" |
| **Size** | Comparative size (larger, smaller, taller, shorter) | "Which is bigger, the car or the bicycle?" |
| **Orientation** | Directional relationship from specific viewpoints | "From the person's perspective, which direction is the dog?" |
| **Instance Location** | Position within image frame (top-left, center, etc.) | "In which part of the image is the flag located?" |
| **Depth** | Distance from camera (closer/farther) | "Which is closer to the camera, the bookshelf or the table?" |
| **Distance** | Distance comparison to reference objects | "Which object is closer to the cat, the ball or the toy?" |
| **Count** | Object counting | "How many cars are there in the image?" |
| **Existence** | Presence verification (yes/no) | "Is there a cat with a red bow in the picture?" |

## Pipeline Stages

### 1. Synthetic Question Generation (`preprocess_data`)

Generates question-answer pairs from Visual Genome scene graphs using Claude models:
- Processes scene graph data (objects, bounding boxes, relationships)
- Generates MCQ questions with 2-4 options
- Assigns difficulty levels (easy/medium/hard) and quality ratings (1-10)
- Applies salience rules to filter trivial questions
- Auto-corrects count questions using lemmatization

### 2. Count Question Auto-Correction (Built-in)

During question generation, count-based answers are automatically validated and corrected:
- Normalizes object names using lemmatization (handles singular/plural: man/men, person/people)
- Validates count answers against the filtered scene graph objects
- Auto-corrects answer options if the count doesn't match

### 3. Post-hoc Count Fixing (`fix_count_questions`) - Optional

A standalone utility to re-validate and fix count questions in existing CSV files:
- Useful if you need to reprocess an old dataset
- Re-runs the same normalization and validation logic
- Updates the CSV file in place

### 4. Balance + Filter + Upload (`generate_hf_data`)

The main function that combines **balancing**, **filtering**, and **upload** in one step:

**Balancing (per paper):**
- **50% relations**: Half of target samples come from the `relation` category
- **50% distributed equally**: Remaining samples split across 8 other categories (~6.25% each)

**Filtering:**
- Within each category, samples are **sorted by rating (descending)** and top N are selected
- Train and val pools are balanced separately (both maintain 50/50 distribution)

**Parameters:**
- `target_samples`: Total samples to keep (e.g., 10000 out of 56K)
- `relation_percent`: Percentage for relations (default 50%)
- `val_split`: Validation split fraction (default 0.1 = 10%)

### 5. GPT-4o Consistency Validation (`validate_with_gpt4o`)

External validation using GPT-4o to ensure semantic correctness at scale:

1. **Pass@2 Criterion**: For each QA pair, GPT-4o attempts to answer twice using the image
2. **Agreement Check**: If either response matches ground truth, the sample passes
3. **Extended Validation**: Failed samples get 2 additional attempts (total 4 responses)
4. **Discard Rule**: Samples where all 4 responses disagree with ground truth are discarded as potentially incorrect or ambiguous

This filtering typically retains **~70-75% of samples**. In the paper: 10K → **~7K (STVQA-7K)**.

### 6. Dataset Variants (`generate_easy_hard_splits`)

Generates difficulty-stratified dataset versions:
- Easy split: simple, clear relationships
- Hard split: medium + hard questions requiring complex reasoning

## Installation

```bash
# Install dependencies
pip install fire pandas anthropic python-dotenv nltk inflect tqdm datasets huggingface_hub

# Download NLTK data (done automatically on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key    # Required for question generation
HF_TOKEN=your_huggingface_token              # Required for HuggingFace upload
OPENAI_API_KEY=your_openai_api_key           # Required for GPT-4o validation
```

## Usage

### STVQA-7K Pipeline (Paper Reproduction)

This is the exact pipeline used to create the **STVQA-7K** dataset in the paper:

```bash
# Step 1: Generate raw QA pairs from scene graphs
# Full dataset: --data_cap=56224 (56K samples)
# Cost-effective: --data_cap=12000 (12K samples, similar final quality)
python data_gen/generate_data.py preprocess_data \
    --data_cap=56224 \
    --model_name=claude-sonnet-4-20250514

# Step 2: Balance + Filter to 10K (50% relations, top-rated per category)
python data_gen/generate_data.py generate_hf_data \
    --input_file="data/spatialthinker_vqa_train.csv" \
    --target_samples=10000 \
    --relation_percent=50 \
    --upload_to_hf=False

# Step 3: Validate with GPT-4o (~75% pass rate → ~7K samples)
python data_gen/generate_data.py validate_with_gpt4o \
    --input_file="data_train.csv" \
    --output_file="data/spatialthinker_vqa_validated.csv"

# Step 4: Upload final STVQA-7K dataset to HuggingFace
python data_gen/generate_data.py generate_hf_data \
    --input_file="data/spatialthinker_vqa_validated.csv" \
    --target_repo="your-username/spatialthinker_vqa_7k" \
    --upload_to_hf=True
```

**Pipeline Summary:**
| Step | Input | Output | Notes |
|------|-------|--------|-------|
| 1. Generate | 56K scene graphs | ~56K raw QA pairs | Claude Sonnet 4 |
| 2. Balance+Filter | 56K samples | 10K balanced | 50% relations, top-rated per category |
| 3. Validate | 10K samples | **~7K validated** | GPT-4o pass@2 criterion |
| 4. Upload | 7K samples | HuggingFace dataset | Balanced train/val split |

> **💰 Cost-Saving Tip**: For reduced API costs, generate **12K samples** instead of 56K in Step 1 (`--data_cap=12000`), then balance/filter to 10K and validate. This produces similar final results with significantly lower Claude/GPT-4o API usage.

---

### Generate Questions from Scene Graphs

```bash
# Generate 100 samples with Claude Haiku 4 (fastest)
python data_gen/generate_data.py preprocess_data \
    --data_cap=100 \
    --model_name=claude-haiku-4-5

# Generate with Claude Sonnet 4 (balanced quality/speed)
python data_gen/generate_data.py preprocess_data \
    --data_cap=1000 \
    --model_name=claude-sonnet-4-5

# Generate with Claude Opus 4 (highest quality)
python data_gen/generate_data.py preprocess_data \
    --data_cap=500 \
    --model_name=claude-opus-4-5

# Resume from checkpoint
python data_gen/generate_data.py preprocess_data \
    --data_cap=56224 \
    --resume=True \
    --resume_file="data/spatialthinker_vqa_train.csv"
```

**Available Models:**
| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `claude-haiku-4-5` | Fastest | Good | Testing, rapid iteration |
| `claude-sonnet-4-5` | Balanced | High | Production generation |
| `claude-opus-4-5` | Slowest | Highest | Final quality pass |
| `claude-sonnet-4-20250514` | Balanced | High | **Original STVQA-7K dataset model** |

### View Dataset Statistics

```bash
python data_gen/generate_data.py print_data_stats \
    --filename="data/spatialthinker_vqa_train.csv"
```

Output:
```
Current data statistics:
{'relation': '20.0%', 'reach': '4.0%', 'size': '0.0%', 'orientation': '0.0%', 
 'instance_location': '0.0%', 'depth': '46.0%', 'distance': '20.0%', 
 'count': '0.0%', 'existence': '10.0%', 'easy': '28.0%', 'medium': '68.0%', 
 'hard': '4.0%', 'total_samples': '50'}
```

### Fix Count Questions (Optional Post-Processing)

Re-validate and fix count-based answers in an existing CSV file. This is **optional** since count auto-correction already runs during `preprocess_data`:

```bash
python data_gen/generate_data.py fix_count_questions \
    --file="data/spatialthinker_vqa_train.csv"
```

### Validate with GPT-4o (Quality Filtering)

Run consistency-based validation using GPT-4o to filter out incorrect/ambiguous samples:

```bash
# Validate all samples (requires OPENAI_API_KEY in .env)
python data_gen/generate_data.py validate_with_gpt4o \
    --input_file="data/spatialthinker_vqa_top10k.csv" \
    --output_file="data/spatialthinker_vqa_validated.csv"

# Validate a subset (e.g., first 1000 samples for testing)
python data_gen/generate_data.py validate_with_gpt4o \
    --input_file="data/spatialthinker_vqa_train.csv" \
    --output_file="data/spatialthinker_vqa_validated.csv" \
    --sample_limit=1000
```

**Note**: GPT-4o validation requires `OPENAI_API_KEY` in your `.env` file. This can be expensive for large datasets.

### Balance, Filter & Upload (`generate_hf_data`)

The main function for processing generated data. Combines **balancing** (category distribution), **filtering** (top-rated per category), and **upload** in one step:

```bash
# Balance to 10K samples (50% relations) and upload
python data_gen/generate_data.py generate_hf_data \
    --input_file="data/spatialthinker_vqa_train.csv" \
    --target_samples=10000 \
    --relation_percent=50 \
    --target_repo="your-username/spatialthinker_vqa_10k" \
    --upload_to_hf=True

# Just balance locally without uploading (outputs: data_train.csv, data_val.csv)
python data_gen/generate_data.py generate_hf_data \
    --input_file="data/spatialthinker_vqa_train.csv" \
    --target_samples=10000 \
    --relation_percent=50 \
    --upload_to_hf=False

# Custom balance: 30% relations, 70% others
python data_gen/generate_data.py generate_hf_data \
    --input_file="data/spatialthinker_vqa_train.csv" \
    --target_samples=5000 \
    --relation_percent=30 \
    --upload_to_hf=False
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_samples` | All data | Total samples to keep (e.g., 10000) |
| `relation_percent` | 50 | Percentage allocated to 'relation' category |
| `val_split` | 0.1 | Validation split fraction (10%) |
| `upload_to_hf` | True | Upload to HuggingFace |
| `target_repo` | - | HuggingFace repository name |

### Generate Easy/Hard Splits

```bash
# Print stats only
python data_gen/generate_data.py easy_hard_split \
    --dataset_name="hunarbatra/spatialthinker_vqa_10k" \
    --print_stats_only=True

# Generate and upload splits
python data_gen/generate_data.py easy_hard_split \
    --dataset_name="hunarbatra/spatialthinker_vqa_10k"
```

### Download Dataset for Review

```bash
python data_gen/generate_data.py data_review \
    --dataset_name="hunarbatra/spatialthinker_vqa_10k"
```

## Data Balancing Details

The `generate_hf_data` function implements the paper's balancing strategy:

1. **Category Distribution (Paper):**
   - **50% relations**: Half of `target_samples` goes to the `relation` category
   - **50% others**: Remaining samples split equally among 8 other categories (~6.25% each)
   - Configurable via `relation_percent` parameter

2. **Quality-Based Selection**: Within each category:
   - Samples are **sorted by rating (descending)**
   - Top N samples are selected based on the category's allocation
   - Both train and val pools are sorted independently

3. **Train/Val Split**: For each category:
   - Split data into train pool (90%) and val pool (10%) first
   - Sort each pool by rating
   - Take top N from each pool to maintain balance in both sets

4. **Answer Distribution**: Answer options are shuffled using deterministic seeding (based on question text) to ensure uniform distribution across A, B, C, D options, preventing models from learning positional biases.

**Example with `target_samples=10000, relation_percent=50`:**
| Category | Total | Train (90%) | Val (10%) |
|----------|-------|-------------|-----------|
| relation | 5,000 | 4,500 | 500 |
| reach | 625 | 562 | 63 |
| size | 625 | 562 | 63 |
| ... | ... | ... | ... |
| **Total** | **10,000** | **~9,000** | **~1,000** |

**Note**: If a category has fewer samples than its allocation, all available samples are used.

## Output Format

Generated samples include:

| Field | Description |
|-------|-------------|
| `image_id` | Visual Genome image identifier |
| `images` | PIL image data |
| `problem` | Full prompt with instructions and question |
| `question_only` | Raw question text |
| `question_with_options` | Question with MCQ options |
| `options` | List of answer options |
| `answer` | Full answer with scene graph and answer tag |
| `answer_only` | Just the answer letter (A/B/C/D) |
| `answer_text` | Answer with option text |
| `category` | Spatial reasoning category |
| `level` | Difficulty level (easy/medium/hard) |
| `rating` | Quality rating (1-10) |

## Source Dataset

The pipeline uses Visual Genome scene graphs:
- **Source**: `JosephZ/vg150_train_sgg_prompt` on HuggingFace
- **Size**: 56,224 images with dense scene graph annotations
- **Annotations**: Objects, bounding boxes (pixel coordinates), and relational triplets

## Scalability

The pipeline is designed to scale:
- Supports up to ~108K samples (full Visual Genome coverage)
- Checkpoint/resume functionality for long runs
- Automatic saving every 10 samples
- Configurable batch processing

## File Structure

```
data_gen/
├── README.md           # This file
├── generate_data.py    # Main pipeline script
├── prompt.py           # Question generation prompts (QUESTION_PREFIX, QUESTION_GEN_PROMPT)
└── utils.py            # Utility functions (extract_json, compute_data_stats)
```

## Citation

If you use this dataset or pipeline, please cite:

```bibtex
@misc{batra2025spatialthinkerreinforcing3dreasoning,  
  title={SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards},  
  author={Hunar Batra and Haoqin Tu and Hardy Chen and Yuanze Lin and Cihang Xie and Ronald Clark},  
  year={2025},  
  eprint={2511.07403},  
  archivePrefix={arXiv},  
  primaryClass={cs.CV},  
  url={https://arxiv.org/abs/2511.07403},  
}
```

## License

This project is released under the MIT License. The Visual Genome dataset has its own licensing terms.

