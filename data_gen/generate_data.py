import os
import io
import re
import ast
import json
import random

import fire
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Literal
from collections import Counter

import nltk
import inflect
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, Features, Value, Sequence
from datasets import Image as HFImage
from huggingface_hub import login, snapshot_download

import anthropic

from utils import extract_json_from_output, compute_data_stats
from prompt import QUESTION_PREFIX, QUESTION_GEN_PROMPT, GPT4O_VALIDATION_PROMPT

# Load environment variables
load_dotenv()

# Setup NLTK (quiet mode to suppress download messages)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
lemmatizer = WordNetLemmatizer()
p = inflect.engine()

# API Clients (lazy initialization)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Constants
SOURCE_DS = "JosephZ/vg150_train_sgg_prompt"
MODEL_NAME = "claude-sonnet-4-5"
TEMPERATURE = 0.0


def call_model(
    prompt: str, 
    model_name: str = MODEL_NAME, 
    temperature: float = TEMPERATURE,
    service: Literal["openai", "anthropic"] = "anthropic"
) -> str:
    print(f"Using Service: {service}")
    if service == "openai":
        if openai_client is None:
            raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY in .env")
        response = openai_client.responses.create(
            model=model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output_text
    elif service == "anthropic":
        if anthropic_client is None:
            raise ValueError("Anthropic API key not set. Please set ANTHROPIC_API_KEY in .env")
        response = anthropic_client.messages.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt + "Only return the JSON object, no additional text."}
            ],
            max_tokens=256,
            temperature=temperature,
        )
        output = response.content[0].text
        try:
            output = output.split("{")[1].split("}")[0]
            output = "{" + output + "}"
        except:
            return output
        return output
    else:
        raise ValueError(f"Invalid service: {service}. Use 'openai' or 'anthropic'.")


def normalize_token(token):
    token = token.lower()
    token = re.sub(r"'s$", '', token)
    if not token.isalpha():
        return None
    return lemmatizer.lemmatize(token, pos='n')  # Normalize to singular noun


def expand_question_words(question: str):
    tokens = word_tokenize(question)
    expanded_words = set()

    for token in tokens:
        token = token.lower()
        token = re.sub(r"'s$", '', token)  # remove possessive 's
        
        if not token.isalpha():
            continue  # skip numbers, symbols, etc.

        # Original token
        expanded_words.add(token)

        # Lemmatized token (singular noun)
        lemma = lemmatizer.lemmatize(token, pos='n')
        expanded_words.add(lemma)

        # Try plural form
        if lemma.isalpha():
            try:
                plural = p.plural(lemma)
                if plural and plural.isalpha():
                    expanded_words.add(plural)
            except Exception:
                pass

        # Try singular form
        if token.isalpha():
            try:
                singular = p.singular_noun(token)
                if singular and singular.isalpha():
                    expanded_words.add(singular)
            except Exception:
                pass

    return expanded_words


def map_objects(objects, question_words):
    filtered_objects = []
    for obj in objects:
        full_id = obj["id"]
        base_name = full_id.split(".")[0].lower()
    
        if base_name in question_words:
            filtered_objects.append(obj)
            
    return filtered_objects


def map_relationships(relationships, question_words):
    filtered_relationships = []
    for rel in relationships:
        subject_id = rel["subject"]
        object_id = rel["object"]
        
        sub_base = subject_id.split(".")[0].lower()
        obj_base = object_id.split(".")[0].lower()
        
        if sub_base in question_words and obj_base in question_words:
            filtered_relationships.append(rel)
    
    return filtered_relationships


def fix_count_questions(file: str = "data/spatialthinker_vqa_train.csv"):
    df = pd.read_csv(file)
    fixed = 0

    for idx, row in df.iterrows():
        if row["category"] != "count":
            continue

        question = row["question_only"]
        try:
            scene = json.loads(re.search(r"<scene>(.*?)</scene>", row["answer"]).group(1))
        except Exception:
            print(f"[Warning] Could not parse scene for row {idx}")
            continue

        options = row["options"]
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except Exception:
                continue

        question_words = expand_question_words(question)
        objects = scene["objects"]

        # Step 1: Normalize object names and count by root
        root_count = Counter()
        raw_to_root = {}
        
        for obj in objects:
            raw_name = obj["id"].split(".")[0].lower()
            root_name = normalize_token(raw_name)
            if root_name:
                raw_to_root[raw_name] = root_name
                root_count[root_name] += 1
                
        matching_total = sum(
            count for root, count in root_count.items() if root in question_words
        )
        matched_roots = [root for root in root_count if root in question_words]

        corrected_answer = None
        count_str = str(matching_total)
                
        try:
            answer = re.search(r"<answer>(.*?)</answer>", row["answer"]).group(1)
            curr_count_str = options[ord(answer.upper()) - ord('A')]
        except:
            curr_count_str = None
            answer = None
            
        if curr_count_str == count_str:
            print(f'Skipping {idx} as answer is already correct | curr_count_str: {curr_count_str}, count_str: {count_str}, options: {options}')
            continue

        # Sanitize options for matching
        sanitized_options = [re.sub(r'^\([A-D]\)\s*', '', opt).strip() for opt in options]
        try:
            index = sanitized_options.index(count_str)
            corrected_answer = chr(ord('A') + index)
        except ValueError:
            # Replace current correct answer's option with correct count
            try:
                original_index = ord(re.search(r"<answer>([A-D])</answer>", row["answer"]).group(1)) - ord('A')
            except:
                original_index = 0

            if 0 <= original_index < len(options):
                options[original_index] = count_str
                corrected_answer = chr(ord('A') + original_index)
            else:
                options[0] = count_str
                corrected_answer = 'A'

        # Step 3: Update DataFrame if changed
        if corrected_answer:
            new_answer = f"<scene>{json.dumps(scene)}</scene>\n<answer>{corrected_answer}</answer>"
            df.at[idx, "answer"] = new_answer
            df.at[idx, "options"] = options
            fixed += 1

    # Save updated file
    df.to_csv(file, index=False)
    print(f"\n✅ Fixed {fixed} count questions in '{file}'")


# python3 data_gen/generate_data.py preprocess_data --data_cap=56224 --resume=True --resume_file="data/spatialthinker_vqa_train.csv"
# python3 data_gen/generate_data.py preprocess_data --data_cap=100 --model_name=claude-sonnet-4-latest
# 
# Available Anthropic models:
#   claude-sonnet-4-5           (default, balanced speed/quality)
#   claude-opus-4-5             (highest quality, slower)
#   claude-haiku-4-5            (fastest, cheapest)
#   claude-sonnet-4-20250514    (original dataset generation model)
def preprocess_data(
    data_cap: int = None, 
    resume: bool = False,
    resume_file: str = "data/spatialthinker_vqa_train.csv",
    model_name: str = "claude-sonnet-4-5",
    temperature: float = 0.0,
):
    print(f"🤖 Using model: {model_name} (temperature: {temperature})")
    
    dataset = load_dataset(SOURCE_DS)["train"]
    
    if data_cap is not None:
        dataset = dataset.select(range(data_cap))

    updated_df = []
    
    if resume:
        df = pd.read_csv(resume_file)
        dataset = dataset.select(range(len(df), len(dataset)))
        print(f'Resuming from {len(df)} rows')
        # get initial processed rows in updated_df list
        updated_df = df.to_dict("records")
        file_save_path = '_'.join(resume_file.split('_')[:-1]) + f'_{data_cap}.csv'
    
    for row in tqdm(dataset, desc="Processing images", total=len(dataset)):
        image_id = row["image_id"]
        image_pil = row["image"]
        width, height = image_pil.size
        
        objects = row["objects"]
        relationships = row["relationships"]
        
        if isinstance(objects, str):
            objects = json.loads(objects)
        if isinstance(relationships, str):
            relationships = json.loads(relationships)
        
        scene_tag_data = json.dumps({"objects": objects, "relationships": relationships})
        
        question_prefix = QUESTION_PREFIX.format(W=width, H=height)
        
        if len(updated_df) > 0:
            data_stats, last_vals = compute_data_stats(updated_df)
            print(f'\nCurrent data statistics:\n{data_stats}')
            
            # Format the stats for the prompt
            stats_str = "\n".join([f"{k}: {v}" for k, v in data_stats.items()])
            stats_str += f"\nDeprioritise repeating the last generation step's (max 3 previous) category and level that were as follows: {last_vals}"
            
            # Replace the placeholder in the prompt with the actual stats
            prompt_with_stats = QUESTION_GEN_PROMPT.replace("$DATA_STATS$", stats_str)
            prompt = f"{prompt_with_stats}\n{scene_tag_data}"
        else:
            # If no data yet, use a simple message
            prompt_with_stats = QUESTION_GEN_PROMPT.replace("$DATA_STATS$", "No data available yet.")
            prompt = f"{prompt_with_stats}\n{scene_tag_data}"
        output = call_model(prompt, service="anthropic", model_name=model_name, temperature=temperature)
        print(f'output: {output}')
        
        json_str = extract_json_from_output(output)
        output_json = json.loads(json_str)
        question = output_json["question"]
        options = output_json["options"]  # MCQ options - might be string or list
        # Convert options to list if it's a string
        if isinstance(options, str):
            options = ast.literal_eval(options)
        answer = output_json["answer"]  # MCQ letter choice (A, B, C, D)
        category = output_json["category"]
        level = output_json["level"]  # easy, medium, hard question
        rating = output_json.get("rating", 0)  # optional rating field
        
        question_words = expand_question_words(question)
        filtered_objects = map_objects(objects, question_words)
        filtered_relationships = map_relationships(relationships, question_words)
        filtered_scene_tag_data = json.dumps({"objects": filtered_objects, "relationships": filtered_relationships})
        
        if category.startswith("REL_"):
            category = "relation"
        
        question_only = question
        # Format question with MCQ options
        options_text = "\n".join(options)
        question_with_options = f"{question}\n\nOptions:\n{options_text}"
        question = f"{question_prefix}\n{question_with_options}"
        
        # Strip (A), (B), etc. prefixes from options for storage
        options_clean = []
        for option in options:
            # Remove (A), (B), (C), (D) prefixes using regex
            clean_option = re.sub(r'^\([A-D]\)\s*', '', option)
            options_clean.append(clean_option)
        options = options_clean
        
        answer = f"<scene>{filtered_scene_tag_data}</scene>\n<answer>{answer}</answer>"
        
        if category == "count":
            root_count = Counter()
            raw_to_root = {}

            for obj in filtered_objects:
                raw_name = obj["id"].split(".")[0].lower()
                root_name = normalize_token(raw_name)
                if root_name:
                    raw_to_root[raw_name] = root_name
                    root_count[root_name] += 1

            matching_total = sum(
                count for root, count in root_count.items() if root in question_words
            )
            matched_roots = [root for root in root_count if root in question_words]

            corrected_answer = None
            count_str = str(matching_total)

            try:
                curr_count_str = options[ord(output_json["answer"].upper()) - ord('A')]
            except:
                curr_count_str = None

            if curr_count_str == count_str:
                pass  # Already correct, no update needed
            else:
                # Sanitize options
                sanitized_options = [re.sub(r'^\([A-D]\)\s*', '', opt).strip() for opt in options]

                try:
                    index = sanitized_options.index(count_str)
                    corrected_answer = chr(ord('A') + index)
                except ValueError:
                    # Replace the current answer's slot with correct count
                    original_index = ord(output_json["answer"].upper()) - ord('A')
                    if 0 <= original_index < len(options):
                        options[original_index] = count_str
                        corrected_answer = output_json["answer"].upper()
                    else:
                        options[0] = count_str
                        corrected_answer = 'A'

            if corrected_answer:
                print(f"[Auto-correct] Count-based answer adjusted to: {corrected_answer}, original option: {output_json['answer']}, correct count: {count_str}")
                answer = f"<scene>{scene_tag_data}</scene>\n<answer>{corrected_answer}</answer>"

        # Create the row data
        cur_row = {
            "image_id": image_id,
            "images": image_pil,
            "problem": question,
            "question_only": question_only,
            "question_with_options": question_with_options,
            "question_words": question_words,
            "options": options,
            "answer": answer,
            "category": category,
            "level": level,
            "rating": rating,
            "full_scene_graph": scene_tag_data
        }
        
        updated_df.append(cur_row)
        
        if len(updated_df) % 10 == 0:
            # save updated_df to csv so far
            df = pd.DataFrame(updated_df)
            if resume:
                df.to_csv(file_save_path, index=False)
            else:
                df.to_csv("data/spatialthinker_vqa_train.csv", index=False)


def print_data_stats(filename: str = "data/spatialthinker_vqa_56224.csv"):
    df = pd.read_csv(filename)
    data_stats, _ = compute_data_stats(df)
    print(f'\nCurrent data statistics:\n{data_stats}')


# python3 data_gen/generate_data.py generate_hf_data --target_samples=10000 --relation_percent=50
# 
# Balancing: 50% relations, 50% distributed equally across 8 other categories
# Filtering: Within each category, samples are sorted by rating and top N are selected
# Train/Val: Split proportionally (default 90/10) with both sets balanced
def generate_hf_data(
    upload_to_hf: bool = True,
    input_file: str = "data/spatialthinker_vqa_56224.csv",
    val_split: float = 0.1,
    target_repo: str = "hunarbatra/spatialthinker_vqa_10k",
    target_samples: int = None,
    relation_percent: float = 50.0,
):
    """
    Balance, filter, and upload dataset to HuggingFace.
    
    Args:
        upload_to_hf: Whether to upload to HuggingFace
        input_file: Path to input CSV file
        val_split: Fraction for validation set (default 0.1 = 10%)
        target_repo: HuggingFace repo name
        target_samples: Total samples to select (e.g., 10000). If None, uses all data.
        relation_percent: Percentage allocated to 'relation' category (default 50%)
    
    Example:
        # Keep 10k samples: 5k relations + 5k others (625 each)
        generate_hf_data(target_samples=10000, relation_percent=50)
    """
    df = pd.read_csv(input_file)
    
    categories = ['relation', 'reach', 'size', 'orientation', 'instance_location', 'depth', 'distance', 'count', 'existence']
    other_categories = [c for c in categories if c != 'relation']
    
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}")
    
    # If no target specified, use all available data
    if target_samples is None:
        target_samples = total_rows
    
    target_samples = min(target_samples, total_rows)
    print(f"Target samples: {target_samples} (relation: {relation_percent}%, others: {100-relation_percent}%)")
    
    # Calculate allocations per category
    relation_total = int(target_samples * relation_percent / 100)
    other_total = target_samples - relation_total
    other_per_category = max(1, int(other_total / len(other_categories)))
    
    # Calculate train/val splits per category
    # Ensure minimum 1 sample for train when total > 0
    allocations = {}
    
    relation_val = max(1, int(relation_total * val_split)) if relation_total > 0 else 0
    relation_train = max(0, relation_total - relation_val)
    allocations['relation'] = {'train': relation_train, 'val': relation_val}
    
    for cat in other_categories:
        cat_val = max(1, int(other_per_category * val_split)) if other_per_category > 1 else 0
        cat_train = max(1, other_per_category - cat_val) if other_per_category > 0 else 0
        allocations[cat] = {'train': cat_train, 'val': cat_val}
    
    print(f"\n📊 Target allocations:")
    print(f"  relation: {allocations['relation']['train']} train + {allocations['relation']['val']} val = {relation_total}")
    print(f"  others: {allocations[other_categories[0]]['train']} train + {allocations[other_categories[0]]['val']} val = ~{other_per_category} each")
    
    # Initialize empty dataframes to store the sampled data
    sampled_train_df = pd.DataFrame()
    sampled_val_df = pd.DataFrame()
    
    # Sample data for each category
    for category in categories:
        category_df = df[df['category'] == category]
        
        if len(category_df) == 0:
            print(f"Warning: No data found for category '{category}'")
            continue
        
        train_target = allocations[category]['train']
        val_target = allocations[category]['val']
        
        # Split into train/val pools first (preserving original behavior)
        if 'rating' in category_df.columns:
            # Split category data into train/val pools
            n_train_pool = min(int((1 - val_split) * len(category_df)), len(category_df) - 1)
            n_train_pool = max(1, n_train_pool)
            
            train_pool = category_df.sample(n=n_train_pool, random_state=42)
            val_pool = category_df.loc[~category_df.index.isin(train_pool.index)]
            
            # Sort each pool by rating (descending) and take top N
            train_pool = train_pool.sort_values(by='rating', ascending=False)
            val_pool = val_pool.sort_values(by='rating', ascending=False)
            
            train_samples = min(train_target, len(train_pool))
            val_samples = min(val_target, len(val_pool))
            
            train_df = train_pool.iloc[:train_samples]
            val_df = val_pool.iloc[:val_samples]
        else:
            # No rating column - random sample
            available = len(category_df)
            train_samples = min(train_target, int(available * (1 - val_split)))
            val_samples = min(val_target, available - train_samples)
            
            shuffled = category_df.sample(n=train_samples + val_samples, random_state=42)
            train_df = shuffled.iloc[:train_samples]
            val_df = shuffled.iloc[train_samples:train_samples + val_samples]
        
        print(f"Category '{category}': {len(train_df)} train, {len(val_df)} val (target: {train_target}/{val_target})")
        
        sampled_train_df = pd.concat([sampled_train_df, train_df], ignore_index=True)
        sampled_val_df = pd.concat([sampled_val_df, val_df], ignore_index=True)
    
    print(f"\n✅ Total sampled: {len(sampled_train_df)} train, {len(sampled_val_df)} val")
    
    # Load the original dataset to get the images
    print("Loading original dataset to get images...")
    original_dataset = load_dataset(SOURCE_DS)["train"]
    
    # Create a mapping from image_id to image for faster lookup
    valid_image_ids = set(sampled_train_df["image_id"]).union(set(sampled_val_df["image_id"]))
    valid_image_ids = set(map(str, valid_image_ids))
    print(f"Total valid image IDs: {len(valid_image_ids)}")
    
    # get a set of image_ids from original_dataset
    original_image_ids = set(original_dataset["image_id"])
    
    original_dataset = original_dataset.filter(lambda x: x["image_id"] in valid_image_ids)
    print(f"Filtered dataset size: {len(original_dataset)}")
    
    image_id_to_image = {}
    for item in tqdm(original_dataset, total=len(original_dataset), desc="Creating image mapping"):
        image_id_to_image[item["image_id"]] = item["image"]
            
    # Add images to the sampled dataframes
    def add_images_to_df(df):
        images = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding images"):
            image_id = row["image_id"]
            if image_id in image_id_to_image:
                images.append(image_id_to_image[image_id])
            else:
                print(f"Warning: Image ID {image_id} not found in original dataset")
                images.append(None)
        df["images"] = images
        return df

    sampled_train_df["image_id"] = sampled_train_df["image_id"].astype(str)
    sampled_val_df["image_id"] = sampled_val_df["image_id"].astype(str)

    sampled_train_df = add_images_to_df(sampled_train_df)
    sampled_val_df = add_images_to_df(sampled_val_df)

    # Convert PIL image to bytes
    def pil_to_bytes(img):
        if isinstance(img, dict) and "bytes" in img:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return {"bytes": buf.getvalue()}
    
    def extract_answer_from_tags(answer_text):
        match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("Answer not found within answer tags")

    # Apply transformations
    sampled_train_df["images"] = sampled_train_df["images"].apply(pil_to_bytes)
    sampled_val_df["images"] = sampled_val_df["images"].apply(pil_to_bytes)
    
    # shuffle options to maintain random distribution of correct answer
    def shuffle_options(row):
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
            
        current_correct_answer = extract_answer_from_tags(row["answer"])
        current_correct_index = ord(current_correct_answer.upper()) - ord('A')
        current_correct_option_text = options[current_correct_index]
        
        # shuffle options with random seed 42
        seed = hash(row["question_only"]) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(options)
    
        # update answer in <answer> tags in "answer" based on that
        new_correct_index = options.index(current_correct_option_text)
        new_correct_answer = chr(ord('A') + new_correct_index)
        updated_answer = row["answer"].split("<answer>")[0].strip() + "\n<answer>" + new_correct_answer + "</answer>"
        
        # update question_with_options with question_only + shuffled options
        updated_question_with_options = row["question_only"] + "\nOptions:"
        for i, option in enumerate(options):
            updated_question_with_options += "\n(" + chr(ord('A') + i) + ") " + option
        
        return options, updated_answer, updated_question_with_options
    
    if len(sampled_train_df) > 0:
        sampled_train_df[["options", "answer", "question_with_options"]] = sampled_train_df.apply(shuffle_options, axis=1, result_type="expand")
    if len(sampled_val_df) > 0:
        sampled_val_df[["options", "answer", "question_with_options"]] = sampled_val_df.apply(shuffle_options, axis=1, result_type="expand")
    
    # print stats for train and val answer distribution A...D
    train_stats = sampled_train_df["answer"].apply(extract_answer_from_tags).value_counts() if len(sampled_train_df) > 0 else pd.Series()
    val_stats = sampled_val_df["answer"].apply(extract_answer_from_tags).value_counts() if len(sampled_val_df) > 0 else pd.Series()
    print("Train answer distribution:", train_stats)
    print("Val answer distribution:", val_stats)
    
    sampled_train_df["answer_only"] = sampled_train_df["answer"].apply(extract_answer_from_tags)
    sampled_val_df["answer_only"] = sampled_val_df["answer"].apply(extract_answer_from_tags)
    
    def update_answer_column(row, gen_answer_option_text: bool = False):
        match = re.search(r"<answer>(.*?)</answer>", row["answer"], re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            options = row["options"]
            if isinstance(options, str):
                options = ast.literal_eval(options)
            correct_option_text = options[ord(answer_text.upper()) - ord('A')]
            if gen_answer_option_text:
                updated_answer = row["answer"].split("<answer>")[0].strip() + "\n<answer>" + f"({answer_text}) {correct_option_text}" + "</answer>"
            else:
                updated_answer = row["answer"].split("<answer>")[0].strip() + "\n<answer>" + correct_option_text + "</answer>"
            return updated_answer
        raise ValueError("Answer not found within answer tags")
    
    if len(sampled_train_df) > 0:
        sampled_train_df["answer_text"] = sampled_train_df.apply(update_answer_column, axis=1)
        sampled_train_df["answer_option_text"] = sampled_train_df.apply(update_answer_column, axis=1, gen_answer_option_text=True)
        sampled_train_df["answer_text_only"] = sampled_train_df["answer_text"].apply(extract_answer_from_tags)
        sampled_train_df["answer_option_text_only"] = sampled_train_df["answer_option_text"].apply(extract_answer_from_tags)
    
    if len(sampled_val_df) > 0:
        sampled_val_df["answer_text"] = sampled_val_df.apply(update_answer_column, axis=1)
        sampled_val_df["answer_option_text"] = sampled_val_df.apply(update_answer_column, axis=1, gen_answer_option_text=True)
        sampled_val_df["answer_text_only"] = sampled_val_df["answer_text"].apply(extract_answer_from_tags)
        sampled_val_df["answer_option_text_only"] = sampled_val_df["answer_option_text"].apply(extract_answer_from_tags)
    else:
        # Ensure empty dataframes have required columns
        for col in ["answer_text", "answer_option_text", "answer_text_only", "answer_option_text_only"]:
            sampled_val_df[col] = pd.Series(dtype='str')
    
    # Same for train if empty
    if len(sampled_train_df) == 0:
        for col in ["answer_text", "answer_option_text", "answer_text_only", "answer_option_text_only"]:
            sampled_train_df[col] = pd.Series(dtype='str')
        
    def update_problem_column(row):
        # Extract image size from existing problem string
        match = re.search(r"Image size:\s*\((\d+)\s*x\s*(\d+)\)", str(row["problem"]))
        image_size = f"Image size: ({match.group(1)} x {match.group(2)})" if match else "Image size: (Unknown)"

        # Build the new problem string with MCQ options
        if 'question_with_options' in row.index and pd.notna(row['question_with_options']):
            question_text = str(row['question_with_options'])
        else:
            question_text = str(row['question_only'])
        
        # Format the QUESTION_PREFIX with width and height from image_size
        width_height_match = re.search(r"(\d+)\s*x\s*(\d+)", image_size)
        if width_height_match:
            W, H = width_height_match.groups()
            formatted_prefix = QUESTION_PREFIX.format(W=W, H=H)
        else:
            formatted_prefix = QUESTION_PREFIX.format(W="Unknown", H="Unknown")
        
        updated_problem = f"{formatted_prefix}\nQ. {question_text}"
        return str(updated_problem)

    # Apply to both dataframes (only if non-empty)
    if len(sampled_train_df) > 0:
        sampled_train_df["problem"] = sampled_train_df.apply(update_problem_column, axis=1)
    if len(sampled_val_df) > 0:
        sampled_val_df["problem"] = sampled_val_df.apply(update_problem_column, axis=1)
    
    # Ensure answer_only column exists for both
    if len(sampled_train_df) > 0 and "answer_only" not in sampled_train_df.columns:
        sampled_train_df["answer_only"] = sampled_train_df["answer"].apply(extract_answer_from_tags)
    if len(sampled_val_df) > 0 and "answer_only" not in sampled_val_df.columns:
        sampled_val_df["answer_only"] = sampled_val_df["answer"].apply(extract_answer_from_tags)
    elif len(sampled_val_df) == 0:
        sampled_val_df["answer_only"] = pd.Series(dtype='str')
    if len(sampled_train_df) == 0 and "answer_only" not in sampled_train_df.columns:
        sampled_train_df["answer_only"] = pd.Series(dtype='str')

    # save as csv
    sampled_train_df.to_csv("data_train.csv", index=False)
    sampled_val_df.to_csv("data_val.csv", index=False)
    
    # Define dataset features - handle both old and new format
    feature_dict = {
        'image_id': Value('string'),
        'images': HFImage(),
        'problem': Value('string'),
        'question_only': Value('string'),
        'answer': Value('string'),
        'answer_only': Value('string'),
        'answer_text': Value('string'),
        'answer_text_only': Value('string'),
        'answer_option_text': Value('string'),
        'answer_option_text_only': Value('string'),
        'category': Value('string'),
        'level': Value('string')
    }
    
    # Add MCQ fields if they exist in the dataframe
    if 'question_with_options' in sampled_train_df.columns:
        feature_dict['question_with_options'] = Value('string')
    if 'options' in sampled_train_df.columns:
        feature_dict['options'] = Sequence(Value('string'))
    if 'rating' in sampled_train_df.columns:
        feature_dict['rating'] = Value('int32')
    
    features = Features(feature_dict)
    
    # Keep only columns that exist in features
    sampled_train_df = sampled_train_df[features.keys()]
    sampled_val_df = sampled_val_df[features.keys()]

    # Convert to Hugging Face datasets
    dataset_train = Dataset.from_pandas(sampled_train_df, features=features)
    dataset_val = Dataset.from_pandas(sampled_val_df, features=features)
        
    TARGET_REPO = target_repo

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": dataset_train,
        "val": dataset_val
    })
    
    # Upload to Hugging Face if specified
    if upload_to_hf:
        print(f"Uploading dataset to Hugging Face: {TARGET_REPO}")
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            print("Warning: HF_TOKEN not found in environment. Please set it to upload to Hugging Face.")
        
        login(token=hf_token)
        
        dataset_dict.push_to_hub(
            TARGET_REPO,
            private=False,
            token=hf_token
        )
        print(f"Dataset uploaded to {TARGET_REPO}")
        
        local_dir = snapshot_download(
            repo_id=TARGET_REPO,
            repo_type="dataset",
            ignore_patterns=["*.lock"],
            token=None
        )
        
        print(f'Dataset downloaded to {local_dir}')


def download_dataset_for_review(
    dataset_name: str = "hunarbatra/spatialthinker_vqa_10k",
):
    data = load_dataset(dataset_name)
    keep_cols = ['image_id', 'question_with_options', 'answer_only', 'category']
    
    data = data.remove_columns([col for col in data.column_names if col not in keep_cols])
    
    file_name = dataset_name.split("/")[-1]
    data.to_csv(f"data/review_{file_name}.csv")


# python3 data/generate_data.py easy_hard_split
def generate_easy_hard_splits(
    dataset_name: str = "hunarbatra/spatialthinker_vqa_10k",
    print_stats_only: bool = False
):  
    data = load_dataset(dataset_name, token=os.getenv("HF_TOKEN"))
    full_data = concatenate_datasets([data["train"], data["val"]])
    full_data = full_data.shuffle(seed=42) 
    
    easy_data = full_data.filter(lambda x: x["level"] == "easy")
    hard_data = full_data.filter(lambda x: x["level"] in ["medium", "hard"])
    
    easy_split = easy_data.train_test_split(test_size=0.1, seed=42)
    hard_split = hard_data.train_test_split(test_size=0.1, seed=42)
    
    if print_stats_only:
        print(f"Easy Data - train size: {len(easy_split['train'])}, val size: {len(easy_split['test'])}")
        print(f"Hard Data - train size: {len(hard_split['train'])}, val size: {len(hard_split['test'])}")
        return
    
    easy_dataset = DatasetDict({
        "train": easy_split["train"],
        "val": easy_split["test"]
    })
    
    hard_dataset = DatasetDict({
        "train": hard_split["train"],
        "val": hard_split["test"]
    })
    
    easy_hf_name = dataset_name + "_easy"
    hard_hf_name = dataset_name + "_hard"
    
    easy_dataset.push_to_hub(easy_hf_name, token=os.getenv("HF_TOKEN"))
    hard_dataset.push_to_hub(hard_hf_name, token=os.getenv("HF_TOKEN"))
    
    for dataset_name in [easy_hf_name, hard_hf_name]:
        local_dir = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            ignore_patterns=["*.lock"],
            token=None
        )
        print(f'downloaded {dataset_name} to {local_dir}')


def filter_by_rating(
    input_file: str = "data/spatialthinker_vqa_train.csv",
    output_file: str = "data/spatialthinker_vqa_top10k.csv",
    top_k: int = 10000,
):
    """
    Select top-k samples based on quality rating.
    
    Questions are rated 1-10 during generation based on complexity 
    and contribution to spatial intelligence.
    """
    df = pd.read_csv(input_file)
    
    if 'rating' not in df.columns:
        print("Warning: 'rating' column not found. Returning all samples.")
        df.to_csv(output_file, index=False)
        return
    
    # Sort by rating descending and take top k
    df_sorted = df.sort_values(by='rating', ascending=False)
    df_top = df_sorted.head(top_k)
    
    print(f"Selected top {len(df_top)} samples from {len(df)} total")
    print(f"Rating range: {df_top['rating'].min()} - {df_top['rating'].max()}")
    
    df_top.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


def _call_gpt4o_vision(question: str, options: list, image_pil, client) -> str:
    """Call GPT-4o with image and question, return predicted answer letter."""
    import base64
    
    # Convert PIL image to base64
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG")
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Format options and prompt
    options_str = "\n".join([f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)])
    prompt = GPT4O_VALIDATION_PROMPT.format(question=question, options=options_str)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"}}
                ]
            }],
            max_tokens=10,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().upper()
        # Extract just the letter
        for char in answer:
            if char in 'ABCD':
                return char
        return answer[0] if answer else None
    except Exception as e:
        print(f"GPT-4o API error: {e}")
        return None


# python data_gen/generate_data.py validate_with_gpt4o --input_file="data/spatialthinker_vqa_top10k.csv"
def validate_with_gpt4o(
    input_file: str = "data/spatialthinker_vqa_train.csv",
    output_file: str = "data/spatialthinker_vqa_validated.csv",
    sample_limit: int = None,
    save_every: int = 100,
):
    """
    Validate QA pairs using GPT-4o as external validator.
    
    Implements consistency-based verification:
    1. Pass@2: Get 2 GPT-4o responses, if either matches ground truth -> pass
    2. Extended: If both fail, get 2 more responses (total 4)
    3. Discard: If all 4 disagree with ground truth -> discard
    
    Typically retains ~75% of samples.
    """
    if openai_client is None:
        raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY in .env")
    
    df = pd.read_csv(input_file)
    
    if sample_limit:
        df = df.head(sample_limit)
    
    print(f"Validating {len(df)} samples with GPT-4o...")
    
    # Get unique image IDs needed
    needed_image_ids = set(df["image_id"].astype(str).tolist())
    print(f"Loading {len(needed_image_ids)} images from source dataset...")
    
    # Load source dataset and filter only needed images (lazy loading)
    source_dataset = load_dataset(SOURCE_DS)["train"]
    filtered_ds = source_dataset.filter(lambda x: str(x["image_id"]) in needed_image_ids)
    image_id_to_image = {str(item["image_id"]): item["image"] for item in filtered_ds}
    print(f"Loaded {len(image_id_to_image)} images")
    
    validated_rows = []
    discarded_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating with GPT-4o"):
        image_id = str(row["image_id"])
        question = row["question_only"]
        
        # Get options
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        
        # Get ground truth answer
        answer_match = re.search(r"<answer>(.*?)</answer>", str(row["answer"]))
        if not answer_match:
            print(f"Warning: Could not extract answer for row {idx}")
            discarded_count += 1
            continue
        ground_truth = answer_match.group(1).strip().upper()
        # Handle cases where answer is text, not letter
        if ground_truth not in 'ABCD':
            # It might be the option text, find the letter
            for i, opt in enumerate(options):
                if opt.lower() == ground_truth.lower():
                    ground_truth = chr(ord('A') + i)
                    break
        
        # Get image
        if image_id not in image_id_to_image:
            print(f"Warning: Image {image_id} not found")
            discarded_count += 1
            continue
        image_pil = image_id_to_image[image_id]
        
        # Pass@2 criterion
        responses = []
        passed = False
        
        for attempt in range(2):
            response = _call_gpt4o_vision(question, options, image_pil, openai_client)
            responses.append(response)
            if response and response == ground_truth:
                passed = True
                break
        
        # Extended validation if initial pass@2 failed
        if not passed:
            for attempt in range(2):
                response = _call_gpt4o_vision(question, options, image_pil, openai_client)
                responses.append(response)
                if response and response == ground_truth:
                    passed = True
                    break
        
        if passed:
            validated_rows.append(row)
        else:
            discarded_count += 1
            print(f"Discarded row {idx}: GT={ground_truth}, GPT-4o responses={responses}")
        
        # Save checkpoint
        if len(validated_rows) > 0 and len(validated_rows) % save_every == 0:
            checkpoint_df = pd.DataFrame(validated_rows)
            checkpoint_df.to_csv(output_file.replace('.csv', '_checkpoint.csv'), index=False)
            print(f"Checkpoint saved: {len(validated_rows)} validated, {discarded_count} discarded")
    
    # Save final results
    validated_df = pd.DataFrame(validated_rows)
    validated_df.to_csv(output_file, index=False)
    
    retention_rate = len(validated_rows) / len(df) * 100
    print(f"\n✅ Validation complete!")
    print(f"   Total: {len(df)}")
    print(f"   Validated: {len(validated_rows)} ({retention_rate:.1f}%)")
    print(f"   Discarded: {discarded_count} ({100-retention_rate:.1f}%)")
    print(f"   Saved to: {output_file}")


if __name__ == "__main__":
    fire.Fire({
        "preprocess_data": preprocess_data,
        "print_data_stats": print_data_stats,
        "generate_hf_data": generate_hf_data,
        "fix_count_questions": fix_count_questions,
        "filter_by_rating": filter_by_rating,
        "validate_with_gpt4o": validate_with_gpt4o,
        "data_review": download_dataset_for_review,
        "easy_hard_split": generate_easy_hard_splits
    })

# 235 million inputs tokens
# 5 million output tokens
