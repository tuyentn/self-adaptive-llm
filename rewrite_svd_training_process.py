#%% Imports
import gc
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm

#%% Config with Hydra
@hydra.main(version_base=None, config_path="cfgs", config_name="zalo_vmlu_config")
def main(cfg):
    """Main function for SVD reinforcement learning on Zalo VMLU dataset."""
    
    # Extract config variables
    num_iters = cfg.num_iters
    test_interval = cfg.test_interval
    batch_size = cfg.batch_size
    seed = cfg.seed
    policy_name = cfg.policy_name
    model_id = cfg.base_model_name
    decomposed_param_file = f"{cfg.base_model.param_folder_path}/{model_id.replace('/', '_')}_decomposed.pt"
    
    # Setup experiment name and logging directories
    if cfg.exp_name is None:
        exp_name = "temp"
    else:
        exp_name = cfg.exp_name
        
    if cfg.run_name is None:
        now = datetime.now()
        run_name = now.strftime("%Y%m%d-%H%M%S")
    else:
        run_name = cfg.run_name
        
    log_dir = f"{cfg.out_dir}/{cfg.task_name}/{policy_name}/{exp_name}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Setup wandb if enabled
    if cfg.wandb_log:
        import wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group_name[:127],
            name=run_name[:127],
            config=config_dict,
        )

    # Dataset loading and preparation
    train_data, train_ix, valid_ix, test_data, transfer_data = prepare_datasets(seed)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.bfloat16
    )
    base_params = model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load decomposed parameters
    if not os.path.exists(decomposed_param_file):
        print(f"ERROR: Decomposed params not found at {decomposed_param_file}")
        return
    else:
        print("Loading decomposed parameters...")
        decomposed_params = torch.load(decomposed_param_file)
    
    # Initialize GPU and VLLM model
    gpu = torch.device("cuda")
    vllm_model = get_vllm_model(model_id)
    
    # Move decomposed params to GPU
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(gpu)
    
    # Initialize policy
    policy = ShakeoffPolicy(
        base_params=base_params,
        decomposed_params=decomposed_params,
        n_bins=cfg.shakeoff_policy.n_bins,
        emb_dim=cfg.shakeoff_policy.emb_dim,
        base_init=cfg.shakeoff_policy.base_init,
        gpu=gpu,
    )
    
    # Initialize optimizer
    optimizer = Adam(
        policy.trainable_params, 
        lr=cfg.optimization_algorithm.learning_rate,
        weight_decay=cfg.optimization_algorithm.weight_decay
    )
    
    # Initialize RNG for batch sampling
    np_random = np.random.RandomState(seed)
    
    # Setup metrics tracking
    metrics_to_log = {}
    best_val_acc = 0.0
    test_at_best = 0.0
    transfer_at_best = 0.0
    
    # Training loop
    print('Starting training...')
    start_time = datetime.now()
    print("Start time:", start_time)
    
    for i in tqdm(range(num_iters), desc="Training loop"):
        # Sample batch
        batch_size_clipped = min(batch_size, len(train_ix))
        batch_ix = np_random.choice(train_ix, size=batch_size_clipped, replace=False)
        
        # Get learnable parameters
        learnable_params = policy.get_learnable_params()
        
        # Forward pass: modify model weights with policy
        new_params = forward(policy, model, base_params, decomposed_params, learnable_params)
        
        # Load weights to VLLM model
        load_hf_params_to_vllm(new_params, vllm_model)
        
        # Evaluate model on batch
        rewards = []
        for ix in batch_ix:
            result = evaluate_sample(vllm_model, train_data[ix])
            reward = 1.0 if result["correct"] else -1.0
            rewards.append(reward)
        
        # Compute batch statistics
        batch_reward = np.mean(rewards)
        metrics_to_log["batch_reward"] = batch_reward
        
        # Update policy with REINFORCE
        optimizer.zero_grad()
        policy.update_with_rewards(rewards=torch.tensor(rewards, device=gpu))
        optimizer.step()
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log progress
        if i % 10 == 0:
            print(f"Iter {i}: reward={batch_reward:.4f}")
        
        # Evaluation and checkpointing
        if i % test_interval == 0:
            # Get new parameters after update
            learnable_params = policy.get_learnable_params()
            forward(policy, model, base_params, decomposed_params, learnable_params)
            load_hf_params_to_vllm(model.state_dict(), vllm_model.llm)
            
            # Evaluate
            train_acc = evaluate_dataset(vllm_model, [train_data[ix] for ix in train_ix[:50]])
            valid_acc = evaluate_dataset(vllm_model, [train_data[ix] for ix in valid_ix[:50]])
            test_acc = evaluate_dataset(vllm_model, test_data[:50])
            transfer_acc = evaluate_dataset(vllm_model, transfer_data[:50])
            
            # Check for best model
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                test_at_best = test_acc
                transfer_at_best = transfer_acc
                print("New best validation accuracy!")
                # Save checkpoint
                torch.save(policy.state_dict(), f"{log_dir}/policy_params.pt")
                torch.save(learnable_params, f"{log_dir}/learnable_params.pt")
            
            # Always save latest
            torch.save(policy.state_dict(), f"{log_dir}/policy_params_latest.pt")
            
            # Log metrics
            metrics = {
                "iter": i,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "test_acc": test_acc,
                "transfer_acc": transfer_acc,
                "best_val_acc": best_val_acc,
                "test_at_best_val": test_at_best,
                "transfer_at_best_val": transfer_at_best,
                **metrics_to_log
            }
            
            if cfg.wandb_log:
                wandb.log(metrics)
                
            # Save to log file
            with open(f"{log_dir}/reinforce_log.json", "a") as f:
                json_data = json.dumps(metrics, indent=4)
                f.write(json_data)
                f.write("\n")
            
            metrics_to_log = {}
    
    # End of training
    end_time = datetime.now()
    print("End time:", end_time)
    print("Total time:", end_time - start_time)

#%% Dataset Functions
def prepare_datasets(dataset_id:str , seed: int = 123):
    """Load and prepare the Zalo VMLU dataset."""
    # Load train and test splits
    train_dataset = load_dataset(dataset_id, split="validation")
    test_dataset = load_dataset(dataset_id, split="test")
    
    # Convert to our format
    train_data = [format_sample(sample) for sample in train_dataset]
    test_data_full = [format_sample(sample) for sample in test_dataset]
    
    # Create train/valid split
    train_size = len(train_data)
    # Set seed for reproducibility
    np.random.seed(seed)
    indices = np.random.permutation(train_size)
    train_split = int(0.8 * train_size)
    train_ix = indices[:train_split].tolist()
    valid_ix = indices[train_split:].tolist()
    
    # Split test into test and transfer based on subjects
    subjects = list(set([sample["metadata"]["subject"] for sample in test_data_full if "subject" in sample["metadata"]]))
    subjects.sort()  # Deterministic ordering
    
    split_idx = int(len(subjects) * 0.8)
    test_subjects = subjects[:split_idx]
    transfer_subjects = subjects[split_idx:]
    
    test_data = [s for s in test_data_full if s["metadata"].get("subject", "") in test_subjects]
    transfer_data = [s for s in test_data_full if s["metadata"].get("subject", "") in transfer_subjects]
    
    print(f"Train size: {len(train_ix)}, Valid size: {len(valid_ix)}")
    print(f"Test size: {len(test_data)}, Transfer size: {len(transfer_data)}")
    
    return train_data, train_ix, valid_ix, test_data, transfer_data

def format_sample(sample):
    """Format a dataset sample into our standard format."""
    question = sample['question']
    answer = sample['answer']
    options = sample['choices']
    question_id = str(sample.get("id", 0))
    metadata = {}
    if "subject" in sample:
        metadata["subject"] = sample["subject"]
    if "level" in sample:
        metadata["level"] = sample["level"]
    
    return {
        "question": question,
        "answer": answer,
        "options": options,
        "question_id": question_id,
        "metadata": metadata
    }

#%%
from string import Template
preamble = 'Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: '

# For Llama models, modify the template to use their chat format
chat_template = Template('''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the user's question with a single letter: A, B, C, D, or E, corresponding to the correct answer to the multiple-choice question.

<|start_header_id|>user<|end_header_id|>

$preamble

$question

$a
$b
$c
$d
$e

Đáp án:
''')

#%% Evaluation Functions
def evaluate_sample(model, sample):
    """Evaluate a single sample."""
    question = sample["question"]
    choices = sample["options"]
    try:
        a = choices[0]
    except:
        a = ''
    try:
        b = choices[1]
    except:
        b = ''
    try:
        c = choices[2]
    except:
        c = ''
    try:
        d = choices[3]
    except:
        d = ''
    try:
        e = choices[4]
    except:
        e = ''

    prompt = chat_template.substitute(
        preamble=preamble, question=question, a=a, b=b, c=c, d=d, e=e)
    result = model.generate([prompt])[0]
    response = result.generation.strip()
    last_part = response.split("Đáp án:")[-1].strip()
    
    # Try to extract the single letter answer
    import re
    letter_match = re.search(r'[A-E]', last_part)
    
    if letter_match:
        prediction = letter_match.group(0)
    else:
        # Fallback if no clear letter is found
        prediction = last_part[:1]  # Take first character as best guess
    
    is_correct = False
    if prediction and prediction.upper() == sample["answer"].upper():
        is_correct = True
    
    return {
        "problem": sample["question"],
        "output": result.generation,
        "prediction": prediction,
        "answer": sample["answer"],
        "correct": is_correct
    }

def evaluate_dataset(model, samples):
    """Evaluate a set of samples and return accuracy."""
    results = []
    for sample in samples:
        result = evaluate_sample(model, sample)
        results.append(result)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    
    return correct / total if total > 0 else 0.0

#%% Model and Policy Functions
def get_vllm_model(model_id):
    """Create and return a VLLM model."""
    llm = vllm.LLM(
        model_id,
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        dtype="bfloat16",
    )
    
    # Disable grad for VLLM model
    m = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for _, param in m.named_parameters():
        param.requires_grad = False
    
    return llm

class ShakeoffPolicy:
    """Policy class for SVD-based weight adaptation."""
    
    def __init__(self, base_params, decomposed_params, n_bins=40, emb_dim=32, base_init=1.0, gpu=None):
        self.base_params = base_params
        self.decomposed_params = decomposed_params
        self.n_bins = n_bins
        self.emb_dim = emb_dim
        self.base_init = base_init
        self.gpu = gpu
        
        # Initialize learnable parameters
        self.embeddings = {}
        for k in base_params:
            if "mlp" in k:
                # Get SVD shape
                s_shape = decomposed_params[f"{k}.S"].shape[0]
                self.embeddings[k] = torch.nn.Parameter(
                    torch.ones(s_shape, device=gpu) * base_init
                )
        
        # Collect trainable parameters
        self.trainable_params = list(self.embeddings.values())
    
    def get_learnable_params(self):
        """Return the current learnable parameters."""
        return {k: v.detach() for k, v in self.embeddings.items()}
    
    def get_mask(self, param):
        """Generate mask from parameter."""
        return param
    
    def state_dict(self):
        """Return the state dictionary for saving."""
        return {k: v.clone() for k, v in self.embeddings.items()}
    
    def load_state_dict(self, state_dict):
        """Load from a state dictionary."""
        for k, v in state_dict.items():
            if k in self.embeddings:
                self.embeddings[k].data.copy_(v)
    
    def update_with_rewards(self, rewards):
        """Update policy using REINFORCE with rewards."""
        # Scale rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Apply REINFORCE update
        for k, param in self.embeddings.items():
            if param.grad is not None:
                param.grad.zero_()
            
            # Compute gradient using reward signal
            grad = -rewards.mean() * (param - self.base_init)
            param.backward(grad, retain_graph=True)

def compose_new_params(policy, param_name, decomposed_params, learnable_params):
    """Compose new parameters from decomposed parameters."""
    mm = policy.get_mask(learnable_params[param_name])
    return (
        decomposed_params[f"{param_name}.U"]
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)
        @ decomposed_params[f"{param_name}.V"].T
    ) * (
        decomposed_params[f"{param_name}.S"].sum()
        / (decomposed_params[f"{param_name}.S"] * mm).sum()
    )

def forward(policy, model, base_params, decomposed_params, learnable_params):
    """Forward pass to modify model weights with policy."""
    new_params = {}
    with torch.no_grad():
        for k in base_params:
            if "mlp" in k:
                new_params[k] = compose_new_params(
                    policy, k, decomposed_params, learnable_params
                )
                model.get_parameter(k).copy_(new_params[k])
            else:
                new_params[k] = base_params[k]
    return new_params

def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """Load weights from HF transformer model to vLLM model."""
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights
    model_param = model.get_parameter("model.embed_tokens.weight")
    model_param.copy_(
        param["model.embed_tokens.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )
    model_param = model.get_parameter("lm_head.weight")
    model_param.copy_(
        param["lm_head.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )

    # Load the final layernorm weights
    model_param = model.get_parameter("model.norm.weight")
    model_param.copy_(
        param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.qkv_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.weight"],
                    param[f"model.layers.{i}.self_attn.k_proj.weight"],
                    param[f"model.layers.{i}.self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load gate_up_proj weights
        model_param = model.get_parameter(f"model.layers.{i}.mlp.gate_up_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.mlp.gate_proj.weight"],
                    param[f"model.layers.{i}.mlp.up_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load o_proj and down_proj weights
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.o_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.self_attn.o_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(f"model.layers.{i}.mlp.down_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.mlp.down_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load layer_norm weights
        model_param = model.get_parameter(f"model.layers.{i}.input_layernorm.weight")
        model_param.copy_(
            param[f"model.layers.{i}.input_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.post_attention_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )

#%% Entry Point
if __name__ == "__main__":
    main()