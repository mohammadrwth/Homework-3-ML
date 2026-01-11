"""
AML HW2: GRPO Training for Mathematical Reasoning
=================================================

This is a simplified GRPO (Group Relative Policy Optimization) implementation
for training language models on mathematical reasoning tasks.

Instructions:
-------------
Fill in the 5 blanks marked with TODO. Each blank should be no more than 10 lines.
Make sure you understand the GRPO algorithm before filling in the blanks.
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# ============================================================================
# Part 1: Dataset Processing
# ============================================================================

class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""

    def __init__(self, split="train", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("gsm8k", "main", split=split)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Extract the final numerical answer
        # Answer format: "#### 42"
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\nFinal answer: #### "


        # ========================================================================
        # TODO 1: Tokenize the prompt
        # ========================================================================
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # END TODO 1
        # ========================================================================

        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "prompt": prompt,
            "answer": answer_number,
        }

    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        # GSM8K answers end with #### followed by the answer
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        return None


# ============================================================================
# Part 2: Reward Function
# ============================================================================

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
    # Look for patterns like "The answer is 42" or "#### 42"
    patterns = [
        r'answer is\s*([-+]?\d+[\d,]*\.?\d*)',
        r'####\s*([-+]?\d+[\d,]*\.?\d*)',
        r'=\s*([-+]?\d+[\d,]*\.?\d*)(?:\s|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    return None


def compute_reward(completions, ground_truth_answers):
    """
    Compute binary rewards: 1 if correct, 0 if incorrect.

    Args:
        completions: List of completion strings
        ground_truth_answers: List of ground truth answers

    Returns:
        rewards: Tensor of shape (batch_size,) with values 0 or 1
    """
    rewards = []
    for completion, gt_answer in zip(completions, ground_truth_answers):
        predicted = extract_answer_from_completion(completion)
        if predicted is not None and abs(predicted - gt_answer) < 1e-3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# Part 3: GRPO Algorithm Implementation
# ============================================================================

def compute_advantages_grpo(rewards, group_size=4):
    """
    Compute advantages using GRPO (Group Relative Policy Optimization).

    In GRPO, for each group of responses to the same prompt:
    - Advantage = reward - mean(group_rewards)

    This encourages the model to generate responses better than average.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_size: Number of responses per prompt

    Returns:
        advantages: Tensor of shape (batch_size,) containing advantages
    """
    # ========================================================================
    # TODO 2: Implement GRPO advantage computation
    # ========================================================================
    assert rewards.numel() % group_size == 0
    group_rewards = rewards.view(-1, group_size)
    advantages = (group_rewards - group_rewards.mean(dim=1, keepdim=True)).view(-1)
    # END TODO 2
    # ========================================================================

    return advantages


def compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=0.2):
    """
    Compute PPO-style clipped policy loss.

    Args:
        logprobs: Current policy log probabilities, shape (batch_size, seq_len)
        old_logprobs: Old policy log probabilities, shape (batch_size, seq_len)
        advantages: Advantages, shape (batch_size,)
        loss_mask: Mask for valid tokens, shape (batch_size, seq_len)
        clip_eps: Clipping epsilon for PPO

    Returns:
        loss: Scalar loss value
    """
    # ========================================================================
    # TODO 3: Implement PPO-style policy loss
    # ========================================================================
    diff = (logprobs - old_logprobs).clamp(-20, 20)
    ratios = torch.exp(diff)
    adv = advantages.unsqueeze(1)
    unclipped = ratios * adv
    clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * adv
    loss = -(torch.min(unclipped, clipped) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # END TODO 3
    # ========================================================================

    return loss


# ============================================================================
# Part 4: Training Loop
# ============================================================================

def generate_completions(model, tokenizer, input_ids, attention_mask,
                        max_new_tokens=256, temperature=1.0, num_samples=4):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    completions = []
    prompt_len = input_ids.shape[1]
    for output in outputs:
        completions.append(tokenizer.decode(output[prompt_len:], skip_special_tokens=True))
    return {"output_ids": outputs, "completions": completions, "prompt_length": prompt_len}

def compute_logprobs_from_model(model, input_ids, attention_mask):
    """
    Compute log probabilities for given sequences.

    Args:
        model: The language model
        input_ids: Input token ids, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)

    Returns:
        logprobs: Log probabilities, shape (batch_size, seq_len)
    """
    # ========================================================================
    # TODO 4: Compute log probabilities
    # ========================================================================
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()
    logp = F.log_softmax(logits, dim=-1)
    tok = logp[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    logprobs = F.pad(tok, (1, 0), value=0.0)

    # END TODO 4
    # ========================================================================

    return logprobs


def train_grpo(
    model,
    tokenizer,
    train_loader,
    optimizer,
    device,
    num_epochs=1,
    group_size=4,
    clip_eps=0.2,
    max_new_tokens=256,
):
    """
    Main GRPO training loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        group_size: Number of samples per prompt for GRPO
        clip_eps: PPO clipping epsilon
        max_new_tokens: Maximum number of tokens to generate
    """

    rewards_list = []
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]

            # ====================================================================
            # TODO 5: Implement the GRPO training step
            # ====================================================================
            gen = generate_completions(model, tokenizer, input_ids, attention_mask, max_new_tokens=max_new_tokens, num_samples=group_size)
            output_ids = gen["output_ids"]; full_attn = (output_ids != tokenizer.pad_token_id).long()
            gt = [a for a in answers for _ in range(group_size)]
            rewards = compute_reward(gen["completions"], gt).to(device); advantages = compute_advantages_grpo(rewards, group_size).to(device)
            loss_mask = full_attn.clone(); loss_mask[:, :gen["prompt_length"]] = 0
            with torch.no_grad(): old_logprobs = compute_logprobs_from_model(model, output_ids, full_attn)
            model.train(); logprobs = compute_logprobs_from_model(model, output_ids, full_attn)
            loss = compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps)
            if not torch.isfinite(loss): optimizer.zero_grad(set_to_none=True); continue
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0); optimizer.step()

            # END TODO 5
            # ====================================================================

            # Logging
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            num_batches += 1
            rewards_list.append(total_reward/num_batches)
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "reward": f"{rewards.mean().item():.4f}",
                "avg_reward": f"{total_reward/num_batches:.4f}",
            })
            
        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
    
    return rewards_list


# ============================================================================
# Part 5: Main Function
# ============================================================================

def main():
    # Configuration
    if len(sys.argv) < 2:
        print("Usage: python grpo_homework.py <MODEL_PATH>")
        sys.exit(1)
    model_path = sys.argv[1]
    print(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    batch_size = 2
    group_size = 16
    num_epochs = 1
    max_new_tokens = 32
    learning_rate = 5e-6

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models need left padding for correct generation
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device.type in ("mps", "cuda") else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Loading dataset...")
    train_dataset = GSM8KDataset(split="train[:200]", tokenizer=tokenizer, max_length=512)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in x],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in x],
                batch_first=True,
                padding_value=0
            ),
            "prompt": [item["prompt"] for item in x],
            "answer": [item["answer"] for item in x],
        }
    )

    print("Setting up optimizer...")
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.lm_head.parameters():
        p.requires_grad_(True)

    N = 4  # set 4 on 16GB, 8 on 24GB+
    for layer in model.model.layers[-N:]:
        for p in layer.parameters():
            p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)


    print("Starting GRPO training...")
    rewards_list = train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    ckpt_dir = "./saved_model"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"The model has been saved to {ckpt_dir}")

    # Plot training curves
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    
    # Smooth the curves using a moving average
    def smooth_curve(data, window_size=10):
        if len(data) < window_size:
            return data
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    plt.figure(figsize=(12, 5))
    # Plot rewards
    smoothed_rewards = smooth_curve(rewards_list)
    plt.plot(rewards_list, alpha=0.3, label='Raw')
    plt.plot(smoothed_rewards, label='Smoothed')
    plt.xlabel('Training Step')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grpo_training_curves.png')
    print(f"Training curves saved to grpo_training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
