"""
AML HW2: GRPO Training for Mathematical Reasoning
=================================================

Simplified GRPO (Group Relative Policy Optimization) for training language models
on mathematical reasoning tasks.

Filled TODOs + GPU-efficiency optimizations:
- device_map="auto" (correct placement)
- bf16/fp16 autocast
- gradient checkpointing enabled
- KV cache disabled during training
- shorter prompts (max_length default 256)
- delete large tensors each step
- TF32 enabled (Ampere+)
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

    def __init__(self, split="train", tokenizer=None, max_length=256):
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
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        # ========================================================================
        # TODO 1: Tokenize the prompt
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids, attention_mask = tok["input_ids"], tok["attention_mask"]
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
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            return float(match.group(1).replace(',', ''))
        return None


# ============================================================================
# Part 2: Reward Function
# ============================================================================

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
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
    """Binary reward: 1 if correct, 0 else."""
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
    GRPO advantage:
      advantage = reward - mean(group_rewards)  (per prompt group)
    """
    # ========================================================================
    # TODO 2: Implement GRPO advantage computation
    r = rewards.view(-1, group_size)
    advantages = (r - r.mean(dim=1, keepdim=True)).view(-1)
    # END TODO 2
    # ========================================================================
    return advantages


def compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=0.2):
    """
    PPO-style clipped policy loss.
    logprobs/old_logprobs: (B, T)
    advantages: (B,)
    loss_mask: (B, T)
    """
    # ========================================================================
    # TODO 3: Implement PPO-style policy loss
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages[:, None].expand_as(logprobs).detach()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss_tok = -torch.min(surr1, surr2) * loss_mask
    loss = loss_tok.sum() / (loss_mask.sum() + 1e-8)
    # END TODO 3
    # ========================================================================
    return loss


# ============================================================================
# Part 4: Training Loop
# ============================================================================

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=128,
    temperature=1.0,
    num_samples=4,
):
    """Generate multiple completions for each prompt."""
    model.eval()

    batch_size = input_ids.shape[0]
    input_ids_repeated = input_ids.repeat_interleave(num_samples, dim=0)
    attention_mask_repeated = attention_mask.repeat_interleave(num_samples, dim=0)

    outputs = model.generate(
        input_ids=input_ids_repeated,
        attention_mask=attention_mask_repeated,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # generation benefits; training uses_cache=False separately
    )

    completions = []
    prompt_len = input_ids.shape[1]
    for output in outputs:
        generated_ids = output[prompt_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return {
        "output_ids": outputs,
        "completions": completions,
        "prompt_length": prompt_len,
    }


def compute_logprobs_from_model(model, input_ids, attention_mask):
    """
    Compute token logprobs for the sequence tokens (aligned with input_ids length).
    Returns logprobs of shape (B, T), with first position = 0 (no previous token).
    """
    # ========================================================================
    # TODO 4: Compute log probabilities
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logp = F.log_softmax(out.logits, dim=-1)
    tgt = input_ids[:, 1:].unsqueeze(-1)
    lp = logp[:, :-1].gather(-1, tgt).squeeze(-1)
    logprobs = torch.cat([lp.new_zeros(lp.size(0), 1), lp], dim=1)
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
    max_new_tokens=128,
    grad_accum_steps=1,
):
    """Main GRPO training loop."""
    rewards_list = []

    # Mixed precision settings
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")

        model.train()
        model.config.use_cache = False  # critical for training memory
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]

            # ====================================================================
            # TODO 5: Implement the GRPO training step
            gen = generate_completions(
                model, tokenizer, input_ids, attention_mask,
                max_new_tokens=max_new_tokens, num_samples=group_size
            )

            # rewards/advantages on CPU then move small tensors to GPU
            gt_rep = np.repeat(np.array(answers, dtype=np.float32), group_size)
            rewards = compute_reward(gen["completions"], gt_rep).to(device)
            advantages = compute_advantages_grpo(rewards, group_size).to(device)

            seq = gen["output_ids"].to(device)
            am = (seq != tokenizer.pad_token_id).long()

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                    old_lp = compute_logprobs_from_model(model, seq, am).detach()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                lp = compute_logprobs_from_model(model, seq, am)
                mask = am.clone()
                mask[:, :gen["prompt_length"]] = 0
                loss = compute_policy_loss(lp, old_lp, advantages, mask, clip_eps=clip_eps)

            loss = loss / max(1, grad_accum_steps)
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # END TODO 5
            # ====================================================================

            total_loss += loss.item() * max(1, grad_accum_steps)
            total_reward += rewards.mean().item()
            num_batches += 1
            rewards_list.append(total_reward / num_batches)

            progress_bar.set_postfix({
                "loss": f"{(loss.item()*max(1,grad_accum_steps)):.4f}",
                "reward": f"{rewards.mean().item():.4f}",
                "avg_reward": f"{total_reward/num_batches:.4f}",
            })

            # free big tensors ASAP
            del gen, seq, am, lp, old_lp, rewards, advantages, mask
            if use_cuda:
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_batches)
        avg_reward = total_reward / max(1, num_batches)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")

    return rewards_list


# ============================================================================
# Part 5: Main Function
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python grpo_homework.py <MODEL_PATH>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU efficiency toggles
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batch_size = 1
    group_size = 4          # must be >1 for GRPO advantages to be non-zero
    num_epochs = 1
    learning_rate = 5e-6
    max_new_tokens = 16
    max_prompt_length = 256
    grad_accum_steps = 1

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Training memory settings
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Loading dataset...")
    train_dataset = GSM8KDataset(split="train[:100]", tokenizer=tokenizer, max_length=max_prompt_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
        grad_accum_steps=grad_accum_steps,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    ckpt_dir = "./saved_model"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"The model has been saved to {ckpt_dir}")

    # Plot training curves
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    def smooth_curve(data, window_size=10):
        if len(data) < window_size:
            return data
        return uniform_filter1d(data, size=window_size, mode="nearest")

    plt.figure(figsize=(12, 5))
    smoothed_rewards = smooth_curve(rewards_list)
    plt.plot(rewards_list, alpha=0.3, label="Raw")
    plt.plot(smoothed_rewards, label="Smoothed")
    plt.xlabel("Training Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grpo_training_curves.png")
    print("Training curves saved to grpo_training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
