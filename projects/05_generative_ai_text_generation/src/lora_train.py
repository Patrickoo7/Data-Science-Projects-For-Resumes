"""
Parameter-efficient fine-tuning with LoRA/QLoRA.
Memory-efficient training for large language models.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import argparse


class LoRATrainer:
    """LoRA fine-tuning trainer."""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True
            )
        elif load_in_8bit:
            model_kwargs['load_in_8bit'] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Prepare for k-bit training
        if load_in_8bit or load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Default target modules for common architectures
        if target_modules is None:
            if 'llama' in model_name.lower():
                target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
            elif 'gpt' in model_name.lower():
                target_modules = ['c_attn', 'c_proj']
            else:
                target_modules = ['q_proj', 'v_proj']

        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, dataset_path: str, max_length: int = 512):
        """Prepare dataset for training."""

        # Load dataset
        if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=dataset_path)
        else:
            dataset = load_dataset(dataset_path)

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        return tokenized_dataset

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = './lora_model',
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100
    ):
        """Train model with LoRA."""

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy='steps' if eval_dataset else 'no',
            save_total_limit=3,
            fp16=True,
            optim='paged_adamw_8bit',
            lr_scheduler_type='cosine',
            report_to='tensorboard',
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model='loss' if eval_dataset else None,
            greater_is_better=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset['train'],
            eval_dataset=eval_dataset['train'] if eval_dataset else None,
            data_collator=data_collator
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")

        return trainer


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning')
    parser.add_argument('--model', type=str, default='gpt2', help='Base model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--output_dir', type=str, default='./lora_model', help='Output directory')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--load_in_4bit', action='store_true', help='4-bit quantization')
    parser.add_argument('--load_in_8bit', action='store_true', help='8-bit quantization')
    args = parser.parse_args()

    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)

    # Initialize trainer
    lora_trainer = LoRATrainer(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )

    # Prepare dataset
    print("Preparing dataset...")
    dataset = lora_trainer.prepare_dataset(args.dataset, args.max_length)

    # Train
    lora_trainer.train(
        train_dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
