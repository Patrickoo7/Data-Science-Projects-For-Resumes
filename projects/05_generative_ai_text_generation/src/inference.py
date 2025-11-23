"""
Text generation inference with multiple sampling strategies.
Supports greedy, beam search, top-k, nucleus sampling.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import argparse
from typing import List, Optional


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation."""

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class TextGenerator:
    """Advanced text generation with multiple strategies."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            'device_map': 'auto' if device == 'cuda' else None,
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        }

        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif load_in_4bit:
            model_kwargs['load_in_4bit'] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )

        if not (load_in_8bit or load_in_4bit) and device == 'cuda':
            self.model = self.model.to(device)

        self.model.eval()

        print(f"Model loaded: {model_name_or_path}")
        print(f"Device: {device}")

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        do_sample: bool = True,
        early_stopping: bool = True,
        stop_sequences: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate text with customizable parameters.

        Args:
            prompt: Input text
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_beams: Beam search width
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for length
            do_sample: Whether to use sampling
            early_stopping: Stop when all beams finish
            stop_sequences: List of sequences to stop generation

        Returns:
            List of generated texts
        """

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)

        # Stopping criteria
        stopping_criteria = None
        if stop_sequences:
            stop_token_ids = []
            for seq in stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(stop_token_ids)
            ])

        # Generation config
        gen_config = GenerationConfig(
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            do_sample=do_sample,
            early_stopping=early_stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria
            )

        # Decode
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prompt from output
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generated_texts.append(text)

        return generated_texts

    def greedy_generate(self, prompt: str, max_length: int = 100) -> str:
        """Greedy decoding (deterministic)."""
        return self.generate(
            prompt,
            max_length=max_length,
            do_sample=False,
            num_beams=1
        )[0]

    def beam_search_generate(
        self,
        prompt: str,
        max_length: int = 100,
        num_beams: int = 5
    ) -> str:
        """Beam search decoding."""
        return self.generate(
            prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )[0]

    def top_k_generate(
        self,
        prompt: str,
        max_length: int = 100,
        top_k: int = 50,
        temperature: float = 0.7
    ) -> str:
        """Top-k sampling."""
        return self.generate(
            prompt,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            temperature=temperature,
            num_beams=1
        )[0]

    def nucleus_generate(
        self,
        prompt: str,
        max_length: int = 100,
        top_p: float = 0.9,
        temperature: float = 0.8
    ) -> str:
        """Nucleus (top-p) sampling."""
        return self.generate(
            prompt,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            top_k=0,
            temperature=temperature,
            num_beams=1
        )[0]

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)[0]
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='Text generation with GPT')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name or path')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Max length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--num_beams', type=int, default=1, help='Beam search')
    parser.add_argument('--strategy', type=str, default='nucleus',
                       choices=['greedy', 'beam', 'top_k', 'nucleus'],
                       help='Generation strategy')
    parser.add_argument('--load_in_8bit', action='store_true', help='8-bit quantization')
    parser.add_argument('--load_in_4bit', action='store_true', help='4-bit quantization')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize generator
    generator = TextGenerator(
        args.model,
        device=device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Strategy: {args.strategy}")
    print("-" * 50)

    # Generate based on strategy
    if args.strategy == 'greedy':
        output = generator.greedy_generate(args.prompt, args.max_length)
    elif args.strategy == 'beam':
        output = generator.beam_search_generate(
            args.prompt, args.max_length, args.num_beams
        )
    elif args.strategy == 'top_k':
        output = generator.top_k_generate(
            args.prompt, args.max_length, args.top_k, args.temperature
        )
    else:  # nucleus
        output = generator.nucleus_generate(
            args.prompt, args.max_length, args.top_p, args.temperature
        )

    print(f"\nGenerated:\n{output}")
    print("-" * 50)


if __name__ == "__main__":
    main()
