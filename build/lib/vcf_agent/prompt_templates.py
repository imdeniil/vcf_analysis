"""
prompt_templates.py
Module for constructing LLM prompts for VCF analysis using YAML prompt contracts.
"""

import os
import yaml
from typing import Dict, Any, Optional

PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../prompts'))


def load_prompt_contract(contract_name: str) -> Dict[str, Any]:
    """
    Loads a YAML prompt contract by name (without .yaml extension) from the prompts directory.
    Returns the contract as a dictionary.
    """
    path = os.path.join(PROMPT_DIR, f"{contract_name}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt contract not found: {path}")
    with open(path, 'r') as f:
        contract = yaml.safe_load(f)
    if contract is None:
        raise ValueError(f"Prompt contract YAML is empty: {path}")
    return contract


def render_prompt(contract: Dict[str, Any], file_paths: list, extra_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Renders a prompt for the LLM using the contract's role, context, and instructions.
    file_paths: list of VCF file paths (1 or 2, depending on task)
    extra_context: optional dict to fill in additional context fields
    Returns the prompt string to send to the LLM.
    """
    role = contract.get('role', '')
    context = contract.get('context', '')
    instructions = contract.get('instructions', '')
    # Compose file info
    file_info = ''
    if len(file_paths) == 1:
        file_info = f"VCF file: {file_paths[0]}"
    elif len(file_paths) == 2:
        file_info = f"VCF file 1: {file_paths[0]}\nVCF file 2: {file_paths[1]}"
    else:
        file_info = f"Files: {', '.join(file_paths)}"
    # Add extra context if provided
    extra = ''
    if extra_context:
        for k, v in extra_context.items():
            extra += f"{k}: {v}\n"
    prompt = f"Role: {role}\nContext: {context}\n{file_info}\n{extra}Instructions:\n{instructions}"
    return prompt


def get_prompt_for_task(task: str, file_paths: list, extra_context: Optional[Dict[str, Any]] = None) -> str:
    """
    High-level function to get a rendered prompt for a given task (e.g., 'vcf_summarization_v1').
    """
    contract = load_prompt_contract(task)
    return render_prompt(contract, file_paths, extra_context) 