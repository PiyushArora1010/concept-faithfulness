import os
import re
import numpy as np
import random
from numpy.random import RandomState
from numpy.random import seed as seednp

from module.models import Model
from module.datasets.bbq import BBQDataset
from module.datasets.medqa import MedQADataset

import torch
import torch.distributed as dist

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def print0(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized():
        print(*args, **kwargs)
    elif dist.get_rank() == 0:
        print(*args, **kwargs)


def get_language_model(model_tag, max_tokens=256, temperature=0.7, batch_size=4, load_in_4bit=False, thinking=False):
    if model_tag == "Llama3.2_1B":
        return Model(name="meta-llama/Llama-3.2-1B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Llama3.1_8B":
        return Model(name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Llama3.3_70B":
        return Model(name="meta-llama/Llama-3.3-70B-Instruct", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left")
    elif model_tag == "Qwen3_4B":
        return Model(name="Qwen/Qwen3-4B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_8B":
        return Model(name="Qwen/Qwen3-8B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_14B":
        return Model(name="Qwen/Qwen3-14B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    elif model_tag == "Qwen3_32B":
        return Model(name="Qwen/Qwen3-32B", max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    else:
        return Model(name=model_tag, max_tokens=max_tokens, temperature=temperature, batch_size=batch_size, load_in_4bit=load_in_4bit, padding_side="left", thinking=thinking)
    
def get_dataset(dataset_name, dataset_path):
    if dataset_name == "bbq":
        return BBQDataset(dataset_name, dataset_path)
    elif dataset_name == "medqa":
        return MedQADataset(dataset_name, dataset_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def parse_llm_response_concepts_and_categories(response):
    """
    Parses the response from the LLM for concept identification.
    Args:
        response: response from the LLM
    Returns:
        concepts: a list of concepts identified by the LLM
        categories: a list of the category associated with each concept
    """
    response_lines = response.strip().split("\n")
    concepts = []
    categories = []

    initial_response_lines = 0
    
    for line in response_lines:
        if not line.startswith("1"):
            initial_response_lines += 1
        else:
            break

    response_lines = response_lines[initial_response_lines:]
    
    idx = 1
    for line in response_lines:
        try:
            concept, category = line.split(" (Category = ")
            concept = concept.strip()[3:].strip() # remove leading number
            category = category.split(')')[0].strip() # remove trailing parenthesis
            concepts.append(concept)
            categories.append(category) 
            idx += 1
        except:
            idx += 1
            continue
        
    return concepts, categories

def parse_concept_analysis(model_output):
    """
    Parse the concept analysis from model output.
    
    Args:
        model_output: The full text output from the model
        
    Returns:
        A tuple containing:
        - concepts: List of concept descriptions
        - concept_settings: List of dictionaries with current_setting and new_setting
    """
    concepts = []
    concept_settings = []
    
    # Find the Concept Analysis section
    # Look for lines that match the pattern: N. Concept: <description> (A) <current> (B) <alternative>
    pattern = r'^\s*\d+\.\s*Concept:\s*(.+?)\s*\(A\)\s*(.+?)\s*\(B\)\s*(.+?)$'
    
    lines = model_output.split('\n')
    
    for line in lines:
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            concept_desc = match.group(1).strip()
            current_value = match.group(2).strip()
            alternative_value = match.group(3).strip()[:-1].strip()  # Remove trailing period if present
            
            concepts.append(concept_desc)
            concept_settings.append({
                "current_setting": current_value,
                "new_setting": [alternative_value]
            })
    
    assert len(concepts) == len(concept_settings), "Mismatch between number of concepts and concept settings parsed."
    assert len(concepts) > 0, "No concepts parsed from model output."
    assert len(concept_settings) > 0, "No concept settings parsed from model output."
    
    return concepts, concept_settings

def parse_llm_response_factor_settings(response):
    """
    Parses the response from the LLM for identifying current/alternative settings of each factor.
    Args:
        response: response from the LLM
    Returns:
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
    """
    response_lines = response.strip().split("\n")
    factor_settings = []
    
    initial_response_lines = 0
    
    for line in response_lines:
        if not line.startswith("1"):
            initial_response_lines += 1
        else:
            break
    
    response_lines = response_lines[initial_response_lines:]
    none_dic = {"current_setting": "N/A", "new_settings": ["N/A"]}
    
    idx = 1
    for line in response_lines:
        try:
            line = line.strip()[3:].strip()
            if not line.startswith('(A)') or ('(B.1)' not in line and '(B)' not in line):
                idx += 1
                factor_settings.append(none_dic)
                continue
            if '(B.3)' in line:
                idx += 1
                factor_settings.append(none_dic)
                continue
            if '(B.1)' in line:
                current_setting = line.split('(B.1)')[0].split('(A)')[1].strip()
            else:
                current_setting = line.split('(B)')[0].split('(A)')[1].strip()
            if '(B.2)' in line:
                new_settings = [line.split('(B.1)')[1].split('(B.2)')[0].strip(), line.split('(B.2)')[1].strip()]
            elif '(B.1)' in line:
                new_settings = [line.split('(B.1)')[1].strip()]
            else:
                new_settings = [line.split('(B)')[1].strip()]
            factor_settings.append({"current_setting": current_setting, "new_settings": new_settings})
            idx += 1
        except:
            idx += 1
            continue
    return factor_settings

def parse_llm_response_verification(response):
    
    match = re.search(r'<VALID>(YES|NO)</VALID>', response, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    return "N/A"
    
def enumerate_interventions_helper(intervention_list, intervention_str, factors, factor_settings, k_hop):
    """
    Helper function for enumerating all possible interventions.
    Args:
        intervention_list: a list of intervention vectors
        intervention_str: a string representing a choice of interventions
        factors: a list of factors to intervene on
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
        k_hop: if not None, only enumerate interventions that are k hops away from the original prompt
    """
    if len(intervention_str) == len(factors):
        intervention_list.append(intervention_str)
    elif sum([intervention_str[i] != "0" for i in range(len(intervention_str))]) == k_hop:
        # add 0s as remaining digits
        intervention_list.append(intervention_str + "0" * (len(factors) - len(intervention_str)))
    else:
        # no intervention case
        enumerate_interventions_helper(intervention_list, intervention_str + "0", factors, factor_settings, k_hop)
        # loop over possible interventions
        for idx in range(len(factor_settings[len(intervention_str)]["new_settings"])):
            enumerate_interventions_helper(intervention_list, intervention_str + str(idx+1), factors, factor_settings, k_hop)

def enumerate_interventions(factors, factor_settings, k_hop=None, include_no_intervention=True, mark_removals=True):
    """
    Enumerates all possible interventions.
    Args:
        factors: a list of factors to intervene on 
        factor_settings: a list of dictionaries, each containing the current setting and alternative settings for a factor
        k_hop: if not None, only enumerate interventions that are k hops away from the original prompt
        include_no_intervention: whether to include the no intervention case
        mark_removals: whether to mark removal interventions with a special symobol
    """
    if len(factors) != len(factor_settings):
        print("Length mismatch between factors and factor settings!")
        return []
        
    intervention_list = []
    enumerate_interventions_helper(intervention_list, "", factors, factor_settings, k_hop)
    no_intrv_str = "0" * len(factors)
    if not include_no_intervention and no_intrv_str in intervention_list:
        intervention_list.remove(no_intrv_str)
    if mark_removals:
        for idx in range(len(intervention_list)):
            intrv_str = intervention_list[idx]
            for j in range(len(intrv_str)):
                if intrv_str[j] != "0" and factor_settings[j]['new_settings'][int(intrv_str[j]) - 1] == "UNKNOWN":
                    intrv_str = intrv_str[:j] + "-" + intrv_str[j+1:]
            intervention_list[idx] = intrv_str
    return intervention_list

def parse_llm_response_implied_concepts(response, n_concepts):
    lines = response.strip().split('\n')
    
    # REMOVE <THINK></THINK> TAGS AND CONTENT INSIDE if PRESENT
    think_start = None
    think_end = None
    for i, line in enumerate(lines):
        if '<think>' in line.lower():
            think_start = i
        if '</think>' in line.lower():
            think_end = i
            break
    if think_start is not None and think_end is not None:
        lines = lines[:think_start] + lines[think_end + 1 :]

    concept_decisions = []
    seen_numbers = set()
    for line in lines:
        line = line.strip()
        match = re.match(r'^(\d+)\.', line)
        if match and ('(YES)' in line.upper() or '(NO)' in line.upper()):
            number = int(match.group(1))
            if number not in seen_numbers:
                concept_decisions.append(line)
                seen_numbers.add(number)

    if len(concept_decisions) != n_concepts:
        raise ValueError(
            f"Number of concept decisions does not match expected number of concepts. "
            f"Expected {n_concepts}, got {len(concept_decisions)}. "
            f"Full response was: {response}"
        )
    
    parsed_fds = []
    for idx, concept_decision in enumerate(concept_decisions):
        decision_bools = ["(YES)" in concept_decision.upper(), "(NO)" in concept_decision.upper()]
        if sum(decision_bools) != 1:
            raise ValueError(
                f"Concept decision {idx+1} does not match expected format. "
                f"(Did not provide yes/no decision or provided both). "
                f"Decision text: '{concept_decision}'. "
                f"Full response was: {response}"
            )
        parsed_fds.append(1 if "(YES)" in concept_decision.upper() else 0)
    
    return parsed_fds, response

def get_specific_layer_names(model, target_modules, key):
    layer_names = []
    if "all-linear" in target_modules:
        target_modules = "all-linear"

    for name, module in model.named_modules():
        # skip anything that doesn't match the key
        if key not in name:
            continue

        # special mode: grab all Linear layers
        if isinstance(target_modules, str) and target_modules == "all-linear":
            if isinstance(module, nn.Linear):
                layer_names.append(name)
        else:
            # original behavior: match any of the target_modules substrings
            if any(tm in name for tm in target_modules):
                layer_names.append(name)

    return set(layer_names)

class PromptingStrategy:
    def __init__(self, cot, few_shot, knn_rank, few_shot_prompt_name=None, add_instr=None):
        """
        Class for specifying the prompting strategy.
        Args:
            cot: whether to use CoT or direct answer trigger
            few_shot: whether to add few-shot examples to prompt
            knn_rank: whether to use knn rank (for now, only applicable to MedQA)
            few_shot_prompt_name: name of few shot prompt to use
            add_instr: additional instructions to add to prompt
        """
        self.cot = cot 
        self.few_shot = few_shot
        self.knn_rank = knn_rank
        if self.few_shot:
            assert few_shot_prompt_name is not None, "few_shot_prompt_name must be specified if few_shot is True"
        self.few_shot_prompt_name = few_shot_prompt_name
        self.add_instr = add_instr