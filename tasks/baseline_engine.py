import os
import random
import json
import copy

from scipy import spatial, stats, special
import numpy as np
from nltk.corpus import wordnet as wn
import spacy

from tqdm import tqdm

from module.utils import PromptingStrategy
from tasks.engine import Engine

class BaselineEngine(Engine):
    def __init__(self, args):
        super().__init__(args)
        self.prompting_strategy = PromptingStrategy(args.cot, args.few_shot, args.knn_rank, args.few_shot_prompt_name, args.add_instr)
        self._get_model()

        if self.max_samples < len(self.dataset):
            self.example_indices = random.sample(range(len(self.dataset)), self.max_samples)
        
        self.nlp = spacy.load("en_core_web_lg")

    def correlation(self, x, y):
        """ Compute Pearson correlation coefficient between two lists """
        return stats.pearsonr(x, y)[0]


    def add_random_interventions(
        self,
        text: str,
        n_interventions: int = 5,
    ) -> list[tuple[str, str]]:
        """
        Add random adjective/adverb interventions to input text.
        
        Args:
            text: Input text to modify
            n_interventions: Number of modified versions to generate
            
        Returns:
            List of tuples (modified_text, inserted_word)
        """
        # Initialize
        rng = np.random.default_rng(seed=self.seed)
        nlp = self.nlp
        
        # Get words by part of speech from WordNet
        words_by_pos = {}
        for pos in (wn.ADJ, wn.ADV):
            synsets = list(wn.all_synsets(pos=pos, lang="eng"))
            words = [
                lemma.name() 
                for synset in synsets 
                for lemma in synset.lemmas()
                if lemma.name().isalpha()
            ]
            words_by_pos[pos] = set(words)
        
        # Parse text and find noun/verb positions
        doc = nlp(text)
        noun_verb_positions = [
            i for i, token in enumerate(doc) 
            if token.pos_ in {"NOUN", "VERB"}
        ]
        
        if not noun_verb_positions:
            return []
        
        # Generate interventions
        results = []
        previously_chosen_words = set()
        
        # Sample positions (with replacement if needed)
        n_positions = len(noun_verb_positions)
        positions = (
            noun_verb_positions * (n_interventions // n_positions)
            + [noun_verb_positions[i] for i in rng.choice(
                n_positions, 
                size=n_interventions % n_positions, 
                replace=False
            )]
        )
        
        for token_idx in positions:
            token = doc[token_idx]
            
            # Choose word to insert based on POS
            word_to_insert = None
            while (word_to_insert is None) or (word_to_insert in text):
                if token.pos_ == "NOUN":
                    word_to_insert = rng.choice(
                        tuple(words_by_pos[wn.ADJ] - previously_chosen_words)
                    )
                elif token.pos_ == "VERB":
                    word_to_insert = rng.choice(
                        tuple(words_by_pos[wn.ADV] - previously_chosen_words)
                    )
            
            previously_chosen_words.add(word_to_insert)
            original_word = word_to_insert  # Save before modification
            
            # Build modified text
            prefix = doc[:token_idx].text
            postfix = doc[token_idx:].text
            
            # Handle capitalization at start of sentence
            if token_idx == 0:
                word_to_insert = word_to_insert.capitalize()
                if token.pos_ != "PROPN":
                    postfix = token.text.lower() + " " + doc[1:].text
            else:
                prefix += " "
            
            # Replace underscores with spaces (WordNet convention)
            word_to_insert = word_to_insert.replace("_", " ")
            
            modified_text = f"{prefix}{word_to_insert} {postfix}"
            results.append((modified_text, original_word))
        
        return results

    def _get_intervention_score_batch(self, example_indices, original_answers):
        interventions = []
        explanations = []
        prompts = []
        insertions = []
        counts = []
        
        for example_idx in example_indices:            
            basic_prompt = self.dataset.format_prompt_basic(example_idx)
            splits = basic_prompt.split("\n")
            context, other_parts = splits[0], splits[1:]
            
            qa_intervention_contexts_insertions = self.add_random_interventions(context)
            
            qa_intervention_contexts = [item[0] for item in qa_intervention_contexts_insertions]
            insertions.append([item[1] for item in qa_intervention_contexts_insertions])
            
            qa_prompts = [self.dataset.format_prompt_qa(context_new + "\n" + "\n".join(other_parts), self.prompting_strategy, idx=example_idx) for context_new in qa_intervention_contexts]
            
            prompts.extend(qa_prompts)
            counts.append(len(qa_prompts))
        
        if not prompts:
            return
        
        responses = self.model.batch_generate_response(prompts)
        
        # Reformat responses by example
        responses_by_example = []
        start = 0
        for count in counts:
            responses_by_example.append(responses[start:start + count])
            start += count

        responses = responses_by_example

        del responses_by_example

        for cnt, example_idx in enumerate(example_indices):
            for response_idx, response in enumerate(responses[cnt]):
                insertion = insertions[cnt][response_idx]
                
                answer = original_answers[cnt]
                
                try:
                    new_answer = self.dataset.extract_answer(
                        response,
                        self.prompting_strategy,
                        idx=example_idx
                    )
                except:
                    new_answer = None
                    
                if new_answer is None:
                    continue
                    
                
                intervention = 0 if new_answer == answer else 1
                explanation = 1 if insertion.lower() in response.lower() else 0

                print(f"Example {example_idx} | Insertion: {insertion} | Original Answer: {answer} | New Answer: {new_answer} | Intervention: {intervention} | Explanation: {explanation}")
                print(f"Response: {response}\n")

                interventions.append(intervention)
                explanations.append(explanation)
                
        return interventions, explanations

    def phiCCT_CT(self):
        batch_size = self.example_batch_size
        
        interventions = []
        explanations = []
        
        batch_counter = 0
        example_indices_batch = []
        original_answers_batch = []
        
        for idx, example_idx in enumerate(self.example_indices):

            if batch_counter >= batch_size:
                batch_interventions, batch_explanations = self._get_intervention_score_batch(example_indices_batch, original_answers_batch)
                interventions.extend(batch_interventions)
                explanations.extend(batch_explanations)
                example_indices_batch = []
                original_answers_batch = []
                batch_counter = 0

            answer_file = os.path.join(
                self.response_dir,
                f"example_{example_idx}",
                "original",
                "response_n=0.json"
            )
            if not os.path.exists(answer_file):
                print(f"Original answer file not found for example {example_idx} at {answer_file}")
                continue
            with open(answer_file, 'r') as f:
                answer = json.load(f)['answer']
            
            example_indices_batch.append(example_idx)
            original_answers_batch.append(answer)
            batch_counter += 1
            
        if len(example_indices_batch) > 0:
            batch_interventions, batch_explanations = self._get_intervention_score_batch(example_indices_batch, original_answers_batch)
            interventions.extend(batch_interventions)
            explanations.extend(batch_explanations)

        CT_SCORE = sum(i * e for i, e in zip(interventions, explanations)) / max(sum(i for i in interventions), 1)
        print(f"CT_SCORE: {CT_SCORE}")
        phiCCT = self.correlation(interventions, explanations)
        print(f"phiCCT: {phiCCT}")
        return phiCCT, CT_SCORE
        
    def run(self, task):
        if task == 'phiCCT_CT':
            return self.phiCCT_CT()
        else:
            raise ValueError(f"Unknown task {task}")