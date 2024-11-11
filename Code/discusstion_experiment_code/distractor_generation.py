from utils import *


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def few_shot(self, question, preprocess_pool):
    if self.config['distractor_generation_function'].get('pick_distractors_per_round') is not None:
        pick_distractors_per_round = self.config['distractor_generation_function']['pick_distractors_per_round']
    else:
        pick_distractors_per_round = 3
    zero_shot = self.config['distractor_generation_function']['zero-shot']
    prompt_layer = {
        "testing_purpose": False,
        "candidate_pool": False,
    }
    if preprocess_pool['reason'] is not None:
        prompt_layer['testing_purpose'] = True
    if preprocess_pool['cand_pool'] is not None:
        prompt_layer['candidate_pool'] = True
        # reverse the order of candidate set since the more connfident candidate should be placed in later place
        preprocess_pool['cand_pool'].reverse()

    result = list()
    target_distractor_count = self.config['distractor_generation_function']['generate_count']
    bad_distractors = list()

    if preprocess_pool['cand_pool'] is not None and len(preprocess_pool['cand_pool']) <= target_distractor_count:
        print("Warning! the candidate pool size is smaller then distractor needed to generate!")
        return preprocess_pool['cand_pool']
    retry = 0
    while len(result) < target_distractor_count and retry < 10:
        has_bad = False
        if not zero_shot:
            # append the few shot prompt into chat
            get_example_prompt(self.model, prompt_layer)
        question_prompt = f"""You are a teacher picking plausable vocabularies as distractors used in the vocabulary exam.
        
**Original Sentence**
    {question['sentence'].replace('[MASK]', "_____")}
    
**Target Word**
{question['answer']}
    
"""
    
        if preprocess_pool['cand_pool'] is not None:
            question_prompt = question_prompt+"\n**Candidate Pool**\n"
            for c in preprocess_pool['cand_pool']:
                question_prompt = question_prompt +f"\'{c}\', "
            question_prompt = question_prompt[:-2]

        if len(bad_distractors) != 0:
            question_prompt = question_prompt+"\n**Words to avoid picking**\n"
            for i in bad_distractors:
                question_prompt = question_prompt+f" {i},"
            # remove the last ','
            question_prompt = question_prompt[:-1]

        if prompt_layer['candidate_pool'] == True:
            quest_prompt = f"""
pick {pick_distractors_per_round} distractors from **Candidate Pool** section for stem given in Original Sentence, response each distractors per line, and starts with enumerate number. Please select your response from words in section **Candidate Pool** only."""
        else:
            quest_prompt = f"""generate {pick_distractors_per_round} distractors for stem given in Original Sentence, restrain your output in following format given below (for example, your generated {pick_distractors_per_round} distractors are: apple, banana, orange, ...)
1. apple
2. banana
3. orange
...
"""

        quest_prompt += f"""\n\nHere are some good distractor examples for this question: {question['distractors'][0]}, {question['distractors'][1]}, {question['distractors'][2]}."""
        question_prompt = question_prompt + "\n" + quest_prompt
        self.model.append_chat(question_prompt, 'user')

        response = self.model.inference(self.config)
        response = self.extract_response(response, preprocess_pool['cand_pool'])
        prev_result_length = len(result)
        for i in response:
            # remove picked elements select by LLM from candidate pool
            if prompt_layer['candidate_pool'] == True:
                result.append(i)
                try:
                    preprocess_pool['cand_pool'].remove(i)
                except ValueError:
                    print(f"The word '{i}' picked by LLM does not exist in candidate list, may it because the low size of candidate pool?")
                    result.remove(i)
                    bad_distractors.append(i)
                    has_bad = True
            else:
                if(i in bad_distractors):
                    print(f"The word '{i}' already picked")
                    has_bad = True
                else:
                    result.append(i)
                    bad_distractors.append(i)
        result = remove_duplicates(result)
        
        # To prevent stuck
        if(len(result) == prev_result_length):
            retry+=1
        elif(has_bad):
            retry+=1
        # Remove the redundant bad selections
        bad_distractors = list(set(bad_distractors))
    if(retry >=10):
        print("Some error occured during candidate selection, appending exceptional distractors into result")
        for b in bad_distractors:
            result.append(b)
        result = remove_duplicates(result)
    if(len(result) < target_distractor_count and prompt_layer['candidate_pool'] == True):
        print("Quantity of picked distractors is smaller then target, filling with candidate pool")
        for b in preprocess_pool['cand_pool']:
            result.append(b)
            result = remove_duplicates(result)
            if(len(result)>=target_distractor_count):
                break
    ## TODO: make avoid generate distractors in non-candidate pool generation
    ## TODO: make selection quantity variable in a round
    return result