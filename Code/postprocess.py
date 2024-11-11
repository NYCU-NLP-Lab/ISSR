import random


def self_answer_correctness(self, question, distractor_pool):
    LLM_config = {
        "temperature": 0.7
    }
    answer = question['answer'].lower()
    if self.config["post_processing_function"]['self-answer'] == True:
        # Change prompt to avoid openai's policy that may refuse to answer
        prompt = f"""
Imagine you are a english teacher that designing a vocabulary test to a second language learner, and you came up with a distrctor candidate {{OPTION1}} 

Qustion:
{{STEM}}

Correct answer:
{{ANSWER}}

Distractor candidate:
{{OPTION1}}

Do you think whether word {{OPTION1}} is a good distractor or not? Response with Yes or No only.
"""

        prompt = prompt.replace("{STEM}", question['sentence'].replace("[MASK]", "_____")).replace("{ANSWER}", question['answer'])
        for dis in distractor_pool:
            send_prompt = prompt.replace("{OPTION1}", dis.lower())
            self.model.append_chat(send_prompt, 'user')
            response = self.model.inference(LLM_config).lower()
            # Gpt refuse to answer
            if "Yes" not in response:
                self.bad_distractor.append(dis)
            else:
                self.good_distractor.append(dis)
            self.model.clear_chat()
        self.good_distractor = list(set(self.good_distractor))
        self.bad_distractor = list(set(self.bad_distractor))
        print(f"good: {self.good_distractor}")
        print(f"bad: {self.bad_distractor}")
        # if self.config['post_processing_function']['error-report'] == True:
        #     # TODO
        #     return suit, non_suit
        # else:
        #     return suit, non_suit

    else:
        self.good_distractor = [x for x in distractor_pool]


def self_answer_same_meaning(self, question, distractor_pool):
    LLM_config = {
        "temperature": 0.7
    }
    answer = question['answer'].lower()
    if self.config["post_processing_function"]['self-answer'] == True:
        # Change prompt to avoid openai's policy that may refuse to answer
        prompt = f"""
You will now see two sentences with only one word difference between them:

Sentence 1:
{{STEM1}}

Sentence 2:
{{STEM2}}

Do these two sentences have the same meaning? Please respond with 'Yes' or 'No' only.
"""
#        prompt = prompt.replace("{STEM1}", question['sentence'].replace("[MASK]", question['answer']))
        for dis in distractor_pool:
            sentence = [question['sentence'].replace("[MASK]", question['answer']), question['sentence'].replace("[MASK]", dis.lower())]
            random.shuffle(sentence)
            send_prompt = prompt.replace("{STEM1}", sentence[0]).replace("{STEM2}", sentence[1])
            self.model.append_chat(send_prompt, 'user')
            response = self.model.inference(LLM_config).lower()
            # Gpt refuse to answer
            if "no" not in response:
                self.bad_distractor.append(dis)
            else:
                self.good_distractor.append(dis)
            self.model.clear_chat()
        self.good_distractor = list(set(self.good_distractor))
        self.bad_distractor = list(set(self.bad_distractor))
        print(f"good: {self.good_distractor}")
        print(f"bad: {self.bad_distractor}")
        # if self.config['post_processing_function']['error-report'] == True:
        #     # TODO
        #     return suit, non_suit
        # else:
        #     return suit, non_suit

    else:
        self.good_distractor = [x for x in distractor_pool]


def self_answer(self, question, distractor_pool):
    LLM_config = {
        "temperature": 0.7
    }
    answer = question['answer'].lower()
    if self.config["post_processing_function"]['self-answer'] == True:
        # Change prompt to avoid openai's policy that may refuse to answer
        if self.model_name == "gpt":
            prompt = f"""
Imagine you are a high school student that studying english, and you are answering question given below:
The following is a vocabulary test that requires selecting one answer from given options to fill in the blank.
Please select the option that fit the context best from below, response with the correct option directly, if you think both options are suitable for the context, response with "BOTH ARE GOOD".

Qustion:
{{STEM}}

options:
{{OPTION1}}
{{OPTION2}}"""
        else:
            prompt = f"""
The following is a vocabulary test that requires selecting one answer from given options to fill in the blank.
Please select the option that fit the context best from below, response with the correct option directly, if you think both options are suitable for the context, response with "BOTH ARE GOOD".

Qustion:
{{STEM}}

options:
{{OPTION1}}
{{OPTION2}}
    """
        prompt = prompt.replace("{STEM}", question['sentence'].replace("[MASK]", "_____"))
        for dis in distractor_pool:
            options = [dis.lower(), answer]
            # randomize the priority of list answer
            random.shuffle(options)
            send_prompt = prompt.replace("{OPTION1}", options[0]).replace("{OPTION2}", options[1])
            self.model.append_chat(send_prompt, 'user')
            response = self.model.inference(LLM_config).lower()
            # Gpt refuse to answer
            if "i'm sorry" in response:
                self.bad_distractor.append(dis)
            elif answer not in response:
                self.bad_distractor.append(dis)
            else:
                self.good_distractor.append(dis)
            self.model.clear_chat()
        self.good_distractor = list(set(self.good_distractor))
        self.bad_distractor = list(set(self.bad_distractor))
        print(f"good: {self.good_distractor}")
        print(f"bad: {self.bad_distractor}")
        # if self.config['post_processing_function']['error-report'] == True:
        #     # TODO
        #     return suit, non_suit
        # else:
        #     return suit, non_suit

    else:
        self.good_distractor = [x for x in distractor_pool]