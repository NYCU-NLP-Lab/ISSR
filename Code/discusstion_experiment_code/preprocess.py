from utils import *
import json
import re

def gen_both_pool_and_reason(self, question):
   cs = pool_generation(self, question)['cand_pool']
   reason = reason_generation(self, question)['reason']
   return {
    "reason": reason,
    "cand_pool": cs,
   }

def pool_generation(self, question):
  if self.config['use_cache_result'] == True:
    return
  # Answer relating
  input = question['sentence'] + " [SEP] " + question['answer']

  # Generate Candidate Set
  cs = list()
  for cand in self.unmasker(input):
      word = cand["token_str"].replace(" ", "")
      if len(word) > 0 and re.match(r'^[a-zA-Z]+$', word):
          cs.append(word)
          # cs.append({"word": word, "s0": cand["score"]})
  new_cs = self._filter_good_cand(cs, question)
  if len(new_cs) < 10:
      for i in cs:
          new_cs.append(i)
          new_cs = list(set(new_cs))
          if(len(new_cs) >= 10):
              break
  return {
      "reason": None,
      "cand_pool": new_cs,
  }

def none(self, question):
  return {
    "reason":None,
    "cand_pool": None
  }

def fast(self, question):
    with open("./dataset/BERT_response_cache.json", "r") as f:
      self.resp = json.load(f)

def reason_generation(self, question):
  LLM_config = {
     "temperature": 0.7
  }
  reason_list = [
      "what may this test question is going to test, response reason only (e.g., why chose \"{ANS}\"), not potential distractors",
      'what students are expected to learn about the usage of word "{ANS}", don\'t response potential distractors',
  ]

  VERBOSE = True
  result = list()
  for reason_need_to_explain in reason_list:
    if question is not None:
      prompt = """[human]
  In the following incomplete multiple-choice question, students need to select the correct answer to fill in the blank, where the target word is "{ANS}"
  Your task is to provide insights into {SLOT}
  Please refrain from generating potential distractors. Focus on key considerations for designing effective distractors in multiple-choice questions.

  Qustion:
  {STEM}

    [gpt]"""
      processed_reason = reason_need_to_explain.replace("{ANS}", question['answer'])
      prompt = prompt.replace("{SLOT}", processed_reason)
      for i in range(1, len(question['distractors'])+1):
          prompt = prompt.replace(f"{{D{i}}}", question['distractors'][i-1])
      prompt = prompt.replace("{ANS}", question['answer'])
      prompt = prompt.replace("{STEM}", question['sentence'].replace("[MASK]", "_____"))
      response = self.model.inference(prompt, LLM_config)
      reason_qa = f"{processed_reason}\n{response}"
      result.append(reason_qa)

  return {
      "reason": result,
      "cand_pool": None,
  }