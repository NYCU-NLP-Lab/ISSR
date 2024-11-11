from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoModel
import torch.nn as nn
import accelerate
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers import pipeline


import argparse
from fastchat.serve.cli import SimpleChatIO
from fastchat.serve.inference import chat_loop
from fastchat.model.model_adapter import (
load_model,
get_generate_stream_function
)
from fastchat.utils import str_to_torch_dtype
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from groq import Groq


class openAIModel():
    def __init__(self, api_key):
        self.client = OpenAI(
          api_key=api_key,
        )
        
        self.chat = list()
        self.notify_using_model = False

    def inference(self, config):
        if config.get("temperature") is not None:
            temperature = config['temperature']
        else:
            temperature = 0

        if config.get("model_name") is not None:
            model = config['model_name']
        else:
            # if no model is specified, use gpt-3.5-turbo-0125
            model = "gpt-3.5-turbo-0125"
        # Notify user the model they are using    
        if not self.notify_using_model:
            print(f"Now inferencing with {model}")
            self.notify_using_model = True
        # for degugging: print the prompt sending to model
        #self.test_prompt()
        
        # Call GPT API and get response
        completion = self.client.chat.completions.create(
            model=model,
            messages=self.chat,
            temperature = temperature,
            n=1 # generate one choice only
        )
        gpt_response = completion.choices[0].message.content
        # print(f"-----[GPT]-----\n{gpt_response}")
        self.clear_chat()
        return gpt_response

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })
        
    def test_prompt(self):
        print(self.chat)


class llama3_70b():
    def __init__(self, api_key):

        self.client = Groq(
     api_key=api_key
        )
        self.chat = list()
        self.notify_using_model = False

    def inference(self, config):
        if config.get("temperature") is not None:
            temperature = config['temperature']
        else:
            temperature = 0

        if config.get("model_name") is not None:
            model = config['model_name']
        else:
            model = "llama3-70b-8192"
        # Notify user the model they are using    
        if not self.notify_using_model:
            print(f"Now inferencing with {model}")
            self.notify_using_model = True

        #self.test_prompt()
        # Call GPT API and get response
        completion = self.client.chat.completions.create(
            model=model,
            messages=self.chat,
            temperature = temperature,
            n=1 # generate one choice only)
        )
        gpt_response = completion.choices[0].message.content
        #print(f"-----[GPT]-----\n{gpt_response}")
        self.clear_chat()
        return gpt_response

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })
        
    def test_prompt(self):
        print(self.chat)



class llama3_8b():
    def __init__(self, api_key):

        self.client = Groq(
            api_key = api_key
        )
        self.chat = list()
        self.notify_using_model = False

    def inference(self, config):
        if config.get("temperature") is not None:
            temperature = config['temperature']
        else:
            temperature = 0

        if config.get("model_name") is not None:
            model = config['model_name']
        else:
            model = "llama3-8b-8192"
        # Notify user the model they are using    
        if not self.notify_using_model:
            print(f"Now inferencing with {model}")
            self.notify_using_model = True

        # self.test_prompt()
        # Call GPT API and get response
        completion = self.client.chat.completions.create(
            model=model,
            messages=self.chat,
            temperature = temperature,
            n=1 # generate one choice only)
        )
        gpt_response = completion.choices[0].message.content
        # print(f"-----[GPT]-----\n{gpt_response}")
        self.clear_chat()
        return gpt_response

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })
        
    def test_prompt(self):
        print(self.chat)



class gemma2():
    def __init__(self, api_key):

        self.client = Groq(
     api_key=api_key
        )
        self.chat = list()
        self.notify_using_model = False

    def inference(self, config):
        if config.get("temperature") is not None:
            temperature = config['temperature']
        else:
            temperature = 0

        if config.get("model_name") is not None:
            model = config['model_name']
        else:
            model = "gemma2-9b-it"
        # Notify user the model they are using    
        if not self.notify_using_model:
            print(f"Now inferencing with {model}")
            self.notify_using_model = True

        #self.test_prompt()
        # Call GPT API and get response
        completion = self.client.chat.completions.create(
            model=model,
            messages=self.chat,
            temperature = temperature,
            n=1 # generate one choice only)
        )
        gpt_response = completion.choices[0].message.content
       # print(f"-----[GPT]-----\n{gpt_response}")
        self.clear_chat()
        return gpt_response

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })
        
    def test_prompt(self):
        print(self.chat)
