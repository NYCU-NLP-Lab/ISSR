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


# 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
class zephyr():
    def __init__(self, model_path, device):
        self.model = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.chat = list()
        self.assistant_template = self.get_assistant_template()
    def inference(self, config):
        if config.get('temperature') is not None:
            temperature = config['temperature']
        else:
            temperature = 0.7
        # self.append_chat("", "assistant")
        prompt = self.model.tokenizer.apply_chat_template(self.chat, tokenize=False, add_generation_prompt=True)

        # for degugging: print the prompt sending to model
        self.test_prompt()


        
        outputs = self.model(prompt, max_new_tokens=2096, do_sample=True, temperature=temperature, top_k=50, top_p=0.95)
        outputs = outputs[0]['generated_text'].split(self.assistant_template)[-1].strip()
        print(f"-----[GPT]-----\n{outputs}")
        # Clear output history
        self.clear_chat()
        return outputs

    # get template by calling the apply_template, model must be loaded!
    def get_assistant_template(self):
        message = [{'role': 'assistant', 'content': ""}]
        prompt = self.model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True).split("\n")[0]
        return prompt

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })

    def test_prompt(self):
        print(self.model.tokenizer.apply_chat_template(self.chat, tokenize=False))


    
# phi_2 model from microsoft/phi-2
class phi_2():
    def __init__(self, model_path, device):
        torch.set_default_device(device)
        self.chat = list()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # print("Model device:", next(self.model.parameters()).device)  # Print device of model parameters
        self.assistant_template = self.get_assistant_template()
    def inference(self, config):
        # self.append_chat("", "assistant")
        inputs = self.tokenizer.apply_chat_template(self.chat, tokenize=False)
        print(inputs)
        input = self.tokenizer(inputs, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True, max_length=2048)
        # predefined max_length=2048
        outputs = self.model.generate(**input, max_length=512, pad_token_id=self.pad_token_id)
        text = self.tokenizer.batch_decode(outputs)[0].split(self.assistant_template)[-1].strip()
        # print(f"RESPONSE: {text}", end = "\n=============\n")
        # Clear output history
        self.clear_chat()
        print(text, end="\n=====\n")
        return text

    # get template by calling the apply_template, model must be loaded!
    def get_assistant_template(self):
        message = [{'role': 'assistant', 'content': ""}]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True).split("\n")[0]
        return prompt

    
    # clear chat history
    def clear_chat(self):
        self.chat = []

    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })

    def test_prompt(self):
        print(self.tokenizer.apply_chat_template(self.chat, tokenize=False))


class vicunaModel():
    def __init__(self, model_path, device):
        # Manually create the namespace
        self.chat = list()
        self.assistant_template = self.get_assistant_template()
        self.args = argparse.Namespace(
            model_path=model_path,
            revision='main',
            device=device,
            gpus=None,
            num_gpus=1,
            max_gpu_memory=None,
            dtype=None,
            load_8bit=False,
            cpu_offloading=False,
            gptq_ckpt=None,
            gptq_wbits=16,
            gptq_groupsize=-1,
            gptq_act_order=False,
            awq_ckpt=None,
            awq_wbits=16,
            awq_groupsize=-1,
            enable_exllama=False,
            exllama_max_seq_len=4096,
            exllama_gpu_split=None,
            exllama_cache_8bit=False,
            enable_xft=False,
            xft_max_seq_len=4096,
            xft_dtype=None,
            conv_template=None,
            conv_system_msg=None,
            temperature=0.7,
            output_path='./out',
            repetition_penalty=1.0,
            max_new_tokens=512,
            no_history=False,
            style='simple',
            multiline=False,
            mouse=False,
            judge_sent_end=False,
            debug=False,
            testset_path=None,
            post_processing='none'
        )
        if self.args.gpus:
            if len(self.args.gpus.split(",")) < self.args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({self.args.num_gpus}) than --gpus {self.args.gpus}!"
                )


        # Model
        self.model, self.tokenizer = load_model(
            self.args.model_path,
            device=self.args.device,
            num_gpus=self.args.num_gpus,
            max_gpu_memory=self.args.max_gpu_memory,
            dtype=self.args.dtype,
            load_8bit=self.args.load_8bit,
            cpu_offloading=self.args.cpu_offloading,
            revision=self.args.revision,
            debug=self.args.debug,
        )
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.chatio = SimpleChatIO(self.args.multiline)

    def inference(self, config):
        # self.append_chat("", "assistant")
        prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False)
        response = chat_loop(
            self.model,
            self.tokenizer,
            self.args.model_path,
            self.generate_stream_func,
            prompt,
            self.args.device,
            self.args.num_gpus,
            self.args.max_gpu_memory,
            str_to_torch_dtype(self.args.dtype),
            self.args.load_8bit,
            self.args.cpu_offloading,
            self.args.conv_template,
            self.args.conv_system_msg,
            self.args.temperature,
            self.args.repetition_penalty,
            self.args.max_new_tokens,
            self.chatio,
            gptq_config=GptqConfig(
                ckpt=self.args.gptq_ckpt or self.args.model_path,
                wbits=self.args.gptq_wbits,
                groupsize=self.args.gptq_groupsize,
                act_order=self.args.gptq_act_order,
            ),
            awq_config=AWQConfig(
                ckpt=self.args.awq_ckpt or self.args.model_path,
                wbits=self.args.awq_wbits,
                groupsize=self.args.awq_groupsize,
            ),
            revision=self.args.revision,
            judge_sent_end=self.args.judge_sent_end,
            debug=self.args.debug,
            history=not self.args.no_history,
            post_processing = self.args.post_processing,
        )
        # print(f"RESPONSE: {response}")
        self.clear_chat()
        return response

    # clear chat history
    def clear_chat(self):
        self.chat = []

    # get template by calling the apply_template, model must be loaded!
    def get_assistant_template(self):
        message = [{'role': 'assistant', 'content': ""}]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True).split("\n")[0]
        return prompt

    
    def append_chat(self, prompt, role):
        self.chat.append({
            "role": role,
            "content": prompt
        })
        
    def test_prompt(self):
        print(self.tokenizer.apply_chat_template(self.chat, tokenize=False))




class openAIModel():
    def __init__(self, api_key):
        self.client = OpenAI(
          api_key="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",  # this is also the default, it can be omitted
        )
        
        # This two template has not been used
        self.user_template = """A good distractor for a multiple-choice question should be carefully crafted to resemble a viable option, making the decision between the correct answer and the distractors more challenging.
Distractor's word length should be close to correct answer, and distractor's pos tag should be same with correct answer, and distractor's word frequency should be close to correct answer.
A good distractor should not be the antonym of correct answer.
The goal is to ensure that test-takers who haven't thoroughly know the meaning of correct answer are more likely to select the distractor.

While the distractor should be a plausible choice, it should not fit well in the context when compared to the correct answer.\n\n"""
        self.gpt_template = """
Got it! I'll keep those guidelines in mind. What's the question you need distractors for?
"""
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
            model = "gpt-4o-mini-2024-07-18"
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


class llama3_70b():
    def __init__(self, api_key):

        self.client = Groq(
     api_key="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"
        )
        # This two template has not been used
        self.user_template = """A good distractor for a multiple-choice question should be carefully crafted to resemble a viable option, making the decision between the correct answer and the distractors more challenging.
Distractor's word length should be close to correct answer, and distractor's pos tag should be same with correct answer, and distractor's word frequency should be close to correct answer.
A good distractor should not be the antonym of correct answer.
The goal is to ensure that test-takers who haven't thoroughly know the meaning of correct answer are more likely to select the distractor.

While the distractor should be a plausible choice, it should not fit well in the context when compared to the correct answer.\n\n"""
        self.gpt_template = """
Got it! I'll keep those guidelines in mind. What's the question you need distractors for?
"""
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
     #api_key="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"
            api_key = "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"
        )
        # This two template has not been used
        self.user_template = """A good distractor for a multiple-choice question should be carefully crafted to resemble a viable option, making the decision between the correct answer and the distractors more challenging.
Distractor's word length should be close to correct answer, and distractor's pos tag should be same with correct answer, and distractor's word frequency should be close to correct answer.
A good distractor should not be the antonym of correct answer.
The goal is to ensure that test-takers who haven't thoroughly know the meaning of correct answer are more likely to select the distractor.

While the distractor should be a plausible choice, it should not fit well in the context when compared to the correct answer.\n\n"""
        self.gpt_template = """
Got it! I'll keep those guidelines in mind. What's the question you need distractors for?
"""
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
     api_key="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"
        )
        # This two template has not been used
        self.user_template = """A good distractor for a multiple-choice question should be carefully crafted to resemble a viable option, making the decision between the correct answer and the distractors more challenging.
Distractor's word length should be close to correct answer, and distractor's pos tag should be same with correct answer, and distractor's word frequency should be close to correct answer.
A good distractor should not be the antonym of correct answer.
The goal is to ensure that test-takers who haven't thoroughly know the meaning of correct answer are more likely to select the distractor.

While the distractor should be a plausible choice, it should not fit well in the context when compared to the correct answer.\n\n"""
        self.gpt_template = """
Got it! I'll keep those guidelines in mind. What's the question you need distractors for?
"""
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
