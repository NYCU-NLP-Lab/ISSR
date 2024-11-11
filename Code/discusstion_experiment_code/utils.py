import nltk
import spacy
import pandas as pd
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re
import json
import matplotlib.pyplot as plt


def init_data():
    with open("./dataset/學測統合資料集v2.json", "r") as f:
        data = json.load(f)
        ref_word = pd.read_excel('./dataset/高中英文參考詞彙表v2.xlsx')
        data = restore_full_passage(data)
        data = append_distractors(data)
        return (data, ref_word)


def mask_questions(data, append_answer=False):
    ret_data = data.copy()
    for qa_pair in ret_data:
        data = qa_pair['questions']
        ans = qa_pair['correct_option_en']
        options = qa_pair['options_en']
        newslide = []
        slide = data.split(' ')
        for i in slide:
            if "___" in i:
                matched = re.search(r"[^_]", i)
                if matched is not None:
                    t = "[MASK]" + i[matched.start():]
                    newslide.append(t)
                else:
                    newslide.append("[MASK]")
            else:
                newslide.append(i)
        if append_answer:
            newslide.append("[SEP]")
            newslide.append(options[ans])
        qa_pair['masked_questions'] = (" ".join(newslide))
    return ret_data


def append_distractors(data):
    ret_data = data.copy()
    for qa_pair in ret_data:
        answer = qa_pair["correct_option_en"]
        distractors = qa_pair["options_en"].copy()
        distractors.remove(answer)
        qa_pair['distractors'] = distractors
    return ret_data


def draw_plot(plot_data, title = "", x_label = "", y_label = "", ylim = None):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create the bar chart
    ax.bar(plot_data.keys(), plot_data.values())

    # Set the x-axis label
    ax.set_xlabel(x_label)

    # Set the y-axis label
    ax.set_ylabel(y_label)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Set the title
    ax.set_title(title)

    # Show the plot
    plt.show()

def restore_full_passage(data):
    ret_data = data.copy()
    for qa_pair in ret_data:
        data = qa_pair['questions']
        ans = qa_pair['correct_option_en']
        options = qa_pair['options_en']
        newslide = []
        slide = data.split(' ')
        origin_word = ans
        for i in slide:
            if "___" in i:
                matched = re.search(r"[^_]", i)
                if matched is not None:
                    t = origin_word + i[matched.start():]
                    newslide.append(t)
                else:
                    newslide.append(origin_word)
            else:
                newslide.append(i)
        qa_pair['origin_questions'] = (" ".join(newslide))
    return ret_data



def __get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatization(word, language_model = None):
    # Parse the word with spaCy
    doc = language_model(word)
    # Get the first token in the document
    token = doc[0]
    # Return the lemma (base form) of the token
    if token.pos_ == 'ADV' and not token.text.endswith('ably') and token.text.endswith('ly'):
      return token.text[:-2].lower()
    return token.lemma_.lower()


""" This section do the disctactor post processing """
def get_pos_tag_of_word(blank_sentence, word):
    filled_sentence = blank_sentence.replace('[MASK]', word)
    tokens = nltk.word_tokenize(filled_sentence)
    pos_tags = nltk.pos_tag(tokens)
    ans_pos=None
    # get the pos of answer
    for token, pos_tag in pos_tags:
        if word == token:
            ans_pos=__get_wordnet_pos(pos_tag)
            break

    # answer may contain multiple words (e.g. phrases), in this case, we skip the candidate filter step
    if ans_pos is None:
        # print("warning, no pos tag returned for word.")
        pass
    return ans_pos



def get_example_prompt(model, prompt_layer):
    # First Example
    ret = f"""**Original Sentence**
Posters of the local rock band were displayed in store windows to promote the sale of their _____ tickets.

**Target Word**
concert

"""
    # Testing purpose
    if prompt_layer['testing_purpose'] == True:
        ret = ret+"""**Testing purpose**
1. what may this test question is going to test, response reason only (e.g., why chose "concert"), not potential distractors
The test question is likely testing the test-taker's understanding of the word "concert" and their ability to select the correct word to fill in the blank based on the context provided. The distractors should closely resemble "concert" in terms of word length, part of speech, and word frequency, but should not be the antonym of "concert" to make the decision more challenging. The goal is to ensure that test-takers who are not familiar with the meaning of "concert" are more likely to select the distractor.
2. what students are expected to learn about the usage of word "concert", don't response potential distractors
Key considerations for designing effective distractors in multiple-choice questions for the word "concert" would include:
- Ensuring that the distractors closely resemble the word "concert" in terms of word length and part of speech
- Making sure that the distractors have similar word frequency to "concert" to make the decision more challenging
- Avoiding the use of antonyms of "concert" as distractors
- Crafting distractors that are plausible choices but do not fit well in the context when compared to the correct answer

"""
    # Candidate pool generated by BERT
    if prompt_layer['candidate_pool'] == True:
        ret = ret+"""**Candidate Pool**
"sports", "proper", "regular", "personal", "clothes", "favorite", "traffic", "traditional", "valuable", "available", "travel", "necessary", "fashionable", "record", "official", "final", "usual", "clothing", "educational", "fashion", "journey"

"""
    # Chain of thought response generated by chat-gpt

    if prompt_layer['candidate_pool'] == True:
        quest_prompt = """pick three distractors from **Candidate Pool** for stem given in Original Sentence, response each distractors per line, and starts with enumerate number"""
    else:
        quest_prompt = """generate three distractors for stem given in Original Sentence, response each distractors per line, and starts with enumerate number"""
    ret = ret+quest_prompt
    model.append_chat(ret, 'user')
    ret = """1. journey
2. traffic
3. record"""
    model.append_chat(ret, 'assistant')

    # Second example
    ret = f"""**Original Sentence**
Maria didn't want to deliver the bad news to David about his failing the job interview. She herself was quite _____ about it.

**Target Word**
upset

"""
    # Testing purpose
    if prompt_layer['testing_purpose'] == True:
        ret = ret+"""**Testing purpose**
1. what may this test question is going to test, response reason only (e.g., why chose "upset"), not potential distractors
The test question is likely testing the test-takers' understanding of synonyms and the ability to choose the most appropriate word for the given context. The distractors should closely resemble "upset" in terms of word length, part of speech, and word frequency, but not be an antonym of "upset." The goal is to make the decision between the correct answer and the distractors more challenging for test-takers who may not thoroughly know the meaning of "upset."
2. what students are expected to learn about the usage of word "upset", don't response potential distractors
Key considerations for designing effective distractors in multiple-choice questions for the word "upset" in the given context:
- The distractor should be a word that is similar in length to "upset" and has the same part of speech (adjective).
- The distractor should have a similar word frequency to "upset" to make the decision between the correct answer and the distractors more challenging.
- The distractor should not be the antonym of "upset" to ensure that test-takers who haven't thoroughly understood the meaning of "upset" are more likely to select the distractor.
- The distractor should be a plausible choice, but not fit well in the context when compared to the correct answer "upset".

"""
    # Candidate pool generated by BERT
    if prompt_layer['candidate_pool'] == True:
        ret = ret+"""**Candidate Pool**
"curious", "careful", "excited", "happy", "afraid", "drowsy", "interested", "serious", "nervous", "concerned", "angry", "crazy", "sorry", "awful", "tired", "sure", "surprised", "upset", "good", "honest", "tragic", "terrible", "proud", "scared", "pleased", "strict"
  
"""
    if prompt_layer['candidate_pool'] == True:
        quest_prompt = """pick three distractors from **Candidate Pool** for stem given in Original Sentence, response each distractors per line, and starts with enumerate number
"""
    else:
        quest_prompt = """generate three distractors for stem given in Original Sentence, response each distractors per line, and starts with enumerate number
"""
    ret = ret+quest_prompt
    model.append_chat(ret, 'user')
    ret = """1. awful
2. drowsy
3. tragic"""
    model.append_chat(ret, 'assistant')




def extract_response(response):
  pattern = re.compile("\d+\. ")
  return [x.lower().replace("\"", "") for x in pattern.sub("", response).strip().split("\n") if x != ""]