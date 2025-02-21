### Prompt used in distractor selector

**Input: Original Sentence**  
Posters of the local rock band were displayed in store windows to promote the sale of their \_\_\_\_\_ tickets.  

**Target Word**  
concert  

**Candidate Pool**  
`sports`, `proper`, `regular`, `personal`, `clothes`, `favorite`, `traffic`, `traditional`, `valuable`, `available`, `travel`, `necessary`, `fashionable`, `record`, `official`, `final`, `usual`, `clothing`, `educational`, `fashion`, `journey`  

Pick three distractors from **Candidate Pool** for the given sentence. Respond with each distractor on a new line, starting with an enumerate number.  

**Output:**  
1. journey  
2. traffic  
3. record  
---

### Prompt used in self-review  

#### Binary Choice Validation for Distractor Suitability  

**Input:** Imagine you are a high school student studying English, and you are answering the following vocabulary test question.  

The following test requires selecting one answer from the given options to fill in the blank. Please select the option that best fits the context. Respond with the correct option directly. If you think both options are suitable, respond with `"BOTH ARE GOOD"`.  

**Question:**  
The newcomer speaks with a strong Irish \_\_\_\_\_; he must be from Ireland.  

**Options:**  
- identity  
- accent  

---

#### Independent Suitability Judgment for Distractor Validation  

**Input:** Imagine you are an English teacher designing a vocabulary test for a second-language learner. You came up with a distractor candidate **"identity"**.  

**Question:**  
The newcomer speaks with a strong Irish \_\_\_\_\_; he must be from Ireland.  

**Correct Answer:**  
accent  

**Distractor Candidate:**  
identity  

**Criteria for question creation:**  
1. The length difference between the answer and the distractor should not exceed 2 characters.  
2. The answer and the distractor should share the same part of speech.  
3. The difficulty levels between the answer and the distractor should be closely matched.  

Do you think the word `"identity"` is a good distractor? Respond with `"Yes"` or `"No"` only.  

---

#### Semantic Consistency Check for Distractor Validation  

You will now see two sentences that differ by only one word:  

**Sentence 1:**  
The newcomer speaks with a strong Irish identity; he must be from Ireland.  

**Sentence 2:**  
The newcomer speaks with a strong Irish accent; he must be from Ireland.  

Do these two sentences have the same meaning? Please respond with `"Yes"` or `"No"` only.  
