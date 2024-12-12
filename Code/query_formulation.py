import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedModel
import transformers
import torch
from accelerate import infer_auto_device_map
import time
#Empty GPU cache
torch.cuda.empty_cache()

script_start_time = time.time()

"""
###################################################-----    Question Generation from I/P     ------###########################################################################
"""

# Model and tokenizer setup
model_path = "/home/joshb/llm/llama2/models/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, torch_dtype=torch.bfloat16)
model.to("cuda")

# Initialize the pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, torch_dtype=torch.bfloat16)

# Load and prepare data
df = pd.read_pickle('/home/joshb/llm/input_data/10000.pkl')
# df = df.dropna().sample(5).reset_index(drop=True)

df['generated_question'] = ''
df['prompt_generation_time'] = ''

for index, row in df.iterrows():
    prompt_with_length_command = f"{row['Essay']}. Generate a question that encapsulates the main theme of the above essay/text."

    start_time = time.time()
    # generated_texts = text_generator(prompt_with_length_command, max_length=max_word, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50, num_beams=2)
    generated_texts = text_generator(prompt_with_length_command, max_length= 8192)
    generated_prompt = generated_texts[0]['generated_text'] if generated_texts else None
    end_time = time.time()
    # generated_prompt = generated_text.split(":")[-1]
    df.at[index, 'generated_question'] = generated_prompt
    df.at[index, 'prompt_generation_time'] = end_time - start_time

"""
#####################################################-----  RegEx To Extract question     ------###########################################################################
"""
import re

pattern = r"(Generate a question that encapsulates the main theme of the above essay\/text\.)((\ *)(\n*))*(.*?\?)"

# Define a function to extract the third group
def extract_fifth_group(generated_question):
    match = re.search(pattern, generated_question)
    if match:
        return match.group(5)  
    return None  

# Apply the function to extract the third group and create a new column
df['prompt'] = df['generated_question'].apply(extract_fifth_group)
df = df.drop(['generated_question'], axis=1)

df.to_pickle('/home/joshb/llm/sonaa/input/10000.pkl')
print(f"Total script runtime: {time.time() - script_start_time:.2f} seconds")