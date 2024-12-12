import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedModel
import transformers
import torch
from accelerate import infer_auto_device_map
import time
#Empty GPU cache
torch.cuda.empty_cache()

script_start_time = time.time()

# Model and tokenizer setup
model_path = "/home/joshb/llm/llama2/models/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True, torch_dtype=torch.bfloat16)
model.to("cuda:1")

# Initialize the pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1, torch_dtype=torch.bfloat16)

# Load and prepare data
df = pd.read_pickle('/home/joshb/llm/sonaa/input/DS1.pkl')
# df = pd.read_pickle('/home/joshb/llm/sonaa/output/4000_output.pkl')
# df = df.dropna().sample(3).reset_index(drop=True)
df['llama3_text'] = ''
df['llama3_time'] = ''

# torch.utils.data.DataLoader(pickle_file_path, batch_size=1)
for index, row in df.iterrows():
    essay_word_count = len(row['Essay'].split())
    essay_word_count = int(1.1 * essay_word_count)
    max_word = int(1.2 * essay_word_count)
    prompt_with_length_command = f"Generate a response between {essay_word_count} to {max_word} words for the following question. {row['prompt']}?"

    start_time = time.time()
    # generated_texts = text_generator(prompt_with_length_command, max_length=max_word, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50, num_beams=2)
    generated_texts = text_generator(prompt_with_length_command, max_length=max_word, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=40, num_beams=2, early_stopping=True)

    end_time = time.time()
    generated_text = generated_texts[0]['generated_text'] if generated_texts else None
    # generated_prompt = generated_text.split("?")[-1]

    df.at[index, 'llama3_text'] = generated_text
    df.at[index, 'llama3_time'] = end_time - start_time
"""
############################################-----    RegEx To extract only the generated essay     ------###########################################################################
"""
import re

# Define the regex pattern
pattern = r"(Generate a response between (\d+) to (\d+) words for the following question\.)((\ *)(\n*))*(.*?\?)(\?*)((\ *)(\n*))*"

# Define a function to clean the text by removing the matched pattern
def remove_pattern(text):
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# Apply the function to the 'passage' column and save the result in a new column
# df['mistral_text'] = df['mistral_text'].apply(remove_pattern)
df['llama3_text'] = df['llama3_text'].apply(remove_pattern)
df.to_pickle('/home/joshb/llm/sonaa/output/DS1.pkl')
print(f"Total script runtime: {time.time() - script_start_time:.2f} seconds")