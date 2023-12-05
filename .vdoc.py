# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
import openai
from openai import OpenAI
import pandas as pd
import os

# Set your OpenAI API key
api_key = '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'
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()


def clean_address(address):
    """
    Send an address to the GPT API for cleaning and standardization.

    Parameters:
    address (str): The address to be cleaned.

    Returns:
    str: The cleaned and standardized address.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an address expert that familiar with all the USPS address standard."}, 
            {"role": "user", "content": f"Clean and standardize this address. I only need the new address. Do not explain.: {address}"}],
        max_tokens=60
    )
    return response.choices[0].message.content

# Load the CSV file with the addresses
df = pd.read_csv('./unstandardized_addresses.csv')  # Adjust the path to your CSV file

# Cleaning each address in the DataFrame
df['cleaned_address'] = df['address'].apply(clean_address)

# Save the cleaned data to a new CSV
df.to_csv('./cleaned_location_data.csv', index=False)  # Adjust the path as needed

#
#
#
#
#
#
#| eval: false
import openai
from openai import OpenAI
import pandas as pd
import os

# Set your OpenAI API key
api_key = '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'
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

test_sentences = [
    "This new song is lit 🔥🔥🔥",
    "Sigh... I guess today was just not my day 😞",
    "Wow, that's awesome!!! 😃👍",
    "I can't stand this! So frustrating! 😡",
    "IDK what's going on, kinda confused rn 🤷‍♂️",
    "LOL, that was hilarious 😂",
    "Ugh, Mondays are the worst 😫",
    "OMG, I just got the job offer!!! 😍",
    "No way, that's cray cray 😜",
    "Why is everyone so glum? Cheer up! 😊"
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Assume the role of a sentiment analysis specialist. Your task is to evaluate the sentiment of given sentences, categorizing them as positive, neutral, or negative. Each sentence should be scored with a fraction representing its sentiment in each category. Ensure that the combined total of these three fractional scores equals 1 for each sentence, with each score ranging from 0 to 1. Provide a balanced and precise sentiment analysis, reflecting the nuanced emotional content of each statement."}, 
        {"role": "user", "content": f"{test_sentences}"}],
)


print(response.choices[0].message)


#
#
#
#
#
#| eval: false
ChatCompletionMessage(content='[\'This new song is lit 🔥🔥🔥\']: \nPositive: 0.9\nNeutral: 0.1\nNegative: 0.0\n\n[\'Sigh... I guess today was just not my day 😞\']: \nPositive: 0.1\nNeutral: 0.2\nNegative: 0.7\n\n["Wow, that\'s awesome!!! 😃👍"]: \nPositive: 0.9\nNeutral: 0.1\nNegative: 0.0\n\n["I can\'t stand this! So frustrating! 😡"]: \nPositive: 0.1\nNeutral: 0.2\nNegative: 0.7\n\n["IDK what\'s going on, kinda confused rn 🤷\\u200d♂️"]: \nPositive: 0.1\nNeutral: 0.8\nNegative: 0.1\n\n[\'LOL, that was hilarious 😂\']: \nPositive: 0.9\nNeutral: 0.1\nNegative: 0.0\n\n[\'Ugh, Mondays are the worst 😫\']: \nPositive: 0.1\nNeutral: 0.3\nNegative: 0.6\n\n[\'OMG, I just got the job offer!!! 😍\']: \nPositive: 0.9\nNeutral: 0.1\nNegative: 0.0\n\n["No way, that\'s cray cray 😜"]: \nPositive: 0.8\nNeutral: 0.2\nNegative: 0.0\n\n[\'Why is everyone so glum? Cheer up! 😊\']: \nPositive: 0.9\nNeutral: 0.1\nNegative: 0.0', role='assistant', function_call=None, tool_calls=None)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
mamba init
mamba activate 8222env3

pip install torch numpy transformers datasets tiktoken wandb tqdm
#
#
#
#
#
#| eval: false

pwd # check your current working directory

cd /Users/lifengren/github/nanoGPT # change the directory to the nanoGPT folder


#
#
#
#
#
#
#
#| eval: false

python data/shakespeare_char/prepare.py
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false

python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| eval: false
from sentence_transformers import SentenceTransformer
import linktransformer as lt
import pandas as pd
# Example usage of lm_merge_df

data2 = {
    "CompanyName": ["TechCorp", "InfoTech Solutions", "GlobalSoft Inc", "DataTech Co", "SoftSys Ltd", "TechCorp"],
    "Industry": ["Technology", "Technology", "Software", "Data Analytics", "Software", "Technology"],
    "Founded_Year": [2005, 1998, 2010, 2012, 2003, 2005]
}

# Create a DataFrame from the data
df2 = pd.DataFrame(data2)

data1 = {
    "CompanyName": ["Tech Corporation", "InfoTech Soln", "GlobalSoft Incorporated", "DataTech Corporation", 
                    "SoftSys Limited", "TechCorp", "AlphaSoft Systems"],
    "Revenue (Millions USD)": [5000, 4500, 3000, 2500, 4000, 5500, 3800],
    "Num_Employees": [10000, 8500, 6000, 5000, 7500, 12000, 7000],
    "Country": ["USA", "Canada", "India", "Germany", "UK", "USA", "Spain"]
}

# Create a DataFrame from the data
df1 = pd.DataFrame(data1)

df_lm_matched = lt.merge(df1, df2, merge_type='1:m', on="CompanyName", model="all-MiniLM-L6-v2",
left_on=None, right_on=None)

print(df_lm_matched)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
