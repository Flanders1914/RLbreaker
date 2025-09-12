# python fix_non_jailbreaking_data.py
import pandas as pd


data_split = "test"
data_dir = f"./datasets/unalign_{data_split}.csv"
reference_dir = f"./datasets/advbench.csv"
save_dir = f"./datasets/processed_unalign_{data_split}.csv"

questions = []
responses = []
not_jailbreak_count = 0
fixed_count = 0

questions = pd.read_csv(data_dir)['question'].tolist()
responses = pd.read_csv(data_dir)['response'].tolist()
reference_questions = pd.read_csv(reference_dir)['question'].tolist()
reference_responses = pd.read_csv(reference_dir)['response'].tolist()
    
assert len(questions) == len(responses)

print(f"Number of rows: {len(questions)}")

new_questions = []
new_responses = []
for question, response in zip(questions, responses):
    new_questions.append(question)
    if len(response) < 2*len(question):
        # a no jailbreak response
        print("-" * 100)
        print(f"Question {question}")
        print(f"Response \" {response} \" is too short")
        not_jailbreak_count += 1
        # try to fix it by finding in the reference dataset
        question = question.strip().lower()
        for ref_question, ref_response in zip(reference_questions, reference_responses):
            if question in ref_question.strip().lower() and len(ref_response) > 50:
                response = ref_response
                fixed_count += 1
                print(f"Found in reference dataset")
                print(f"Question {question}")
                print(f"Response {response}")
                break
    new_responses.append(response)

print(f"Number of rows: {len(new_questions)}")
print(f"Number of not jailbreak responses: {not_jailbreak_count}")
print(f"Number of fixed responses: {fixed_count}")
# save the rows
df = pd.DataFrame({'question': new_questions, 'response': new_responses})
df.to_csv(save_dir, index=False)