import pandas as pd

def concat_datasets(file_name:str):
    train_df = pd.read_csv(f"./datasets/{file_name}_train.csv")
    val_df = pd.read_csv(f"./datasets/{file_name}_val.csv")
    test_df = pd.read_csv(f"./datasets/{file_name}_test.csv")
    df = pd.concat([train_df, val_df, test_df])
    df.to_csv(f"./datasets/{file_name}.csv", index=False)

def count_dataset(file_name:str):
    df = pd.read_csv(f"./datasets/{file_name}.csv")
    print(f"Number of rows in {file_name}: {len(df)}")

def extract_questions(file_name:str):
    df = pd.read_csv(f"./datasets/{file_name}.csv")
    text = df["question"].tolist()
    index = list(range(1, len(text) + 1))
    df = pd.DataFrame({"index": index, "text": text})
    df.to_csv(f"./datasets/questions/{file_name}_questions.csv", index=False)

def print_data(file_name:str):
    df = pd.read_csv(f"./datasets/{file_name}.csv")
    # random sample 10 rows
    df = df.sample(10)
    questions = df["question"].tolist()
    responses = df["response"].tolist()
    for q, r in zip(questions, responses):
        print(f"Question: {q}")
        print(f"Response: {r}")
        print("-" * 100)

if __name__ == "__main__":
    extract_questions("processed_unalign_test")
    extract_questions("processed_unalign_train")
    extract_questions("processed_unalign_val")