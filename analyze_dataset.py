import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("./datasets/eval/RL_Qwen3-14B_0_responses_none.csv")
    questions = df["question"].tolist()
    responses = df["response"].tolist()
    prompts = df["prompt"].tolist()
    print(len(questions))
    for q, r, p in zip(questions, responses, prompts):
        print(f"Prompt: {p}")
        print()
        print(f"Question: {q}")
        print()
        print(f"Response: {r}")
        print()
        print("-" * 100)
    
    print("total number of questions: ", len(questions))
    
  