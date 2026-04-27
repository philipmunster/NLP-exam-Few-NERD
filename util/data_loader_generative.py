import pandas as pd
import random



def data_loader_generative(filepath, n, k, test_size, seed):

    #Setting random seed for reproducibility
    random.seed(seed)


    ###################
    #Load test.txt 
    ###################
    sentences, labels = [], []
    current_tokens, current_labels = [], []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    labels.append(current_labels[:])
                    current_tokens, current_labels = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)

    if current_tokens:  # last sentence if no trailing newline
        sentences.append(" ".join(current_tokens))
        labels.append(current_labels[:])

    df = pd.DataFrame({"sentence": sentences, "labels": labels})
    df["classes"] = df["labels"].apply(lambda l: set(x for x in l if x != "O"))



    ###################
    #sample n classes present in the loaded data using the random seed
    ###################
    available_classes = sorted(set(cls for s in df["classes"] for cls in s))
    classes = random.sample(available_classes, n)


    ###################
    #sample examples for each class containing k ~ 2k words with that class label
    ###################
    support_sentences, support_labels = [], []

    for cls in classes:
        candidates = df[df["classes"].apply(lambda s: cls in s)].sample(frac=1, random_state=seed)        
        token_count = 0
        cls_sentences, cls_labels = [], []

        for _, row in candidates.iterrows():
            cls_token_count = sum(1 for l in row["labels"] if l == cls)
            if token_count + cls_token_count > 2 * k:
                continue
            cls_sentences.append(row["sentence"])
            masked_labels = [l if l == cls else "O" for l in row["labels"]]
            cls_labels.append(masked_labels)
            token_count += cls_token_count
            if token_count >= k:
                break

        support_sentences.extend(cls_sentences)
        support_labels.extend(cls_labels)

    support_df = pd.DataFrame({"sentence": support_sentences, "labels": support_labels})



    ###################
    # sample test examples (unseen sentences only)
    ###################
    used_sentences = set(support_df["sentence"])
    test_sentences, test_labels = [], []

    for cls in classes:
        candidates = df[
            df["classes"].apply(lambda s: cls in s) &
            ~df["sentence"].isin(used_sentences)
        ].sample(frac=1, random_state=seed)

        token_count = 0
        cls_sentences, cls_labels = [], []

        for _, row in candidates.iterrows():
            cls_token_count = sum(1 for l in row["labels"] if l == cls)
            if token_count + cls_token_count > 2 * test_size:
                continue
            cls_sentences.append(row["sentence"])
            masked_labels = [l if l == cls else "O" for l in row["labels"]]
            cls_labels.append(masked_labels)
            token_count += cls_token_count
            if token_count >= test_size:
                break

        test_sentences.extend(cls_sentences)
        test_labels.extend(cls_labels)

    test_df = pd.DataFrame({"sentence": test_sentences, "labels": test_labels})

    return classes, support_df, test_df

