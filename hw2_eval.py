import re
import numpy as np
from stqdm import stqdm

from collections import defaultdict
from glob import glob
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer


def f1_score(pred_lines, true_lines):
    def to_dict(lines):
        entities = defaultdict(list)
        file = None
        for line in lines:
            line = line.strip().split()
            if len(line) == 1:  # file_name, e.g. 1136.ann
                file = line[0]
                continue
            if len(line) == 3:  # entity, e.g. PROFESSION 66 73
                tag, l, r = line
                entities[file].append((tag, l, r))
        return entities

    pred_entities = to_dict(pred_lines)
    true_entities = to_dict(true_lines)

    precision, recall, precision_total, recall_total = 0, 0, 0, 0
    for file in true_entities:
        if file not in pred_entities:
            precision_total += len(pred_entities[file])
            recall_total += len(true_entities[file])
            continue

        for entity in set(pred_entities[file]):
            precision_total += 1
            if entity in true_entities[file]:
                precision += 1

        for entity in set(true_entities[file]):
            recall_total += 1
            if entity[1:] in [e[1:] for e in pred_entities[file]]:  # recall проверяет только границы сущности
                recall += 1

    if precision_total > 0:
        precision /= precision_total
    if recall_total > 0:
        recall /= recall_total

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1


def parse_annotations(annotations):
    labels = [re.split(r"[\t\n ]", x.strip()) for x in annotations if ";" not in x]
    labels = [x for x in labels if x[0][0] == "T"]
    labels = sorted(labels, key=lambda x: (int(x[2]), -int(x[3])))

    res = [labels[0]]
    for label in labels:
        if int(res[-1][3]) < int(label[2]):
            res.append(label)
    return res


def preprocess_correct(data):
    res = []
    for file in data:
        with open(file) as f:
            parsed = parse_annotations(f.readlines())
            parsed = [f"{p[1]} {p[2]} {p[3]}" for p in parsed]
        res.append(re.split(r"[/\\]", file)[-1])
        res.extend(parsed)
    return res


def list_files(path):
    all_files = glob(path)
    txt_files = [f for f in all_files if f[-4:] == ".txt"]
    ann_files = {f[:-4] for f in all_files if f[-4:] == ".ann"}
    result = [
        txt[:-4] + ".ann"
        for txt in sorted(txt_files)
        if txt[:-4] in ann_files
    ]
    return result


def predict(pipeline, test_files):
    texts = []
    for path in tqdm(test_files, desc="Prepare test data", mininterval=1):
        path = path[:-4] + ".txt"
        with open(path, encoding="utf-8") as f:
            text = f.read()
        texts.append(text)
    pred = pipeline(texts)
    result = [(f, p, t) for f, p, t in zip(test_files, pred, texts)]
    return postprocess_predicitons(result)


def postprocess_predicitons(pred):
    result = []
    for path, p, t in pred:
        result.append(re.split(r"[/\\]", path)[-1])
        for e in p:
            result.append(f"{e['entity_group']} {e['start']} {e['end']}")
    return result


def evaluate(model_dir, test_path):
    test_data = list_files(test_path)
    true_lines = preprocess_correct(test_data)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, truncation=True)
    print('222')
    tokenizer.model_max_length = 512
    p = pipeline("token-classification", model=model_dir, tokenizer=tokenizer, aggregation_strategy="average", device="cpu", stride=256)
    print('333')
    pred_lines = predict(p, test_data)
    print('444')
    return f1_score(pred_lines, true_lines)


if __name__ == "__main__":
    result = evaluate("folder_model/nerel_finetuned_last_checkpoint", "pages/test/*")
    print(f"F1-score: {np.round(100 * result, 1)}%")
