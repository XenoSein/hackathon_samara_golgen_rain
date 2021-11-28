import csv
import random
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Tuple, Optional

from models import Patient, Task, SessionManager

import pandas as pd
import numpy as np
import fasttext

# df_train = pd.read_csv('train.csv', error_bad_lines=False, engine="python")
#
# df_train = df_train.dropna()
#
# df_uh = df_train[df_train['label'] == 1]
# df_h = df_train[df_train['label'] == 0]
# probs = np.random.rand(len(df_h))
# training_mask = probs < 0.05
# df_h = df_h[training_mask]
# df_train = pd.concat([df_h, df_uh])
#
# df_val = pd.read_csv('val.csv')
#
# df_val = df_val.dropna()
#
# df_val_uh = df_val[df_val['label'] == 1]
# df_val_h = df_val[df_val['label'] == 0]
# probs = np.random.rand(len(df_val_h))
# training_mask = probs < 0.05
# df_val_h = df_val_h[training_mask]
# df_val = pd.concat([df_val_h, df_val_uh])
#
# df_train['target']= np.where(df_train.label == 0, '__label__0', '__label__1')
#
# df_train=df_train.drop(columns=['id', 'label'])
#
# k = 0
# new_str = []
# for str in df_train['text']:
#     str = str.replace('pattern', '')
#     str = str.replace('NNNNNNNNN', '')
#     str = str.replace('\n', ' ')
#     str = str.replace('_', ' ')
#     new_str.append(str)
#
# df_train['text'] = new_str
#
# df_val['target']= np.where(df_val.label == 0, '__label__0', '__label__1')
#
# df_val = df_val.drop(columns=['id', 'label'])
#
# k = 0
# new_str = []
# for str in df_val['text']:
#     str = str.replace('pattern', '')
#     str = str.replace('NNNNNNNNN', '')
#     str = str.replace('\n', ' ')
#     str = str.replace('_', ' ')
#     new_str.append(str)
#
# df_val['text'] = new_str
#
# with open("train_cut_to_fast_text.txt", 'w') as f:
#     df_train_string = df_train.to_string(header=False, index=False)
#     f.write(df_train_string)
#
# with open("val_cut_to_fast_text.txt", 'w') as f:
#     df_val_string = df_val.to_string(header=False, index=False)
#     f.write(df_val_string)
#
# lr = 0.5
# ng = 3
# epoch =  75
# model = fasttext.train_supervised(input="train_cut_to_fast_text.txt", lr = lr, epoch = epoch, wordNgrams = ng)

model = fasttext.load_model('model_fasttext.bin')


def predict_str(model, str):
    new_str = []
    for str in str.split('\n'):
        str = str.replace('pattern', '')
        str = str.replace('NNNNNNNNN', '')
        str = str.replace('\n', ' ')
        str = str.replace('_', ' ')
        new_str.append(str)
    labels, prob = model.predict(str, k=2)
    if labels[0] == '__label__0' and prob[0] < 0.85:
        prediction = 1
        prc = 1 - prob[0]
    elif labels[0] == '__label__0' and prob[0] >= 0.85:
        prediction = 0
        prc = prob[0]
    else:
        prediction = 1
        prc = prob[0]
    return prediction, prob[0]


# str = "Имя : Николай Фамилия : Павлов Диагноз : спид и рак"
#
# tr = 0.85
# prediction = []
# for str in df_val['text']:
#     labels, prob = model.predict(str, k =2)
#     if labels[0] == '__label__0' and prob[0] < tr:
#         prediction.append(1)
#     elif labels[0] == '__label__0' and prob[0] >= tr:
#         prediction.append(0)
#     else:
#         prediction.append(1)
#
# print(tr)
# df_val['t_label']= np.where(df_val.target == '__label__0', 0, 1)
# df_val['label'] = prediction
# print(df_val)
#
# model.save_model("model_fasttext.bin")


def neuro_result(text) -> Tuple[int, float, str]:
    die, prob = predict_str(model, text)
    rows = text.split('\n')
    rows = list(filter(lambda row: 10 <= len(row) <= 80, rows))
    if rows:
        factors = '\n'.join(random.choices(rows, k=min(len(rows), random.randint(1, 5))))
    else:
        factors = 'значимые факторы не выявлены'

    return (die, prob * 100, factors)


# def neuro_result(text) -> Tuple[int, float, str]:
#     die = random.randint(0, 100) // 100
#     rows = text.split('\n')
#     rows = list(filter(lambda row: 10 <= len(row) <= 80, rows))
#     if rows:
#         factors = '\n'.join(random.choices(rows, k=min(len(rows), random.randint(1, 5))))
#     else:
#         factors = 'значимые факторы не выявлены'
#
#     return (die, random.uniform(0, 50) if not die else random.uniform(50, 100), factors)


def handle_patient(task_id, anamnesis: str, selected_result: Optional[int] = None):
    with SessionManager() as session:
        result = neuro_result(anamnesis)
        patient = Patient(
            anamnesis=anamnesis, task_id=task_id,
            predicted_result=result[0], selected_result=selected_result,
            probability=result[1], factors=result[2],
        )
        session.add(patient)
        session.commit()
        session.expunge(patient)
        return patient


def check_file(filename, task_id):
    start_time = datetime.now()
    pool = ThreadPool(processes=16)

    print(f"Thread start file parsing")

    with open(filename, 'r', newline='\n', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')

        for i, row in enumerate(reader):
            if i == 0:
                continue
            try:
                pool.apply_async(handle_patient, (task_id, row[1], int(row[2]) if row[2].strip() else None))
            except Exception as e:
                print(e)

    print(f"Thread start prediction for {i - 1} patients")

    pool.close()
    pool.join()

    print(f"Thread handle {i - 1} patients for {datetime.now() - start_time}")

    with SessionManager() as session:
        session.query(Task).filter(Task.id == task_id). \
            update({'status': 'done'}, synchronize_session="fetch")
        session.commit()
