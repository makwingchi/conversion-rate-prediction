import json 
import numpy as np

def get_digits(digit_str):
    digit_str = digit_str[1:-1]
    return [eval(item) for item in digit_str.split(' ')]

preds = []
log_keys = []
with open('infer.txt', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if(len(line) < 3 or line[:4] != "time"):
            continue
        items = line.split(",")
        log_key_str = items[2].split(':')[1]
        pred_str = items[3].split(':')[1]
        log_key = get_digits(log_key_str)
        pred = get_digits(pred_str)
        #print(len(log_key))
        #print(log_key[0])
        for j in range(len(log_key)):
            print(str(log_key[j])+","+str(pred[j]))
