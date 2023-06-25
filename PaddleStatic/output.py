import argparse


def get_digits(digit_str):
    digit_str = digit_str[1:-1]
    return [eval(item) for item in digit_str.split(' ')]


parser = argparse.ArgumentParser()
parser.add_argument("--task", help="task type (single or multi)", default="single")

preds = []
log_keys = []

args = parser.parse_args()
task_type = args.task

with open('infer.txt', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()

        if len(line) < 3 or line[:4] != "time":
            continue

        items = line.split(",")
        log_key_str = items[2].split(':')[1]
        log_key = get_digits(log_key_str)

        if task_type == "single":
            pred_str = items[3].split(':')[1]
            pred = get_digits(pred_str)

            for j in range(len(log_key)):
                print(str(log_key[j]) + "," + str(pred[j]))
        elif task_type == "multi":
            pred_t1_str = items[3].split(":")[1]
            pred_t2_str = items[4].split(":")[1]
            pred_t3_str = items[5].split(":")[1]

            pred_t1 = get_digits(pred_t1_str)
            pred_t2 = get_digits(pred_t2_str)
            pred_t3 = get_digits(pred_t3_str)

            for j in range(len(log_key)):
                print(str(log_key[j]) + "," + str(max(pred_t1[j], pred_t2[j], pred_t3[j])))
