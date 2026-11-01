import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

ground_truth_file = 'data/public_test/dapan.jsonl'
submission_file = sys.argv[1] if len(sys.argv) > 1 else 'output/public_test/submission_10.jsonl'

gt = {}
with open(ground_truth_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        gt[item['id']] = item

sub = {}
with open(submission_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        sub[item['id']] = item

total = len(gt)
correct = 0
incorrect = 0
abstained = 0

print('=' * 80)
print(f'EVALUATING: {submission_file} against {ground_truth_file}')
print('=' * 80)

for qid, target in gt.items():
    pred = sub.get(qid)
    target_type = target.get('type')
    target_ans = target.get('answer')
    if pred is None:
        status = 'MISSING'
        incorrect += 1
    elif pred.get('type') == 'abstain' and target_type == 'abstain':
        status = 'CORRECT (ABSTAIN)'
        correct += 1
    elif pred.get('type') == 'abstain' and target_type != 'abstain':
        status = 'ABSTAINED (0 PTS)'
        abstained += 1
    elif pred.get('type') == target_type:
        pred_ans = pred.get('answer')
        if target_type == 'number':
            try:
                is_match = abs(float(pred_ans) - float(target_ans)) < 1e-2
            except:
                is_match = False
        elif target_type == 'list':
            p_set = set(str(x).strip().lower() for x in (pred_ans if isinstance(pred_ans, list) else []))
            t_set = set(str(x).strip().lower() for x in (target_ans if isinstance(target_ans, list) else []))
            is_match = (p_set == t_set) or (len(t_set) > 0 and len(p_set & t_set) >= len(t_set) * 0.8)
        else:
            is_match = (pred_ans == target_ans)
        if is_match:
            status = 'CORRECT (+1)'
            correct += 1
        else:
            status = 'WRONG (-1)'
            incorrect += 1
    else:
        status = 'WRONG TYPE (-1)'
        incorrect += 1
    pred_str = pred.get('answer') if pred else 'None'
    pred_type = pred.get('type') if pred else 'None'
    print(f'[{qid}] {status}')
    print(f'   Target: type={target_type}, ans={target_ans}')
    print(f'   Pred  : type={pred_type}, ans={pred_str}')
    print('-' * 80)

score = correct * 1 - incorrect * 1
print(f'TOTAL: {total} | CORRECT: {correct} | INCORRECT: {incorrect} | ABSTAINED: {abstained}')
print(f'SCORE: {score}/{total}')
