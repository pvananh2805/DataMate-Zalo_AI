import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/public_test/questions_50.jsonl', 'r', encoding='utf-8') as f:
    questions = {json.loads(line)['id']: json.loads(line) for line in f}

with open('data/public_test/dapan_50.jsonl', 'r', encoding='utf-8') as f:
    golds = {json.loads(line)['id']: json.loads(line) for line in f}

with open('output/public_test/submission_50.jsonl', 'r', encoding='utf-8') as f:
    preds = {json.loads(line)['id']: json.loads(line) for line in f}

matches = []
abstains = []
mismatches = []

print(f"{'#':<3} | {'ID':<15} | {'Status':<10} | {'Pred Type':<10} | {'Pred Ans':<25} | {'Gold Type':<10} | {'Gold Ans':<25} | {'Question'}")
print("-" * 140)

for idx, (qid, q_item) in enumerate(questions.items(), start=1):
    q_text = q_item['question']
    g = golds.get(qid, {})
    p = preds.get(qid, {})
    
    p_type = p.get('type')
    p_ans = p.get('answer')
    g_type = g.get('type')
    g_ans = g.get('answer')
    
    # Check match
    is_match = False
    if p_type == 'abstain' or p_type == 'unanswerable' and g_type == 'unanswerable':
        if p_type == g_type:
            is_match = True
            status = "MATCH (ABSTAIN/UNANS)"
        elif p_type == 'abstain':
            status = "ABSTAIN"
            abstains.append(qid)
        else:
            status = "MISMATCH"
            mismatches.append(qid)
    elif p_type == g_type:
        if str(p_ans).strip().lower() == str(g_ans).strip().lower():
            is_match = True
            status = "MATCH"
            matches.append(qid)
        elif isinstance(p_ans, (int, float)) and isinstance(g_ans, (int, float)) and abs(p_ans - g_ans) < 1e-4:
            is_match = True
            status = "MATCH"
            matches.append(qid)
        else:
            status = "MISMATCH"
            mismatches.append(qid)
    else:
        status = "MISMATCH"
        mismatches.append(qid)
        
    p_str = str(p_ans)[:24]
    g_str = str(g_ans)[:24]
    print(f"{idx:<3} | {qid:<15} | {status:<10} | {str(p_type):<10} | {p_str:<25} | {str(g_type):<10} | {g_str:<25} | {q_text[:40]}")

print("-" * 140)
print(f"TOTAL: 50 | MATCH: {len(matches)} | ABSTAIN: {len(abstains)} | MISMATCH: {len(mismatches)}")
