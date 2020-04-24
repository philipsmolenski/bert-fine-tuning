import pandas as pd

df_bert_test = pd.read_csv('bert/fine-tune_data/test.tsv', sep='\t')
df_ans = pd.read_csv('bert/fine-tune_data/ans.tsv', sep='\t', header=None)
df_result = pd.read_csv('bert/bert_output/test_results.tsv', sep='\t', header=None)

df_ans.columns = ['x', 'id', 'tactic', 'theorem', 'params']

df_map_result = pd.DataFrame({'guid': df_bert_test['guid'],
    'text': df_bert_test['text'],
    'label': df_result.idxmax(axis=1)})
df_map_result.sample(10)

def evaluate (preds=df_map_result, ans=df_ans):
    gr = ans.groupby('theorem')
    n = preds.shape[0]
    sum = 0.0
    for i in range (n):
        text = preds['text'][i]
        label = preds['label'][i]
        good_tacs = gr.get_group(text)['tactic'].unique()
        if (label in good_tacs):
            sum += 1
            
    return sum / n

print(evaluate())
