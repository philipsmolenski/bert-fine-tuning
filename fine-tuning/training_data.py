import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bert/fine-tune_data/all.csv')

#Remove duplicates and too long sentences
print(df.shape)
df = df[df['theorem'].str.len().lt(5000)]
print(df.shape)
df = df.drop_duplicates(subset=['tactic', 'theorem'],keep='first')
print(df.shape)

#Split data in such a way that the same theorem does not appear in two different sets
train_inds, test_inds = next(GroupShuffleSplit(test_size=0.1, n_splits=2, random_state=42).split(df, groups=df['theorem']))
df_train = df.iloc[train_inds]
print(df_train.shape)
df_test_dev = df.iloc[test_inds]

eval_inds, test_inds = next(GroupShuffleSplit(test_size=0.5, n_splits=2, random_state=42).split(df_test_dev, groups=df_test_dev['theorem']))
df_test = df_test_dev.iloc[test_inds]
df_dev = df_test_dev.iloc[eval_inds]


df_bert_train = pd.DataFrame({'guid': df_train['id'],
    'alpha': df_train['params'],
    'label': df_train['tactic'],
    'text': df_train['theorem']})

df_bert_dev = pd.DataFrame({'guid': df_dev['id'],
    'alpha': df_dev['params'],
    'label': df_dev['tactic'],
    'text': df_dev['theorem']})

df_bert_test = pd.DataFrame({'guid': df_test['id'],
    'text': df_test['theorem']})

df_bert_train = df_bert_train[['alpha', 'guid', 'label', 'text']]
df_bert_dev = df_bert_dev[['alpha', 'guid', 'label', 'text']]

#No need to evaluate the same sample several times
df_bert_test = df_bert_test.drop_duplicates(subset=['text'],keep='first')

#Save answers
df_test.to_csv('bert/fine-tune_data/ans.tsv', sep='\t', index=False, header=False)

#And train-valid-test sets
df_bert_train.to_csv('bert/fine-tune_data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('bert/fine-tune_data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('bert/fine-tune_data/test.tsv', sep='\t', index=False, header=True)

