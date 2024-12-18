import re

import pandas as pd
from data.vocabulary import Vocabulary


def blank_non_sub_expression(row):
    x = row['X']
    y = row['Y']
    if 'MIN' in row['X'] or 'MAX' in row['X'] or 'SM' in row['X']:
        x = x.replace('MIN', 'I').replace('MAX', 'A').replace('SM', 'S')
        y = y.replace('MIN', 'I').replace('MAX', 'A').replace('SM', 'S')
    start_subexpr = x.find(y)
    end_subexpr = start_subexpr + len(y)
    y_hat = ''
    for idx, x in enumerate(x):
        if idx == (start_subexpr-1) or idx == (end_subexpr):
            y_hat += '/'
        elif start_subexpr <= idx < end_subexpr:
            y_hat += x
        else:
            y_hat += '?'
    return y_hat.replace('I', 'MIN').replace('A', 'MAX').replace('S', 'SM')


def blank_non_sub_expressions(row):
    x = row['X']
    if 'MIN' in x or 'MAX' in x or 'SM' in x:   # listops
        sub_expression_re = re.compile(r'(\[[IAS]\d\d\])|([IAS]\d\d)')
        x = x.replace('MIN', 'I').replace('MAX', 'A').replace('SM', 'S')
    elif 'a' in x or 'b' in x or 'x' in x or 'y' in x:
        if len(x) <= 12 and x.count('(') == 0:  # algebra
            sub_expression_re = re.compile(r'[\-+]?\d{0,2}[\-abxy*]*')
        else:
            sub_expression_re = re.compile(r'\([\-+]?\d{0,2}[\-abxy*]*[\-+][\-+]?\d{0,2}[\-abxy*]*\)')
    else:   # arithmetic
        if len(x) <= 3:
            sub_expression_re = re.compile(r'[\-+]?\d{1,2}')
        else:
            sub_expression_re = re.compile(r'\([\-+]?\d{1,2}[\-*+][\-+]?\d{1,2}\)')

    sub_expressions = sub_expression_re.findall(x)
    if isinstance(sub_expressions[0], tuple):
        sub_expressions = [sub_expression for tup in sub_expressions for sub_expression in tup if sub_expression != '']

    sub_expressions_positions = []
    for sub_expression in sub_expressions:
        start_subexpr = x.find(sub_expression)
        end_subexpr = start_subexpr + len(sub_expression)
        sub_expressions_positions += list(range(start_subexpr, end_subexpr))

    y_hat = ''
    for idx, x in enumerate(x):
        if idx in sub_expressions_positions:
            y_hat += x
        else:
            y_hat += '?'
    return y_hat.replace('I', 'MIN').replace('A', 'MAX').replace('S', 'SM')


def main():
    for task_name in ['listops', 'arithmetic', 'algebra']:
        print(f'Processing task {task_name}')
        for split_name in ['train', 'valid_iid', 'valid_ood', 'test']:
            print(f'Processing split {split_name}')
            df = pd.read_csv(f'../datasets/{task_name}_controlled_select/{split_name}.csv')
            df['Y'] = df.apply(blank_non_sub_expressions, axis=1)
            df.to_csv(f'../datasets/{task_name}_select_enc/{split_name}.csv', index=False)


if __name__ == "__main__":
    main()
