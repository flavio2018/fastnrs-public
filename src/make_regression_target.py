import pandas as pd


def main():
	for set_name in ["train", "valid_iid", "valid_ood"]:
		df = pd.read_csv(f'../datasets/alltask_controlled_select/{set_name}.csv')
		df['y_regr'] = df['X'].apply(y_regr_given_x)
		df.to_csv(f'../datasets/alltask_select_regr/{set_name}.csv', index=False)


def y_regr_given_x(x):
	if '[' in x:	# listops
		return 1
	elif ('a' in x) or ('b' in x) or ('x' in x) or ('y' in x):	# algebra
		return 5
	else:	# arithmetic
		return 3


if __name__ == '__main__':
	main()