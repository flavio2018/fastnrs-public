import hydra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@hydra.main(config_path="../conf/", config_name="plot_test_dist", version_base='1.2')
def main(cfg):
	df = pd.read_csv(f'../datasets/{cfg.dataset_name}_controlled_solve/test.csv')
	df['x_len'] = df['X'].apply(str).apply(len)
	ax = sns.histplot(data=df, x='x_len', hue='nesting', bins=50)
	plt.savefig(f'../out/plots/test_set_distribution_{cfg.dataset_name}.png')


if __name__ == '__main__':
	main()