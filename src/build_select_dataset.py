#!/usr/bin/env python


import hydra
import pandas as pd
from tqdm import trange
import utils


@hydra.main(config_path="../conf/", version_base='1.2')
def main(cfg):
	utils.make_dir_if_not_exists(cfg)
	gen = utils.get_generator(cfg)
	
	build_set(gen, cfg, 'train', cfg[cfg.dataset_name].splits.train)
	build_set(gen, cfg, 'valid_iid', cfg[cfg.dataset_name].splits.valid_iid)
	build_set(gen, cfg, 'valid_ood', cfg[cfg.dataset_name].splits.valid_ood)
	build_set(gen, cfg, 'test', cfg[cfg.dataset_name].splits.test)
	
	easy = '_easy' if cfg.easy else ''
	utils.dump_config(cfg, f"../datasets/{cfg.dataset_name}{cfg.variant_name}_{cfg.task}{easy}/")


def build_set(generator, cfg, split, splits_parameters):
	if 'train' in split:
		generator_split = 'train'
		num_set_samples = cfg.num_train_samples
	elif 'valid' in split:
		generator_split = 'valid'
		num_set_samples = cfg.num_valid_samples
	elif 'test' in split:
		generator_split = 'test'
		num_set_samples = cfg.num_valid_samples

	SAMPLES_PER_BATCH = 10
	df_dict = {
		'X': [],
		'Y': [],
		'nesting': [],
		'num_operands': [],
	}
	if cfg.dataset_name == 'listops':
		df_dict['extra'] = []

	max_trange = (num_set_samples // len(splits_parameters)) // SAMPLES_PER_BATCH

	for split_parameters in splits_parameters:

		if len(split_parameters) == 2:	# standard param spec
			nesting, num_operands = split_parameters
			task = cfg.task
			extra = None
			generator.easy = True if num_operands < 2 else cfg.easy
		
		elif len(split_parameters) == 3:  # listops train param spec
			nesting, num_operands, extra = split_parameters
			generator.easy = True if extra == 'easy' else cfg.easy
			task = f"{cfg.task}_{extra}" if extra == 'step' else cfg.task
			
		
		for _ in trange(max_trange):
			X, Y = generator.generate_samples(SAMPLES_PER_BATCH, nesting=nesting, num_operands=num_operands, split=generator_split, exact=True, task=task)
			df_dict['X'] += X
			df_dict['Y'] += Y
			df_dict['nesting'] += [nesting for _ in range(SAMPLES_PER_BATCH)]
			df_dict['num_operands'] += [num_operands for _ in range(SAMPLES_PER_BATCH)]
			df_dict['extra'] += [extra for _ in range(SAMPLES_PER_BATCH)]

	total = len(df_dict['X'])
	unique = len(set(df_dict['X']))
	print(f"Num {split} samples: {total}")
	print(f"Num unique {split} samples: {unique} ({unique/total*100:0.0f}%)")
	
	df = pd.DataFrame(df_dict)
	easy = '_easy' if cfg.easy else ''
	df.to_csv(f'../datasets/{cfg.dataset_name}{cfg.variant_name}_{cfg.task}{easy}/{split}.csv', index=False)


if __name__ == '__main__':
	main()