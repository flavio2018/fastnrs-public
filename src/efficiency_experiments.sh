# fnrs listops
python main.py --config-name test_textseg_nrs_listops_solve model.selector_encoder.ckpt=2024-07-24_21-50_train_textseg_listops.pth model.solver.ckpt=2024-06-15_15-08_train_transformer_listops_solve_atomic.pth model.solver.deterministic=False wandb_disabled=False
# fnrs logic
python main.py --config-name test_textseg_nrs_logic_solve model.selector_encoder.ckpt=2024-07-27_03-38_train_textseg_logic.pth model.solver.ckpt=2024-05-31_14-55_train_transformer_logic_solve_atomic.pth model.solver.deterministic=False wandb_disabled=False
# nrs listops
python main.py --config-name test_selsolcom_listops_solve model.selector.ckpt=2024-06-15_14-02_train_transformer_listops_select.pth model.selector.deterministic=False model.solver.ckpt=2024-06-15_15-08_train_transformer_listops_solve_atomic.pth model.solver.deterministic=False model.zoom_selector=True model.length_threshold=60 wandb_disabled=False data.eval_batch_size=25 model.n_multi=1000
# nrs logic
python main.py --config-name=test_selsolcom_logic_solve model.selector.ckpt=2024-05-30_20-48_train_transformer_logic_select.pth model.selector.deterministic=False model.solver.ckpt=2024-05-31_14-55_train_transformer_logic_solve_atomic.pth model.solver.deterministic=False wandb_disabled=False data.eval_batch_size=4 model.n_multi=10
# nrs algebra
python main.py --config-name test_selsolcom_algebra_solve model.selector.ckpt=2024-06-09_23-34_train_transformer_algebra_select.pth model.selector.deterministic=False model.solver.ckpt=2024-01-04_02-52_train_transformer_algebra_solve_atomic.pth model.solver.deterministic=False model.zoom_selector=True model.length_threshold=60 wandb_disabled=False data.eval_batch_size=25 model.n_multi=1000
# nrs arithmetic
python main.py --config-name test_selsolcom_arithmetic_solve model.selector.ckpt=2024-06-08_12-02_train_transformer_arithmetic_select.pth model.selector.deterministic=False model.solver.ckpt=2024-04-19_07-32_train_transformer_arithmetic_solve_atomic.pth model.solver.d_model=256 model.solver.ff_mul=1 model.solver.num_heads=4 model.solver.num_layers_dec=4 model.solver.num_layers_enc=4 model.solver.deterministic=False model.zoom_selector=True model.length_threshold=50 wandb_disabled=False data.eval_batch_size=25 model.n_multi=1000
# nrs alltask
python main.py --config-name test_selsolcom_alltask_solve model.selector.ckpt=2024-07-04_09-48_train_transformer_alltask_select.pth model.selector.deterministic=False model.solver.ckpt=2024-07-01_17-13_train_transformer_alltask_solve_atomic.pth model.solver.deterministic=False model.zoom_selector=True model.length_threshold=null wandb_disabled=False data.eval_batch_size=25 model.n_multi=100
