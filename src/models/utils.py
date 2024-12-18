import hydra
import os
import torch
import logging
from models.transformer import Transformer
from models.selsolcom import SelSolCom
from models.width_regressor import WidthRegressor
from models.dynamic_selector import DynamicSelector
from models.selector_encoder import SelectorEncoder
from models.encselsolcom import EncSelSolCom


def build_model(cfg, vocab):
    if cfg.model.name == "transformer":
        model = build_transformer(cfg.model, cfg.device, vocab)
    elif cfg.model.name == "selsolcom":
        model = build_selsolcom(cfg.model, cfg.device, vocab)
    elif cfg.model.name == "encselsolcom":
        model = build_encselsolcom(cfg.model, cfg.device, vocab)
    elif cfg.model.name == "width_regressor":
        model = build_regressor(cfg.model, cfg.device, vocab)
    elif cfg.model.name == "dynamic_selector":
        model = build_dynamic_selector(cfg.model, cfg.device, vocab)
    elif cfg.model.name == "selector_encoder":
        model = build_selector_encoder(cfg.model, cfg.device, vocab)
    else:
        assert False, "Unknown model."

    return model


def build_selector_encoder_base(cfg_model, device, vocab):
    return SelectorEncoder(
        d_model=cfg_model.d_model,
        ff_mul=cfg_model.ff_mul,
        num_heads=cfg_model.num_heads,
        num_layers_enc=cfg_model.num_layers_enc,
        vocabulary=vocab,
        label_pe_enc=cfg_model.label_pe_enc,
        max_range_pe=cfg_model.max_range_pe,
        diag_mask_width_below=cfg_model.diag_mask_width_below,
        diag_mask_width_above=cfg_model.diag_mask_width_above,
        average_attn_weights=cfg_model.average_attn_weights,
        store_attn_weights=cfg_model.store_attn_weights,
        mha_init_gain=cfg_model.mha_init_gain,
        dropout=cfg_model.dropout,
        use_pe_enc=cfg_model.use_pe_enc,
        device=device,
    ).to(device)


def build_dynamic_selector(cfg_model, device, vocabulary):
    width_regressor = build_regressor(cfg_model.width_regressor, device, vocabulary)
    if cfg_model.width_regressor.ckpt is not None:
        logging.info(
            f"Loading WidthRegressor from ckpt: {cfg_model.width_regressor.ckpt}..."
        )
        width_regressor.load_model_weights(cfg_model.width_regressor.ckpt)
        logging.info("Done.")
    selector = build_transformer(cfg_model.selector, device, vocabulary)
    return DynamicSelector(width_regressor, selector)


def build_selsolcom(cfg_model, device, vocabularies):
    selector = build_transformer(cfg_model.selector, device, vocabularies["selector"])
    solver = build_transformer(cfg_model.solver, device, vocabularies["solver"])
    return SelSolCom(
        selector,
        solver,
        vocabularies["selsolcom"],
        cfg_model.n_multi,
        cfg_model.zoom_selector,
        cfg_model.length_threshold,
    ).to(device)


def build_encselsolcom(cfg_model, device, vocabularies):
    selector_encoder = build_selector_encoder(
        cfg_model.selector_encoder, device, vocabularies["selector"]
    )
    solver = build_transformer(cfg_model.solver, device, vocabularies["solver"])
    return EncSelSolCom(
        selector_encoder, solver, vocabularies["selsolcom"], cfg_model.n_multi
    )


def build_selector_encoder(cfg_model, device, vocabulary):
    if cfg_model.ckpt is not None:
        logging.info(f"Loading model from ckpt: {cfg_model.ckpt}...")
        torch_ckpt = torch.load(
            os.path.join(
                hydra.utils.get_original_cwd(), f"../checkpoints/{cfg_model.ckpt}"
            )
        )
        logging.info(f"Model last saved at iteration n. {torch_ckpt['update']}.")
        if "model_cfg" in torch_ckpt:
            torch_ckpt_model_cfg = torch_ckpt["model_cfg"]
            logging.info("Found model cfg in ckpt, building model using that.")
            cfg = torch_ckpt_model_cfg
            cfg = cast_numbers(cfg)
        else:
            logging.info("Model cfg not found in cfg, building model using input cfg.")
            cfg = cfg_model
        model = build_selector_encoder_base(cfg, device, vocabulary)
        model.load_model_weights(cfg_model.ckpt)
        return model
    else:
        return build_selector_encoder_base(cfg_model, device, vocabulary)


def build_transformer(cfg_model, device, vocabulary):
    if cfg_model.ckpt is not None:
        logging.info(f"Loading model from ckpt: {cfg_model.ckpt}...")
        torch_ckpt = torch.load(
            os.path.join(
                hydra.utils.get_original_cwd(), f"../checkpoints/{cfg_model.ckpt}"
            )
        )
        logging.info(f"Model last saved at iteration n. {torch_ckpt['update']}.")
        if "model_cfg" in torch_ckpt:
            torch_ckpt_model_cfg = torch_ckpt["model_cfg"]
            logging.info("Found model cfg in ckpt, building model using that.")
            if "selector" in torch_ckpt_model_cfg:  # dynamic selector
                torch_ckpt_model_cfg = torch_ckpt_model_cfg["selector"]
            torch_ckpt_model_cfg.deterministic = cfg_model.deterministic
            cfg = torch_ckpt_model_cfg
            cfg = cast_numbers(cfg)
        else:
            logging.info("Model cfg not found in cfg, building model using input cfg.")
            cfg = cfg_model
        model = build_transformer_base(cfg, device, vocabulary)
        model.load_model_weights(cfg_model.ckpt)
        return model
    else:
        return build_transformer_base(cfg_model, device, vocabulary)


def build_transformer_base(cfg_model, device, vocabulary):
    if not "mha_init_gain" in cfg_model:  # backwards compatibility
        import omegaconf

        dict_cfg = dict(cfg_model)
        dict_cfg["mha_init_gain"] = 1
        cfg_model = omegaconf.OmegaConf.create(dict_cfg)

    return Transformer(
        d_model=cfg_model.d_model,
        ff_mul=cfg_model.ff_mul,
        num_heads=cfg_model.num_heads,
        num_layers_enc=cfg_model.num_layers_enc,
        num_layers_dec=cfg_model.num_layers_dec,
        vocabulary=vocabulary,
        label_pe_enc=cfg_model.label_pe_enc,
        label_pe_dec=cfg_model.label_pe_dec,
        deterministic=cfg_model.deterministic,
        n_multi=cfg_model.n_multi,
        temperature=cfg_model.temperature,
        max_range_pe=cfg_model.max_range_pe,
        diag_mask_width_below=cfg_model.diag_mask_width_below,
        diag_mask_width_above=cfg_model.diag_mask_width_above,
        average_attn_weights=cfg_model.average_attn_weights,
        store_attn_weights=cfg_model.store_attn_weights,
        mha_init_gain=cfg_model.mha_init_gain,
        num_recurrent_steps=cfg_model.num_recurrent_steps,
        multi_fwd_threshold=cfg_model.multi_fwd_threshold,
        dropout=cfg_model.dropout,
        device=device,
    ).to(device)


def build_regressor(cfg_model, device, vocabulary):
    return WidthRegressor(
        n_layers=cfg_model.n_layers,
        vocabulary=vocabulary,
        d_model=cfg_model.d_model,
        ff_mul=cfg_model.ff_mul,
        num_heads=cfg_model.num_heads,
        dropout=cfg_model.dropout,
        label_pe=cfg_model.label_pe,
        max_range_pe=cfg_model.max_range_pe,
        device=device,
    ).to(device)


def cast_numbers(cfg_model):
    int_fields = [
        "d_model",
        "ff_mul",
        "num_heads",
        "num_layers_enc",
        "num_layers_dec",
        "diag_mask_width_above",
    ]
    float_fields = ["dropout", "mha_init_gain"]

    for field in int_fields:
        if field in cfg_model and cfg_model[field] is not None:
            cfg_model[field] = int(cfg_model[field])

    for field in float_fields:
        if field in cfg_model and cfg_model[field] is not None:
            cfg_model[field] = float(cfg_model[field])

    return cfg_model
