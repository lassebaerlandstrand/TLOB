from models.binctabl import BiN_CTABL
from models.deeplob import DeepLOB
from models.fuselob import FuseLOB
from models.mlplob import MLPLOB
from models.patchlob import PatchLOB  # noqa: F401
from models.original.mlplob import MLPLOB as MLPLOBOriginal
from models.original.tlob import TLOB as TLOBOriginal
from models.tlob import TLOB


def pick_model(
    model_type,
    hidden_dim,
    num_layers,
    seq_size,
    num_features,
    num_heads=8,
    is_sin_emb=False,
    dataset_type=None,
    use_fast_attention=True,
    num_horizons=1,
    dropout=0.0,
    **kwargs,
):
    if model_type == "MLPLOB":
        return MLPLOB(
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            dataset_type,
            num_horizons=num_horizons,
        )
    elif model_type == "MLPLOB_ORIGINAL":
        if num_horizons != 1:
            raise ValueError(
                "MLPLOB_ORIGINAL supports only single-horizon mode (num_horizons must be 1)."
            )
        return MLPLOBOriginal(
            hidden_dim, num_layers, seq_size, num_features, dataset_type
        )
    elif model_type == "TLOB":
        return TLOB(
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            num_heads,
            is_sin_emb,
            dataset_type,
            use_fast_attention=use_fast_attention,
            num_horizons=num_horizons,
            dropout=dropout,
        )
    elif model_type == "TLOB_ORIGINAL":
        if num_horizons != 1:
            raise ValueError(
                "TLOB_ORIGINAL supports only single-horizon mode (num_horizons must be 1)."
            )
        return TLOBOriginal(
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            num_heads,
            is_sin_emb,
            dataset_type,
        )
    elif model_type == "BINCTABL":
        return BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, 3, 1)
    elif model_type == "DEEPLOB":
        return DeepLOB()
    elif model_type == "PATCHLOB":
        return PatchLOB(
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            num_heads,
            is_sin_emb,
            dataset_type,
            use_fast_attention=use_fast_attention,
            num_horizons=num_horizons,
            dropout=dropout,
        )
    elif model_type == "FUSELOB":
        return FuseLOB(
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            num_heads,
            is_sin_emb,
            dataset_type,
            use_fast_attention=use_fast_attention,
            num_horizons=num_horizons,
            dropout=dropout,
            max_events_per_window=kwargs.get("max_events_per_window", 64),
            n_event_features=kwargs.get("n_event_features", 7),
            n_perceiver_queries=kwargs.get("n_perceiver_queries", 8),
            event_encoder_layers=kwargs.get("event_encoder_layers", 2),
            snap_encoder_layers=kwargs.get("snap_encoder_layers", 2),
            event_heads=kwargs.get("event_heads", 4),
        )
    else:
        raise ValueError("Model not found")
