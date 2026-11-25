"""The batched-decode attention path must match the float32 reference.

Covers B=1 and B=4, left padding (rows with fewer real keys), full attention
and a sliding window, and a relative-position extent shorter than S.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

Hq, Hkv, D = 24, 8, 128


def case(B, S, window, R, offsets, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    pos_bias = mx.random.normal((B, 1, Hq, R)).astype(mx.bfloat16)
    sinks = mx.random.normal((B, 1, Hq)).astype(mx.bfloat16)
    me = SimpleNamespace(scaling=D**-0.5, sliding_window=window)
    end = mx.array(offsets) if B > 1 else offsets[0]
    cache = SimpleNamespace(offset=end)

    new = layers.Attention._attention(me, q, k, v, pos_bias, sinks, cache)
    qp = layers._positions(end - 1, 1)
    kp = layers._key_positions(cache, S, 1, end)
    ref = layers._attention_reference(
        q, k, v, pos_bias, sinks, scale=me.scaling, window=window,
        query_positions=qp, key_positions=kp,
    )
    err = mx.abs(new.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    scale = mx.abs(ref.astype(mx.float32)).max().item()
    ok = new.shape == ref.shape and err < 2e-2 * max(scale, 1.0)
    print(f"B={B} S={S} window={window} R={R} offsets={offsets}: shape={new.shape} "
          f"max_err={err:.2e} (ref max {scale:.2f}) {'OK' if ok else 'FAIL'}")
    return ok


results = [
    case(1, 300, 0, 128, [300]),
    case(4, 300, 0, 128, [300, 300, 300, 300]),
    case(4, 300, 0, 128, [300, 250, 120, 1]),     # left-padded rows
    case(4, 300, 100, 100, [300, 250, 120, 7]),   # sliding window
    case(4, 2048, 0, 1024, [2048, 2000, 1500, 900], seed=1),
]
print("ALL OK" if all(results) else "SOME FAILED")
