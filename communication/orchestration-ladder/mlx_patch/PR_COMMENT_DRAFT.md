### Batched decode makes the Metal buffer cache grow until the GPU runs out of memory

On the PR head (`e2b5bbd`), `mlx_lm.server` with `--decode-concurrency 4` crashes on long generations (16k-token budgets) with:

```
[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

This is on an M4 Max with 128 GB, running `pipenetwork/Inkling-2-Mini-MLX-bf16`. Weights plus KV caches use about 20 GB. The process footprint swings between 23 GB and 66 GB, dropping back each time `BatchGenerator` calls `mx.clear_cache()` (every 512 generation steps). With two servers on one machine, both die at the same time.

**Cause.** `Attention._attention` takes the fused SDPA path for decode only when `B == 1`. A batched decode step (`L == 1, B > 1`) falls through to `_attention_reference`, which casts the whole key cache to float32 and builds several float32 temporaries of size `(B, H, 1, S)`. Each step's temporaries are one token longer than the previous step's, so the buffer cache can never reuse them. Freed buffers pile up until the next clear. Live memory stays under 1 GB, while the cache grows to tens of GB.

**Repro.** Below are 4 copies of one prompt in a `BatchGenerator`, logging `mx.get_active_memory()` / `mx.get_cache_memory()` every 100 steps. "Window peak" is the cache's highest value between two clears (every 512 steps). The 4-stream speeds come from runs on an otherwise idle machine. I can share the script if useful.

| | live memory | buffer cache, window peak | tok/s, 4 streams |
|---|---|---|---|
| B=4, ~1k ctx, head | < 1 GB | 18 GB | 88–95 |
| B=1, ~1k ctx, head | 0.2 GB | 0.2 GB | 17.5 (one stream) |
| B=4, ~1k ctx, patched | < 1 GB | 0.8 GB (2,500 steps) | 76–81 |
| B=4, ~8k ctx, head, `set_cache_limit(8GB)` | 1.8 GB | stays at the 8 GB limit | 67–69 |
| B=4, ~8k ctx, patched, `set_cache_limit(8GB)` | 1.7 GB | 2–4 GB | 76 |

The B=1 run shared the GPU with another server, so its speed isn't comparable to the 4-stream rows. That run was also killed by an out-of-memory error after ~1,000 steps, while its own footprint was about 18 GB: the other server's cache growth had used up the GPU.

**Possible fix.** For `L == 1`, run the fused SDPA once per row with that row's mask and sinks. The `B == 1` case is unchanged.

<details><summary>diff against e2b5bbd</summary>

```diff
@@ -331,24 +331,36 @@
 
         query_positions = _positions(end - L, L)
         key_positions = _key_positions(cache, S, L, end)
-        if L == 1 and B == 1:
-            # The fused attention takes the sinks of one row.
-            distance = query_positions.reshape(1) - key_positions.reshape(S)
+        if L == 1:
+            # The fused attention takes the sinks of one row, so a batched
+            # decode step runs it once per row. The reference path would cast
+            # the whole key cache to float32 every step, and those growing
+            # temporaries can't be reused by the buffer cache.
+            query_positions = mx.broadcast_to(query_positions.reshape(-1, 1), (B, 1))
+            key_positions = mx.broadcast_to(key_positions.reshape(-1, S), (B, S))
             R = position_bias.shape[-1]
-            bias = mx.take(position_bias[0, 0], mx.clip(distance, 0, R - 1), axis=-1)
-            bias = mx.where(distance < R, bias, mx.zeros_like(bias))
-            visible = (distance >= 0) & (key_positions.reshape(S) >= 0)
-            if self.sliding_window:
-                visible = visible & (distance < self.sliding_window)
-            mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
-            attn_output = mx.fast.scaled_dot_product_attention(
-                query,
-                key,
-                value,
-                scale=self.scaling,
-                mask=mask[None, :, None, :],
-                sinks=sinks[0, 0].astype(query.dtype),
-            )
+            rows = []
+            for b in range(B):
+                distance = query_positions[b] - key_positions[b]
+                bias = mx.take(position_bias[b, 0], mx.clip(distance, 0, R - 1), axis=-1)
+                bias = mx.where(distance < R, bias, mx.zeros_like(bias))
+                visible = (distance >= 0) & (key_positions[b] >= 0)
+                if self.sliding_window:
+                    visible = visible & (distance < self.sliding_window)
+                mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
+                rows.append(
+                    mx.fast.scaled_dot_product_attention(
+                        query[b : b + 1],
+                        key[b : b + 1],
+                        value[b : b + 1],
+                        scale=self.scaling,
+                        mask=mask[None, :, None, :],
+                        sinks=sinks[b, 0].astype(query.dtype),
+                    )
+                )
+            attn_output = rows[0] if B == 1 else mx.concatenate(rows, axis=0)
             return attn_output.transpose(0, 2, 1, 3)
```

</details>

**Checks run**
- `tests/models/test_inkling2.py` from the PR head: all 8 tests pass, including `test_batch_matches_single`.
- Against `_attention_reference` on random inputs (B=1 and B=4, left-padded rows, a sliding window, `R < S`), the max absolute error is 2e-3 to 1.6e-2. That's bf16 rounding, the same as the existing `B == 1` fused path shows against the reference.
- Greedy decoding, one prompt alone vs. the same prompt in a batch of 4: the outputs match for the first 669 of 872 characters with the patch, and 569 on head. Some divergence is expected from other batched kernels.
- In a long run (two servers, 4 streams each), each process stays at 19–23 GB, down from swings up to 66 GB.

**Trade-off.** At short context (~1k tokens), the per-row loop is about 15% slower than the reference path. At ~8k it's about 10% faster, and its throughput stays roughly flat as context grows.

**Not resolved.** In one earlier run I also saw `[metal::malloc] Resource limit (499000) exceeded`. This bug predicts thousands of leftover buffers, not ~499k, so that crash may have a separate cause. It hasn't recurred since the patch, but the patched runs so far are short.

Happy to send this as a PR against the branch if that helps.
