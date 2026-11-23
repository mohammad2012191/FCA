#!/usr/bin/env python3
"""
fca_video.py: Factorized Context Attention (FCA) on long video, with the read of GridProbe.

K*K frames are sampled and placed on a K x K grid, one frame per block. GridProbe answers the question once per row and once per
column as 2K separate prompts, builds a map from the confidences, and answers once more on the frames the map selects. FCA keeps that
read and computes the 2K probes from one encoding of the frames, kept per video and reused by every question on it.

Arms (--arms): mono (one pass over the K*K frames), mono_cache (the same with a prefix cache), gridprobe (GridProbe as published),
fca (this paper). Models: Qwen3-VL and InternVL3.5 (the -HF checkpoints). Benchmarks: lvb (LongVideoBench), video_mme.
--needle_json runs the video needle test, --sparse_prefill xattention and --compressor NAME run baselines of the monolithic read
with the released code of those methods on PYTHONPATH.

Requires the public code of GridProbe as the package GridProbe (its eval folder) next to this file.

  python fca_video.py --check                                                  # equivalence tests on tiny random models, CPU
  python fca_video.py --benchmark video_mme --data_dir DATA/Video-MME --duration_bin long --arms mono,gridprobe,fca
  python fca_video.py --benchmark video_mme --arms mono,gridprobe,fca --summary
"""
import argparse, json, math, os, sys, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "GridProbe"))

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", default="lvb", choices=["lvb", "video_mme", "video_mme_v2"])
ap.add_argument("--lvb_json", default=None)
ap.add_argument("--lvb_dir", default=None)
ap.add_argument("--data_dir", default=None, help="Video-MME / Video-MME-v2 root")
ap.add_argument("--duration_bin", default=None, help="keep one duration bin (Video-MME long: long)")
ap.add_argument("--with_subtitle", action="store_true", help="subtitles in the request, as GridProbe on LongVideoBench")
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--attn", default="flash_attention_2")
ap.add_argument("--arms", default="mono,fca", help="from mono, mono_cache, gridprobe, fca")
ap.add_argument("--K", type=int, default=12)
ap.add_argument("--probe_pixels", type=int, default=224 * 224, help="probe and store resolution, GridProbe's 224x224")
ap.add_argument("--focus_pixels", type=int, default=0, help="focused pass resolution, 0 = native (GridProbe)")
ap.add_argument("--selector", default="skew", help="GridProbe's frame-budget rule")
ap.add_argument("--pr_min_M", type=int, default=2)
ap.add_argument("--pr_max_M", type=int, default=None)
ap.add_argument("--pr_gamma", type=float, default=0.25)
ap.add_argument("--pr_subtract_min", action="store_true", default=True)
ap.add_argument("--copy_batch", type=int, default=24, help="probe sequences per forward")
ap.add_argument("--fps", type=float, default=0.0, help="clock of the stored timestamps; 0 = the one qwen_vl_utils gives the monolithic read")
ap.add_argument("--needle_json", default=None, help="needle test: a jsonl of {image, question, answers} (AnswerMap/prep_vqa.py's TextVQA file); one "
                "needle image replaces a frame of each haystack video, and the question is asked about it with four options")
ap.add_argument("--needle_images", default=None, help="folder of the needle images")
ap.add_argument("--needle_n", type=int, default=33, help="needles per length (one haystack video each, in the benchmark's order); 33 = three per "
                "depth, as the text needle test")
ap.add_argument("--needle_depths", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1", help="needle positions as a share of the video, cycled over "
                "the needles (the 11 depths of the text needle test)")
ap.add_argument("--needle_Ks", default="", help="needle test: grid sizes run one after another with one model load, e.g. 12,16,24,32,45 for 144 to "
                "2,025 frames (replaces --K); the same needles, haystacks and depths at every length")
ap.add_argument("--mono_frames", type=int, default=0, help="needle test: the monolithic arms read this many uniformly spaced frames of the "
                "K*K (0 = all of them); for a model whose window cannot hold K*K frames")
ap.add_argument("--visual_batch", type=int, default=0, help="frames per vision-tower call when building the store (0 = all at once)")
ap.add_argument("--encode_batch", type=int, default=0, help="local spans per language-model call when building the store (0 = all at once)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--n_shards", type=int, default=1)
ap.add_argument("--shard_index", type=int, default=0)
ap.add_argument("--out_dir", default="runs_video")
ap.add_argument("--summary", action="store_true")
ap.add_argument("--flops", action="store_true", help="TFLOPs per question of the saved rows of --arms with GridProbe's language-model counter "
                "(processor only, no model), then exit")
ap.add_argument("--check", action="store_true")
ap.add_argument("--sparse_prefill", default="none", choices=["none", "xattention"], help="XAttention (ICML 2025) as the prefill of the mono and "
                "mono_cache arms, in its own --out_dir; its released code (x-attention, Block-Sparse-Attention) must be on PYTHONPATH")
ap.add_argument("--xattn_stride", type=int, default=8, help="XAttention: antidiagonal stride")
ap.add_argument("--xattn_threshold", type=float, default=0.9, help="XAttention: block threshold for every head")
ap.add_argument("--check_sparse", action="store_true", help="on the GPU: XAttention through the hook on the benchmark's first video against "
                "the dense read, then exit")
ap.add_argument("--compressor", default="none", choices=["none", "vidcom2", "fastvid", "visionzip", "holitom", "dycoke"], help="training-free token "
                "reduction of the monolithic read of Qwen3-VL (--arms mono only, in its own --out_dir): the method's Qwen3VLModel_forward from the qwen "
                "branch of github.com/xuyang-liu16/VidCom2, which must be on PYTHONPATH, attached as that repo's lmms-eval wrapper attaches it")
ap.add_argument("--r_ratio", type=float, default=0.25, help="--compressor: share of the video tokens kept (the repo's default)")
args = ap.parse_args()
ARMS = [a for a in args.arms.split(",") if a]


# ============================================================== the store and the probes (model-level, no data code)
class FCAVideo:
    """FCA over a Qwen3-VL language model. prefix_ids: the chat template up to the video. ts_ids(f): the timestamp tokens of frame f.
    A book holds the layout of one video (blocks, positions, store offsets) and its store of keys and values."""

    def __init__(self, model, prefix_ids, dev):
        import torch
        self.t, self.model, self.dev = torch, model, dev
        cfg = model.config; tc = cfg.text_config
        self.lm, self.head = model.model.language_model, model.lm_head
        self.vs, self.ve, self.vp = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.video_token_id
        self.n_layers, self.H = tc.num_hidden_layers, tc.num_key_value_heads
        self.d = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        self.merge = cfg.vision_config.spatial_merge_size; self.prefix_ids = list(prefix_ids); self._pad_noise = False
        self.R = 3                                                                              # rotary axes (multi-axis rotary positions)

    def lm_call(self, ie, att, pos, vm, ds, use_cache):
        """The language model on a padded batch: Qwen3-VL takes (3, B, L) positions and the DeepStack features, a 1-D model (B, L)."""
        if self.R == 3: return self.lm(inputs_embeds=ie, attention_mask=att, position_ids=pos, visual_pos_masks=vm, deepstack_visual_embeds=ds, use_cache=use_cache)
        return self.lm(inputs_embeds=ie, attention_mask=att, position_ids=pos[0], use_cache=use_cache)

    def visual(self, pixel_values_videos, video_grid_thw, batch=0):
        """Merged visual tokens and DeepStack features of every frame: (n, g, D) and a list of (n, g, D), plus the merged grid (gh, gw).
        Every frame is one temporal group, and the vision tower attends within a group only, so a frame's tokens do not depend on the
        other frames (checked). batch > 0 runs the tower on that many frames at a time (the patches are stored frame by frame), which
        gives the same tokens (checked)."""
        t = self.t
        if batch and video_grid_thw.shape[0] == 1 and int(video_grid_thw[0, 0]) > batch:
            T, h, w = (int(x) for x in video_grid_thw[0]); per = h * w; assert pixel_values_videos.shape[0] == T * per, (pixel_values_videos.shape, T, h, w)
            parts = [self.visual(pixel_values_videos[a * per:min(a + batch, T) * per], t.tensor([[min(a + batch, T) - a, h, w]])) for a in range(0, T, batch)]
            return t.cat([x[0] for x in parts], 0), [t.cat([x[1][i] for x in parts], 0) for i in range(len(parts[0][1]))], parts[0][2]
        with t.no_grad():
            out = self.model.model.get_video_features(pixel_values_videos.to(self.dev), video_grid_thw.to(self.dev))
        emb, deep = (out[0], out[1]) if isinstance(out, tuple) else (out.pooler_output, out.deepstack_features)
        if isinstance(emb, (list, tuple)): emb = t.cat(list(emb), 0)
        n = int(video_grid_thw[:, 0].sum()); gh, gw = int(video_grid_thw[0, 1]) // self.merge, int(video_grid_thw[0, 2]) // self.merge
        return emb.view(n, gh * gw, -1), [x.view(n, gh * gw, -1) for x in deep], (gh, gw)

    def layout(self, n, K, ts_ids, grid):
        """Blocks of the n frames in reading order and their stair positions. A block is the frame's timestamp, <vision_start>, its
        gh*gw visual tokens and <vision_end>, with Qwen3-VL's positions inside it (text sequential; the group at a constant temporal
        index, height and width offsets; the next text max(gh, gw) on). Frame f sits at row f // K, column f % K, and its block starts
        at P + (row + column) * stride on all three axes. The store is token-major: prefix rows, then the blocks in frame order, so a row
        (local span) is one run of the store."""
        gh, gw = grid; G = max(gh, gw); P = len(self.prefix_ids); ids, pos, voff, ext = [], [], [], []
        for f in range(n):
            ts = list(ts_ids(f)); s = len(ts) + 1
            ids.append(ts + [self.vs] + [self.vp] * (gh * gw) + [self.ve]); voff.append(s); ext.append(s + G + 1)
            txt = list(range(s)); r = np.repeat(np.arange(gh), gw); c = np.tile(np.arange(gw), gh)
            pos.append(np.stack([np.array(txt + [s] * (gh * gw) + [s + G]),
                                 np.array(txt + list(s + r) + [s + G]), np.array(txt + list(s + c) + [s + G])]))
        stride = max(ext); base = np.array([P + (f // K + f % K) * stride for f in range(n)])
        lens = np.array([len(x) for x in ids]); offs = P + np.concatenate([[0], np.cumsum(lens[:-1])])
        local = [list(range(i * K, min(i * K + K, n))) for i in range(math.ceil(n / K))]
        strided = [list(range(j, n, K)) for j in range(K) if j < n]
        return {"n": n, "K": K, "P": P, "ids": ids, "pos": pos, "voff": voff, "ext": np.array(ext), "stride": stride, "base": base,
                "lens": lens, "offs": offs, "local": local, "strided": strided}

    def sequence(self, book, members, req_ids=None, q_start=None):
        """ids, (3, L) positions and the visual slots [(start, frame)] of the prompt, the members' blocks and optionally a request."""
        P = book["P"]; ids = list(self.prefix_ids); pos = [np.tile(np.arange(P), (self.R, 1))]; slots = []
        for m in members:
            slots.append((len(ids) + book["voff"][m], m)); ids += book["ids"][m]; pos.append(book["pos"][m] + int(book["base"][m]))
        if req_ids is not None: ids += list(req_ids); pos.append(np.tile(q_start + np.arange(len(req_ids)), (self.R, 1)))
        return ids, np.concatenate(pos, 1), slots

    def embeds(self, seqs, emb, deep):
        """A right-padded batch: inputs_embeds, attention mask, (3, B, L) positions, the visual mask, and the DeepStack features in the
        order of the visual mask (row-major over the batch, as the language model reads them)."""
        t = self.t; B, Lm = len(seqs), max(len(s[0]) for s in seqs); E = self.lm.embed_tokens
        ids = t.zeros((B, Lm), dtype=t.long); att = t.zeros((B, Lm), dtype=t.long); pos = t.zeros((self.R, B, Lm), dtype=t.long)
        vmask = t.zeros((B, Lm), dtype=t.bool); g = emb.shape[1]; rows = []
        for b, (sid, spos, slots) in enumerate(seqs):
            L = len(sid); ids[b, :L] = t.tensor(sid); att[b, :L] = 1; pos[:, b, :L] = t.from_numpy(spos); pos[:, b, L:] = int(spos.max()) + 1
            for s0, f in slots: vmask[b, s0:s0 + g] = True; rows.append((b, s0, f))
        with t.no_grad(): ie = E(ids.to(self.dev))
        for b, s0, f in rows: ie[b, s0:s0 + g] = emb[f].to(ie.dtype)
        order = [f for b, s0, f in sorted(rows)]
        ds = [t.cat([x[f] for f in order], 0).to(ie.dtype) for x in deep] if order else None
        return ie, att.to(self.dev), pos.to(self.dev), vmask.to(self.dev), ds

    def encode(self, book, emb, deep, batch=0):
        """The store: one causal pass per local span, keys and values of every layer copied out. The rows run in one right-padded batch,
        or batch of them at a time (a row sees only itself, so the store is the same: checked)."""
        t = self.t; P = book["P"]; rows = book["local"]; step = batch or len(rows)
        buf = t.empty((2, self.n_layers, P + int(book["lens"].sum()), self.H, self.d), dtype=self.model.dtype, device=self.dev)
        for r0 in range(0, len(rows), step):
            part = rows[r0:r0 + step]; seqs = [self.sequence(book, members) for members in part]; ie, att, pos, vm, ds = self.embeds(seqs, emb, deep)
            with t.no_grad(): out = self.lm_call(ie, att, pos, vm, ds, True)
            cache = out.past_key_values
            for r, members in enumerate(part):
                a = int(book["offs"][members[0]]); L = int(sum(book["lens"][m] for m in members))
                for l in range(self.n_layers):
                    k, v = cache.layers[l].keys[r], cache.layers[l].values[r]                   # (H, L, d)
                    if r0 + r == 0: buf[0, l, :P].copy_(k[:, :P].transpose(0, 1)); buf[1, l, :P].copy_(v[:, :P].transpose(0, 1))
                    buf[0, l, a:a + L].copy_(k[:, P:P + L].transpose(0, 1)); buf[1, l, a:a + L].copy_(v[:, P:P + L].transpose(0, 1))
            del out, cache, ie, ds
        book["buf"] = buf; return book

    def gather(self, book, spans):
        """A batched cache for B spans: per layer (B, kv heads, T_max, d), every span right-aligned (prefix, then its blocks in order),
        pad slots zero and masked."""
        t = self.t; from transformers import DynamicCache
        buf, offs, lens, P = book["buf"], book["offs"], book["lens"], book["P"]; B = len(spans)
        T = [P + int(sum(lens[m] for m in members)) for members in spans]; Tm = max(T); att = t.zeros((B, Tm), dtype=t.long); rows = []
        for b, members in enumerate(spans):
            dst = Tm - T[b]; att[b, dst:] = 1; runs = [(0, P)]
            for m in members:
                a0, a1 = int(offs[m]), int(offs[m] + lens[m])
                if a0 == runs[-1][1]: runs[-1] = (runs[-1][0], a1)
                else: runs.append((a0, a1))
            row = []
            for a0, a1 in runs: row.append((a0, a1, dst)); dst += a1 - a0
            rows.append(row)
        cache = DynamicCache()
        for l in range(self.n_layers):
            Kb = t.zeros((B, Tm, self.H, self.d), dtype=buf.dtype, device=self.dev); Vb = t.zeros_like(Kb)
            if self._pad_noise: Kb.normal_(); Vb.normal_()                                    # self-test only: the pads must be masked
            for b, row in enumerate(rows):
                for a0, a1, dst in row: Kb[b, dst:dst + a1 - a0].copy_(buf[0, l, a0:a1]); Vb[b, dst:dst + a1 - a0].copy_(buf[1, l, a0:a1])
            cache.update(Kb.transpose(1, 2), Vb.transpose(1, 2), l); del Kb, Vb
        return cache, att

    def q_start(self, book, members): m = members[-1]; return int(book["base"][m] + book["ext"][m])

    def ask(self, book, spans, req_ids):
        """The request after each of B spans in one forward: last-token logits (B, vocab)."""
        t = self.t; cache, att = self.gather(book, spans); B, q = len(spans), len(req_ids)
        pos = (t.tensor([self.q_start(book, m) for m in spans])[:, None] + t.arange(q)[None]).to(self.dev)
        with t.no_grad():
            out = self.lm(input_ids=t.tensor([list(req_ids)] * B, device=self.dev), position_ids=pos[None].expand(3, B, q) if self.R == 3 else pos,
                          attention_mask=t.cat([att, t.ones((B, q), dtype=t.long)], 1).to(self.dev), past_key_values=cache, use_cache=True)
            lg = self.head(out.last_hidden_state[:, -1])
        del cache; return lg

    def reference(self, book, members, req_ids, emb, deep):
        """Self-test: one plain causal pass over the prompt, the members' blocks and the request, at the same positions."""
        t = self.t; seq = self.sequence(book, members, req_ids, self.q_start(book, members)); ie, att, pos, vm, ds = self.embeds([seq], emb, deep)
        with t.no_grad(): out = self.lm_call(ie, att, pos, vm, ds, False)
        return self.head(out.last_hidden_state[:, -1])[0]

    def probes(self, book, req_ids, B):
        """Logits of the K local and the K strided sequences, in that order."""
        spans = book["local"] + book["strided"]; out = []
        for b in range(0, len(spans), B): out.append(self.ask(book, spans[b:b + B], req_ids))
        return self.t.cat(out, 0), len(book["local"])


class FCAInternVL(FCAVideo):
    """FCA over an InternVL language model: 1-D rotary positions, no DeepStack. A block is one frame as the monolithic read writes it
    (<img>, its 256 visual tokens, </img> and the newline after it), cut from the tokenized monolithic prompt; frame f sits at row f // K,
    column f % K, and its block starts at P + (row + column) * stride with consecutive positions inside it."""

    def __init__(self, model, prefix_ids, dev):
        import torch
        self.t, self.model, self.dev = torch, model, dev
        cfg = model.config; tc = cfg.text_config
        self.lm, self.head = model.model.language_model, model.lm_head
        self.vp = cfg.image_token_id; self.n_layers, self.H = tc.num_hidden_layers, tc.num_key_value_heads
        self.d = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        self.prefix_ids = list(prefix_ids); self._pad_noise = False; self.R = 1

    def visual(self, pixel_values, batch=0):
        """Visual tokens of every frame, (n, 256, D): InternVL encodes each frame (one 448 x 448 tile) alone, so batch > 0 frames per
        call give the same tokens."""
        t = self.t; cfg = self.model.config
        if batch and pixel_values.shape[0] > batch:
            return t.cat([self.visual(pixel_values[a:a + batch])[0] for a in range(0, pixel_values.shape[0], batch)], 0), [], None
        with t.no_grad():
            out = self.model.model.get_image_features(pixel_values=pixel_values.to(self.dev, self.model.dtype), vision_feature_layer=cfg.vision_feature_layer,
                                                      vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        if not t.is_tensor(out): out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        return out, [], None

    def layout(self, n, K, block, grid=None):
        """Blocks of the n frames (block(f): the frame's token ids) at stair positions; the store is token-major as for Qwen3-VL."""
        P = len(self.prefix_ids); ids = [list(block(f)) for f in range(n)]; ext = np.array([len(x) for x in ids]); stride = int(ext.max())
        return {"n": n, "K": K, "P": P, "ids": ids, "pos": [np.arange(e)[None] for e in ext], "voff": [x.index(self.vp) for x in ids], "ext": ext,
                "stride": stride, "base": np.array([P + (f // K + f % K) * stride for f in range(n)]), "lens": ext.copy(),
                "offs": P + np.concatenate([[0], np.cumsum(ext[:-1])]), "local": [list(range(i * K, min(i * K + K, n))) for i in range(math.ceil(n / K))],
                "strided": [list(range(j, n, K)) for j in range(K) if j < n]}


def grid_read(row_lp, col_lp, letters):
    """GridProbe's stage-1 read (two_stage_eval.stage1_probe) on given letter log-probs: confidences, map, ranking, ensemble."""
    import torch, torch.nn.functional as F
    row_confs = F.softmax(row_lp, dim=-1).max(dim=-1).values.numpy(); col_confs = F.softmax(col_lp, dim=-1).max(dim=-1).values.numpy()
    M = row_confs[:, None] * col_confs[None, :]; K = M.shape[1]; order = np.argsort(M.flatten())[::-1]
    return {"row_lp": row_lp, "col_lp": col_lp, "row_confs": row_confs, "col_confs": col_confs, "M_mult": M,
            "row_preds": [letters[i] for i in row_lp.argmax(-1).tolist()], "col_preds": [letters[i] for i in col_lp.argmax(-1).tolist()],
            "golden": [(int(i // K), int(i % K)) for i in order], "ensemble_pred": letters[torch.cat([row_lp, col_lp], 0).mean(0).argmax().item()]}


def needle_place(n, depth, tps):
    """First frame of the needle: the frame at this depth, moved back onto a temporal patch of tps frames and inside the video."""
    pos = min(int(round(depth * (n - 1))), n - tps); return pos - pos % tps


# ============================================================== XAttention, the sparse-prefill baseline of the monolithic read
class SparsePrefill:
    """XAttention (Xu et al., ICML 2025) on the monolithic read, as fca_text.py runs it: a wrapper registered in transformers' attention
    registry under the backend's name, so every attention call passes through it and nothing in the model changes. It is off unless an arm
    switches it on (mono, mono_cache). When on, a call from a language-model layer over one unpadded sequence whose queries and keys are the
    same positions (the whole prompt, or the video prefix of the cached read) goes to the released Xattention_prefill with the key-value heads
    repeated; every other call goes to the backend: the vision tower (it calls the same registry, non-causal within a frame), a question over
    the cache, decoding. The share of causal 128 x 128 blocks it keeps is read from the block mask it hands its kernel, for the TFLOPs pass."""

    def __init__(self, model, backend, stride, threshold, fns=None):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.t, self.registry, self.backend, self.orig = torch, ALL_ATTENTION_FUNCTIONS, backend, ALL_ATTENTION_FUNCTIONS[backend]
        self.layer_of = {id(l.self_attn): i for i, l in enumerate(model.model.language_model.layers)}
        self.stride, self.thr, self.on = stride, threshold, False; self.reset()
        self.prefill = fns["xattention_prefill"] if fns else self.release()
        type(ALL_ATTENTION_FUNCTIONS).register(backend, self)

    def release(self):
        """The released code, unchanged; its kernel call is wrapped to read the block mask it receives (no extra compute)."""
        import xattn.src.Xattention as X
        kernel = getattr(X, "block_sparse_attn_func", None)
        assert kernel is not None, "xattn.src.Xattention no longer imports block_sparse_attn_func at module level: the kept share cannot be read"
        def counted(*a, **kw):
            m = next((x for x in list(a) + list(kw.values()) if self.t.is_tensor(x) and x.dtype == self.t.bool and x.dim() == 4), None)
            if m is not None: self.count(m)
            return kernel(*a, **kw)
        X.block_sparse_attn_func = counted
        return X.Xattention_prefill

    def reset(self):
        self.calls, self.other, self.kept, self.total = 0, 0, 0, 0

    def count(self, m):
        """Kept blocks of a (batch, heads, query blocks, key blocks) mask, against the causal blocks (the queries end where the keys end)."""
        b, h, nq, nk = m.shape; self.kept = self.kept + m.sum(); self.total += b * h * (nq * (nk - nq) + nq * (nq + 1) // 2)

    def density(self):
        return float(self.kept) / self.total if self.total else None

    def remove(self):
        type(self.registry).register(self.backend, self.orig)

    def __call__(self, module, query, key, value, attention_mask, **kw):
        b, h, q_len, d = query.shape; lm = id(module) in self.layer_of
        whole = (self.on and lm and b == 1 and q_len > 1 and key.shape[2] == q_len
                 and (attention_mask is None or attention_mask.dim() != 2 or bool(attention_mask.all())))
        if not whole:
            self.other += int(self.on and not lm)                                                    # vision-tower calls while on
            return self.orig(module, query, key, value, attention_mask, **kw)
        n_rep = h // key.shape[1]
        k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key                                 # transformers' repeat_kv: kv head g serves query heads g*n_rep onwards
        v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
        scaling = kw.get("scaling") or d ** -0.5
        assert abs(scaling - d ** -0.5) < 1e-6, "XAttention's kernels use 1/sqrt(head_dim)"
        out = self.prefill(query, k, v, stride=self.stride, norm=1, threshold=self.thr, use_triton=True)   # (1, heads, q, d), blocks of 128
        self.calls += 1
        return out.transpose(1, 2).contiguous(), None


# ============================================================== self-test on a tiny random Qwen3-VL (CPU, fp32)
def checks():
    """The store path must equal an explicit causal read of the same prompt, blocks and request at the same positions for every local
    span; a strided span must differ from a read in which its blocks see each other (the one approximation); batches of mixed lengths
    with noise in the pad slots must equal one-at-a-time; a frame's visual tokens must not depend on the other frames; and the block
    layout must give Qwen3-VL's own positions (get_rope_index) when the blocks are laid out as one video. Says nothing about accuracy."""
    import torch as t
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    import transformers
    t.manual_seed(0); rng = np.random.default_rng(0)
    rope = {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True}     # transformers 5 reads rope_parameters, 4.57 rope_scaling
    rope = ({"rope_parameters": {**rope, "rope_theta": 10000.0}} if int(transformers.__version__.split(".")[0]) >= 5
            else {"rope_scaling": rope, "rope_theta": 10000.0})
    cfg = Qwen3VLConfig(text_config=dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                                         num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, **rope),
                        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=64,
                                           deepstack_visual_indexes=[0, 1]),
                        image_token_id=500, video_token_id=501, vision_start_token_id=502, vision_end_token_id=503)
    model = Qwen3VLForConditionalGeneration(cfg).eval(); model.config._attn_implementation = "eager"
    vp = Qwen3VLVideoProcessor(); n, K = 7, 3                                                   # 7 frames on a 3 x 3 grid: the last row is short
    frames = rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8); tps = vp.temporal_patch_size
    def vis(fr):
        o = vp(videos=[np.repeat(fr, tps, axis=0)], do_sample_frames=False, do_resize=False, return_tensors="pt")
        return o["pixel_values_videos"], o["video_grid_thw"]
    fca = FCAVideo(model, [1, 2, 3, 4], "cpu")
    emb, deep, grid = fca.visual(*vis(frames)); ok = True
    # a frame's visual tokens alone equal its tokens inside the whole video
    d_vit = max(float((fca.visual(*vis(frames[f:f + 1]))[0][0] - emb[f]).abs().max()) for f in range(n))
    print(f"  vision: frame alone vs inside the {n}-frame video, max |d| {d_vit:.2e} -> {'OK' if d_vit < 1e-5 else 'FAIL'}"); ok &= d_vit < 1e-5
    ts = lambda f: [10 + f % 5] * (1 + f % 2)                                                 # timestamps of two lengths: blocks of two extents
    book = fca.encode(fca.layout(n, K, ts, grid), emb, deep); req = list(range(20, 32))
    # layout against Qwen3-VL's own position rule: the blocks laid out back to back as one video
    seq_ids = list(fca.prefix_ids) + [x for f in range(n) for x in book["ids"][f]]; mine, cur = [np.tile(np.arange(4), (3, 1))], 4
    for f in range(n): mine.append(book["pos"][f] + cur); cur += int(book["ext"][f])
    ids_t = t.tensor([seq_ids]); mm = (ids_t == cfg.video_token_id).int() * 2
    vgt = t.tensor([[n, grid[0] * fca.merge, grid[1] * fca.merge]])
    try: theirs, _ = model.model.get_rope_index(ids_t, mm_token_type_ids=mm, video_grid_thw=vgt)            # transformers 5
    except TypeError: theirs, _ = model.model.get_rope_index(ids_t, None, vgt)                             # 4.57: (input_ids, image_grid_thw, video_grid_thw)
    d_pos = int((theirs[:, 0] - t.from_numpy(np.concatenate(mine, 1))).abs().max())
    print(f"  layout: block positions laid out as one video vs get_rope_index, max |d| {d_pos} -> {'OK' if d_pos == 0 else 'FAIL'}"); ok &= d_pos == 0
    worst = 0.0
    for i, members in enumerate(book["local"]):
        d = float((fca.ask(book, [members], req)[0] - fca.reference(book, members, req, emb, deep)).abs().max()); worst = max(worst, d)
    print(f"  local spans: store + gather vs explicit causal read, worst max |dlogit| {worst:.2e} -> {'OK' if worst < 1e-4 else 'FAIL'}"); ok &= worst < 1e-4
    members = book["strided"][0]; d_s = float((fca.ask(book, [members], req)[0] - fca.reference(book, members, req, emb, deep)).abs().max())
    print(f"  strided span 0 vs a read whose blocks see each other, max |dlogit| {d_s:.2e} (must NOT be 0) -> {'OK' if d_s > 1e-4 else 'FAIL'}"); ok &= d_s > 1e-4
    spans = book["local"] + book["strided"]; seq = t.cat([fca.ask(book, [s], req) for s in spans])
    fca._pad_noise = True; bat = t.cat([fca.ask(book, spans[b:b + 3], req) for b in range(0, len(spans), 3)]); fca._pad_noise = False
    d_b = float((seq - bat).abs().max())
    print(f"  {len(spans)} spans one at a time vs batches of 3 (mixed lengths, noise in the pad slots), max |dlogit| {d_b:.2e} -> {'OK' if d_b < 1e-4 else 'FAIL'}"); ok &= d_b < 1e-4
    steps = [int(book["base"][s[k + 1]] - book["base"][s[k]]) for s in spans for k in range(len(s) - 1)]
    print(f"  stair: consecutive blocks of every span one stride apart ({book['stride']}): {set(steps) == {book['stride']}}; "
          f"largest position {int(max(book['base'] + book['ext']))} -> {'OK' if set(steps) == {book['stride']} else 'FAIL'}"); ok &= set(steps) == {book["stride"]}
    emb2, deep2, _ = fca.visual(*vis(frames), batch=2)                                          # the vision tower two frames at a time
    d_vb = max(float((emb2 - emb).abs().max()), max(float((a - b).abs().max()) for a, b in zip(deep2, deep)))
    print(f"  vision: two frames per call vs all at once, max |d| {d_vb:.2e} -> {'OK' if d_vb < 1e-5 else 'FAIL'}"); ok &= d_vb < 1e-5
    buf_all = book["buf"].clone(); d_eb = float((fca.encode(fca.layout(n, K, ts, grid), emb, deep, batch=1)["buf"] - buf_all).abs().max())
    print(f"  store: one local span per forward vs all at once, max |d| {d_eb:.2e} -> {'OK' if d_eb < 1e-4 else 'FAIL'}"); ok &= d_eb < 1e-4
    place = [needle_place(nn, dd, tt) for nn, dd, tt in ((144, 0.0, 2), (144, 0.5, 2), (144, 1.0, 2), (2025, 1.0, 2), (2025, 1.0, 1), (2025, 0.3, 2))]
    good = place == [0, 72, 142, 2022, 2024, 606]
    print(f"  needle: first frame for (frames, depth, patch) cases {place} -> {'OK' if good else 'FAIL'}"); ok &= good
    ok &= hook_check_qwen(model, cfg, vis(frames), seq_ids, req)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


def fake_xattention(q, k, v, stride, norm, threshold, use_triton):
    """XAttention keeping every block: causal attention over the repeated heads (the stand-in of the CPU checks)."""
    import torch
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def hook_check_qwen(model, cfg, video, seq, req):
    """The XAttention hook on the tiny Qwen3-VL (SDPA) with the stand-in: the hooked uncached read and the hooked cached read (prefix, then the
    question over the cache) must equal the plain ones; only the language model's prefills may reach the method, one call per layer, while the
    vision tower's calls pass through to the backend."""
    import torch as t
    model.set_attn_implementation("sdpa"); pv, vgt = video; ids = t.tensor([list(seq) + req]); hl = len(seq)    # seq: the template and the frames' blocks
    mm = (ids == cfg.video_token_id).int() * 2
    try: pos, _ = model.model.get_rope_index(ids, mm_token_type_ids=mm, video_grid_thw=vgt)            # transformers 5
    except TypeError: pos, _ = model.model.get_rope_index(ids, None, vgt)                             # 4.57
    sp = SparsePrefill(model, "sdpa", 8, 0.9, fns={"xattention_prefill": fake_xattention}); n_layers = len(sp.layer_of)
    def reads(on):
        sp.on = on; sp.reset(); calls = []
        try:
            with t.no_grad():
                full = model(input_ids=ids, position_ids=pos, pixel_values_videos=pv, video_grid_thw=vgt).logits[0, -1]; calls.append(sp.calls)
                cache = model.model(input_ids=ids[:, :hl], position_ids=pos[:, :, :hl], pixel_values_videos=pv, video_grid_thw=vgt, use_cache=True).past_key_values
                calls.append(sp.calls)
                later = model(input_ids=ids[:, hl:], position_ids=pos[:, :, hl:], past_key_values=cache, use_cache=True).logits[0, -1]; calls.append(sp.calls)
        finally: sp.on = False
        return full, later, calls, sp.other
    try: (pf, pl, _, _), (hf, hl_, calls, other) = reads(False), reads(True)
    finally: sp.remove(); model.set_attn_implementation("eager")
    d = max(float((pf - hf).abs().max()), float((pl - hl_).abs().max()))
    good = d < 1e-4 and calls == [n_layers, 2 * n_layers, 2 * n_layers] and other > 0
    print(f"  XAttention hook (stand-in keeping every block): uncached and cached reads vs plain, max |dlogit| {d:.2e}; sparse calls after the "
          f"read, the cached prefix and the question over the cache {calls} (one per layer, {n_layers}, in each prefill only); {other} vision-tower "
          f"calls passed to the backend -> {'OK' if good else 'FAIL'}")
    return good


IVL = "internvl" in args.model.lower()
if args.check and not IVL:
    checks(); sys.exit(0)


# ============================================================== data, model and the three arms (GridProbe's code wherever it exists)
import torch
from GridProbe.eval.grid_sampled_ensemble_eval import extract_frames_uniform, build_inputs, get_letter_token_id_variants, letters_log_probs_from_logits
import GridProbe.eval.two_stage_eval as ts_eval

def benchmark():
    """GridProbe's loaders, prompts, letters and answer check for the chosen benchmark."""
    if args.benchmark == "lvb":
        from GridProbe.eval.longvideobench import LETTERS_LVB, build_prompt_lvb, check_answer_lvb, load_lvb
        samples = load_lvb(args.lvb_json, lvb_dir=args.lvb_dir, with_subtitle=args.with_subtitle)
        return samples, LETTERS_LVB, lambda s: build_prompt_lvb(s["question"], s["options"], subtitle=s.get("subtitle") if args.with_subtitle else None), check_answer_lvb
    if args.benchmark == "video_mme_v2":
        from GridProbe.eval.video_mme_v2 import LETTERS_V2, build_prompt_v2, check_answer_v2, load_video_mme_v2
        samples = load_video_mme_v2(args.data_dir, with_subtitle=args.with_subtitle)
        return samples, LETTERS_V2, lambda s: build_prompt_v2(s["question"], s["options"], subtitle=s.get("subtitle") if args.with_subtitle else None), check_answer_v2
    from GridProbe.eval.grid_sampled_ensemble_eval import load_video_mme
    from GridProbe.eval.video_mme import build_prompt, check_answer
    return load_video_mme(args.data_dir), ["A", "B", "C", "D"], lambda s: build_prompt(s["question"], s["options"]), check_answer

NEEDLE_CUE = "One frame of this video is a still image that does not belong to the video. About that image: "

def needle_samples(samples):
    """The needle test's questions. Haystacks are the benchmark's videos in order, one per needle. A needle is an image with a question
    about it (TextVQA as AnswerMap/prep_vqa.py writes it); its four options are its most frequent gold answer and the gold answers of three
    other needles, shuffled with a fixed seed. Needle i sits at depth needle_depths[i mod len]."""
    import collections
    from GridProbe.eval.longvideobench import build_prompt_lvb
    rng = np.random.default_rng(0); videos = list(dict.fromkeys(s["video_path"] for s in samples)); depths = [float(x) for x in args.needle_depths.split(",")]
    recs = [json.loads(l) for l in open(args.needle_json, encoding="utf-8") if l.strip()]
    gold = [collections.Counter(str(a).strip().lower() for a in (r.get("answers") or [r["answer"]])).most_common(1)[0][0] for r in recs]
    out = []
    for i in range(min(args.needle_n, len(recs), len(videos))):
        others = sorted({g for g in gold if g != gold[i]}); opts = [others[j] for j in rng.choice(len(others), 3, replace=False)] + [gold[i]]
        opts = [opts[j] for j in rng.permutation(4)]; d = depths[i % len(depths)]
        out.append(dict(id=f"needle{i:03d}_n{args.K * args.K}_d{d:g}", video_path=f"{videos[i]}#needle{i}@{args.K * args.K}", video_file=videos[i], depth=d,
                        needle_image=os.path.join(args.needle_images or os.path.dirname(args.needle_json), recs[i]["image"]),
                        question=NEEDLE_CUE + recs[i]["question"], options=opts, answer="ABCD"[opts.index(gold[i])], duration_bin=f"depth {d:g}"))
    return out, ["A", "B", "C", "D"], lambda s: build_prompt_lvb(s["question"], s["options"]), lambda pred, ans: pred == ans

def letterbox(img, size):
    """The image scaled to fit a frame of this size, centred on black, so the needle has the haystack's frame size without distortion."""
    from PIL import Image
    W, H = size; r = min(W / img.width, H / img.height); im = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))))
    out = Image.new("RGB", (W, H)); out.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); return out

def needle_scores(rec, cells, K):
    """Where the needle sits in a probing arm's map: is the map's peak on it, its rank among the K*K cells, and did the select read keep it."""
    import torch.nn.functional as F
    conf = lambda lp: F.softmax(torch.tensor(lp), dim=-1).max(dim=-1).values.numpy()
    M = conf(rec["row_lp"])[:, None] * conf(rec["col_lp"])[None, :]; order = np.argsort(M.flatten())[::-1].tolist()
    cell = lambda f: (f // K) * M.shape[1] + f % K
    return dict(peak_hit=bool(order[0] in {cell(f) for f in cells}), needle_rank=1 + min(order.index(cell(f)) for f in cells),
                needle_selected=bool(set(cells) & set(int(f) for f in rec["frames"])))

def needle_tag():
    """The needle test keeps one file per length: _n<frames> in the name."""
    return f"_n{args.K * args.K}" if args.needle_json else ""

def needle_Ks():
    return [int(k) for k in args.needle_Ks.split(",") if k] if args.needle_json and args.needle_Ks else [args.K]

def out_path(arm):
    shard = f"_s{args.shard_index}of{args.n_shards}" if args.n_shards > 1 else ""                 # one file per shard; the summary reads them all
    return os.path.join(args.out_dir, f"{args.benchmark}_{arm}_{args.model.split('/')[-1]}{needle_tag()}{shard}.jsonl")

def all_paths(arm):
    import glob
    return sorted(glob.glob(os.path.join(args.out_dir, f"{args.benchmark}_{arm}_{args.model.split('/')[-1]}{needle_tag()}*.jsonl")))

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def sparse_prefill(model):
    if args.sparse_prefill == "none": return None
    print(f"sparse prefill: XAttention, stride {args.xattn_stride}, threshold {args.xattn_threshold}", flush=True)
    return SparsePrefill(model, args.attn, args.xattn_stride, args.xattn_threshold)

def monolithic(run, arm_fn, *a):
    """A monolithic arm with XAttention switched on for its duration when --sparse_prefill is set; the row records its calls and kept share."""
    sp = run.sparse
    if sp is None: return arm_fn(*a)
    sp.on = True; sp.reset()
    try: rec = arm_fn(*a)
    finally: sp.on = False
    rec.update(sparse_prefill="xattention", xattn_stride=sp.stride, xattn_threshold=sp.thr, sparse_calls=sp.calls, xattn_density=sp.density())
    return rec


class Runner:
    def __init__(self, letters):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn).to(self.dev).eval()
        if args.compressor != "none":                    # the repo's forward of Qwen3VLModel drops video tokens after the vision encoder, on a forward without a cache
            import importlib, types
            os.environ["COMPRESSOR"] = args.compressor; os.environ["R_RATIO"] = str(args.r_ratio)
            fwd = importlib.import_module(f"token_compressor.{args.compressor}.models.qwen3_vl").Qwen3VLModel_forward
            self.model.model.forward = types.MethodType(fwd, self.model.model)
            print(f"compressor {args.compressor}: Qwen3VLModel.forward replaced, keeping {args.r_ratio} of the video tokens", flush=True)
        self.proc = AutoProcessor.from_pretrained(args.model); self.tok = self.proc.tokenizer; self.letters = letters
        self.letter_variants = get_letter_token_id_variants(self.tok, letters); ts_eval.LETTERS = letters  # GridProbe's functions read their module's LETTERS
        SENT = "@@REQUEST@@"; vtag = "<|vision_start|><|video_pad|><|vision_end|>"
        s = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "video", "video": "x"}, {"type": "text", "text": SENT}]}],
                                          tokenize=False, add_generation_prompt=True)
        assert vtag in s and SENT in s, s
        self.head_text, self.tail_text = s[:s.index(vtag)], s[s.index(SENT) + len(SENT):]
        self.fca = FCAVideo(self.model, self.tok.encode(self.head_text, add_special_tokens=False), self.dev)
        enc = lambda x: self.tok.encode(x, add_special_tokens=False)                              # the pieces must tokenize as the whole does
        assert enc(self.head_text + "<0.0 seconds>") == enc(self.head_text) + enc("<0.0 seconds>")
        assert enc("<|vision_end|>Question: x" + self.tail_text) == enc("<|vision_end|>") + enc("Question: x" + self.tail_text)
        self.fps = args.fps or self.mono_fps(); self._book = None; self._mono = None; self.sparse = sparse_prefill(self.model)
        print(f"loaded {args.model} ({args.attn}); prefix {len(self.fca.prefix_ids)} tokens; stored timestamps at f / {self.fps:g} s", flush=True)

    def mono_fps(self):
        """The clock qwen_vl_utils gives a list of frames (the monolithic read's timestamps), read from its own output."""
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        img = Image.new("RGB", (224, 224)); msgs = [{"role": "user", "content": [{"type": "video", "video": [img] * 4}, {"type": "text", "text": "x"}]}]
        out = process_vision_info(msgs, return_video_kwargs=True); vk = out[2] if len(out) >= 3 else {}
        fps = (vk or {}).get("fps", 2.0); return float(fps[0] if isinstance(fps, (list, tuple)) else fps)

    def book(self, key, frames):
        """The store of this video, kept while consecutive questions share the video."""
        if self._book is not None and self._book["key"] == key: return self._book, True, 0.0
        self._book = None
        if self.dev == "cuda": torch.cuda.empty_cache()
        sync(); t0 = time.time(); tps = self.proc.video_processor.temporal_patch_size
        inp = build_inputs(self.proc, "x", [f for f in frames for _ in range(tps)], args.probe_pixels)   # GridProbe's probe preprocessing
        emb, deep, grid = self.fca.visual(inp["pixel_values_videos"], inp["video_grid_thw"], args.visual_batch)
        assert emb.shape[0] == len(frames), (emb.shape, len(frames))
        b = self.fca.layout(len(frames), args.K, lambda f: self.tok.encode(f"<{f / self.fps:.1f} seconds>", add_special_tokens=False), grid)
        b = self.fca.encode(b, emb, deep, args.encode_batch); sync(); b.update(key=key, encode_s=time.time() - t0, grid=grid); self._book = b; return b, False, b["encode_s"]

    def fca_arm(self, sample, frames, prompt_text):
        b, shared, enc_s = self.book(sample["video_path"], frames); req = self.tok.encode(prompt_text + self.tail_text, add_special_tokens=False)
        sync(); t0 = time.time(); lg, nL = self.fca.probes(b, req, args.copy_batch)
        lp = torch.stack([letters_log_probs_from_logits(lg[i], self.letter_variants, self.dev) for i in range(lg.shape[0])]).cpu()
        probe = grid_read(lp[:nL], lp[nL:], self.letters); sync(); probe_s = time.time() - t0
        return self.second_stage(probe, frames, prompt_text, dict(shared=int(shared), encode_s=enc_s, probe_s=probe_s, req_tokens=len(req),
                                 store_tokens=int(b["P"] + b["lens"].sum()), stride=int(b["stride"]), max_pos=int(max(b["base"] + b["ext"])) + len(req)))

    def gridprobe_arm(self, sample, frames, prompt_text):
        sync(); t0 = time.time()
        probe = ts_eval.stage1_probe(self.model, self.proc, prompt_text, frames, args.K, self.letter_variants, self.dev, args.probe_pixels)
        sync(); return self.second_stage(probe, frames, prompt_text, dict(shared=0, encode_s=0.0, probe_s=time.time() - t0))

    def second_stage(self, probe, frames, prompt_text, rec):
        """GridProbe's budget rule and focused pass on the map of either arm."""
        M_eff = ts_eval.select_M(args.selector, probe["M_mult"], args); sync(); t0 = time.time()
        st2 = ts_eval.stage2_focused(self.model, self.proc, prompt_text, frames, probe["golden"], M_eff, args.K, self.letter_variants, self.dev, args.focus_pixels)
        sync()
        rec.update(direct=probe["ensemble_pred"], pred=st2["pred"], M_eff=int(M_eff), frames=st2["frame_indices"], select_s=time.time() - t0,
                   row_lp=probe["row_lp"].tolist(), col_lp=probe["col_lp"].tolist(), row_preds=probe["row_preds"], col_preds=probe["col_preds"])
        rec["latency_direct"] = rec["encode_s"] + rec["probe_s"]; rec["latency_select"] = rec["latency_direct"] + rec["select_s"]; return rec

    def mono_arm(self, sample, frames, prompt_text):
        sync(); t0 = time.time()
        r = ts_eval.run_baseline_full(self.model, self.proc, prompt_text, frames, self.letter_variants, self.dev, 0)
        sync(); return {"pred": r["pred"], "latency": time.time() - t0, "lp": r["lp"].tolist()}

    def mono_cache_arm(self, sample, frames, prompt_text):
        """The monolithic read with a prefix cache (text Table 1's prefix-cache row, for video): GridProbe's inputs (build_inputs at the
        processor's resolution); the template and the frames up to the last <vision_end> are encoded on the first question about a video and
        kept, and every question runs only its own text over that cache at the positions of the whole sequence. A first question's time
        includes processing the frames; a later one's does not, since a cached video needs no frames."""
        sync(); t_in = time.time()
        inp = build_inputs(self.proc, prompt_text, frames, 0); ids = inp["input_ids"].to(self.dev); lst = ids[0].tolist()
        hl = len(lst) - lst[::-1].index(self.model.config.vision_end_token_id)
        vgt = inp["video_grid_thw"].to(self.dev); mm = inp.get("mm_token_type_ids"); mdl = self.model.model
        try: pos, _ = mdl.get_rope_index(ids, mm_token_type_ids=mm.to(self.dev), video_grid_thw=vgt)        # transformers 5
        except (TypeError, AttributeError): pos, _ = mdl.get_rope_index(ids, None, vgt)                     # 4.57: (input_ids, image_grid_thw, video_grid_thw)
        key = (sample["video_path"], tuple(lst[:hl])); shared = self._mono is not None and self._mono["key"] == key
        sync(); t0 = time.time() if shared else t_in
        with torch.no_grad():
            if not shared:
                self._mono = None
                if self.dev == "cuda": torch.cuda.empty_cache()
                kw = dict(input_ids=ids[:, :hl], pixel_values_videos=inp["pixel_values_videos"].to(self.dev), video_grid_thw=vgt,
                          position_ids=pos[:, :, :hl], use_cache=True)
                if mm is not None: kw["mm_token_type_ids"] = mm[:, :hl].to(self.dev)
                self._mono = {"key": key, "cache": mdl(**kw).past_key_values}
            cache = self._mono["cache"]
            out = self.model(input_ids=ids[:, hl:], position_ids=pos[:, :, hl:], past_key_values=cache, use_cache=True, logits_to_keep=1)
            lp = letters_log_probs_from_logits(out.logits[0, -1], self.letter_variants, self.dev); cache.crop(hl)   # the cache holds the video again
        sync()
        return {"pred": self.letters[int(lp.argmax())], "latency": time.time() - t0, "shared": int(shared), "lp": lp.cpu().tolist(),
                "tokens": len(lst), "prefix_tokens": hl}


def blank():
    from PIL import Image
    return Image.new("RGB", (224, 224))

def build_internvl(proc, prompt_text, frames, px):
    """The rebuttal's InternVL inputs (vlm_backends._build_internvl): frames resized to a square side from a pixel budget (a multiple of 28)
    and passed as images in the chat message; each is one 448 x 448 tile, 256 visual tokens."""
    if px and px > 0: side = max(56, (int(math.sqrt(px)) // 28) * 28); frames = [f.resize((side, side)) for f in frames]
    msg = [{"role": "user", "content": [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": prompt_text}]}]
    return proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")


class RunnerInternVL:
    """InternVL3.5 (the -HF checkpoint) on the rebuttal's protocol (crossfamily_eval.py): probes, focused pass and monolithic read all at
    the probe resolution (224 x 224, since InternVL tiles every frame and full resolution overflows its 40,960-token context), GridProbe's
    skew budget and read; the attention kernel is --attn (the rebuttal used SDPA)."""

    def __init__(self, letters, model=None, proc=None, dev=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn).to(self.dev).eval()
        self.proc = proc or AutoProcessor.from_pretrained(args.model); self.tok = self.proc.tokenizer; self.letters = letters
        self.letter_variants = get_letter_token_id_variants(self.tok, letters); ts_eval.LETTERS = letters
        self.img0, self.img1 = self.tok.convert_tokens_to_ids([self.proc.start_image_token, self.proc.end_image_token]); self.px = args.probe_pixels
        prefix, _, _ = self.cut(build_internvl(self.proc, "x", [blank(), blank()], self.px)["input_ids"][0].tolist())
        self.fca = FCAInternVL(self.model, prefix, self.dev); self._book = None; self._mono = None
        self.sparse = sparse_prefill(self.model) if model is None else None                             # a model passed in is the CPU check's
        print(f"loaded {args.model} ({args.attn}); prefix {len(prefix)} tokens; frames at {self.px} pixels", flush=True)

    def cut(self, ids):
        """Prefix, blocks and request of a tokenized prompt: a block runs from a frame's <img> to the next frame's; the last one ends after
        its </img> and the separator the others carry."""
        s = [i for i, x in enumerate(ids) if x == self.img0]; blocks = [ids[a:b] for a, b in zip(s, s[1:])]
        last = ids[s[-1]:]; e = last.index(self.img1) + 1; sep = blocks[0][blocks[0].index(self.img1) + 1:] if blocks else []
        if sep and last[e:e + len(sep)] == sep: e += len(sep)
        return ids[:s[0]], blocks + [last[:e]], ids[s[-1] + e:]

    def request(self, prompt_text):
        return self.cut(build_internvl(self.proc, prompt_text, [blank(), blank()], self.px)["input_ids"][0].tolist())[2]

    def score(self, prompt_text, frames):
        """GridProbe's read of one prompt: letter log-probs of the last token (and the input length)."""
        inp = build_internvl(self.proc, prompt_text, frames, self.px)
        with torch.no_grad():
            out = self.model(input_ids=inp["input_ids"].to(self.dev), attention_mask=inp["attention_mask"].to(self.dev),
                             pixel_values=inp["pixel_values"].to(self.dev, self.model.dtype), logits_to_keep=1)
        return letters_log_probs_from_logits(out.logits[0, -1], self.letter_variants, self.dev).cpu(), int(inp["input_ids"].shape[1])

    def book(self, video_path, frames):
        if self._book is not None and self._book["key"] == video_path: return self._book, True, 0.0
        self._book = None
        if self.dev == "cuda": torch.cuda.empty_cache()
        sync(); t0 = time.time(); inp = build_internvl(self.proc, "x", frames, self.px); prefix, blocks, _ = self.cut(inp["input_ids"][0].tolist())
        assert prefix == self.fca.prefix_ids and len(blocks) == len(frames), (len(prefix), len(blocks), len(frames))
        emb, deep, _ = self.fca.visual(inp["pixel_values"], args.visual_batch); assert emb.shape[0] == len(frames), (emb.shape, len(frames))
        b = self.fca.encode(self.fca.layout(len(frames), args.K, lambda f: blocks[f]), emb, deep, args.encode_batch)
        sync(); b.update(key=video_path, encode_s=time.time() - t0); self._book = b; return b, False, b["encode_s"]

    def fca_arm(self, sample, frames, prompt_text):
        b, shared, enc_s = self.book(sample["video_path"], frames); req = self.request(prompt_text)
        sync(); t0 = time.time(); lg, nL = self.fca.probes(b, req, args.copy_batch)
        lp = torch.stack([letters_log_probs_from_logits(lg[i], self.letter_variants, self.dev) for i in range(lg.shape[0])]).cpu()
        probe = grid_read(lp[:nL], lp[nL:], self.letters); sync(); probe_s = time.time() - t0
        return self.second_stage(probe, frames, prompt_text, dict(shared=int(shared), encode_s=enc_s, probe_s=probe_s, req_tokens=len(req),
                                 store_tokens=int(b["P"] + b["lens"].sum()), stride=int(b["stride"]), max_pos=int(max(b["base"] + b["ext"])) + len(req)))

    def gridprobe_arm(self, sample, frames, prompt_text):
        from GridProbe.eval.grid_sampled_ensemble_eval import row_indices, col_indices
        sync(); t0 = time.time()
        row = [self.score(prompt_text, [frames[i] for i in idx])[0] for idx in row_indices(args.K)]
        col = [self.score(prompt_text, [frames[i] for i in idx])[0] for idx in col_indices(args.K)]
        probe = grid_read(torch.stack(row), torch.stack(col), self.letters); sync()
        return self.second_stage(probe, frames, prompt_text, dict(shared=0, encode_s=0.0, probe_s=time.time() - t0))

    def second_stage(self, probe, frames, prompt_text, rec):
        """GridProbe's budget and focused pass as the rebuttal runs it: the top M_eff frames of the map in time order, at the probe resolution."""
        M_eff = ts_eval.select_M(args.selector, probe["M_mult"], args); K = args.K
        idx = sorted(r * K + c for r, c in probe["golden"][:M_eff] if r * K + c < len(frames))
        sync(); t0 = time.time(); lp, _ = self.score(prompt_text, [frames[i] for i in idx]); sync()
        rec.update(direct=probe["ensemble_pred"], pred=self.letters[int(lp.argmax())], M_eff=int(M_eff), frames=idx, select_s=time.time() - t0,
                   row_lp=probe["row_lp"].tolist(), col_lp=probe["col_lp"].tolist(), row_preds=probe["row_preds"], col_preds=probe["col_preds"])
        rec["latency_direct"] = rec["encode_s"] + rec["probe_s"]; rec["latency_select"] = rec["latency_direct"] + rec["select_s"]; return rec

    def mono_arm(self, sample, frames, prompt_text):
        sync(); t0 = time.time(); lp, n = self.score(prompt_text, frames); sync()
        return {"pred": self.letters[int(lp.argmax())], "latency": time.time() - t0, "lp": lp.tolist(), "tokens": n}

    def mono_cache_arm(self, sample, frames, prompt_text):
        """The monolithic read with a prefix cache up to the last frame's block; positions continue from the cache (1-D)."""
        sync(); t_in = time.time()
        inp = build_internvl(self.proc, prompt_text, frames, self.px); ids = inp["input_ids"].to(self.dev); lst = ids[0].tolist()
        hl = len(lst) - len(self.cut(lst)[2]); key = (sample["video_path"], tuple(lst[:hl]))
        shared = self._mono is not None and self._mono["key"] == key; sync(); t0 = time.time() if shared else t_in
        with torch.no_grad():
            if not shared:
                self._mono = None
                if self.dev == "cuda": torch.cuda.empty_cache()
                out = self.model.model(input_ids=ids[:, :hl], pixel_values=inp["pixel_values"].to(self.dev, self.model.dtype), use_cache=True)
                self._mono = {"key": key, "cache": out.past_key_values}
            cache = self._mono["cache"]
            out = self.model(input_ids=ids[:, hl:], past_key_values=cache, use_cache=True, logits_to_keep=1)
            lp = letters_log_probs_from_logits(out.logits[0, -1], self.letter_variants, self.dev); cache.crop(hl)
        sync()
        return {"pred": self.letters[int(lp.argmax())], "latency": time.time() - t0, "shared": int(shared), "lp": lp.cpu().tolist(),
                "tokens": len(lst), "prefix_tokens": hl}


def checks_internvl():
    """The InternVL store on a tiny random InternVL with the real token ids and processor: the first row of the store read with a request
    equals GridProbe's row probe on the same frames (same tokens, same positions); every local span from the store equals an explicit causal
    read; a strided span differs from a read whose blocks see each other; batches with noise in the pad slots equal one at a time; the
    cached monolithic read equals the uncached one."""
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, InternVLConfig, InternVLForConditionalGeneration
    real = AutoConfig.from_pretrained(args.model); torch.manual_seed(0); rng = np.random.default_rng(0); tc = real.text_config
    cfg = InternVLConfig(vision_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=[448, 448], patch_size=[14, 14]),
                         text_config=dict(model_type=tc.model_type, vocab_size=tc.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=16384),
                         image_token_id=real.image_token_id, downsample_ratio=real.downsample_ratio, vision_feature_layer=-1, vision_feature_select_strategy="default")
    model = InternVLForConditionalGeneration(cfg).eval(); args.K = 3; n = 7
    run = RunnerInternVL(["A", "B", "C", "D"], model=model, proc=AutoProcessor.from_pretrained(args.model), dev="cpu"); fca = run.fca
    frames = [Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)) for _ in range(n)]
    q = "Question: what happens? A. a B. b C. c D. d\nAnswer with the option's letter."; req = run.request(q)
    book, _, _ = run.book("v.mp4", frames); emb, deep, _ = fca.visual(build_internvl(run.proc, "x", frames, run.px)["pixel_values"]); ok = True
    lp_row0 = letters_log_probs_from_logits(fca.ask(book, [book["local"][0]], req)[0], run.letter_variants, "cpu").cpu()
    d0 = float((lp_row0 - run.score(q, frames[:args.K])[0]).abs().max())
    print(f"  row 0 of the store vs GridProbe's row probe on the same frames, max |d letter log-prob| {d0:.2e} -> {'OK' if d0 < 1e-4 else 'FAIL'}"); ok &= d0 < 1e-4
    worst = max(float((fca.ask(book, [m], req)[0] - fca.reference(book, m, req, emb, deep)).abs().max()) for m in book["local"])
    print(f"  local spans: store + gather vs explicit causal read, worst max |dlogit| {worst:.2e} -> {'OK' if worst < 1e-4 else 'FAIL'}"); ok &= worst < 1e-4
    d_s = float((fca.ask(book, [book["strided"][0]], req)[0] - fca.reference(book, book["strided"][0], req, emb, deep)).abs().max())
    print(f"  strided span 0 vs a read whose blocks see each other, max |dlogit| {d_s:.2e} (must NOT be 0) -> {'OK' if d_s > 1e-4 else 'FAIL'}"); ok &= d_s > 1e-4
    spans = book["local"] + book["strided"]; single = torch.cat([fca.ask(book, [m], req) for m in spans], 0)
    fca._pad_noise = True; batch = torch.cat([fca.ask(book, spans[b:b + 3], req) for b in range(0, len(spans), 3)], 0); fca._pad_noise = False
    d_b = float((single - batch).abs().max())
    print(f"  {len(spans)} spans one at a time vs batches of 3 (noise in the pad slots), max |dlogit| {d_b:.2e} -> {'OK' if d_b < 1e-4 else 'FAIL'}"); ok &= d_b < 1e-4
    d_c = 0.0
    for qq in (q, "Question: which colour? A. red B. blue C. green D. grey\nAnswer with the option's letter."):
        r = run.mono_cache_arm({"video_path": "v.mp4"}, frames, qq); d_c = max(d_c, float(np.abs(np.array(r["lp"]) - run.score(qq, frames)[0].numpy()).max()))
    print(f"  cached monolithic read (first and later question) vs uncached, max |d letter log-prob| {d_c:.2e} -> {'OK' if d_c < 1e-4 else 'FAIL'}"); ok &= d_c < 1e-4
    ok &= hook_check_internvl(run, frames, q)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


def hook_check_internvl(run, frames, q):
    """The XAttention hook on the tiny InternVL (SDPA) with the stand-in, through the arms as main() calls them: the hooked mono arm and the
    hooked cached arm (a first and a later question) must give the plain letter log-probs, with one call per layer in each prefill and none for
    a question over the cache."""
    prev = run.model.model.language_model.config._attn_implementation; run.model.set_attn_implementation("sdpa")
    qq, v = "Question: which colour? A. red B. blue C. green D. grey\nAnswer with the option's letter.", {"video_path": "v.mp4"}
    def arms():
        run._mono = None
        return [monolithic(run, run.mono_arm, v, frames, q), monolithic(run, run.mono_cache_arm, v, frames, q), monolithic(run, run.mono_cache_arm, v, frames, qq)]
    run.sparse = None; plain = arms()
    run.sparse = sp = SparsePrefill(run.model, "sdpa", 8, 0.9, fns={"xattention_prefill": fake_xattention}); n_layers = len(sp.layer_of)
    try: hooked = arms()
    finally: sp.remove(); run.sparse = None; run.model.set_attn_implementation(prev)
    d = max(float(np.abs(np.array(a["lp"]) - np.array(b["lp"])).max()) for a, b in zip(plain, hooked)); calls = [r["sparse_calls"] for r in hooked]
    good = d < 1e-4 and calls == [n_layers, n_layers, 0] and hooked[2]["shared"] == 1
    print(f"  XAttention hook (stand-in keeping every block): mono and cached arms vs plain, max |d letter log-prob| {d:.2e}; sparse calls "
          f"{calls} (mono, first and later cached question; {n_layers} layers) -> {'OK' if good else 'FAIL'}")
    return good


def check_sparse():
    """On the GPU, before any run with --sparse_prefill: the released XAttention through the hook on the benchmark's first video against the
    dense read. Keeping every block (threshold 1.0) must give the dense read's answer, letter log-probs within bf16 noise and one sparse call per
    language-model layer; on the cached read a first question must make those calls and a later one none. Then the run's threshold is reported."""
    assert args.sparse_prefill != "none", "set --sparse_prefill"
    samples, letters, prompt_of, _ = benchmark(); samples = [s for s in samples if os.path.exists(s["video_path"])]
    if args.duration_bin: samples = [s for s in samples if s["duration_bin"] == args.duration_bin]
    s0 = samples[0]; pair = [s for s in samples if s["video_path"] == s0["video_path"]][:2]
    run = (RunnerInternVL if IVL else Runner)(letters); sp = run.sparse; n_layers = len(sp.layer_of); ok = True
    frames = extract_frames_uniform(s0["video_path"], args.K * args.K)
    def mono(on, thr):
        sp.thr = thr
        if not on: return run.mono_arm(s0, frames, prompt_of(s0))
        return monolithic(run, run.mono_arm, s0, frames, prompt_of(s0))
    dense = np.array(mono(False, 1.0)["lp"])
    print(f"check_sparse: XAttention on {s0['id']} ({args.benchmark}), {args.attn}, {n_layers} language-model layers", flush=True)
    for thr, bound in ((1.0, 0.5), (args.xattn_threshold, None)):
        r = mono(True, thr); d = float(np.abs(np.array(r["lp"]) - dense).max()); same = int(np.argmax(r["lp"])) == int(dense.argmax())
        good = bound is None or (same and d <= bound and r["sparse_calls"] == n_layers); ok &= good
        print(f"  mono, threshold {thr}: {r['sparse_calls']} sparse calls, {sp.other} vision-tower calls passed on, kept {r['xattn_density']} of the causal "
              f"blocks, max |d letter log-prob| {d:.3f}, same answer {same}" + (f" -> {'OK' if good else 'FAIL'}" if bound is not None else ""), flush=True)
    if len(pair) == 2:
        run._mono = None; calls = [monolithic(run, run.mono_cache_arm, s, frames, prompt_of(s))["sparse_calls"] for s in pair]
        good = calls == [n_layers, 0]; ok &= good
        print(f"  mono_cache, two questions on one video: sparse calls {calls} -> {'OK' if good else 'FAIL'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


def main(run=None):
    samples, letters, prompt_of, check = benchmark()
    n_all = len(samples); samples = [s for s in samples if os.path.exists(s["video_path"])]           # as GridProbe: a question whose video is missing is skipped
    if len(samples) < n_all: print(f"{args.benchmark}: {n_all - len(samples)} of {n_all} questions skipped, video file missing", flush=True)
    if args.duration_bin: samples = [s for s in samples if s["duration_bin"] == args.duration_bin]
    if args.needle_json: samples, letters, prompt_of, check = needle_samples(samples)
    order = {}; [order.setdefault(s["video_path"], len(order)) for s in samples]
    samples = sorted(samples, key=lambda s: order[s["video_path"]])                              # questions on one video in a row: one store
    if args.n_shards > 1: samples = [s for s in samples if order[s["video_path"]] % args.n_shards == args.shard_index]   # shards by video, so reuse survives
    if args.limit: samples = samples[:args.limit]
    assert args.sparse_prefill == "none" or set(ARMS) <= {"mono", "mono_cache"}, "XAttention is a prefill of the monolithic read: --arms mono,mono_cache"
    assert args.compressor == "none" or (set(ARMS) == {"mono"} and not IVL and args.sparse_prefill == "none"), "--compressor reduces the monolithic read of Qwen3-VL: --arms mono, no --sparse_prefill"
    for arm in set(ARMS) & {"mono", "mono_cache"}:                                                   # never mix dense and sparse rows in one file
        first = open(out_path(arm), encoding="utf-8").readline() if os.path.exists(out_path(arm)) else ""
        assert not first.strip() or json.loads(first).get("sparse_prefill", "none") == args.sparse_prefill, f"{out_path(arm)} holds another prefill's rows: use another --out_dir"
        assert not first.strip() or json.loads(first).get("compressor", "none") == args.compressor, f"{out_path(arm)} holds rows of another compressor: use another --out_dir"
    os.makedirs(args.out_dir, exist_ok=True); run = run or (RunnerInternVL if IVL else Runner)(letters)   # one model for every length of the needle test
    done = {arm: {json.loads(l)["id"] for l in open(out_path(arm))} if os.path.exists(out_path(arm)) else set() for arm in ARMS}
    print(f"{args.benchmark}: {len(samples)} questions on {len({s['video_path'] for s in samples})} videos (shard {args.shard_index} of {args.n_shards}), K={args.K}, arms {ARMS}", flush=True)
    frames_key, frames, needle = None, None, None
    for si, s in enumerate(samples):
        todo = [a for a in ARMS if s["id"] not in done[a]]
        if not todo: continue
        prompt_text = prompt_of(s)
        if s["video_path"] != frames_key:
            frames = extract_frames_uniform(s.get("video_file", s["video_path"]), args.K * args.K); frames_key = s["video_path"]; needle = None
            if frames is not None and args.needle_json:                                          # the needle replaces tps frames of the haystack
                from PIL import Image
                tps = 1 if IVL else run.proc.video_processor.temporal_patch_size; pos = needle_place(len(frames), s["depth"], tps)
                img = letterbox(Image.open(s["needle_image"]).convert("RGB"), frames[0].size); frames[pos:pos + tps] = [img] * tps
                alone = run.mono_arm(s, [img] * tps, prompt_text)["pred"]                         # the needle with no haystack: can the model answer it at all
                needle = dict(needle_pos=pos, needle_len=tps, n_frames=len(frames), depth=s["depth"], alone_correct=bool(check(alone, s["answer"])))
        if frames is None: print(f"    frame extraction failed: {s['id']}", flush=True); continue
        for arm in todo:
            fn = {"fca": run.fca_arm, "gridprobe": run.gridprobe_arm, "mono": run.mono_arm, "mono_cache": run.mono_cache_arm}[arm]
            fr, kept = frames, None
            if needle and arm in ("mono", "mono_cache") and args.mono_frames and len(frames) > args.mono_frames:
                kept = sorted(set(np.linspace(0, len(frames) - 1, args.mono_frames).round().astype(int).tolist())); fr = [frames[i] for i in kept]
            rec = monolithic(run, fn, s, fr, prompt_text) if arm in ("mono", "mono_cache") else fn(s, fr, prompt_text)
            rec.update(id=s["id"], video=s["video_path"], bin=s.get("duration_bin"), gold=s["answer"], arm=arm, attn=args.attn,
                       correct=bool(check(rec["pred"], s["answer"])))
            if args.compressor != "none": rec.update(compressor=args.compressor, r_ratio=args.r_ratio)
            if needle:
                cells = list(range(needle["needle_pos"], needle["needle_pos"] + needle["needle_len"])); rec.update(needle)
                if arm in ("mono", "mono_cache"): rec.update(frames_read=len(fr), needle_seen=bool(kept is None or set(cells) & set(kept)))
                else: rec.update(needle_scores(rec, cells, args.K))
            if "direct" in rec: rec["correct_direct"] = bool(check(rec["direct"], s["answer"]))
            with open(out_path(arm), "a", encoding="utf-8") as fo: fo.write(json.dumps(rec) + "\n")
        if (si + 1) % 10 == 0: print(f"    {si + 1}/{len(samples)}", flush=True)
    summary(); return run


def summary():
    print(f"{args.benchmark}, {args.model.split('/')[-1]}, K={args.K}. Accuracy (%) and seconds per question; first-of-video vs shared for FCA.")
    rows = {}
    for arm in ARMS:
        ps = all_paths(arm)
        if not ps: print(f"{arm:10s} no file {out_path(arm)}"); continue
        rows[arm] = R = [json.loads(l) for p in ps for l in open(p, encoding="utf-8")]
        line = f"{arm:10s} n={len(R):5d}  select/answer {100 * np.mean([r['correct'] for r in R]):5.1f}"
        if "correct_direct" in R[0]: line += f"  direct {100 * np.mean([r['correct_direct'] for r in R]):5.1f}  M_eff {np.mean([r['M_eff'] for r in R]):5.1f}"
        if arm == "mono": line += f"  s/q {np.mean([r['latency'] for r in R]):6.2f}"
        elif arm == "mono_cache":
            first = [r for r in R if not r["shared"]]; sh = [r for r in R if r["shared"]]
            line += f"  s/q {np.mean([r['latency'] for r in R]):6.2f}  | first-of-video {np.mean([r['latency'] for r in first]):6.2f} (n={len(first)})"
            if sh: line += f", shared {np.mean([r['latency'] for r in sh]):6.2f} (n={len(sh)})"
        else:
            first = [r for r in R if not r["shared"]]; sh = [r for r in R if r["shared"]]
            line += f"  direct s/q {np.mean([r['latency_direct'] for r in R]):6.2f}  select s/q {np.mean([r['latency_select'] for r in R]):6.2f}"
            if sh: line += f"  | first-of-video direct {np.mean([r['latency_direct'] for r in first]):6.2f} (n={len(first)}), shared {np.mean([r['latency_direct'] for r in sh]):6.2f} (n={len(sh)})"
        dens = [r["xattn_density"] for r in R if r.get("xattn_density") is not None]
        if dens: line += f"  | XAttention stride {R[0]['xattn_stride']}, threshold {R[0]['xattn_threshold']}, kept {np.mean(dens):.3f} of the causal blocks"
        bins = sorted({str(r.get("bin")) for r in R}); line += "  | by bin " + ", ".join(f"{b}: {100 * np.mean([r['correct'] for r in R if str(r.get('bin')) == b]):.1f}" for b in bins)
        print(line)
        if "needle_pos" in R[0]:                                                                 # the needle test: needles the model answers alone, and where the map puts them
            A = [r for r in R if r["alone_correct"]]; pct = lambda k, S: f"{100 * np.mean([r[k] for r in S]):5.1f}" if S else "  n/a"
            line = f"{'':10s} needle, {R[0]['n_frames']} frames: answered alone {len(A)}/{len(R)}; accuracy on those {pct('correct', A)}"
            if "correct_direct" in R[0]: line += f" (direct {pct('correct_direct', A)}); map peak on the needle {pct('peak_hit', A)}, needle kept by the select read {pct('needle_selected', A)}, median rank {np.median([r['needle_rank'] for r in A]) if A else float('nan'):.0f} of {args.K * args.K}"
            else: line += f"; needle among the {R[0]['frames_read']} frames read {pct('needle_seen', A)}"
            print(line)
    if "mono" in rows and "mono_cache" in rows:                                                 # the cache must not change the read
        m = {r["id"]: r for r in rows["mono"]}; C = [r for r in rows["mono_cache"] if r["id"] in m]
        if C: print(f"cached vs uncached monolithic read on {len(C)} questions: same answer {100 * np.mean([r['pred'] == m[r['id']]['pred'] for r in C]):.1f}%, "
                    f"max |d letter log-prob| {max(float(np.abs(np.array(r['lp']) - np.array(m[r['id']]['lp'])).max()) for r in C):.3f}")
    if "fca" in rows and "gridprobe" in rows:                                                   # where FCA's probes depart from GridProbe's
        g = {r["id"]: r for r in rows["gridprobe"]}; F_ = [r for r in rows["fca"] if r["id"] in g]
        if F_:
            agree = lambda fam: np.mean([np.mean([a == b for a, b in zip(r[fam], g[r["id"]][fam])]) for r in F_])
            print(f"FCA vs GridProbe on {len(F_)} questions: same row-probe answer {100 * agree('row_preds'):.1f}%, same column-probe answer {100 * agree('col_preds'):.1f}%, "
                  f"same direct {100 * np.mean([r['direct'] == g[r['id']]['direct'] for r in F_]):.1f}%, same select {100 * np.mean([r['pred'] == g[r['id']]['pred'] for r in F_]):.1f}%")


def flops_pass():
    """TFLOPs per question of the saved rows with GridProbe's counter (two_stage_eval_crossmodel's lm_flops_per_forward, language model only,
    every token through the head): monolithic f(n_full); GridProbe 2K f(n_probe) + f(n_focus), as GridProbe records it; FCA the K row passes
    of its store on the first question about a video, 2K copies of the request over one span each, and the same focused pass; the cached
    monolithic read f(n_full) on a first question and only its question tokens over the cached video on a later one. Token counts come from
    the processor on blank frames of each video's size (no model, one decoded frame per video). An XAttention prefill keeps its measured share of
    the attention term and adds its antidiagonal scoring, a product of the queries and keys reduced by the stride (1/stride of QK^T)."""
    from functools import lru_cache
    from types import SimpleNamespace
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from GridProbe.eval.two_stage_eval_crossmodel import infer_lm_dims
    proc = AutoProcessor.from_pretrained(args.model); tok = proc.tokenizer
    D = infer_lm_dims(SimpleNamespace(config=AutoConfig.from_pretrained(args.model))); K = args.K

    def f(nq, nk):                                   # GridProbe's lm_flops_per_forward for nq query tokens over nk keys (nq = nk: its own count)
        d, ffn, dkv, L, V = D["d_model"], D["d_ffn"], D["d_kv"], D["n_layers"], D["vocab_size"]
        return nq * L * (2 * (d * d + 2 * d * dkv) + 4 * nk * d + 2 * d * d + 6 * d * ffn) + 2 * nq * d * V

    def fx(n, r):                                    # an XAttention prefill of n tokens: the kept share of the attention term, plus the scoring
        d, L = D["d_model"], D["n_layers"]
        return f(n, n) - n * L * 4 * n * d * (1 - r["xattn_density"]) + n * L * 2 * n * d / r["xattn_stride"]

    @lru_cache(maxsize=None)
    def base(path, n, px):                           # tokens of the inputs for n frames of this video and the one-word prompt "x"
        if IVL: return int(build_internvl(proc, "x", [blank()] * n, px)["input_ids"].shape[1]) - len(tok.encode("x", add_special_tokens=False))
        wh = extract_frames_uniform(path, 1)[0].size
        return int(build_inputs(proc, "x", [Image.new("RGB", wh)] * n, px)["input_ids"].shape[1]) - len(tok.encode("x", add_special_tokens=False))

    samples, _, prompt_of, _ = benchmark(); S = {s["id"]: s for s in samples}; result = {}
    n_tok = lambda s, n, px: base(None if IVL else s["video_path"], n, px) + len(tok.encode(prompt_of(s), add_special_tokens=False))
    mono_px, focus_px = (args.probe_pixels, args.probe_pixels) if IVL else (0, args.focus_pixels)   # InternVL: everything at the probe resolution
    for arm in ARMS:
        R = [json.loads(l) for p in all_paths(arm) for l in open(p, encoding="utf-8")]; per = []
        for r in R:
            s = S.get(r["id"])
            if s is None: continue
            focus = f(*(2 * [n_tok(s, len(r["frames"]), focus_px)])) if arm in ("gridprobe", "fca") else 0
            sparse = r.get("xattn_density") is not None
            if arm == "mono": n = n_tok(s, K * K, mono_px); x = direct = fx(n, r) if sparse else f(n, n)
            elif arm == "mono_cache" and sparse: x = direct = fx(r["prefix_tokens"], r) + f(r["tokens"] - r["prefix_tokens"], r["tokens"])   # a first question
            elif arm == "mono_cache": x = direct = f(r["tokens"], r["tokens"]) if not r["shared"] else f(r["tokens"] - r["prefix_tokens"], r["tokens"])
            elif arm == "gridprobe": n = n_tok(s, K, args.probe_pixels); direct = 2 * K * f(n, n); x = direct + focus
            else:
                span = r["store_tokens"] / K                                                    # prefix and one row (or column) of blocks, on average
                direct = (0 if r["shared"] else K * f(span, span)) + 2 * K * f(r["req_tokens"], span + r["req_tokens"]); x = direct + focus
            per.append((x / 1e12, direct / 1e12, int(r.get("shared", 0))))
        if not per: print(f"{arm:10s} no rows"); continue
        a = np.array(per); first, later = a[a[:, 2] == 0], a[a[:, 2] == 1]
        result[arm] = {"n": len(a), "tflops": float(a[:, 0].mean()), "tflops_direct": float(a[:, 1].mean()),
                       "first": float(first[:, 0].mean()) if len(first) else None, "later": float(later[:, 0].mean()) if len(later) else None}
        print(f"{arm:10s} n={len(a):5d}  TFLOPs per question {a[:, 0].mean():7.1f} (direct read {a[:, 1].mean():7.1f})"
              + (f"  | first of a video {first[:, 0].mean():7.1f}, later {later[:, 0].mean():7.1f}" if len(later) else ""), flush=True)
    path = os.path.join(args.out_dir, f"{args.benchmark}_flops_{args.model.split('/')[-1]}.json"); json.dump(result, open(path, "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    if args.check: checks_internvl()
    elif args.check_sparse: check_sparse()
    elif args.summary:
        for K in needle_Ks(): args.K = K; summary()
    elif args.flops: flops_pass()
    else:
        run = None
        for K in needle_Ks(): args.K = K; run = main(run)
