"""
fca_image.py: Factorized Context Attention inside AnswerMap's protocol.

AnswerMap (answermap.probe) cuts the probe image into K row bands and K column bands, pastes each band back as its own image and reads
P(yes) to its yes/no prompt: 2K separate encodings. FCA encodes the whole probe image once and answers the same 2K questions as masked
sequences over that one encoding. The sequence for band k is a copy of AnswerMap's request (prompt, end of turn, assistant turn) that
attends to the template before the image (with <vision_start>), to the image tokens whose centre lies in band k, and to itself; never to
<vision_end>, which has read the whole image. Keys keep the positions of the whole-image read, and every copy sits where the request
follows the image in that read. Same prompt (YESNO_PROMPT), same thinking-off template, same Yes/No variants and read-out, same marginals,
map and expectation point as answermap.probe, so AnswerMap's experiment scripts score both with the same code.

    fca = FCABands(vlm)                         # vlm = answermap.VLM
    res = fca.probe(image, query, cfg)          # the dict answermap.probe returns
    M = fca.multigrid(image, query, cfg, Ks)    # answermap.multigrid_map with FCA bands at every grid; one encoding for all grids
    python fca_image.py --image IMG --query Q   # self-test on the real model (K=1 against the plain read, batched against single)

Requires the public code of AnswerMap (answermap.py) on PYTHONPATH.
"""
import hashlib
import numpy as np
import answermap as am


def fill_empty(p):
    """A band with no image token (images thinner than K token rows) takes the value of its nearest band that has one."""
    p = np.array(p, float); ok = np.where(~np.isnan(p))[0]
    for i in np.where(np.isnan(p))[0]: p[i] = p[ok[np.argmin(np.abs(ok - i))]]
    return p


class FCABands:
    def __init__(self, vlm):
        self.vlm, self.t, self.model = vlm, vlm.torch, vlm.model
        cfg = self.model.config; self.PAD, self.VE = cfg.image_token_id, cfg.vision_end_token_id
        self.merge = cfg.vision_config.spatial_merge_size; self.n_layers = cfg.text_config.num_hidden_layers
        self._head = None; self._allow_all = False; self.last = {}

    def _positions(self, inp):
        mdl = self.model.model
        try: pos, _ = mdl.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"], image_grid_thw=inp["image_grid_thw"])
        except (TypeError, KeyError): pos, _ = mdl.get_rope_index(inp["input_ids"], inp["image_grid_thw"], None, inp.get("attention_mask"))   # 4.57
        return pos                                                                           # (3, 1, L), the model's own M-RoPE positions

    def _request(self, image, query):
        """AnswerMap's conversation for one image and query (its template call, thinking off), split at <vision_end>: the head does not
        depend on the query, the copy is everything after it."""
        conv = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": am.YESNO_PROMPT.format(cond=query)}]}]]
        inp = self.vlm._inputs(conv); ids = inp["input_ids"][0].tolist(); hl = ids.index(self.VE) + 1
        return inp, hl, self._positions(inp)

    def head(self, image, query):
        """The image encoded once (template, <vision_start>, image tokens, <vision_end>), kept while the same image is asked again."""
        t = self.t; inp, hl, pos = self._request(image, query); key = hashlib.md5(image.tobytes()).hexdigest() + str(image.size)
        if self._head is None or self._head["key"] != key:
            with t.no_grad():
                out = self.model.model(input_ids=inp["input_ids"][:, :hl], pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"],
                                       position_ids=pos[:, :, :hl], use_cache=True)
            c = out.past_key_values; ids = inp["input_ids"][0, :hl].tolist(); img = [i for i, x in enumerate(ids) if x == self.PAD]
            _, h, w = [int(v) for v in inp["image_grid_thw"][0]]
            self._head = {"key": key, "K": [c.layers[l].keys for l in range(self.n_layers)], "V": [c.layers[l].values for l in range(self.n_layers)],
                          "img": img, "hh": h // self.merge, "ww": w // self.merge, "hl": hl}
            assert len(img) == self._head["hh"] * self._head["ww"]
            del out, c
        return self._head, inp["input_ids"][0, hl:], pos[:, 0, hl:]

    def band_sets(self, hd, K):
        """Head token indices each of the 2K copies reads: the template before the first image token, then its band's image tokens."""
        hh, ww = hd["hh"], hd["ww"]; r = np.repeat(np.arange(hh), ww); c = np.tile(np.arange(ww), hh)
        rb = np.floor((r + 0.5) * K / hh).astype(int); cb = np.floor((c + 0.5) * K / ww).astype(int)       # band of each token's centre
        pre = list(range(hd["img"][0])); img = np.array(hd["img"])
        if self._allow_all: return [list(range(hd["hl"]))] * (2 * K), -1                     # self-test only: every copy reads the whole head
        return [pre + img[rb == k].tolist() for k in range(K)] + [pre + img[cb == k].tolist() for k in range(K)], len(pre)

    def p_yes(self, hd, sets, cids, cpos):
        """P(yes) of each copy, all copies in one forward over a right-aligned gather of their head tokens (pads masked)."""
        t = self.t; from transformers import DynamicCache
        B, T, q, dev = len(sets), max(len(s) for s in sets), len(cids), self.model.device
        att = t.zeros((B, T), dtype=t.long, device=dev); cache = DynamicCache()
        for l in range(self.n_layers):
            Kh, Vh = hd["K"][l][0], hd["V"][l][0]                                                # (H, hl, d)
            Kb = t.zeros((B,) + (Kh.shape[0], T, Kh.shape[2]), dtype=Kh.dtype, device=dev); Vb = t.zeros_like(Kb)
            if getattr(self, "_pad_noise", False): Kb.normal_(); Vb.normal_()
            for b, s in enumerate(sets):
                idx = t.tensor(s, device=dev); Kb[b, :, T - len(s):] = Kh[:, idx]; Vb[b, :, T - len(s):] = Vh[:, idx]
                if l == 0: att[b, T - len(s):] = 1
            cache.update(Kb, Vb, l)
        with t.no_grad():
            out = self.model.model.language_model(input_ids=cids[None].expand(B, q).to(dev), position_ids=cpos[:, None, :].expand(3, B, q).to(dev),
                                                  attention_mask=t.cat([att, t.ones((B, q), dtype=t.long, device=dev)], 1),
                                                  past_key_values=cache, use_cache=True)
            lg = self.model.lm_head(out.last_hidden_state[:, -1]).float()
        lp = t.log_softmax(lg, -1); yes = t.tensor(self.vlm.yes_ids, device=dev); no = t.tensor(self.vlm.no_ids, device=dev)
        return t.softmax(t.stack([t.logsumexp(lp[:, yes], -1), t.logsumexp(lp[:, no], -1)], -1), -1)[:, 0].cpu().numpy()   # AnswerMap's read-out

    def probe(self, image, query, cfg, K=None):
        """answermap.probe with FCA bands: the same dict (c_row, c_col, M, point)."""
        if isinstance(image, str): image = am.load_image(image, cfg.max_side)
        K = int(K or cfg.K); W, H = image.size; hd, cids, cpos = self.head(image, query); sets, npre = self.band_sets(hd, K)
        vals = self.p_yes(hd, sets, cids, cpos).astype(float); empty = np.array([len(s) == npre for s in sets]); vals[empty] = np.nan
        c_row, c_col = fill_empty(vals[:K]), fill_empty(vals[K:]); M = np.outer(c_row, c_col)
        pr, pc = c_row / (c_row.sum() + 1e-12), c_col / (c_col.sum() + 1e-12)
        er, ec = float((np.arange(K) * pr).sum()), float((np.arange(K) * pc).sum())
        self.last = {"empty_bands": int(empty.sum()), "image_tokens": len(hd["img"]), "copy_tokens": int(len(cids))}
        return {"image": image, "size": (W, H), "c_row": c_row, "c_col": c_col, "M": M, "point": ((ec + 0.5) * W / K, (er + 0.5) * H / K)}

    def multigrid(self, image, query, cfg, Ks=(3, 5), combine="product"):
        """answermap.multigrid_map, its own upsampling and combination, with FCA bands at every grid (the image encoded once for all)."""
        orig = am.probe; am.probe = lambda vlm, img, q, c, K=None: self.probe(img, q, c, K)
        try: return am.multigrid_map(self.vlm, image, query, cfg, Ks, combine)
        finally: am.probe = orig


def self_test(vlm, image, query, cfg):
    """On the real model: with every copy allowed the whole head, K=1 must equal AnswerMap's plain read of the whole image (same tokens,
    same computation, bf16 noise only); the 2K band copies in one batch must equal the copies one at a time with noise in the pad slots;
    and the band map must differ from the K=1 read."""
    fca = FCABands(vlm); fca._allow_all = True
    p_fca = fca.probe(image, query, cfg, K=1)["c_row"][0]; fca._allow_all = False
    p_plain = vlm.p_yes_batch([[image]], query)[0]
    print(f"  K=1, every copy reading the whole head, vs AnswerMap's plain read of the image: {p_fca:.4f} vs {p_plain:.4f}, |d| {abs(p_fca - p_plain):.4f}")
    hd, cids, cpos = fca.head(image, query); sets, _ = fca.band_sets(hd, cfg.K)
    single = np.array([fca.p_yes(hd, [s], cids, cpos)[0] for s in sets]); fca._pad_noise = True; batch = fca.p_yes(hd, sets, cids, cpos); fca._pad_noise = False
    print(f"  {len(sets)} band copies one at a time vs one batch with noise in the pad slots: max |dP| {np.abs(single - batch).max():.4f}")
    res = fca.probe(image, query, cfg); ref = am.probe(vlm, image, query, cfg)
    print(f"  K={cfg.K}: FCA rows {np.round(res['c_row'], 2).tolist()} cols {np.round(res['c_col'], 2).tolist()} ({fca.last})")
    print(f"        AnswerMap strips rows {np.round(ref['c_row'], 2).tolist()} cols {np.round(ref['c_col'], 2).tolist()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None); ap.add_argument("--query", default=None)
    ap.add_argument("--data", default=None, help="an AnswerMap jsonl: test on its first row (with --images_dir)"); ap.add_argument("--images_dir", default=".")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct"); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--probe_side", type=int, default=512); ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    cfg = am.Config(model_name=a.model_name, K=a.K, max_side=a.probe_side, min_pixels=a.min_pixels, attn=a.attn)
    if a.data:
        import json, os
        r = json.loads(open(a.data, encoding="utf-8").readline()); a.image = os.path.join(a.images_dir, r["image"]); a.query = r.get("query") or r.get("question")
    vlm = am.VLM(cfg); print(f"self-test on {a.image}: {a.query!r}"); self_test(vlm, am.load_image(a.image, a.probe_side), a.query, cfg)
