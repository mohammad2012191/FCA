#!/usr/bin/env python3
"""
fca_text.py: Factorized Context Attention (FCA) on long text, next to the reads it is compared with.

The text is cut into blocks of --chunk_size tokens on a K x K grid. A local span is K consecutive blocks (a row), a strided span is
every K-th block (a column). FCA encodes each block once, within its local span, at stair positions, and keeps the keys and values
of every layer in one store. A request is answered by 2K sequences: each gathers the stored keys and values of one span, appends a
copy of the request, and the model runs over the request tokens only. The scores are pooled into an answer (direct) and an evidence
map, whose top K blocks are read once more (select). The store carries no question, so later questions on the same text reuse it.

Arms (--arms, comma separated):
  mask_stair    FCA, the method of the paper (local-span encoding, stair positions)
  periscope     grid probing as prompts (the emulated read): 2K separate prompts per question
  window        the monolithic read: the longest prefix that fits the window
  window_reuse  the monolithic read with a prefix cache kept per text
  fca_append, mono_conv   conversations with long turns (--dataset conversation): the store grows by rows / the monolithic read
  Ablations: mask_span (document positions), mask_stair_alone (blocks encoded alone), mask_stair_both (strided spans encoded too),
  mask_doc, mask_chunk, mask_all (one query over every block), mask_all_anchor.
Baselines of the monolithic read (--sparse_prefill xattention | tokensparse) import the released code of those methods from PYTHONPATH.

Datasets (--dataset): infinitebench (En.MC), longbench (LongBench v2), scbench, needle, conversation.
Every forward is a padded batch over a key-value cache, so any attention backend serves it (--attn flash_attention_2, sdpa, ...).
Results are written to --out_dir as one jsonl file per arm, and runs resume from those files.

  python fca_text.py --check                                                   # equivalence tests on a tiny random model, CPU
  python fca_text.py --dataset infinitebench --attn flash_attention_2 --arms window_reuse,periscope,mask_stair
  python fca_text.py --dataset infinitebench --arms window_reuse,periscope,mask_stair --summary
"""
import argparse, os, json, time, math, string, re, random, hashlib
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="dense attention only: Qwen3.5 has Gated DeltaNet layers that no mask can restrict")
ap.add_argument("--dataset", default="longbench", choices=["longbench", "infinitebench", "scbench", "needle", "conversation"])
ap.add_argument("--split", default="short,medium", help="longbench: comma list of short, medium, long; empty = all. scbench: qa_eng (open-ended) or choice_eng")
ap.add_argument("--arms", default="window,periscope,mask_doc,mask_chunk", help="from window, window_reuse, periscope, mask_doc, mask_chunk, mask_span, mask_stair, mask_stair_alone, "
                "mask_stair_both (ablation d: strided spans also encoded in their own pass), and the one-query ablations mask_all (parallel encoding) "
                "and mask_all_anchor (with Star Attention's anchor)")
ap.add_argument("--device_map", default="", help="window arms only: 'balanced' spreads the layers evenly over the visible GPUs (Qwen3-32B's 64 GB of "
                "bf16 weights and its 131k cache need two A100s); empty = one GPU")
ap.add_argument("--chunk_size", type=int, default=500)
ap.add_argument("--window", type=int, default=262144, help="the model's context length: the window read fits this, spans longer than this are skipped (as in Periscope)")
ap.add_argument("--span_batch", type=int, default=4, help="Periscope spans per forward pass")
ap.add_argument("--chunk_batch", type=int, default=32, help="chunks per forward pass in the masked arms")
ap.add_argument("--copy_batch", type=int, default=8, help="question copies (spans) per forward pass in the masked arms; 1 = one span at a time")
ap.add_argument("--no_pin", action="store_true", help="do not page-lock an offloaded chunk cache (pageable host memory, staged copies)")
ap.add_argument("--offload", action="store_true", help="always keep the chunk cache on the CPU, gather each span to the GPU")
ap.add_argument("--host_cache_gb", type=float, default=0.0, help="cap on host memory for an offloaded chunk cache; 0 = detect from the cgroup / SLURM limit / physical RAM")
ap.add_argument("--gpu_cache_gb", type=float, default=24.0, help="chunk caches estimated above this size go to the CPU automatically (LongBench medium reaches 550k tokens = 81 GB)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out_dir", default="runs_text")
ap.add_argument("--check", action="store_true", help="equivalence tests on a tiny random model (CPU), then exit")
ap.add_argument("--summary", action="store_true", help="read the saved rows of --arms in --out_dir and print the combined table (accuracy all / in window / beyond window, costs, cache reuse), then exit")
ap.add_argument("--no_reuse", action="store_true", help="masked arms: encode the book again for every question (the cold-start cost on every row)")
ap.add_argument("--attn", default="sdpa", help="attention backend passed to the loader: sdpa, flash_attention_2, eager, flex_attention, or a kernels-hub name such as kernels-community/flash-attn (the tiny --check model stays on sdpa or eager)")
ap.add_argument("--gen_tokens", type=int, default=0, help="decode timing: after the answer token, greedily generate this many tokens through the read's cache and record ms per token; 0 = off. The text is discarded and the time is excluded from latency")
ap.add_argument("--prefill_chunk", type=int, default=32768, help="window_reuse: tokens per prefill forward when the document is encoded into its cache")
ap.add_argument("--reuse_margin", type=int, default=1024, help="window_reuse: the document is cut once per book to the window minus this margin, so the cache is shared by every question")
ap.add_argument("--cache_dir", default=None)
ap.add_argument("--sparse_prefill", default="none", choices=["none", "xattention", "tokensparse"], help="sparse-prefill baselines for the monolithic arms only: "
                "XAttention (ICML 2025) on window_reuse, Token Sparse Attention (ICML 2026) on window; their released code must be on PYTHONPATH")
ap.add_argument("--xattn_stride", type=int, default=8, help="XAttention: antidiagonal stride")
ap.add_argument("--xattn_threshold", default="0.9", help="XAttention: block threshold for every head, or 'profile' for the authors' per-head thresholds of Llama-3.1-8B at this stride")
ap.add_argument("--ts_coverage", type=float, default=0.005, help="Token Sparse Attention: drop the tokens that together receive less than this share of the window's attention")
ap.add_argument("--ts_window", type=int, default=128, help="Token Sparse Attention: the last tokens whose attention scores the others")
ap.add_argument("--ts_kernel", type=int, default=7, help="Token Sparse Attention: pooling width of the scores")
ap.add_argument("--ts_min_tokens", type=int, default=1024, help="Token Sparse Attention: tokens always kept")
ap.add_argument("--ts_layers", default="", help="Token Sparse Attention: comma list of sparse layers; empty = the authors' profile (Llama-3.1-8B, Mistral-Nemo), else the second half")
ap.add_argument("--check_sparse", action="store_true", help="on the GPU: the sparse-prefill hook on one real prompt against the dense backend, then exit")
ap.add_argument("--max_turns", type=int, default=5, help="scbench: the first turns of every session, as SCBench's default")
ap.add_argument("--max_new", type=int, default=40, help="open-ended answers: greedy tokens until an end-of-turn token or this many (SCBench's cap)")
ap.add_argument("--needle_lengths", default="32768,65536,131072,262144,524288", help="needle: haystack lengths in tokens of the model's tokenizer")
ap.add_argument("--needle_depths", type=int, default=11, help="needle: depths from 0 to 100%% in this many even steps")
ap.add_argument("--needle_trials", type=int, default=3, help="needle: seeded needles per length and depth")
ap.add_argument("--conv_n", type=int, default=8, help="conversation: number of conversations")
ap.add_argument("--conv_turns", type=int, default=20, help="conversation: turns per conversation (at most 38, one needle word per turn)")
ap.add_argument("--conv_turn_tokens", type=int, default=32768, help="conversation: tokens of new text that every turn adds")
ap.add_argument("--conv_K", type=int, default=0, help="conversation, fca_append: columns of the grid, fixed for the whole conversation; "
                "0 = ceil(sqrt(blocks of the full conversation)), so the grid is square at the last turn")
ap.add_argument("--yarn", type=float, default=0.0, help="monolithic arms only: YaRN rope scaling with this factor, as the Qwen3 model cards prescribe "
                "beyond their native 32k (Qwen3-14B and 32B: 4.0 for 131k); FCA reads with native positions; 0 = off")
ap.add_argument("--yarn_orig", type=int, default=32768, help="YaRN: the model's original context length (original_max_position_embeddings)")
ap.add_argument("--select_budget", default="k", choices=["k", "skew"], help="blocks the select read-out reads: k = K top blocks of the map; "
                "skew = GridProbe's skewness selector on the map of span confidences (2 to the window)")
ap.add_argument("--select_mode", default="reread", choices=["reread", "store", "both"], help="the select read-out of the masked arms: reread encodes the K "
                "selected blocks again with the request; store gathers their stored keys and values, re-rotated to consecutive positions; both records "
                "the two from the same probes (pred is the reread, pred_store the gather)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
ARMS = [a for a in args.arms.split(",") if a]
assert not args.device_map or set(ARMS) <= {"window", "window_reuse"}, "--device_map splits the monolithic read only: the masked arms keep their store on one GPU"
SPLITS = [s for s in args.split.split(",") if s]
if args.dataset == "scbench": SPLITS = [s for s in SPLITS if s in ("qa_eng", "choice_eng")][:1] or ["qa_eng"]
SPLIT_TAG = "enmc" if args.dataset == "infinitebench" else "niah" if args.dataset == "needle" else "conv" if args.dataset == "conversation" else (f"sc_{SPLITS[0]}" if args.dataset == "scbench" else ("-".join(SPLITS) or "all"))

def host_ram_bytes():
    """The memory this process may use on the host: the smallest of the cgroup limit (a SLURM job runs inside one), SLURM's
    per-node allocation, the physical RAM, and --host_cache_gb. The kernel kills the process above the cgroup limit, so the
    physical RAM alone is the wrong number."""
    cands = {}
    try: cands["physical"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception: pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 1 << 60: cands["cgroup"] = int(v)
        except Exception: pass
    if os.environ.get("SLURM_MEM_PER_NODE", "").isdigit(): cands["slurm"] = int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 * 1024
    if args.host_cache_gb > 0: cands["--host_cache_gb"] = args.host_cache_gb * 1e9
    if not cands: return float("inf"), "unknown"
    src = min(cands, key=cands.get); return cands[src], src
PREAMBLE = ("Read the book and answer the question. Be very concise in your answer.\n\n" if (args.dataset == "scbench" and SPLITS[0] == "qa_eng")      # SCBench's own instructions
            else "Read the book and answer the question.\n\n" if args.dataset in ("infinitebench", "scbench")
            else "Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.\n\n"
            if args.dataset == "needle" else "")                                                    # RULER's instruction for its needle tasks
MARK = "\x00SPAN\x00"                                                                             # where the text goes in a prompt, to split it into prefix and request

# ============================================================== Periscope, copied verbatim from periscope/core.py and periscope/reader.py
def letters(n):
    """Answer letters for n options and the abstain letter after them (4 options: A-D and E)."""
    if not 2 <= n <= 25:
        raise ValueError("between 2 and 25 options")
    return list(string.ascii_uppercase[:n]), string.ascii_uppercase[n]

def chat(tokenizer, body):
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def qa_body(text, question, options, abstain):
    ys, ab = letters(len(options))
    lines = ["%s. %s" % (y, o) for y, o in zip(ys, options)]
    if abstain:
        lines.append("%s. Unsure" % ab)
    return "%s\n\nQuestion: %s\n%s\nAnswer with a single letter:" % (text, question, "\n".join(lines))

def qa_prompt(tokenizer, text, question, options, abstain):
    return chat(tokenizer, PREAMBLE + qa_body(text, question, options, abstain))

# ---- the request of a read: the conversation so far and the current question with the cue of the read
YESNO = "Does the text above contain the information needed to answer this question? Answer Yes or No."

def turn_block(question, options, cue):
    """One user turn without the text. cue: 'letter' (options, a single letter), 'abstain' (the same with the Unsure option, the probes of
    multiple choice), 'gen' (an open answer, SCBench's 'Question: ...\\nAnswer:'), 'yesno' (the relevance probe of open-ended questions)."""
    if cue == "gen": return "Question: %s\nAnswer:" % question
    if cue == "yesno": return "Question: %s\n%s" % (question, YESNO)
    return qa_body("", question, options, cue == "abstain")[2:]                                    # the multiple-choice block of qa_body, without its text

def build_prompt(tokenizer, text, ex, cue):
    """The whole chat of one read, as SCBench's multi-turn protocol builds it: the first user message holds the instruction, the text and the
    first question of the conversation; every earlier turn is answered by the assistant with its gold answer; the last user message is the
    current question with the cue of this read. With no history this is exactly qa_prompt, so the single-turn benchmarks are unchanged, and the
    part before the text (the stored prefix) never depends on the history or the cue."""
    hist = ex.get("history") or []; past_cue = "gen" if ex["options"] is None else "letter"
    turns = [(h["question"], h["options"], past_cue, h["answer"]) for h in hist] + [(ex["question"], ex["options"], cue, None)]
    msgs = []
    for i, (q, o, c, a) in enumerate(turns):
        body = turn_block(q, o, c)
        msgs.append({"role": "user", "content": (PREAMBLE + text + "\n\n" + body) if i == 0 else body})
        if a is not None: msgs.append({"role": "assistant", "content": a})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def read_spec(ex):
    """The cues and answers of a turn: multiple choice probes with the Unsure option and reads a letter; an open question probes with Yes or
    No (its probe score is the log-odds of Yes against No, Yes playing the answer and No the abstain option) and generates its answer."""
    if ex["options"] is None: return "yesno", "gen", ["Yes"], "No"
    ys, ab = letters(len(ex["options"])); return "abstain", "letter", ys, ab

def request_text(tokenizer, ex, cue):
    """What follows the text in every read of this turn: the rest of the first user message, the earlier turns, the current turn, the cue."""
    return build_prompt(tokenizer, MARK, ex, cue).split(MARK)[1]

# ---- open-ended scoring: InfiniteBench's En.QA metric (normalized token F1), and SCBench's long-dialogue match kept for reference
def normalize_answer(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation)); s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def qa_f1(pred, golds):
    best = 0.0; p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split(); common = sum((Counter(p) & Counter(gt)).values())
        if common: prec, rec = common / len(p), common / len(gt); best = max(best, 2 * prec * rec / (prec + rec))
    return best

def sc_match(pred, golds):
    """SCBench's scorer for this task (get_score_one_longdialogue_qa_eng): a gold string, upper-cased, inside the upper-cased answer."""
    pred = pred.strip().upper(); return float(any(g.upper() in pred for g in golds))

def row_score(r, key, metric="main"):
    """The score of the prediction under key: accuracy for multiple choice; for an open answer (gold is a list) token F1, or SCBench's match."""
    if not isinstance(r["gold"], list): return float(r[key] == r["gold"])
    return qa_f1(r[key], r["gold"]) if metric == "main" else sc_match(r[key], r["gold"])

READOUTS = {"pred": "select", "direct": "direct", "pred_store": "store"}                          # the read-outs a row can carry

def collapse(topmap, keys):
    out = {}
    for k, v in topmap.items():
        s = k.strip()
        if s in keys:
            out[s] = max(out.get(s, -1e9), float(v))
    floor = (min(float(v) for v in topmap.values()) - 2.0) if topmap else -1e9
    return out, floor

def log_odds(topmap, answers, abstain):
    """Eq. 1: the log-odds of each answer against the abstain option, with no default clamp."""
    lp, floor = collapse(topmap, set(answers) | {abstain})
    has_a = abstain in lp
    out = {}
    for y in answers:
        if y in lp and has_a:
            out[y] = lp[y] - lp[abstain]
        elif y in lp:
            out[y] = lp[y] - floor
        elif has_a:
            out[y] = floor - lp[abstain]
        else:
            out[y] = floor - 50.0
    return out

def read_scores(topmap, answers):
    lp, floor = collapse(topmap, set(answers))
    return {y: lp.get(y, floor) for y in answers}

def chunk(tokenizer, text, chunk_size):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    decoded = tokenizer.batch_decode(pieces, skip_special_tokens=True)
    return [t for t in (x.strip() for x in decoded) if t]

def grid(chunks):
    """K = ceil(sqrt(N)), padded to K^2 with empty chunks that are dropped from the spans. Returns K, local [(i, text)], strided [(j, text)]."""
    k = math.ceil(math.sqrt(len(chunks)))
    padded = chunks + [""] * (k * k - len(chunks))
    cells = [padded[i * k:(i + 1) * k] for i in range(k)]
    local = [(i, " ".join(c for c in cells[i] if c)) for i in range(k)]
    strided = [(j, " ".join(cells[i][j] for i in range(k) if cells[i][j])) for j in range(k)]
    return k, [(i, s) for i, s in local if s], [(j, s) for j, s in strided if s]

def compose(local, strided):
    """Eq. 3: score(y) = max_i L_i(y) + max_j S_j(y)."""
    answers = sorted({y for d in list(local.values()) + list(strided.values()) for y in d})
    out = {}
    for y in answers:
        s = max(d[y] for d in local.values())
        if strided:
            s += max(d[y] for d in strided.values())
        out[y] = s
    return out

def evidence_map(local, strided, answers):
    """Eq. 4: M[i,j] = max_y (L_i(y) + S_j(y)), cell (i, j) being chunk i*K + j."""
    return {(i, j): max(li[y] + sj[y] for y in answers) for i, li in local.items() for j, sj in strided.items()}

def skew_budget(L, S, ys, k, n_chunks):
    """GridProbe's skewness selector (two_stage_eval.skewness_selector) as the budget of the select read: each span's confidence is its top
    probability over the answers (the sigmoid of the Yes log-odds for an open question), the map is the outer product of row and column
    confidences over the real blocks, and the budget is ceil(n / (1 + 3 (|skew| + 0.5 max(0, excess kurtosis)))) blocks, at least 2 and at
    most what the window holds."""
    def conf(d):
        v = np.array([d[y] for y in ys], float)
        return float(1 / (1 + np.exp(-v[0]))) if len(v) == 1 else float(1 / np.exp(v - v.max()).sum())
    cr = {int(i): conf(d) for i, d in L.items()}; cc = {int(j): conf(d) for j, d in S.items()}
    flat = np.array([cr[i] * cc[j] for i in cr for j in cc if i * k + j < n_chunks]); hi = max(2, (args.window - 2048) // args.chunk_size)
    mu, sd = flat.mean(), flat.std()
    if sd < 1e-10: return int(min(len(flat), hi))
    skew = ((flat - mu) ** 3).mean() / sd ** 3; kurt = ((flat - mu) ** 4).mean() / sd ** 4 - 3.0
    return int(np.clip(np.ceil(len(flat) / (1 + 3 * (abs(skew) + 0.5 * max(0.0, kurt)))), 2, hi))

def select(cells, k, n_chunks, budget):
    picked = []
    for i, j in sorted(cells, key=cells.get, reverse=True):
        if i * k + j < n_chunks:
            picked.append(i * k + j)
        if len(picked) == budget:
            break
    return sorted(picked)

def fit_prefix(tokenizer, ids, build, window, manner="prefix"):
    def cut(n):
        if manner == "prefix" or n >= len(ids):
            return ids[:n]
        return ids[:n // 2] + ids[-(n - n // 2):]
    overhead = len(tokenizer.encode(build(""), add_special_tokens=False)) + 16
    n = min(len(ids), max(0, window - overhead))
    while n > 0:
        text = tokenizer.decode(cut(n), skip_special_tokens=True)
        if len(tokenizer.encode(build(text), add_special_tokens=False)) < window:
            break
        n = int(n * 0.98)
    return cut(n)

# the same grid in chunk INDICES (what the masked arms gather): local i = chunks iK..iK+K-1, strided j = chunks j, j+K, ...
def grid_indices(n, k):
    local = [(i, [m for m in range(i * k, min((i + 1) * k, n))]) for i in range(k)]
    strided = [(j, [m for m in range(j, n, k)]) for j in range(k)]
    return [(i, ix) for i, ix in local if ix], [(j, ix) for j, ix in strided if ix]

# ============================================================== data (as scripts/run_qa.py)
ENMC_URL = "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/longbook_choice_eng.jsonl"
NEEDLE_WORDS = ["anchor", "apple", "bridge", "candle", "castle", "cobalt", "compass", "copper", "crystal", "dolphin", "falcon", "forest", "glacier", "harbor",
                "island", "jasmine", "lantern", "maple", "marble", "meadow", "mirror", "orchid", "pepper", "pigeon", "planet", "puzzle", "quartz", "rocket",
                "saddle", "silver", "spider", "tiger", "tulip", "velvet", "violin", "walnut", "willow", "zebra"]

def make_needles(tok, haystack, lengths, n_depths, trials):
    """RULER's single needle: 'One of the special magic numbers for {word} is: {7-digit number}.' in a haystack of L tokens of the model's tokenizer
    (the haystack cut to L minus the needle, the needle inserted), at the sentence boundary nearest to depth d of the text, for n_depths depths from
    0 to 100% and `trials` needles each, word and number seeded by (L, depth, trial). The question is RULER's; the answer is scored by string match."""
    ids = tok.encode(haystack, add_special_tokens=False); out = []
    assert len(ids) >= max(lengths), f"haystack of {len(ids)} tokens is shorter than {max(lengths)}"
    for L in lengths:
        for di in range(n_depths):
            d = di / max(n_depths - 1, 1)
            for tr in range(trials):
                rng = random.Random(f"{L}-{di}-{tr}"); word = rng.choice(NEEDLE_WORDS); value = str(rng.randint(1000000, 9999999))
                needle = f"One of the special magic numbers for {word} is: {value}."
                body = tok.decode(ids[:L - len(tok.encode(" " + needle + " ", add_special_tokens=False))], skip_special_tokens=True)
                bounds = [0] + [m.end() for m in re.finditer(r"[.!?][\"'”’]?\s+", body)] + [len(body)]
                b = min(bounds, key=lambda x: abs(x - d * len(body)))
                text = (body[:b] + (" " if b and not body[b - 1].isspace() else "") + needle + " " + body[b:]).strip()
                out.append({"id": f"L{L}-d{round(100 * d)}-t{tr}", "context": text, "question": f"What is the special magic number for {word} mentioned in the provided text?",
                            "options": None, "gold": [value], "split": f"niah_{L // 1024}k", "length": L, "depth": round(100 * d)})
    return out

def make_conversations(tok, haystack, n_conv, n_turns, turn_tokens):
    """Conversations with long turns. Turn t of a conversation adds the next turn_tokens tokens of the haystack with one needle sentence inside
    (RULER's 'One of the special magic numbers for {word} is: {7-digit number}.', a different word in every turn, at a seeded random sentence
    boundary) and asks for the number of the needle of one turn so far, chosen uniformly among turns 0..t. One example per turn: 'segments' is
    the list of every turn's text of the conversation (shared by its examples), 'context' the conversation's text up to this turn, 'target_turn'
    the turn that holds the asked needle and 'age' how many turns ago that was. Answers are scored by string match, as the needle test."""
    assert n_turns <= len(NEEDLE_WORDS), f"at most {len(NEEDLE_WORDS)} turns: one needle word per turn"
    ids = tok.encode(haystack, add_special_tokens=False); need = n_conv * n_turns * turn_tokens; out = []
    assert len(ids) >= need, f"haystack of {len(ids)} tokens is shorter than the {need} the conversations need"
    for c in range(n_conv):
        rng = random.Random(f"conv-{c}"); words = rng.sample(NEEDLE_WORDS, n_turns); segs, vals = [], []
        for t_i in range(n_turns):
            value = str(rng.randint(1000000, 9999999)); needle = f"One of the special magic numbers for {words[t_i]} is: {value}."; a = (c * n_turns + t_i) * turn_tokens
            body = tok.decode(ids[a:a + turn_tokens - len(tok.encode(" " + needle + " ", add_special_tokens=False))], skip_special_tokens=True)
            bounds = [0] + [m.end() for m in re.finditer(r"[.!?][\"'”’]?\s+", body)] + [len(body)]; at = rng.random() * len(body)
            b = min(bounds, key=lambda x: abs(x - at))
            segs.append((body[:b] + (" " if b and not body[b - 1].isspace() else "") + needle + " " + body[b:]).strip()); vals.append(value)
        for t_i in range(n_turns):
            tgt = rng.randint(0, t_i)
            out.append({"id": f"c{c}-t{t_i}", "context": "\n\n".join(segs[:t_i + 1]), "question": f"What is the special magic number for {words[tgt]} mentioned in the provided text?",
                        "options": None, "gold": [vals[tgt]], "split": "conv", "conv": c, "turn": t_i, "target_turn": tgt, "age": t_i - tgt, "segments": segs, "n_turns": n_turns})
    return out

def load_examples():
    from datasets import load_dataset
    out = []
    if args.dataset == "conversation":
        # the haystack: the distinct En.MC novels in order, joined; every turn takes the next --conv_turn_tokens tokens of it
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir); parts, seen = [], set()
        for r in load_dataset("json", data_files=ENMC_URL, split="train"):
            if r["context"] not in seen: seen.add(r["context"]); parts.append(r["context"])
        out = make_conversations(tok, "\n\n".join(parts), args.conv_n, args.conv_turns, args.conv_turn_tokens)
        return out[:args.limit] if args.limit else out
    if args.dataset == "needle":
        # the haystack: the distinct En.MC novels in order, joined until they cover the longest length
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir); lengths = [int(x) for x in args.needle_lengths.split(",") if x]; parts, seen, chars = [], set(), 0
        for r in load_dataset("json", data_files=ENMC_URL, split="train"):
            if r["context"] in seen: continue
            seen.add(r["context"]); parts.append(r["context"]); chars += len(r["context"])
            if chars > 6 * max(lengths): break                                                    # about 4 characters per token, with room to spare
        out = make_needles(tok, "\n\n".join(parts), lengths, args.needle_depths, args.needle_trials)
    elif args.dataset == "infinitebench":
        url = ENMC_URL
        for r in load_dataset("json", data_files=url, split="train"):
            options = list(r["options"])
            if len(options) != 4 or r["answer"][0] not in options:
                continue
            out.append({"id": str(r["id"]), "context": r["context"], "question": r["input"], "options": options, "gold": "ABCD"[options.index(r["answer"][0])], "split": "enmc"})
    elif args.dataset == "scbench":
        # SCBench (microsoft/SCBench, ICLR 2025): InfiniteBench En.QA / En.MC grouped into sessions over one novel. One example per turn, in order,
        # the first --max_turns turns of each session, the earlier turns with their gold answers as the history (SCBench's multi-turn mode)
        task = SPLITS[0]
        for r in load_dataset("microsoft/SCBench", f"scbench_{task}", split="test"):
            hist = []
            for t_i, turn in enumerate(list(r["multi_turns"])[:args.max_turns]):
                opts = list(turn["options"]) if task == "choice_eng" else None
                gold = "ABCD"[opts.index(turn["answer"])] if opts else [turn["answer"]]
                out.append({"id": f"{r['id']}-{t_i}", "context": r["context"], "question": turn["input"], "options": opts, "gold": gold,
                            "split": f"sc_{task}", "history": list(hist), "turn": t_i})
                hist.append({"question": turn["input"], "options": opts, "answer": turn["answer"]})
    else:
        for r in load_dataset("THUDM/LongBench-v2", split="train"):
            if SPLITS and r["length"] not in SPLITS:
                continue
            out.append({"id": r["_id"], "context": r["context"], "question": r["question"], "options": [r["choice_A"], r["choice_B"], r["choice_C"], r["choice_D"]],
                        "gold": r["answer"].strip().upper(), "split": r["length"], "domain": r["domain"]})
    return out[:args.limit] if args.limit else out

# ============================================================== sparse-prefill baselines
class _OneGroup:
    """The attention module seen with one key-value group: Token Sparse Attention hands the backend keys already repeated to every query head,
    and a backend that repeats keys itself (SDPA without GQA support) must not repeat them again. Every other attribute is the module's."""
    num_key_value_groups = 1
    def __init__(self, module): self._m = module
    def __getattr__(self, name): return getattr(self._m, name)


class SparsePrefill:
    """XAttention (Xu et al., ICML 2025) and Token Sparse Attention (Jo et al., ICML 2026), the training-free sparse-prefill baselines, as a wrapper
    around the attention backend. transformers looks the backend up by name in its attention registry at every attention call, so registering this
    object under that name routes every call through it; nothing in the model changes. It is off unless LM switches it on inside a monolithic arm.
    When on, a forward with one unpadded sequence whose queries and keys are the same positions (a whole prompt or document in one piece, nothing
    cached before it) goes through the method; every other forward (a request over a cache, decoding, batches) goes to the original backend.
    Both methods are prefill-only and repeat the key-value heads to the query heads, as their releases do. Their code is imported unchanged from
    the two repositories (github.com/mit-han-lab/x-attention, github.com/dongwonjo/Token-Sparse-Attention) on PYTHONPATH."""
    TS_PROFILE = {"meta-llama/Llama-3.1-8B-Instruct": list(range(15, 31)),                         # the authors' profiled sparse layers (their set_model)
                  "mistralai/Mistral-Nemo-Instruct-2407": [14, 15] + list(range(20, 38))}

    def __init__(self, method, backend, model_name, n_layers, fns=None):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.t, self.method, self.backend, self.on, self.n_sparse, self.kept = torch, method, backend, False, 0, []
        self.registry = ALL_ATTENTION_FUNCTIONS; self.orig = ALL_ATTENTION_FUNCTIONS[backend]
        self.fns = fns if fns is not None else self.release(method)
        if method == "xattention":
            self.stride = args.xattn_stride
            if args.xattn_threshold == "profile":
                prof = self.fns["llama_profile"](self.stride)
                assert len(prof) == n_layers and all(len(r) == len(prof[0]) for r in prof), "the per-head profile exists for Llama-3.1-8B (32 layers x 32 heads) only"
                self.thr = [torch.tensor(r, dtype=torch.float32) for r in prof]
            else:
                self.thr = [float(args.xattn_threshold)] * n_layers
            print(f"sparse prefill: XAttention, stride {self.stride}, threshold {args.xattn_threshold}", flush=True)
        else:
            self.layers = [int(x) for x in args.ts_layers.split(",") if x.strip()] or self.TS_PROFILE.get(model_name, list(range(n_layers // 2, n_layers)))
            self.clusters = []
            for i in range(n_layers):
                c = self.fns["SparseCluster"](); c.layer_idx, c.min_tokens, c.window_size, c.kernel_size = i, args.ts_min_tokens, args.ts_window, args.ts_kernel
                c.coverage = args.ts_coverage if i in self.layers else 0.0; self.clusters.append(c)
            print(f"sparse prefill: Token Sparse Attention, coverage {args.ts_coverage}, window {args.ts_window}, pool {args.ts_kernel}, "
                  f"at least {args.ts_min_tokens} tokens, sparse layers {self.layers}", flush=True)
        type(ALL_ATTENTION_FUNCTIONS).register(backend, self)

    @staticmethod
    def release(method):
        """The released implementations, unchanged."""
        if method == "xattention":
            from xattn.src.Xattention import Xattention_prefill
            from xattn.threshold import llama_threshold
            return {"xattention_prefill": Xattention_prefill, "llama_profile": lambda s: getattr(llama_threshold, f"llama_fuse_{s}")}
        from sparse_attn.token_sparse.sparse_cluster import SparseCluster
        return {"SparseCluster": SparseCluster}

    def remove(self):
        type(self.registry).register(self.backend, self.orig)

    def __call__(self, module, query, key, value, attention_mask, **kw):
        b, h, q_len, d = query.shape
        whole = (self.on and b == 1 and q_len > 1 and key.shape[2] == q_len
                 and (attention_mask is None or attention_mask.dim() != 2 or bool(attention_mask.all())))
        if not whole: return self.orig(module, query, key, value, attention_mask, **kw)
        n_rep = h // key.shape[1]
        k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key                                 # transformers' repeat_kv: kv head g serves query heads g*n_rep onwards
        v = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
        if self.method == "xattention":
            scaling = kw.get("scaling") or d ** -0.5
            assert abs(scaling - d ** -0.5) < 1e-6, "XAttention's kernels use 1/sqrt(head_dim)"
            thr = self.thr[module.layer_idx]; thr = thr.to(query.device) if self.t.is_tensor(thr) else thr
            out = self.fns["xattention_prefill"](query, k, v, stride=self.stride, norm=1, threshold=thr, use_triton=True)   # (1, heads, q, d), blocks of 128
            self.n_sparse += 1
            return out.transpose(1, 2).contiguous(), None
        c = self.clusters[module.layer_idx]; idx = c.get_index(query, k)
        if idx is None: return self.orig(module, query, key, value, attention_mask, **kw)             # a dense layer, or a prompt too short to prune
        q2, k2, v2 = c.compress_qkv(query, k, v, idx)                                                 # each head keeps its own tokens, in their original order
        kw2 = {n: x for n, x in kw.items() if n not in ("position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")}
        out, _ = self.orig(_OneGroup(module), q2, k2, v2, None, **kw2)                               # causal attention among the kept tokens
        self.n_sparse += 1; self.kept.append(idx.shape[-1] / q_len)
        return c.decompress_output(out, idx).view(b, q_len, h, d), None                               # a dropped token gets a zero attention output


# ============================================================== the model
def apply_yarn(cfg):
    """YaRN as the Qwen3 model cards prescribe: rope type yarn, --yarn as the factor, --yarn_orig as the original length. max_position_embeddings is
    set to factor x original as well, since some transformers versions take the factor from that ratio rather than from the factor field."""
    y = {"rope_type": "yarn", "factor": float(args.yarn), "original_max_position_embeddings": int(args.yarn_orig)}
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp: cfg.rope_parameters = {**rp, **y}             # transformers 5
    else: cfg.rope_scaling = y                                                                     # transformers 4
    cfg.max_position_embeddings = int(args.yarn * args.yarn_orig)

_STORE_LAYER = None

def store_layer_class():
    """The layer of LM.lazy_cache, defined once. Each instance holds its own layer of the store and the runs to copy, so the store is released as
    soon as the cache is dropped. A class defined inside lazy_cache closed over the store, and a class object lives until Python's cyclic
    garbage collector runs: every book's store stayed on the GPU and the first runs went out of memory after a few books."""
    global _STORE_LAYER
    if _STORE_LAYER is None:
        import torch as t
        from transformers.cache_utils import DynamicLayer

        class StoreLayer(DynamicLayer):
            def __init__(s, K, V, runs, shape, dtype, dev):
                super().__init__(); s.K, s.V, s.runs, s.shape, s.dt, s.dev = K, V, runs, shape, dtype, dev; s.T = sum(a1 - a0 for a0, a1 in runs)
            def update(s, k, v, cache_kwargs=None):
                if not s.is_initialized: s.dtype, s.device, s.is_initialized = k.dtype, k.device, True   # what lazy_initialization does (4.x and 5.x)
                Kb = t.empty((s.T,) + tuple(s.shape), dtype=s.dt, device=s.dev); Vb = t.empty_like(Kb); dst = 0; nb = (s.dev == "cuda")
                for a0, a1 in s.runs:
                    Kb[dst:dst + a1 - a0].copy_(s.K[a0:a1], non_blocking=nb); Vb[dst:dst + a1 - a0].copy_(s.V[a0:a1], non_blocking=nb); dst += a1 - a0
                keys = t.cat([Kb.transpose(0, 1)[None], k], dim=2); values = t.cat([Vb.transpose(0, 1)[None], v], dim=2); del Kb, Vb
                return keys, values
            def get_seq_length(s, cache_position=None): return s.T
        _STORE_LAYER = StoreLayer
    return _STORE_LAYER


class LM:
    def __init__(self, model_name, tiny=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        if tiny:                                                                             # a random 2-layer Qwen3 for the equivalence check, CPU, fp32
            from transformers import Qwen3Config, Qwen3ForCausalLM
            cfg = Qwen3Config(vocab_size=len(self.tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=16, max_position_embeddings=4096, rope_theta=10000.0, tie_word_embeddings=False, attn_implementation=args.attn if args.attn in ("sdpa", "eager") else "sdpa")
            if args.yarn: apply_yarn(cfg)
            torch.manual_seed(0); self.model = Qwen3ForCausalLM(cfg).eval(); self.dev = "cpu"
        else:
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
            assert getattr(cfg, "model_type", "") in ("qwen3", "qwen2", "llama", "mistral", "gemma3_text"), f"{cfg.model_type}: this script needs a model whose every layer is standard attention"
            if args.yarn: apply_yarn(cfg)
            print(f"loading {model_name} ({cfg.model_type}, {cfg.num_hidden_layers} layers, {cfg.num_key_value_heads} kv heads, max positions {cfg.max_position_embeddings}"
                  f"{f', YaRN factor {args.yarn} over {args.yarn_orig}' if args.yarn else ''}), {args.attn}, bf16", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=args.device_map or self.dev, cache_dir=args.cache_dir, attn_implementation=args.attn,
                                                              **({"config": cfg} if args.yarn else {})).eval()
        self.n_layers = self.model.config.num_hidden_layers
        if args.yarn:                                                                              # the factor the rotary embedding really uses: YaRN's attention scaling is 0.1 ln(factor) + 1
            sc = float(getattr(self.model.model.rotary_emb, "attention_scaling", 1.0))
            assert abs(sc - (0.1 * math.log(args.yarn) + 1)) < 1e-4, f"YaRN not applied with factor {args.yarn}: attention scaling {sc}"
            print(f"YaRN on: rotary attention scaling {sc:.4f} (factor {args.yarn})", flush=True)
        self.sparse = SparsePrefill(args.sparse_prefill, args.attn, model_name, self.n_layers) if (not tiny and args.sparse_prefill != "none") else None
        # the chat template around the span: prefix before the text, suffix after it (the question block and the assistant prompt)
        mark = "\x00SPAN\x00"; full = qa_prompt(self.tok, mark, "Q", ["a", "b", "c", "d"], True)
        self.prefix_text = full.split(mark)[0]
        self.prefix_ids = self.tok.encode(self.prefix_text, add_special_tokens=False)
        # the tokens that end an answer: the tokenizer's end of sequence and every end-of-turn id of the model's generation config (SCBench stops on these)
        stop = {self.tok.eos_token_id}
        for src in (getattr(getattr(self.model, "generation_config", None), "eos_token_id", None), getattr(self.model.config, "eos_token_id", None)):
            stop |= set(src) if isinstance(src, (list, tuple)) else {src}
        self.stop_ids = {i for i in stop if i is not None}

    # ---- one forward, top-20 log-probabilities at the last position, as vLLM returns them (token string -> logprob)
    def top20(self, logits_last):
        t = self.torch; lp = t.log_softmax(logits_last.float(), -1); v, i = lp.topk(20, -1); out = []
        for row_v, row_i in zip(v.tolist(), i.tolist()):
            out.append({self.tok.decode([tid]): val for tid, val in zip(row_i, row_v)})
        return out

    def forward_last(self, ids_batch, position_ids=None):
        """Logits at the last real token for a batch of token-id lists (left padded)."""
        t = self.torch
        enc = self.tok.pad({"input_ids": ids_batch}, padding=True, return_tensors="pt").to(self.dev)
        with t.no_grad():
            out = self.model(**enc, use_cache=False, logits_to_keep=1)
        return out.logits[:, -1]

    def new_cache(self, n_total):
        """A DynamicCache whose layers write into preallocated buffers of n_total positions instead of concatenating: transformers' default
        layer copies the whole cache on every append, which doubles the peak during a chunked prefill (the 70 GB peak of the first window_reuse
        run) and copies the cache once per generated token. Every inherited method (seq length, mask sizes, crop) sees keys/values as views."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer
        t = self.torch

        class Prealloc(DynamicLayer):
            def __init__(s): super().__init__(); s.buf_k = s.buf_v = None; s.n = 0
            def update(s, k, v, cache_kwargs=None):
                if not s.is_initialized:                                                       # what lazy_initialization does in every transformers version;
                    s.dtype, s.device, s.is_initialized = k.dtype, k.device, True                 # its signature is (key) in 4.x and (key, value) in 5.x
                m = k.shape[2]
                if s.buf_k is None or s.n + m > s.buf_k.shape[2]:                                  # allocate, or grow once if a caller overruns
                    cap = max(n_total, s.n + m); nk = t.empty((k.shape[0], k.shape[1], cap, k.shape[3]), dtype=k.dtype, device=k.device); nv = t.empty_like(nk)
                    if s.buf_k is not None: nk[:, :, :s.n] = s.buf_k[:, :, :s.n]; nv[:, :, :s.n] = s.buf_v[:, :, :s.n]
                    s.buf_k, s.buf_v = nk, nv
                s.buf_k[:, :, s.n:s.n + m] = k; s.buf_v[:, :, s.n:s.n + m] = v; s.n += m
                s.keys, s.values = s.buf_k[:, :, :s.n], s.buf_v[:, :, :s.n]; return s.keys, s.values
            def crop(s, n):
                s.n = min(s.n, max(n, 0)); s.keys, s.values = s.buf_k[:, :, :s.n], s.buf_v[:, :, :s.n]
            def get_seq_length(s, cache_position=None): return s.n
        cache = DynamicCache(); cache.layers = [Prealloc() for _ in range(self.n_layers)]; return cache

    def prefill(self, ids, extra=0):
        """Encode one sequence in --prefill_chunk pieces into a preallocated cache sized for the sequence plus `extra` later tokens.
        Returns the last-token logits and the cache. Used by the window read and the select reread when --gen_tokens is set, and by window_reuse."""
        t = self.torch; cache = self.new_cache(len(ids) + extra); lg = None
        step = len(ids) if (self.sparse is not None and self.sparse.on) else args.prefill_chunk           # the sparse prefills select over the whole sequence in one forward
        with t.no_grad():
            for s in range(0, len(ids), step):
                out = self.model(input_ids=t.tensor([ids[s:s + step]], device=self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values; lg = out.logits[:, -1]
        return lg, cache

    def forward_cached(self, ids):
        """One sequence with its cache kept: last-token logits and the cache, for the decode timing."""
        return self.prefill(ids, extra=args.gen_tokens + 1)

    def release(self):
        """Drop every cache held across questions (the masked store, the window_reuse cache) and return the GPU memory. Called between arms."""
        self.drop_book(); self._wcache = None; self._conv = None; self._mconv = None
        if self.dev == "cuda": self.torch.cuda.empty_cache()

    def decode_timed(self, cache, logits_last, n):
        """Greedy decode of n tokens through the cache, timed. A measurement of ms per generated token over this context; the text is
        discarded and the cache is left with the n tokens appended (callers crop or drop it)."""
        t = self.torch; nxt = logits_last.argmax(-1, keepdim=True)
        if self.dev == "cuda": t.cuda.synchronize()
        t0 = time.time()
        with t.no_grad():
            for _ in range(n):
                out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if self.dev == "cuda": t.cuda.synchronize()
        dt = time.time() - t0
        return {"decode_s": dt, "decode_ms_per_tok": 1000 * dt / n, "gen_tokens": n}

    def generate(self, cache, logits_last, max_new):
        """The answer of an open question: greedy tokens through the cache from the read's last logits until an end-of-turn token or max_new tokens,
        as SCBench's loop. Returns the text, the number of tokens and the seconds after the first token; the cache keeps the generated tokens."""
        t = self.torch; nxt = logits_last.argmax(-1, keepdim=True); toks = []
        if self.dev == "cuda": t.cuda.synchronize()
        t0 = time.time()
        with t.no_grad():
            while len(toks) < max_new:
                tid = int(nxt.item())
                if tid in self.stop_ids: break
                toks.append(tid)
                if len(toks) == max_new: break
                out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True, logits_to_keep=1); nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if self.dev == "cuda": t.cuda.synchronize()
        self._last_gen = toks                                                                       # for the checks
        return self.tok.decode(toks, skip_special_tokens=True).strip(), len(toks), time.time() - t0

    def store_cache(self, book, picked):
        """The prompt and the selected blocks gathered from the store in document order, every key re-rotated from its stored position to the
        consecutive positions after the prompt (stair positions put blocks of one anti-diagonal at the same place). Returns the cache and its length."""
        P, lens, bases = book["P"], book["lens"], book["bases"]; T = P + int(sum(int(lens[m]) for m in picked))
        cache, _ = self.gather(book, [picked])                                                     # one span, so no padding
        stored = np.concatenate([np.arange(P)] + [int(bases[m]) + np.arange(int(lens[m])) for m in picked])
        self.rerotate(cache, self.torch.tensor(np.arange(T) - stored)); return cache, T

    def rerotate(self, cache, delta):
        """Rotate every cached key by delta[t] positions. Keys are cached with their rotary rotation applied and rotary rotations compose, so a key
        cached at position p and rotated by d is the key the model computes at p + d. Done in float32 with the model's own rotary embedding, its
        attention scaling divided out (a rotation by d must not scale twice)."""
        t = self.torch; rot = self.model.model.rotary_emb
        assert not hasattr(self.model.model, "rotary_emb_local"), "one rotary embedding for every layer only (Gemma 3's local layers use another)"
        cos, sin = rot(t.zeros(1, dtype=t.float32, device=self.dev), delta[None].to(self.dev))      # (1, T, head_dim)
        sc = float(getattr(rot, "attention_scaling", 1.0) or 1.0); cos, sin = (cos / sc)[:, None].float(), (sin / sc)[:, None].float()
        for layer in cache.layers:
            k = layer.keys.float(); h = k.shape[-1] // 2
            layer.keys = (k * cos + t.cat((-k[..., h:], k[..., :h]), dim=-1) * sin).to(layer.values.dtype)

    def read_store(self, book, picked, ex):
        """The select read-out without a second encoding: the stored keys and values of the K selected blocks are gathered after the prompt in
        document order, every key re-rotated from its stored position to consecutive positions, and the turn's request runs right after them, its
        positions continuing from the cache. Each block keeps the context of its own local span (the selected blocks do not attend to one another,
        as in the strided probes). Multiple choice reads the letter; an open question generates over this cache. Same return as read()."""
        t = self.torch; _, cue, ys, _ = read_spec(ex); T = book["P"] + int(sum(int(book["lens"][m]) for m in picked))
        req = self.tok.encode(request_text(self.tok, ex, cue), add_special_tokens=False)
        if T + len(req) >= args.window: return None, None, T + len(req), None
        cache, _ = self.store_cache(book, picked)
        with t.no_grad():
            lg = self.model(input_ids=t.tensor([req], device=self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        self._last_store = (req, lg)                                                                  # for the checks
        if cue == "gen":
            txt, n_gen, gen_s = self.generate(cache, lg, args.max_new); del cache
            return txt, None, len(req), {"gen_s": gen_s, "gen_n": n_gen}
        scores = read_scores(self.top20(lg)[0], ys)
        dec = self.decode_timed(cache, lg, args.gen_tokens) if (args.gen_tokens > 0 and args.select_mode == "store") else None
        del cache; return max(ys, key=scores.get), scores, len(req), dec

    @staticmethod
    def crop_cache(cache, n):
        """Drop every cached position beyond n (the suffix and any generated tokens), keeping the document's keys and values."""
        if hasattr(cache, "crop"): cache.crop(n); return
        for layer in cache.layers: layer.keys, layer.values = layer.keys[:, :, :n], layer.values[:, :, :n]

    # ---- the window read and the Periscope spans
    def read(self, text, ex):
        """One read of the text with the turn's request, without the abstain option; None if the prompt does not fit the window. Multiple choice
        reads the letter, and with --gen_tokens keeps the cache and decodes that many tokens for the timing (the fourth element); an open question
        generates its answer (up to --max_new tokens), returned as the prediction with its generation time in the fourth element."""
        _, cue, ys, _ = read_spec(ex); ids = self.tok.encode(build_prompt(self.tok, text, ex, cue), add_special_tokens=False)
        if len(ids) >= args.window: return None, None, len(ids), None
        if cue == "gen":
            lg, cache = self.prefill(ids, extra=args.max_new + 1); txt, n_gen, gen_s = self.generate(cache, lg, args.max_new); del cache
            self._last = (ids, lg); return txt, None, len(ids), {"gen_s": gen_s, "gen_n": n_gen}
        if args.gen_tokens <= 0:
            lg = self.forward_last([ids]); scores = read_scores(self.top20(lg)[0], ys); dec = None
        else:
            lg, cache = self.forward_cached(ids); scores = read_scores(self.top20(lg)[0], ys)
            dec = self.decode_timed(cache, lg, args.gen_tokens); del cache
        self._last = (ids, lg)                                                                  # for the checks
        return max(ys, key=scores.get), scores, len(ids), dec

    def window_read(self, ex): return self._sparse_arm(self._window_read, ex)

    def window_reuse(self, ex):
        assert self.sparse is None or self.sparse.method != "tokensparse", "Token Sparse Attention scores tokens with the question, so its prefill cannot be shared across questions"
        return self._sparse_arm(self._window_reuse, ex)

    def _sparse_arm(self, arm, ex):
        """A monolithic arm with the sparse-prefill baseline switched on for its duration (a plain call without --sparse_prefill); the row records the
        method, how many attention calls it served, and for Token Sparse Attention the mean share of tokens it kept."""
        if self.sparse is None: return arm(ex)
        self.sparse.on, self.sparse.n_sparse, self.sparse.kept = True, 0, []
        try: rec = arm(ex)
        finally: self.sparse.on = False
        rec.update(sparse_prefill=args.sparse_prefill, sparse_calls=self.sparse.n_sparse)
        if self.sparse.kept: rec["ts_kept"] = float(np.mean(self.sparse.kept))
        return rec

    def _window_read(self, ex):
        """The longest part of the text that fits the window with the turn's request: its beginning, or for SCBench its beginning and its end
        (SCBench truncates the middle)."""
        _, cue, _, _ = read_spec(ex); ids = self.tok.encode(ex["context"], add_special_tokens=False)
        kept = fit_prefix(self.tok, ids, lambda tx: build_prompt(self.tok, tx, ex, cue), args.window, "middle" if args.dataset == "scbench" else "prefix")
        pred, scores, n, dec = self.read(self.tok.decode(kept, skip_special_tokens=True), ex)
        rec = {"pred": pred, "scores": scores, "tokens": n, "passes": 1, "truncated": int(len(kept) < len(ids))}
        if dec: rec.update(dec)
        return rec

    def _window_reuse(self, ex):
        """The monolithic read with its cache reused across a book's questions, which is what prefix caching gives an end user: the document is
        cut once per book to the window minus --reuse_margin (so the cut does not depend on the question), encoded once in prefill chunks into one
        cache kept on the GPU, then each question's suffix (no abstain option, as the window read) runs against the cache at the positions after
        the document, the answer is read from the last token, and the cache is cropped back to the document for the next question."""
        t = self.torch; key = ex.get("book") or hashlib.md5(ex["context"].encode("utf-8")).hexdigest()
        wc = getattr(self, "_wcache", None); shared = (wc is not None and wc["key"] == key and not args.no_reuse); t_enc = time.time()
        if not shared:
            self._wcache = None; wc = None
            if self.dev == "cuda": t.cuda.empty_cache()
            ids = self.tok.encode(ex["context"], add_special_tokens=False); budget = max(1, args.window - len(self.prefix_ids) - args.reuse_margin)
            middle = args.dataset == "scbench" and len(ids) > budget                                 # SCBench keeps the beginning and the end
            kept = (ids[:budget // 2] + ids[-(budget - budget // 2):]) if middle else ids[:budget]; text = self.tok.decode(kept, skip_special_tokens=True)
            # the window read tokenizes its whole prompt as one string, and a token can merge across the text's boundaries, so the shared
            # document part is the longest common token prefix of two full prompts with different questions: exactly the window read's tokens
            a = self.tok.encode(qa_prompt(self.tok, text, "A?", ["a", "b", "c", "d"], False), add_special_tokens=False)   # the two questions differ from their first token,
            b = self.tok.encode(qa_prompt(self.tok, text, "B!", ["e", "f", "g", "h"], False), add_special_tokens=False)   # so the common prefix ends with the question cue
            n_doc = 0
            while n_doc < min(len(a), len(b)) and a[n_doc] == b[n_doc]: n_doc += 1
            doc_ids = a[:n_doc]; _, cache = self.prefill(doc_ids, extra=args.reuse_margin + len(self.prefix_ids) + max(args.gen_tokens, args.max_new, 0) + 1)   # room for the suffix and the decode
            wc = {"key": key, "cache": cache, "n_doc": n_doc, "doc_ids": doc_ids, "text": text, "truncated": int(len(kept) < len(ids)),
                  "prefills": 1 if (self.sparse is not None and self.sparse.on) else math.ceil(len(doc_ids) / args.prefill_chunk)}; self._wcache = wc
        encode_s = time.time() - t_enc; t_tok = time.time()
        _, cue, ys, _ = read_spec(ex); n = wc["n_doc"]
        full = self.tok.encode(build_prompt(self.tok, wc["text"], ex, cue), add_special_tokens=False)
        tok_s = time.time() - t_tok; fallback = int(full[:n] != wc["doc_ids"]); fcache = None
        if fallback:                                                                               # a merge moved with this question: read it without the cache
            if cue == "gen": lg, fcache = self.prefill(full, extra=args.max_new + 1)
            else: lg = self.forward_last([full])
            suffix_ids = full; ask_s = 0.0
        else:
            suffix_ids = full[n:]; t_ask = time.time()
            with t.no_grad():
                out = self.model(input_ids=t.tensor([suffix_ids], device=self.dev), past_key_values=wc["cache"], use_cache=True, logits_to_keep=1)
            lg = out.logits[:, -1]; ask_s = time.time() - t_ask
        self._last = (full, lg)                                                                    # for the checks
        if cue == "gen":                                                                           # the open answer, generated through the same cache
            pred, n_gen, gen_s = self.generate(fcache if fallback else wc["cache"], lg, args.max_new); scores = None; dec = {"gen_s": gen_s, "gen_n": n_gen}; del fcache
        else:
            scores = read_scores(self.top20(lg)[0], ys); pred = max(ys, key=scores.get)
            dec = self.decode_timed(wc["cache"], lg, args.gen_tokens) if (args.gen_tokens > 0 and not fallback) else None
        self.crop_cache(wc["cache"], n)
        rec = {"pred": pred, "scores": scores, "tokens": len(suffix_ids) + (0 if shared else n), "content_tokens": 0 if shared else n, "passes": 1 + (0 if shared else wc["prefills"]),
               "truncated": wc["truncated"], "shared": int(shared), "encode_s": encode_s, "ask_s": ask_s, "tok_s": tok_s, "fallback": fallback, "doc_tokens": n}
        if dec: rec.update(dec)
        return rec

    def periscope(self, ex):
        """Periscope as released: 2K span prompts, each its own sequence, batched --span_batch at a time. An open question probes each span with
        the Yes or No relevance question."""
        probe, _, ys, ab = read_spec(ex); chunks = chunk(self.tok, ex["context"], args.chunk_size) or ["empty"]
        k, local, strd = grid(chunks); spans = local + strd
        prompts = [self.tok.encode(build_prompt(self.tok, s, ex, probe), add_special_tokens=False) for _, s in spans]
        keep = [i for i, p in enumerate(prompts) if len(p) < args.window]; tops = {}
        for b in range(0, len(keep), args.span_batch):
            idx = keep[b:b + args.span_batch]; lg = self.forward_last([prompts[i] for i in idx])
            for i, tm in zip(idx, self.top20(lg)): tops[i] = tm
        L = {local[i][0]: log_odds(tops[i], ys, ab) for i in range(len(local)) if i in tops}
        S = {strd[j][0]: log_odds(tops[len(local) + j], ys, ab) for j in range(len(strd)) if len(local) + j in tops}
        return self._readouts(ex, chunks, k, L, S, tokens=sum(len(prompts[i]) for i in keep), passes=math.ceil(len(keep) / args.span_batch), encoded=sum(len(prompts[i]) for i in keep))

    def _readouts(self, ex, chunks, k, L, S, tokens, passes, encoded, book=None):
        """Direct and select from the probe scores. Multiple choice: direct is the composed answer and select reads the K top blocks of the map
        once more. An open question has no direct answer (its probes score Yes against No) and select generates it. Select re-encodes the blocks'
        text with the request (reread), or, given the store of a masked arm, gathers their stored keys and values (store); --select_mode both runs
        the two on the same blocks, the reread as pred and the gather as pred_store, each with its own time (reread_s, store_s)."""
        _, cue, ys, _ = read_spec(ex); mode = args.select_mode if book is not None else "reread"
        out = {"k": k, "n_chunks": len(chunks), "tokens": tokens, "passes": passes, "content_tokens": encoded, "direct": None, "pred": None, "select_mode": mode}
        if not L: return out
        scores = compose(L, S); cells = evidence_map(L, S, ys)
        out.update(direct=max(ys, key=scores.get) if cue == "letter" else None, scores=scores, peak=list(max(cells, key=cells.get)),
                   L={str(i): d for i, d in L.items()}, S={str(j): d for j, d in S.items()})
        budget = k if args.select_budget == "k" else skew_budget(L, S, ys, k, len(chunks))
        picked = select(cells, k, len(chunks), budget); out["selected"] = picked; out["budget"] = budget
        if mode in ("reread", "both"):
            t_r = time.time(); pred, rs, n, dec = self.read(" ".join(chunks[i] for i in picked), ex); dec = dec or {}
            out.update(pred=pred, read_scores=rs, read_tokens=n, reread_s=time.time() - t_r - dec.get("decode_s", 0.0) - dec.get("gen_s", 0.0), **dec)
            out["passes"] += 1; out["tokens"] += n
        if mode in ("store", "both"):
            t_s = time.time(); pred, rs, n, dec = self.read_store(book, picked, ex); dec = dec or {}
            store_s = time.time() - t_s - dec.get("decode_s", 0.0) - dec.get("gen_s", 0.0)
            if mode == "store": out.update(pred=pred, read_scores=rs, read_tokens=n, store_s=store_s, **dec)
            else: out.update(pred_store=pred, store_scores=rs, store_tokens=n, store_s=store_s, **{name + "_store": v for name, v in dec.items()})
            out["passes"] += 1; out["tokens"] += n
        return out

    # ---- the masked arms: chunks encoded once, keys and values kept token-major in one buffer, spans gathered from it in batches
    def cache_bytes(self, n_tokens):
        cfg = self.model.config; kvh = cfg.num_key_value_heads; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return n_tokens * self.n_layers * 2 * kvh * hd * 2

    def _kv_shape(self):
        cfg = self.model.config; return cfg.num_key_value_heads, (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)

    def alloc_store(self, n_rows):
        """One buffer (2, layers, rows, kv heads, head_dim) for the keys and values of the prefix (rows 0..P) and of every chunk, token-major
        so a chunk is a contiguous slice. On the GPU when the document fits --gpu_cache_gb, else in host memory page-locked with
        cudaHostRegister so each gathered run is one direct DMA; pageable with a note if locking is unavailable or refused."""
        t = self.torch; H, d = self._kv_shape(); est = self.cache_bytes(n_rows)
        on_cpu = args.offload or est > args.gpu_cache_gb * 1e9
        if on_cpu and self.dev == "cuda" and not getattr(self, "_said_offload", False):
            print(f"    chunk cache estimated at {est / 1e9:.0f} GB: keeping it in host memory, gathering spans to the GPU", flush=True); self._said_offload = True
        buf = t.empty((2, self.n_layers, n_rows, H, d), dtype=self.model.dtype, device="cpu" if on_cpu else self.dev); pinned = False
        if on_cpu and self.dev == "cuda" and not args.no_pin:
            try: rc = t.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), 0); code = int(rc)       # an enum in some torch builds
            except Exception as e: rc, code = repr(e), -1
            pinned = (code == 0)
            if not pinned and not getattr(self, "_said_pin", False):
                print(f"    could not page-lock the host cache ({rc}): pageable memory, staged copies", flush=True); self._said_pin = True
        return {"buf": buf, "pinned": pinned, "on_cpu": on_cpu, "K": [buf[0, l] for l in range(self.n_layers)], "V": [buf[1, l] for l in range(self.n_layers)]}

    def drop_book(self):
        """Release the current book: unlock its host buffer first, so the memory really returns."""
        book = getattr(self, "_book", None); self._book = None
        if book is not None and book.get("pinned"):
            try: self.torch.cuda.cudart().cudaHostUnregister(book["buf"].data_ptr())
            except Exception: pass
        del book
        if self.dev == "cuda": self.torch.cuda.empty_cache()

    def encode_chunks(self, chunk_ids, mode):
        """Keys and values of every chunk under the mode's isolation and positions, written into the store.
        doc / chunk / stair_alone: every chunk encoded alone after the prefix (batches of --chunk_batch, right padded, explicit positions),
        at its document position, at the start, or at its stair position.
        anchor: every chunk after the prefix and a copy of the first chunk, as Star Attention's first phase, at the positions of chunk mode shifted by
        the anchor; only the chunk's own keys and values are kept (the first chunk is kept from its own pass, without an anchor).
        span / stair: one causal pass per local span, the K chunks sliced out of it.
        stair_both (ablation d): stair, plus one causal pass per strided span into a second copy of every chunk, which the strided spans read
        (chunk ids n..2n-1 of the book, see span_rows), so every probe reads its span as one causal pass. Returns the book: store, offs, lens, bases, P."""
        t = self.torch; P = len(self.prefix_ids); lens = np.array([len(c) for c in chunk_ids]); offs = P + np.cumsum([0] + list(lens[:-1]))
        k = math.ceil(math.sqrt(len(chunk_ids))); stride = int(lens.max()); n = len(chunk_ids); S = int(lens.sum()); two = mode == "stair_both"
        if mode in ("doc", "span"): bases = offs                                                 # document positions
        elif mode == "chunk": bases = np.full(len(chunk_ids), P)                                   # every chunk at the start
        elif mode == "anchor": bases = np.array([P] + [P + int(lens[0])] * (len(chunk_ids) - 1))     # every chunk right after the anchor (the first chunk at the start)
        elif mode in ("stair", "stair_alone", "stair_both"): bases = np.array([P + (m // k + m % k) * stride for m in range(len(chunk_ids))])   # chunk j of span i at (i + j) * stride
        else: raise ValueError(mode)
        book = self.alloc_store(P + (2 if two else 1) * S); K, V = book["K"], book["V"]; book.update(offs=offs, lens=lens, bases=bases, P=P, mode=mode, n=n)
        if two:                                                                                   # the strided copy, laid out column by column so a strided span is one run of rows
            offs2 = np.zeros(n, dtype=np.int64); o = P + S
            for m in sorted(range(n), key=lambda m: (m % k, m // k)): offs2[m] = o; o += int(lens[m])
            book.update(offs=np.concatenate([offs, offs2]), lens=np.concatenate([lens, lens]), bases=np.concatenate([bases, bases]))
        def put(cache, r, s_src, e_src, s_dst):                                                # rows s_src..e_src of batch row r of a pass -> store rows from s_dst
            for l in range(self.n_layers):
                K[l][s_dst:s_dst + e_src - s_src].copy_(cache.layers[l].keys[r][:, s_src:e_src].transpose(0, 1))
                V[l][s_dst:s_dst + e_src - s_src].copy_(cache.layers[l].values[r][:, s_src:e_src].transpose(0, 1))
        if mode in ("span", "stair", "stair_both"):                                          # one causal pass per local span (and per strided span), chunks sliced out of it
            n_local = len(range(0, n, k))
            spans = [list(range(i, min(i + k, n))) for i in range(0, n, k)] + ([list(range(j, n, k)) for j in range(min(k, n))] if two else [])
            for s_i, members in enumerate(spans):
                ids = list(self.prefix_ids) + [x for m in members for x in chunk_ids[m]]
                pos = list(range(P)) + [int(bases[m]) + j for m in members for j in range(len(chunk_ids[m]))]
                with t.no_grad():
                    out = self.model.model(input_ids=t.tensor([ids], device=self.dev), position_ids=t.tensor([pos], device=self.dev), use_cache=True)
                cache = out.past_key_values
                if s_i == 0: put(cache, 0, 0, P, 0)
                put(cache, 0, P, len(ids), int(offs[members[0]] if s_i < n_local else offs2[members[0]]))   # a span's chunks are consecutive rows of its copy
                del out, cache
            return book
        anchor = list(chunk_ids[0]) if mode == "anchor" else []
        for b in range(0, len(chunk_ids), args.chunk_batch):
            heads = [list(self.prefix_ids) + (anchor if m > 0 else []) for m in range(b, min(b + args.chunk_batch, len(chunk_ids)))]   # what precedes each chunk
            ids = [h + c for h, c in zip(heads, chunk_ids[b:b + args.chunk_batch])]; L_ = [len(x) for x in ids]; T = max(L_)
            inp = t.full((len(ids), T), self.tok.pad_token_id, dtype=t.long); att = t.zeros((len(ids), T), dtype=t.long); pos = t.zeros((len(ids), T), dtype=t.long)
            for r, x in enumerate(ids):                                                   # right padding here: positions are explicit, the mask hides the pad
                inp[r, :len(x)] = t.tensor(x); att[r, :len(x)] = 1
                H, base = len(heads[r]), int(bases[b + r])                                    # the prefix (and the anchor) at 0..H, the chunk from its base
                pos[r, :H] = t.arange(H); pos[r, H:len(x)] = base + t.arange(len(x) - H); pos[r, len(x):] = base + len(x) - H
            with t.no_grad():
                out = self.model.model(input_ids=inp.to(self.dev), attention_mask=att.to(self.dev), position_ids=pos.to(self.dev), use_cache=True)
            cache = out.past_key_values
            for r in range(len(ids)):
                if b + r == 0: put(cache, 0, 0, P, 0)
                put(cache, r, len(heads[r]), L_[r], int(offs[b + r]))                       # the chunk's rows only, never the anchor's
            del out, cache
        return book

    def gather(self, book, spans):
        """A batched cache for B spans: per layer (B, kv heads, T_max, head_dim), every span right-aligned (prefix, then its chunks in
        order), pad slots zero and masked. Consecutive chunks are copied as one run; from the page-locked host buffer a run is one DMA.
        Returns the cache and the (B, T_max) pad mask."""
        t = self.torch; K, V, offs, lens, P = book["K"], book["V"], book["offs"], book["lens"], book["P"]; H, d = self._kv_shape()
        from transformers import DynamicCache
        B = len(spans); T = [P + int(sum(lens[m] for m in members)) for members in spans]; Tm = max(T)
        att = t.zeros((B, Tm), dtype=t.long); rows = []
        for b, members in enumerate(spans):
            dst = Tm - T[b]; att[b, dst:] = 1; runs = [(0, P)]
            for m in members:
                a0, a1 = int(offs[m]), int(offs[m] + lens[m])
                if a0 == runs[-1][1]: runs[-1] = (runs[-1][0], a1)
                else: runs.append((a0, a1))
            row = []
            for a0, a1 in runs: row.append((a0, a1, dst)); dst += a1 - a0
            rows.append(row)
        cache = DynamicCache(); nb = (self.dev == "cuda")
        for l in range(self.n_layers):
            Kb = t.zeros((B, Tm, H, d), dtype=self.model.dtype, device=self.dev); Vb = t.zeros_like(Kb)
            if getattr(self, "_pad_noise", False): Kb.normal_(); Vb.normal_()                     # self-test only: the pads must be masked, not merely zero
            for b, row in enumerate(rows):
                for a0, a1, dst in row:
                    Kb[b, dst:dst + a1 - a0].copy_(K[l][a0:a1], non_blocking=nb); Vb[b, dst:dst + a1 - a0].copy_(V[l][a0:a1], non_blocking=nb)
            cache.update(Kb.transpose(1, 2), Vb.transpose(1, 2), l); del Kb, Vb               # update concatenates: the (B, heads, T, d) copy the model reads
        return cache, att

    def ask_spans(self, book, spans, suffix_ids, q_starts):
        """The question copies of B spans in one forward: each copy sits at its own positions after its span's last chunk, attends causally
        to itself and to its own span's cache slots only (the 2D mask hides the pads; batch rows never mix). Last-token logits, (B, vocab)."""
        t = self.torch; cache, att = self.gather(book, spans); B, q = len(spans), len(suffix_ids)
        inp = t.tensor([suffix_ids] * B, device=self.dev); pos = (t.tensor(q_starts)[:, None] + t.arange(q)[None]).to(self.dev)
        mask = t.cat([att, t.ones((B, q), dtype=t.long)], 1).to(self.dev)
        with t.no_grad():
            out = self.model(input_ids=inp, position_ids=pos, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
        del cache; return out.logits[:, -1]

    def lazy_cache(self, book, members):
        """A cache over the prompt and the given chunks in document order whose layers come from the store only while the model runs them: when the
        request reaches layer l, that layer's keys and values are copied to the GPU (one transfer per run of consecutive chunks), joined with the
        request's own, used, and dropped. The GPU holds one layer of the context at a time (3 GB at 750k tokens instead of the whole store), so one
        request can read every chunk of any book. One forward only: the request's keys are not kept."""
        from transformers import DynamicCache
        K, V, offs, lens, P = book["K"], book["V"], book["offs"], book["lens"], book["P"]
        runs = [(0, P)]
        for m in members:
            a0, a1 = int(offs[m]), int(offs[m] + lens[m])
            if a0 == runs[-1][1]: runs[-1] = (runs[-1][0], a1)
            else: runs.append((a0, a1))
        SL = store_layer_class(); cache = DynamicCache()
        cache.layers = [SL(K[l], V[l], runs, self._kv_shape(), self.model.dtype, self.dev) for l in range(self.n_layers)]; return cache

    def masked_all(self, ex, mode):
        """One query over every chunk of a single encoding (ablation b): parallel encoding (mode chunk: every chunk alone at the chunk start, as PCW)
        or with an anchor (mode anchor: every chunk encoded after the first chunk, as Star Attention). The request (multiple choice, no abstain
        option, as the monolithic and select reads) runs once over the prompt and every chunk, right after the longest chunk, through lazy_cache."""
        assert ex["options"] is not None, "one query over all blocks reads a letter: multiple choice only"
        t = self.torch; _, cue, ys, _ = read_spec(ex)
        book, shared, encode_s, skip = self.get_book(ex, mode)
        if skip: return skip
        lens, bases = book["lens"], book["bases"]; members = list(range(len(lens))); T = book["P"] + int(lens.sum())
        req = self.tok.encode(request_text(self.tok, ex, cue), add_special_tokens=False); q0 = int(max(int(bases[m]) + int(lens[m]) for m in members))
        t_ask = time.time(); cache = self.lazy_cache(book, members)
        with t.no_grad():
            lg = self.model(input_ids=t.tensor([req], device=self.dev), position_ids=(q0 + t.arange(len(req)))[None].to(self.dev), past_key_values=cache,
                            use_cache=True, logits_to_keep=1).logits[:, -1]
        del cache; self._last_all = lg; scores = read_scores(self.top20(lg)[0], ys)
        return {"pred": max(ys, key=scores.get), "scores": scores, "direct": None, "k": book["k"], "n_chunks": len(members), "tokens": len(req) + (0 if shared else T),
                "content_tokens": 0 if shared else T, "passes": 1 + (0 if shared else book["passes"]), "shared": int(shared), "encode_s": encode_s,
                "ask_s": time.time() - t_ask, "doc_tokens": T, "max_pos": q0 + len(req), "offloaded": int(book["on_cpu"]), "pinned": int(book["pinned"])}

    def get_book(self, ex, mode):
        """The store of this text under this mode: the one kept from the previous question when the text is the same (shared), else encoded now.
        Returns (book, shared, encode seconds, None), or (None, False, 0, a skipped row) when the store would not fit the host."""
        key = (hashlib.md5(ex["context"].encode("utf-8")).hexdigest(), mode); book = getattr(self, "_book", None)
        shared = (book is not None and book["key"] == key and not args.no_reuse); t_enc = time.time()
        if not shared:
            self.drop_book(); book = None
            chunks = chunk(self.tok, ex["context"], args.chunk_size) or ["empty"]; k, _, _ = grid(chunks)
            chunk_ids = [self.tok.encode(c, add_special_tokens=False) for c in chunks]
            est = self.cache_bytes(sum(len(c) for c in chunk_ids) * (2 if mode == "stair_both" else 1)); limit, src = host_ram_bytes()
            if est > args.gpu_cache_gb * 1e9 and est + 12e9 > 0.8 * limit:                       # the offloaded cache would not fit the host either (12 GB for everything else)
                print(f"    skipping: chunk cache of {est / 1e9:.0f} GB against a host limit of {limit / 1e9:.0f} GB ({src})", flush=True)
                return None, False, 0.0, {"pred": None, "direct": None, "k": k, "n_chunks": len(chunks), "tokens": 0, "passes": 0, "content_tokens": 0, "shared": 0,
                                          "skipped": f"cache of {est / 1e9:.0f} GB exceeds the host limit of {limit / 1e9:.0f} GB ({src})"}
            book = self.encode_chunks(chunk_ids, mode)
            book.update(key=key, chunks=chunks, chunk_ids=chunk_ids, k=k, passes={"span": k, "stair": k, "stair_both": 2 * k}.get(mode, math.ceil(len(chunks) / args.chunk_batch))); self._book = book
        return book, shared, time.time() - t_enc, None

    def masked(self, ex, mode):
        probe, _, ys, ab = read_spec(ex); P = len(self.prefix_ids)
        book, shared, encode_s, skip = self.get_book(ex, mode)
        if skip: return skip
        chunks, k, lens, bases = book["chunks"], book["k"], book["lens"], book["bases"]
        n = len(chunks); local, strd = grid_indices(n, k); passes = 0 if shared else book["passes"]
        suffix_ids = self.tok.encode(request_text(self.tok, ex, probe), add_special_tokens=False)          # the conversation so far and the probe cue
        content = P + int(lens.sum()); tokens = 0 if shared else content; t_cop = time.time()
        spans = []                                                                            # (family, index, members, question start)
        for fam, lst in (("L", local), ("S", strd)):
            for i, members in lst:
                members = self.span_rows(book, fam, members)
                if mode != "chunk": q_start = int(bases[members[-1]]) + int(lens[members[-1]])      # right after the span's last chunk, in that mode's positions
                else: q_start = P + int(max(lens[m] for m in members))                           # right after "a chunk"
                if P + int(sum(lens[m] for m in members)) + len(suffix_ids) < args.window: spans.append((fam, i, members, q_start))
        spans.sort(key=lambda sp: int(sum(lens[m] for m in sp[2])))                           # similar lengths share a batch: little padding
        L, S = {}, {}; B = max(1, args.copy_batch); n_batches = 0
        for b in range(0, len(spans), B):
            batch = spans[b:b + B]; lg = self.ask_spans(book, [sp[2] for sp in batch], suffix_ids, [sp[3] for sp in batch]); n_batches += 1
            for (fam, i, _, _), tm in zip(batch, self.top20(lg)): (L if fam == "L" else S)[i] = log_odds(tm, ys, ab)
            tokens += len(suffix_ids) * len(batch)
        rec = self._readouts(ex, chunks, k, L, S, tokens=tokens, passes=passes + n_batches, encoded=0 if shared else content, book=book)
        rec.update(shared=int(shared), encode_s=encode_s, copies_s=time.time() - t_cop - sum(rec.get(x, 0.0) for x in ("decode_s", "gen_s", "gen_s_store")), doc_tokens=content, copy_batch=B, offloaded=int(book["on_cpu"]), pinned=int(book["pinned"])); return rec

    # ---- conversations with long turns: the store grows by rows (fca_append), the monolithic cache by tokens (mono_conv)
    def conv_append(self, st, chunk_texts):
        """Append blocks to the store of a conversation. The grid has a fixed number of columns K and grows by rows: block m sits at row m // K
        and column m % K for good, at the stair position P + (m // K + m % K) * stride, and its keys and values at the next free rows of the
        store. Only the rows that receive new blocks are encoded, one causal pass per row over the prefix and the row's blocks; a row that was
        partly filled is encoded again as a whole, so every row is always one causal pass over all its blocks, as a row of the fixed store.
        Returns the number of rows encoded."""
        t = self.torch; P = len(self.prefix_ids); K, stride, book = st["K"], st["stride"], st["book"]; Kb, Vb = book["K"], book["V"]
        n0 = len(st["ids"])
        for c in chunk_texts:
            ids = self.tok.encode(c, add_special_tokens=False)
            if len(ids) > stride: ids = ids[:stride]; st["clipped"] += 1                           # a block longer than the stride would overlap the next one
            m = len(st["ids"]); st["ids"].append(ids); st["chunks"].append(c); st["lens"].append(len(ids)); st["offs"].append(P + st["used"])
            st["used"] += len(ids); st["bases"].append(P + (m // K + m % K) * stride)
        n1 = len(st["ids"]); assert P + st["used"] <= book["buf"].shape[2], f"store of {book['buf'].shape[2]} rows is too small for {P + st['used']}"
        rows = list(range(n0 // K, (n1 - 1) // K + 1)) if n1 > n0 else []
        for i in rows:
            members = list(range(i * K, min((i + 1) * K, n1)))
            ids = list(self.prefix_ids) + [x for m in members for x in st["ids"][m]]
            pos = list(range(P)) + [st["bases"][m] + j for m in members for j in range(st["lens"][m])]
            with t.no_grad():
                out = self.model.model(input_ids=t.tensor([ids], device=self.dev), position_ids=t.tensor([pos], device=self.dev), use_cache=True)
            cache = out.past_key_values; a = st["offs"][members[0]]; n_tok = len(ids) - P
            for l in range(self.n_layers):
                k, v = cache.layers[l].keys[0], cache.layers[l].values[0]                        # (kv heads, tokens, head_dim)
                if not st["prefix_done"]: Kb[l][:P].copy_(k[:, :P].transpose(0, 1)); Vb[l][:P].copy_(v[:, :P].transpose(0, 1))
                Kb[l][a:a + n_tok].copy_(k[:, P:].transpose(0, 1)); Vb[l][a:a + n_tok].copy_(v[:, P:].transpose(0, 1))
            st["prefix_done"] = True; del out, cache
        book.update(offs=np.array(st["offs"]), lens=np.array(st["lens"]), bases=np.array(st["bases"]), n=n1, chunks=st["chunks"], chunk_ids=st["ids"])
        return len(rows)

    def conv_state(self, conv, n_turns, cap_tokens=None):
        """A new, empty store for one conversation, sized for all its turns."""
        import gc
        P = len(self.prefix_ids); self.drop_book(); self._conv = None; gc.collect()               # the old store must be gone: two stores do not fit the host
        if self.dev == "cuda": self.torch.cuda.empty_cache()
        K = args.conv_K or math.ceil(math.sqrt(n_turns * args.conv_turn_tokens / args.chunk_size))
        cap = P + (cap_tokens if cap_tokens is not None else int(1.05 * n_turns * args.conv_turn_tokens) + 4096)
        est = self.cache_bytes(cap); limit, src = host_ram_bytes()
        if est > args.gpu_cache_gb * 1e9 and est + 12e9 > 0.8 * limit:
            raise SystemExit(f"the store of this conversation needs {est / 1e9:.0f} GB of host memory against a limit of {limit / 1e9:.0f} GB ({src}): "
                             f"lower --conv_turns or --conv_turn_tokens")
        book = self.alloc_store(cap); book.update(P=P, mode="stair", k=K, key=("conv", conv), passes=0)
        st = {"conv": conv, "turn": -1, "K": K, "stride": args.chunk_size + 16, "book": book, "ids": [], "chunks": [], "lens": [], "offs": [], "bases": [],
              "used": 0, "prefix_done": False, "clipped": 0}
        self._conv = st; self._book = book; return st

    def fca_append(self, ex):
        """FCA over a conversation with long turns: the text of every turn enters the store once, as new rows of a grid with K fixed columns
        (conv_append), and the turn's question is then answered as any request, by the 2K-or-more span sequences over the store (one per row and
        one per column) and the select read-out. Nothing stored is encoded again, and the request holds the question only."""
        t = self.torch; P = len(self.prefix_ids); st = getattr(self, "_conv", None)
        if self.dev == "cuda": t.cuda.synchronize()
        t_enc = time.time()
        if st is None or st["conv"] != ex["conv"] or st["turn"] > ex["turn"] or getattr(self, "_book", None) is not st["book"]:
            st = None                                                                             # release the previous conversation's store before the new one is allocated
            st = self.conv_state(ex["conv"], ex["n_turns"])
        used0, new_rows = st["used"], 0
        for u in range(st["turn"] + 1, ex["turn"] + 1):                                             # normally one turn; more when a run resumes inside a conversation
            new_rows += self.conv_append(st, chunk(self.tok, ex["segments"][u], args.chunk_size) or ["empty"]); st["turn"] = u
        if self.dev == "cuda": t.cuda.synchronize()
        encode_s = time.time() - t_enc; new_tokens = st["used"] - used0
        book, K = st["book"], st["K"]; lens, bases = book["lens"], book["bases"]; n = len(lens); R = math.ceil(n / K)
        probe, _, ys, ab = read_spec(ex); suffix_ids = self.tok.encode(request_text(self.tok, ex, probe), add_special_tokens=False); t_cop = time.time()
        spans = []
        for fam, lst in (("L", [(i, list(range(i * K, min((i + 1) * K, n)))) for i in range(R)]), ("S", [(j, list(range(j, n, K))) for j in range(min(K, n))])):
            for i, members in lst:
                q0 = int(bases[members[-1]]) + int(lens[members[-1]])
                if P + int(sum(lens[m] for m in members)) + len(suffix_ids) < args.window and q0 + len(suffix_ids) < args.window: spans.append((fam, i, members, q0))
        spans.sort(key=lambda sp: int(sum(lens[m] for m in sp[2])))
        L, S = {}, {}; B = max(1, args.copy_batch); n_batches = 0; tokens = new_tokens
        for b in range(0, len(spans), B):
            batch = spans[b:b + B]; lg = self.ask_spans(book, [sp[2] for sp in batch], suffix_ids, [sp[3] for sp in batch]); n_batches += 1
            for (fam, i, _, _), tm in zip(batch, self.top20(lg)): (L if fam == "L" else S)[i] = log_odds(tm, ys, ab)
            tokens += len(suffix_ids) * len(batch)
        rec = self._readouts(ex, book["chunks"], K, L, S, tokens=tokens, passes=new_rows + n_batches, encoded=new_tokens, book=book)
        rec.update(shared=int(new_tokens == 0), encode_s=encode_s, copies_s=time.time() - t_cop - sum(rec.get(x, 0.0) for x in ("decode_s", "gen_s", "gen_s_store")),
                   new_tokens=new_tokens, doc_tokens=P + st["used"], rows=R, K=K, spans=len(spans), copy_batch=B, offloaded=int(book["on_cpu"]), pinned=int(book["pinned"]),
                   max_pos=int(max(bases) + max(lens)) + len(suffix_ids), clipped=st["clipped"])
        hit = [m for m, c in enumerate(book["chunks"]) if ex["gold"][0] in c]                        # the block that holds the asked needle, when one block holds it whole
        if hit and L and S:
            cells = evidence_map(L, S, ys); order = sorted(cells, key=cells.get, reverse=True); cell = (hit[0] // K, hit[0] % K)
            rec.update(needle_block=hit[0], needle_rank=(order.index(cell) + 1) if cell in cells else None, map_cells=len(cells),
                       needle_selected=int(hit[0] in (rec.get("selected") or [])))
        return rec

    def mono_conv(self, ex):
        """The monolithic read over a conversation with long turns, with a prefix cache that grows with the conversation: every turn appends its
        text to the cache (each new token attends to everything before it) and then answers its question over the cache. When the conversation
        no longer fits the window, the read keeps the most recent whole turns that fit, as a chat does, and the cache of those turns is encoded
        again, since the tokens before them are gone. The row says which turns were kept and whether the asked needle was among them."""
        t = self.torch; P = len(self.prefix_ids); st = getattr(self, "_mconv", None); budget = max(1, args.window - P - args.reuse_margin - args.max_new - 64)
        if st is None or st["conv"] != ex["conv"]:
            self._mconv = None; st = None
            if self.dev == "cuda": t.cuda.empty_cache()
            st = {"conv": ex["conv"], "seg_n": {}, "cache": None, "doc_ids": [], "first": None}; self._mconv = st
        t_tok = time.time()
        for u in range(ex["turn"] + 1):
            if u not in st["seg_n"]: st["seg_n"][u] = len(self.tok.encode(ex["segments"][u], add_special_tokens=False)) + 2
        first = ex["turn"]; tot = st["seg_n"][first]
        while first > 0 and tot + st["seg_n"][first - 1] <= budget: first -= 1; tot += st["seg_n"][first]
        text = "\n\n".join(ex["segments"][first:ex["turn"] + 1])
        if tot > budget: text = self.tok.decode(self.tok.encode(text, add_special_tokens=False)[-budget:], skip_special_tokens=True)   # one turn longer than the window
        q = lambda question: self.tok.encode(build_prompt(self.tok, text, {"question": question, "options": None}, "gen"), add_special_tokens=False)
        a, b, full = q("A?"), q("B!"), q(ex["question"]); n_doc = 0                               # the text's tokens: what two prompts with different questions share
        while n_doc < min(len(a), len(b)) and a[n_doc] == b[n_doc]: n_doc += 1
        while n_doc > 0 and full[:n_doc] != a[:n_doc]: n_doc -= 1                                    # and what this turn's prompt shares with them
        doc_ids = full[:n_doc]; tok_s = time.time() - t_tok
        if self.dev == "cuda": t.cuda.synchronize()
        t_enc = time.time(); keep = 0
        if st["cache"] is not None and st["first"] == first:                                      # the cache of the earlier turns is a prefix of this turn's text
            while keep < min(len(st["doc_ids"]), n_doc) and st["doc_ids"][keep] == doc_ids[keep]: keep += 1
            self.crop_cache(st["cache"], keep)
        else:
            st["cache"] = None
            if self.dev == "cuda": t.cuda.empty_cache()
            st["cache"] = self.new_cache(min(args.window, P + int(1.05 * ex["n_turns"] * args.conv_turn_tokens) + 4096 + args.reuse_margin) + args.max_new + 64)
        cache = st["cache"]
        with t.no_grad():
            for s_ in range(keep, n_doc, args.prefill_chunk):
                out = self.model(input_ids=t.tensor([doc_ids[s_:min(s_ + args.prefill_chunk, n_doc)]], device=self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
        st.update(cache=cache, doc_ids=doc_ids, first=first)
        if self.dev == "cuda": t.cuda.synchronize()
        encode_s = time.time() - t_enc; suffix = full[n_doc:]; t_ask = time.time()
        with t.no_grad():
            out = self.model(input_ids=t.tensor([suffix], device=self.dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        lg = out.logits[:, -1]; ask_s = time.time() - t_ask; self._last = (full, lg)
        pred, n_gen, gen_s = self.generate(cache, lg, args.max_new); self.crop_cache(cache, n_doc)
        return {"pred": pred, "scores": None, "tokens": len(suffix) + n_doc - keep, "content_tokens": n_doc - keep, "passes": 1 + math.ceil(max(n_doc - keep, 0) / args.prefill_chunk),
                "truncated": int(first > 0), "first_turn": first, "needle_in_context": int(ex["target_turn"] >= first), "shared": int(keep > 0), "kept_tokens": keep,
                "encode_s": encode_s, "ask_s": ask_s, "tok_s": tok_s, "gen_s": gen_s, "gen_n": n_gen, "doc_tokens": n_doc}

    def mask_doc(self, ex): return self.masked(ex, "doc")
    def mask_chunk(self, ex): return self.masked(ex, "chunk")
    def mask_span(self, ex): return self.masked(ex, "span")
    def mask_stair(self, ex): return self.masked(ex, "stair")
    def mask_stair_alone(self, ex): return self.masked(ex, "stair_alone")
    def mask_stair_both(self, ex): return self.masked(ex, "stair_both")

    @staticmethod
    def span_rows(book, fam, members):
        """The store ids a span reads: its chunks, or in stair_both for a strided span their copy from its own pass (ids n..2n-1)."""
        return [m + book["n"] for m in members] if (book["mode"] == "stair_both" and fam == "S") else members
    def mask_all(self, ex): return self.masked_all(ex, "chunk")
    def mask_all_anchor(self, ex): return self.masked_all(ex, "anchor")

# ============================================================== equivalence check on a tiny random model
def checks():
    """The cache-gather path must equal an explicit single-sequence pass with a block mask and the same positions, for every position mode,
    on a tiny random Qwen3 (CPU, fp32); and the batched copies must equal the one-at-a-time copies with random values in the pad slots.
    This tests the store, the gather, the positions, the prefix handling and the batching; it says nothing about accuracy."""
    def ref_mask(m2d):
        """An explicit (L, L) boolean mask for a test reference, in the form the backend expects: boolean for sdpa, additive for eager."""
        t = lm.torch; m = m2d[None, None]
        return t.zeros(m.shape, dtype=lm.model.dtype).masked_fill(~m, float("-inf")) if args.attn == "eager" else m
    import torch
    lm = LM(args.model, tiny=True); t = torch
    text = " ".join(f"Sentence number {i} talks about topic {i % 7} and item {i * 3}." for i in range(60))
    question = "Which topic appears?"; options = ["topic 1", "topic 2", "topic 3", "topic 4"]
    args.chunk_size = 25; chunks = chunk(lm.tok, text, args.chunk_size); n = len(chunks); k = math.ceil(math.sqrt(n)); local, strd = grid_indices(n, k)
    chunk_ids = [lm.tok.encode(c, add_special_tokens=False) for c in chunks]; P = len(lm.prefix_ids)
    suffix_ids = lm.tok.encode(qa_prompt(lm.tok, "\x00SPAN\x00", question, options, True).split("\x00SPAN\x00")[1], add_special_tokens=False)
    print(f"tiny model: {n} chunks of about {args.chunk_size} tokens, K={k}, prefix {P} tokens, suffix {len(suffix_ids)} tokens")
    print("prefix text:", repr(lm.prefix_text)); print("suffix text ends:", repr(lm.tok.decode(suffix_ids)[-60:]))
    full = lm.tok.encode(qa_prompt(lm.tok, chunks[0], question, options, True), add_special_tokens=False)
    print("prefix + chunk + suffix == processor prompt for one chunk:", full == lm.prefix_ids + chunk_ids[0] + suffix_ids)
    def q_of(mode, bases, members): return int(bases[members[-1]]) + len(chunk_ids[members[-1]]) if mode != "chunk" else P + max(len(chunk_ids[m]) for m in members)
    def batched_vs_single(book, mode, bases):
        """All 2K spans one at a time against batches of 3 (local and strided spans mixed, so lengths differ and padding is exercised),
        with random values written into the pad slots: only the mask can keep them out."""
        allspans = [ms for _, ms in local] + [ms for _, ms in strd]
        seq = t.stack([lm.ask_spans(book, [ms], suffix_ids, [q_of(mode, bases, ms)])[0] for ms in allspans])
        lm._pad_noise = True
        bat = t.cat([lm.ask_spans(book, allspans[b:b + 3], suffix_ids, [q_of(mode, bases, ms) for ms in allspans[b:b + 3]]) for b in range(0, len(allspans), 3)])
        lm._pad_noise = False; d = float((seq.float() - bat.float()).abs().max())
        print(f"  mode {mode:5s}: {len(allspans)} spans one at a time vs batches of 3 (mixed lengths, random values in the pad slots), max |dlogit| {d:.2e} -> {'OK' if d < 1e-4 else 'FAIL'}")
    for mode in ("doc", "chunk", "stair_alone"):
        book = lm.encode_chunks(chunk_ids, mode); bases = book["bases"]; worst = 0.0
        for name, members in (("local 0", local[0][1]), ("local last", local[-1][1]), ("strided 0", strd[0][1]), ("strided last", strd[-1][1])):
            # cache-gather path, one span
            q_start = q_of(mode, bases, members)
            lg_a = lm.ask_spans(book, [members], suffix_ids, [q_start])[0]
            # explicit single sequence: prefix, member chunks, suffix, with a block mask and the same positions
            ids = list(lm.prefix_ids); pos = list(range(P)); blocks = [(0, P)]
            for m in members:
                s0 = len(ids); ids += chunk_ids[m]; st = int(bases[m]); pos += [st + x for x in range(len(chunk_ids[m]))]; blocks.append((s0, len(ids)))
            s0 = len(ids); ids += suffix_ids; pos += [q_start + x for x in range(len(suffix_ids))]; Lq = len(ids)
            m = t.zeros((Lq, Lq), dtype=t.bool); m[:P, :P] = t.tril(t.ones((P, P), dtype=t.bool))
            for (a, b) in blocks[1:]: m[a:b, :P] = True; m[a:b, a:b] = t.tril(t.ones((b - a, b - a), dtype=t.bool))
            m[s0:, :s0] = True; m[s0:, s0:] = t.tril(t.ones((Lq - s0, Lq - s0), dtype=t.bool))
            with t.no_grad():
                lg_b = lm.model(input_ids=t.tensor([ids]), attention_mask=ref_mask(m), position_ids=t.tensor([pos]), use_cache=False, logits_to_keep=1).logits[0, -1]
            d = float((lg_a.float() - lg_b.float()).abs().max()); worst = max(worst, d)
            print(f"  mode {mode:5s} {name:12s}: {len(members)} chunks, cache-gather vs explicit block mask, max |dlogit| {d:.2e}")
        # and the block mask is NOT the plain causal read: chunks do not see each other
        with t.no_grad():
            lg_c = lm.model(input_ids=t.tensor([ids]), position_ids=t.tensor([pos]), use_cache=False, logits_to_keep=1).logits[0, -1]
        print(f"  mode {mode:5s}: vs a plain causal read of the same sequence (chunks seeing each other), max |dlogit| {float((lg_b.float() - lg_c.float()).abs().max()):.2e} (must NOT be 0)")
        if mode == "doc":                                                                        # the Periscope arm's path: left-padded batch, default positions, last-token logits
            lg_d = lm.forward_last([ids, ids[:40]])
            with t.no_grad():
                ref0 = lm.model(input_ids=t.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0, -1]
                ref1 = lm.model(input_ids=t.tensor([ids[:40]]), use_cache=False, logits_to_keep=1).logits[0, -1]
            print(f"  forward_last: unpadded row vs plain read {float((lg_d[0].float() - ref0.float()).abs().max()):.2e}, left-padded row vs plain read {float((lg_d[1].float() - ref1.float()).abs().max()):.2e} (RoPE is relative: both about 0)")
        print(f"  mode {mode}: worst equivalence error {worst:.2e}  -> {'OK' if worst < 1e-3 else 'FAIL'}")
        batched_vs_single(book, mode, bases)
    # span and stair modes: blocks are whole local spans. Explicit reference: one sequence [prefix, all chunks in order, suffix]; chunk m attends to
    # the prefix, to earlier chunks of ITS OWN local span and to itself; the suffix attends to the prefix, the member chunks and itself.
    def span_ref(bases, members, sfx, q_start):
        """The explicit reference of a probe in span and stair modes, last-token logits."""
        ids = list(lm.prefix_ids); pos = list(range(P)); span_of = {}; starts = {}
        for m in range(n):
            starts[m] = (len(ids), len(ids) + len(chunk_ids[m])); ids += chunk_ids[m]; pos += [int(bases[m]) + x for x in range(len(chunk_ids[m]))]; span_of[m] = m // k
        s0 = len(ids); ids += sfx; pos += [q_start + x for x in range(len(sfx))]; Lq = len(ids)
        msk = t.zeros((Lq, Lq), dtype=t.bool); msk[:P, :P] = t.tril(t.ones((P, P), dtype=t.bool))
        for m in range(n):
            a, b = starts[m]; msk[a:b, :P] = True
            for m2 in range(n):
                if span_of[m2] == span_of[m] and m2 < m: a2, b2 = starts[m2]; msk[a:b, a2:b2] = True
            msk[a:b, a:b] = t.tril(t.ones((b - a, b - a), dtype=t.bool))
        msk[s0:, :P] = True
        for m in members: a, b = starts[m]; msk[s0:, a:b] = True
        msk[s0:, s0:] = t.tril(t.ones((Lq - s0, Lq - s0), dtype=t.bool))
        with t.no_grad():
            return lm.model(input_ids=t.tensor([ids]), attention_mask=ref_mask(msk), position_ids=t.tensor([pos]), use_cache=False, logits_to_keep=1).logits[0, -1]
    for mode in ("span", "stair"):
        book = lm.encode_chunks(chunk_ids, mode); bases = book["bases"]; worst = 0.0
        for name, members in (("local 0", local[0][1]), ("local last", local[-1][1]), ("strided 0", strd[0][1]), ("strided last", strd[-1][1])):
            q_start = q_of(mode, bases, members)
            lg_a = lm.ask_spans(book, [members], suffix_ids, [q_start])[0]
            lg_b = span_ref(bases, members, suffix_ids, q_start)
            d = float((lg_a.float() - lg_b.float()).abs().max()); worst = max(worst, d)
            # positions inside the gathered span: no two member chunks may overlap, and the block should be contiguous in stair mode
            ranges = sorted((int(bases[m]), int(bases[m]) + len(chunk_ids[m])) for m in members); overlap = any(ranges[i][1] > ranges[i + 1][0] for i in range(len(ranges) - 1))
            print(f"  mode {mode:5s} {name:12s}: {len(members)} chunks, cache-gather vs explicit span-block mask, max |dlogit| {d:.2e}, positions {ranges[0][0]}..{ranges[-1][1]}, overlap {overlap}")
        print(f"  mode {mode:5s}: worst equivalence error {worst:.2e}  -> {'OK' if worst < 1e-3 else 'FAIL'}; largest position used {int(max(bases)) + max(len(c) for c in chunk_ids)} (document length {sum(len(c) for c in chunk_ids)} tokens)")
        # a local span equals Periscope's local computation: a plain causal read of [prefix + span chunks + suffix] at the same positions
        members = local[0][1]; ids = list(lm.prefix_ids) + [x for m in members for x in chunk_ids[m]] + suffix_ids; q0 = q_of(mode, bases, members)
        pos = list(range(P)) + [int(bases[m]) + j for m in members for j in range(len(chunk_ids[m]))] + [q0 + x for x in range(len(suffix_ids))]
        # explicit causal mask: with no mask and gapped positions, transformers infers packed sequences from the position jumps and splits the span
        cm = t.tril(t.ones((len(ids), len(ids)), dtype=t.bool))
        with t.no_grad(): lg_p = lm.model(input_ids=t.tensor([ids]), position_ids=t.tensor([pos]), attention_mask=ref_mask(cm), use_cache=False, logits_to_keep=1).logits[0, -1]
        lg_s = lm.ask_spans(book, [members], suffix_ids, [q0])[0]
        print(f"  mode {mode:5s}: local span 0 vs a plain causal read of the same span (Periscope's local call), max |dlogit| {float((lg_s.float() - lg_p.float()).abs().max()):.2e} (must be about 0)")
        batched_vs_single(book, mode, bases)

    # ablation (d), stair_both: every span, local or strided, read from the store must equal a plain causal read of [prefix, its chunks, suffix] at
    # stair positions (its own pass); under stair the strided spans must not (their chunks were encoded within their local spans)
    def plain_read(bases, members, q0):
        ids = list(lm.prefix_ids) + [x for m in members for x in chunk_ids[m]] + suffix_ids
        pos = list(range(P)) + [int(bases[m]) + j for m in members for j in range(len(chunk_ids[m]))] + [q0 + x for x in range(len(suffix_ids))]
        cm = t.tril(t.ones((len(ids), len(ids)), dtype=t.bool))
        with t.no_grad(): return lm.model(input_ids=t.tensor([ids]), position_ids=t.tensor([pos]), attention_mask=ref_mask(cm), use_cache=False, logits_to_keep=1).logits[0, -1]
    for mode in ("stair", "stair_both"):
        book = lm.encode_chunks(chunk_ids, mode); bases = book["bases"]; worst = {"L": 0.0, "S": 0.0}; allrows = []
        for fam, lst in (("L", local), ("S", strd)):
            for _, members in lst:
                rows, q0 = LM.span_rows(book, fam, members), q_of(mode, bases, members); allrows.append((rows, q0))
                d = float((lm.ask_spans(book, [rows], suffix_ids, [q0])[0].float() - plain_read(bases, members, q0).float()).abs().max()); worst[fam] = max(worst[fam], d)
        ok = worst["L"] < 1e-4 and (worst["S"] < 1e-4 if mode == "stair_both" else worst["S"] > 1e-4)
        print(f"  mode {mode}: every span vs a plain causal read of it at stair positions, worst max |dlogit| local {worst['L']:.2e}, strided {worst['S']:.2e} "
              f"({'both about 0' if mode == 'stair_both' else 'strided must NOT be 0'}) -> {'OK' if ok else 'FAIL'}")
    seq = t.stack([lm.ask_spans(book, [r], suffix_ids, [q])[0] for r, q in allrows]); lm._pad_noise = True
    bat = t.cat([lm.ask_spans(book, [r for r, _ in allrows[b:b + 3]], suffix_ids, [q for _, q in allrows[b:b + 3]]) for b in range(0, len(allrows), 3)]); lm._pad_noise = False
    d_b = float((seq.float() - bat.float()).abs().max())
    print(f"  mode stair_both: {len(allrows)} spans one at a time vs batches of 3 (random values in the pad slots), max |dlogit| {d_b:.2e} -> {'OK' if d_b < 1e-4 else 'FAIL'}")
    ex_b = {"context": text, "question": question, "options": options, "gold": "A"}
    r_b = lm.masked(ex_b, "stair_both"); saved = args.copy_batch; args.copy_batch = 1; lm.drop_book(); r_b1 = lm.masked(ex_b, "stair_both"); args.copy_batch = saved; lm.drop_book()
    dd = max(abs(r_b["scores"][y] - r_b1["scores"][y]) for y in r_b["scores"])
    print(f"masked() stair_both with --copy_batch {saved} vs 1: passes {r_b['passes']} vs {r_b1['passes']}, direct {r_b['direct']}, max |dscore| {dd:.2e} -> {'OK' if dd < 1e-4 else 'FAIL'}")

    # the production path end to end: --copy_batch as set vs one span at a time must give the same scores; cache reuse across questions on the
    # same text: second call must reuse and give identical scores; a different text must re-encode
    ex = {"context": text, "question": question, "options": options, "gold": "A"}
    r1 = lm.masked(ex, "stair"); saved = args.copy_batch; args.copy_batch = 1; r1_one = lm.masked(ex, "stair"); args.copy_batch = saved
    dd = max(abs(r1["scores"][y] - r1_one["scores"][y]) for y in r1["scores"])
    print(f"masked() with --copy_batch {saved} vs 1 (stair): passes {r1['passes']} vs {r1_one['passes']}, max |dscore| {dd:.2e} -> {'OK' if dd < 1e-4 else 'FAIL'}")
    r1 = lm.masked(ex, "chunk"); r2 = lm.masked(dict(ex, question="Which item appears?"), "chunk"); r3 = lm.masked(dict(ex, context=text + " Extra sentence."), "chunk")
    r1b = lm.masked(ex, "chunk")
    print(f"reuse: first call shared={r1['shared']} encode {r1['encode_s']:.2f}s | same text, new question shared={r2['shared']} encode {r2['encode_s']:.2f}s | new text shared={r3['shared']} | "
          f"same question again identical scores: {r1b['scores'] == lm.masked(ex, 'chunk')['scores']}")
    # window_reuse: the chunked prefill plus a suffix over the cache must equal the window read's single forward; the second question of the
    # same text must reuse the cache (cropped back after the first) and still equal its own window read; decode timing must not change the answer
    saved_pc = args.prefill_chunk; args.prefill_chunk = 300                                   # the 1016-token prompt prefills in four pieces
    ex2 = dict(ex, question="Which item appears?"); w1 = lm.window_read(ex); ids_w, lg_w = lm._last; w2 = lm.window_read(ex2)
    u1 = lm.window_reuse(ex); ids_u, lg_u = lm._last; u2, u1b = lm.window_reuse(ex2), lm.window_reuse(ex)
    dl = float((lg_w.float() - lg_u.float()).abs().max())
    dw = max(max(abs(w1["scores"][y] - u1["scores"][y]) for y in w1["scores"]), max(abs(w2["scores"][y] - u2["scores"][y]) for y in w2["scores"]), max(abs(u1["scores"][y] - u1b["scores"][y]) for y in u1["scores"]))
    fb = u1["fallback"] + u2["fallback"] + u1b["fallback"]
    print(f"window_reuse: same tokens as the window read {ids_w == ids_u} (lengths {len(ids_w)}, {len(ids_u)}), max |dlogit| {dl:.2e}, first shared={u1['shared']} prefills {u1['passes'] - 1}, "
          f"second shared={u2['shared']}, fallbacks {fb}, max |dscore| {dw:.2e} -> {'OK' if dl < 1e-4 and fb == 0 else 'FAIL'}")
    saved = args.gen_tokens; args.gen_tokens = 5; g_w, g_u, g_m = lm.window_read(ex), lm.window_reuse(ex), lm.masked(ex, "stair"); args.gen_tokens = saved; args.prefill_chunk = saved_pc
    same = (g_w["pred"] == w1["pred"]) and (g_u["pred"] == u1["pred"]) and (g_m["pred"] == r1["pred"] if r1.get("pred") else True)
    print(f"decode timing: window {g_w['decode_ms_per_tok']:.1f} ms/tok, window_reuse {g_u['decode_ms_per_tok']:.1f}, select reread {g_m['decode_ms_per_tok']:.1f}, reread_s {g_m['reread_s']:.3f}, answers unchanged {same} -> {'OK' if same else 'FAIL'}")
    open_checks(lm, ex, text, chunks, chunk_ids, local, strd, k, P, q_of, span_ref, ref_mask)
    all_checks(lm, ex, chunk_ids, P, ref_mask)
    needle_checks(lm, text)
    conv_checks(lm)
    # the Periscope functions on the same text
    kk, loc, st = grid(chunks); print(f"grid: K={kk}, {len(loc)} local spans, {len(st)} strided spans; local 0 has {len(lm.tok.encode(loc[0][1]))} tokens")
    print("log_odds on a fake top-20:", log_odds({" B": -0.5, " E": -1.5, "A": -3.0}, ["A", "B", "C", "D"], "E"))
    sparse_checks(lm, ex)
    yarn_checks(ex, w1)


def yarn_checks(ex, native):
    """--yarn on a second tiny model (factor 4 over 256 positions): its rotary embedding must carry YaRN's attention scaling 0.1 ln 4 + 1, the read must
    differ from the native read, and window_reuse must still equal the single-pass window read from the same tokens."""
    saved = (args.yarn, args.yarn_orig, args.prefill_chunk); args.yarn, args.yarn_orig, args.prefill_chunk = 4.0, 256, 300
    try:
        lm = LM(args.model, tiny=True); sc = float(lm.model.model.rotary_emb.attention_scaling)
        w = lm.window_read(ex); ids_w = lm._last[0]; u = lm.window_reuse(ex); ids_u = lm._last[0]
    finally: args.yarn, args.yarn_orig, args.prefill_chunk = saved
    want = 0.1 * math.log(4) + 1
    d_wu = max(abs(w["scores"][y] - u["scores"][y]) for y in w["scores"]); d_nat = max(abs(w["scores"][y] - native["scores"][y]) for y in w["scores"])
    ok = abs(sc - want) < 1e-4 and ids_w == ids_u and d_wu < 1e-4 and d_nat > 1e-4
    print(f"yarn: rotary attention scaling {sc:.4f} (expected {want:.4f}); under YaRN window_reuse vs the window read: same tokens {ids_w == ids_u}, max |dscore| {d_wu:.2e}; "
          f"vs the native read {d_nat:.2e} (must NOT be 0) -> {'OK' if ok else 'FAIL'}")


def conv_checks(lm):
    """The growing store of fca_append on the tiny model: (1) three turns appended one after another give the same store as all their blocks
    appended at once, so encoding a partly filled row again changes nothing that was stored; (2) with K = ceil(sqrt(blocks)) and the stride
    of the fixed store, the grown store equals the store encode_chunks builds in stair mode; (3) the conversation generator; (4) the two arms
    run over a three-turn conversation, the store of fca_append grows by the new turn only, and mono_conv reuses its cache."""
    t = lm.torch; saved = (args.chunk_size, args.conv_K, args.conv_turn_tokens, args.window, args.prefill_chunk, args.reuse_margin, args.max_new)
    args.chunk_size = 25; segs = [" ".join(f"Turn {u} sentence {i} mentions item {i * 7 + u} and place {i % 5}." for i in range(n)) for u, n in enumerate((14, 9, 17))]
    parts = [chunk(lm.tok, s_, args.chunk_size) for s_ in segs]; flat = [c for p_ in parts for c in p_]; n = len(flat); k = math.ceil(math.sqrt(n))
    ids = [lm.tok.encode(c, add_special_tokens=False) for c in flat]; stride = max(len(x) for x in ids); cap = sum(len(x) for x in ids) + 8
    def grown(groups, K):
        args.conv_K = K; st = lm.conv_state(0, 3, cap_tokens=cap); st["stride"] = stride; rows = [lm.conv_append(st, g) for g in groups]
        return st["book"]["buf"][:, :, :len(lm.prefix_ids) + st["used"]].clone(), rows, st
    K_fix = 4; inc, rows_inc, _ = grown(parts, K_fix); one, rows_one, _ = grown([flat], K_fix)
    d1 = float((inc - one).abs().max())
    print(f"conversation store: {n} blocks in 3 turns, K={K_fix}: appended turn by turn (rows encoded {rows_inc}) vs all at once ({rows_one}), max |d| {d1:.2e} -> {'OK' if d1 < 1e-4 else 'FAIL'}")
    sq, _, st = grown(parts, k); ref = lm.encode_chunks(ids, "stair"); d2 = float((sq - ref["buf"][:, :, :sq.shape[2]]).abs().max())
    same_pos = list(st["book"]["bases"]) == list(ref["bases"]) and list(st["book"]["offs"]) == list(ref["offs"])
    print(f"conversation store with K={k}: grown store vs the fixed stair store, max |d| {d2:.2e}, same positions and offsets {same_pos} -> {'OK' if d2 < 1e-4 and same_pos else 'FAIL'}")
    hay = " ".join(f"Filler sentence {i} about subject {i % 9}." for i in range(900)); args.conv_turn_tokens = 120
    exs = make_conversations(lm.tok, hay, 2, 3, args.conv_turn_tokens)
    ok = all(e["gold"][0] in e["segments"][e["target_turn"]] and e["gold"][0] in e["context"] and e["target_turn"] <= e["turn"] for e in exs)
    ok &= all(sum(v in e["context"] for v in {x["gold"][0] for x in exs if x["conv"] == e["conv"]}) <= e["turn"] + 1 for e in exs)
    print(f"conversation generator: {len(exs)} turns, lengths {[len(lm.tok.encode(s_, add_special_tokens=False)) for s_ in exs[0]['segments']]} tokens for {args.conv_turn_tokens}, "
          f"every asked needle in its target turn and in the context -> {'OK' if ok else 'FAIL'}")
    args.conv_K = 0; args.window = 4096; args.prefill_chunk = 64; args.reuse_margin = 64; args.max_new = 4; lm.release(); grew, good = [], True
    for e in exs[:3]:
        r = lm.fca_append(e); grew.append(r["new_tokens"]); good &= r["pred"] is not None and r["rows"] == math.ceil(len(lm._conv["ids"]) / lm._conv["K"])
    again = lm.fca_append(exs[2]); good &= again["new_tokens"] == 0 and again["shared"] == 1
    print(f"fca_append over 3 turns: new tokens per turn {grew}, a repeated turn adds {again['new_tokens']}; store {again['doc_tokens']} tokens, "
          f"{again['rows']} rows of K={again['K']}, {again['spans']} spans -> {'OK' if good and all(g > 0 for g in grew) else 'FAIL'}")
    lm.release(); kept, good = [], True
    for e in exs[:3]:
        r = lm.mono_conv(e); kept.append(r["kept_tokens"]); full, lg = lm._last
        ref_lg = lm.forward_last([full]); good &= float((lg - ref_lg).abs().max()) < 1e-3
    print(f"mono_conv over 3 turns: cache tokens reused per turn {kept}, last-token logits equal an uncached read -> {'OK' if good and kept[0] == 0 and kept[1] > 0 and kept[2] > kept[1] else 'FAIL'}")
    args.window = 300; lm.release(); r = [lm.mono_conv(e) for e in exs[:3]][-1]
    print(f"mono_conv with a window of 300 tokens: at turn 3 it keeps turns from {r['first_turn'] + 1}, truncated {r['truncated']} -> {'OK' if r['truncated'] == 1 and r['first_turn'] > 0 else 'FAIL'}")
    lm.release(); (args.chunk_size, args.conv_K, args.conv_turn_tokens, args.window, args.prefill_chunk, args.reuse_margin, args.max_new) = saved

def needle_checks(lm, text):
    """The needle generator on a small haystack (the check text three times, lengths 300 and 600 tokens, 5 depths, 2 needles): each sample holds one
    needle with its number, has the requested length, and puts the needle at the requested depth in tokens (to within a sentence of the haystack)."""
    exs = make_needles(lm.tok, " ".join([text] * 3), [300, 600], 5, 2); worst_len = worst_dep = 0.0; ok = True
    for e in exs:
        ids = lm.tok.encode(e["context"], add_special_tokens=False); ci = e["context"].find("One of the special magic numbers")
        word = e["question"].split(" for ")[1].split(" mentioned")[0]; n_needle = len(lm.tok.encode(f"One of the special magic numbers for {word} is: {e['gold'][0]}.", add_special_tokens=False))
        dep = len(lm.tok.encode(e["context"][:ci], add_special_tokens=False)) / (len(ids) - n_needle)                 # the needle's start within the haystack
        ok &= e["context"].count("special magic number") == 1 and f"for {word} is: {e['gold'][0]}." in e["context"]
        worst_len = max(worst_len, abs(len(ids) - e["length"]) / e["length"]); worst_dep = max(worst_dep, abs(dep - e["depth"] / 100))
    ok &= worst_len < 0.02 and worst_dep < 0.04
    print(f"needle: {len(exs)} samples, length off by at most {100 * worst_len:.1f}%, depth in the haystack off by at most {100 * worst_dep:.1f} points (half a sentence of about 12 tokens), "
          f"one needle with its number in each -> {'OK' if ok else 'FAIL'}")
    print("  needle at depth 50%, 300 tokens:", repr(exs[4]["context"][max(0, exs[4]["context"].find('One of') - 60):exs[4]["context"].find('One of') + 80]))


def all_checks(lm, ex, chunk_ids, P, ref_mask):
    """One query over all blocks (ablation b), both encodings: the store-backed cache must equal the ordinary gather of every layer at once, and the
    read must equal one explicit sequence under a block mask: prefix, the first chunk, then each later chunk (after its own copy of the first
    chunk for the anchor mode), then the request, which sees the prefix and every chunk but never an anchor copy. A second question shares the store."""
    import torch as t
    n = len(chunk_ids); req = lm.tok.encode(request_text(lm.tok, ex, "letter"), add_special_tokens=False); a = len(chunk_ids[0])
    for mode in ("chunk", "anchor"):
        lm.release(); r = lm.masked_all(ex, mode); lg_a = lm._last_all[0]; book = lm._book; bases, lens = book["bases"], book["lens"]
        q0 = int(max(int(bases[m]) + int(lens[m]) for m in range(n)))
        lg_g = lm.ask_spans(book, [list(range(n))], req, [q0])[0]
        ids = list(lm.prefix_ids); pos = list(range(P)); blocks = []
        for m in range(n):
            s_a = None
            if mode == "anchor" and m > 0: s_a = len(ids); ids += chunk_ids[0]; pos += list(range(P, P + a))
            s0 = len(ids); ids += chunk_ids[m]; pos += [int(bases[m]) + j for j in range(len(chunk_ids[m]))]; blocks.append((s_a, s0, len(ids)))
        s_q = len(ids); ids += req; pos += [q0 + j for j in range(len(req))]; Lq = len(ids)
        msk = t.zeros((Lq, Lq), dtype=t.bool); msk[:P, :P] = t.tril(t.ones((P, P), dtype=t.bool))
        for s_a, s0, e in blocks:
            if s_a is not None:
                msk[s_a:s0, :P] = True; msk[s_a:s0, s_a:s0] = t.tril(t.ones((s0 - s_a, s0 - s_a), dtype=t.bool)); msk[s0:e, s_a:s0] = True
            msk[s0:e, :P] = True; msk[s0:e, s0:e] = t.tril(t.ones((e - s0, e - s0), dtype=t.bool)); msk[s_q:, s0:e] = True
        msk[s_q:, :P] = True; msk[s_q:, s_q:] = t.tril(t.ones((Lq - s_q, Lq - s_q), dtype=t.bool))
        with t.no_grad():
            lg_b = lm.model(input_ids=t.tensor([ids]), attention_mask=ref_mask(msk), position_ids=t.tensor([pos]), use_cache=False, logits_to_keep=1).logits[0, -1]
        d_g, d_b = float((lg_a - lg_g).abs().max()), float((lg_a - lg_b).abs().max()); r2 = lm.masked_all(ex, mode)
        ok = d_g < 1e-4 and d_b < 1e-3 and r2["shared"] == 1 and r2["scores"] == r["scores"]
        print(f"one query over all blocks, {mode:6s}: {n} chunks{' (each after an anchor copy)' if mode == 'anchor' else ''}, request at position {q0}; store-backed cache vs a full gather "
              f"max |dlogit| {d_g:.2e}, vs the explicit mask {d_b:.2e}; a second question shares the store {r2['shared']} -> {'OK' if ok else 'FAIL'}")
    # the store of a book must be released as soon as the next book replaces it, without waiting for Python's garbage collector (the first GPU runs
    # kept every book's store alive through the cache's layer class and went out of memory)
    import gc, weakref
    lm.release(); gc.collect(); gc.disable()
    try:
        lm.masked_all(ex, "chunk"); w = weakref.ref(lm._book["buf"]); lm.masked_all(dict(ex, context=ex["context"] + " One more sentence."), "chunk"); freed = w() is None
    finally: gc.enable()
    print(f"one query over all blocks: the previous book's store is freed when the next book arrives, with the garbage collector off {freed} -> {'OK' if freed else 'FAIL'}")
    lm.release()


def open_checks(lm, ex, text, chunks, chunk_ids, local, strd, k, P, q_of, span_ref, ref_mask):
    """Open questions and conversations (SCBench) and the store read-out, on the tiny model: the prompt pieces, the yes/no probe, the re-rotation,
    the store gather, generation through a cache, and the open arms end to end with both select read-outs."""
    import torch as t
    question, options = ex["question"], ex["options"]
    ex_o = {"context": text, "question": "Which item goes with topic 3?", "options": None, "gold": ["item 9"], "turn": 1,
            "history": [{"question": "Which topic comes first?", "options": None, "answer": "topic 0"}]}
    ex_m = dict(ex, history=[{"question": "Which topic comes first?", "options": options, "answer": "topic 1"}], turn=1)
    enc = lambda s: lm.tok.encode(s, add_special_tokens=False)
    # 1) prompts: no history is exactly the published prompt; the stored prefix, a block and the request make the whole conversation
    same_mc = (build_prompt(lm.tok, text, ex, "abstain") == qa_prompt(lm.tok, text, question, options, True)
               and build_prompt(lm.tok, text, ex, "letter") == qa_prompt(lm.tok, text, question, options, False))
    pieces = all(lm.prefix_ids + chunk_ids[0] + enc(request_text(lm.tok, e, c)) == enc(build_prompt(lm.tok, chunks[0], e, c))
                 for e, c in ((ex_o, "yesno"), (ex_o, "gen"), (ex_m, "abstain"), (ex_m, "letter")))
    print(f"prompts: no history == qa_prompt {same_mc}; prefix + block + request == the whole conversation (open and multiple choice, with history) {pieces} "
          f"-> {'OK' if same_mc and pieces else 'FAIL'}")
    print("  open request with history, yes/no probe:", repr(request_text(lm.tok, ex_o, "yesno")))
    # 2) the yes/no probe with the history over the stair store against the explicit span-block mask
    book = lm.encode_chunks(chunk_ids, "stair"); bases = book["bases"]; yn = enc(request_text(lm.tok, ex_o, "yesno"))
    d_yn = max(float((lm.ask_spans(book, [ms], yn, [q_of("stair", bases, ms)])[0] - span_ref(bases, ms, yn, q_of("stair", bases, ms))).abs().max()) for ms in (local[1][1], strd[1][1]))
    print(f"yes/no probe with history (stair), local 1 and strided 1, cache-gather vs explicit span-block mask, max |dlogit| {d_yn:.2e} -> {'OK' if d_yn < 1e-3 else 'FAIL'}")
    # 3) re-rotation: a block encoded alone at p and rotated by d has, in every layer, the keys of the same block encoded at p + d
    ids0 = chunk_ids[0]; cm = ref_mask(t.tril(t.ones((len(ids0), len(ids0)), dtype=t.bool)))
    def keys_at(p0):
        with t.no_grad(): return lm.model.model(input_ids=t.tensor([ids0]), position_ids=t.arange(p0, p0 + len(ids0))[None], attention_mask=cm, use_cache=True).past_key_values
    d_rot = 0.0
    for p0, dd in ((50, 37), (300, -123)):
        c = keys_at(p0); lm.rerotate(c, t.full((len(ids0),), dd)); ref = keys_at(p0 + dd)
        d_rot = max(d_rot, max(float((a.keys - b.keys).abs().max()) for a, b in zip(c.layers, ref.layers)))
    print(f"re-rotation: keys at p rotated by d vs keys computed at p + d, every layer, d = 37 and -123, max |d| {d_rot:.2e} -> {'OK' if d_rot < 1e-4 else 'FAIL'}")
    # 4) the store gather lays the selected blocks at consecutive positions: layer 0 keys depend on the token and its position only, so after the
    #    re-rotation they equal a fresh encoding of [prefix + blocks] at 0..T. Chunks 1 and K lie on one anti-diagonal: the same stored positions
    picked = [1, k]; sc_, T_ = lm.store_cache(book, picked); seq = list(lm.prefix_ids) + [x for m in picked for x in chunk_ids[m]]
    with t.no_grad(): fresh = lm.model.model(input_ids=t.tensor([seq]), use_cache=True).past_key_values
    d0 = float((sc_.layers[0].keys - fresh.layers[0].keys).abs().max()); dl = float((sc_.layers[-1].keys - fresh.layers[-1].keys).abs().max())
    print(f"store gather of chunks {picked} (both stored from position {int(bases[1])}): {T_} tokens, layer 0 keys vs a fresh encoding at consecutive positions max |d| {d0:.2e}, "
          f"last layer {dl:.2e} (not 0: each block keeps the context it was encoded with) -> {'OK' if d0 < 1e-4 and T_ == len(seq) else 'FAIL'}")
    # 5) with no shift (span mode, local span 0: stored at the consecutive positions already) the store read-out is the probe path with the request
    saved = (args.select_mode, args.max_new, args.prefill_chunk); args.max_new = 6; book_s = lm.encode_chunks(chunk_ids, "span"); ms = local[0][1]; d_st = 0.0
    for e in (ex_m, ex_o):
        req = enc(request_text(lm.tok, e, read_spec(e)[1])); lm.read_store(book_s, ms, e)
        d_st = max(d_st, float((lm._last_store[1][0] - lm.ask_spans(book_s, [ms], req, [P + sum(len(chunk_ids[m]) for m in ms)])[0]).abs().max()))
    print(f"store read-out with no shift vs the gathered probe with the same request (multiple choice and open), max |dlogit| {d_st:.2e} -> {'OK' if d_st < 1e-4 else 'FAIL'}")
    # 6) generation through a cache equals greedy full re-reads, token by token
    ids_g = enc(build_prompt(lm.tok, chunks[0], ex_o, "gen")); args.prefill_chunk = 40
    lg, cache = lm.prefill(ids_g, extra=8); lm.generate(cache, lg, 8); toks = list(lm._last_gen); sq = list(ids_g); ref = []
    with t.no_grad():
        while len(ref) < 8:
            tid = int(lm.model(input_ids=t.tensor([sq]), use_cache=False, logits_to_keep=1).logits[0, -1].argmax())
            if tid in lm.stop_ids: break
            ref.append(tid); sq.append(tid)
    print(f"generate: {len(toks)} greedy tokens through a chunked-prefill cache vs full re-reads, identical {toks == ref} -> {'OK' if toks == ref else 'FAIL'}")
    # 7) the open arms end to end, select both: window and window_reuse give the same answer from the same tokens, the next turn of the conversation
    #    reuses the document cache, the masked arm gives both read-outs (no direct answer for an open question), multiple choice with history too
    args.select_mode, args.prefill_chunk = "both", 300; lm.release()
    m_o, m_m = lm.masked(ex_o, "stair"), lm.masked(ex_m, "stair")
    w_o = lm.window_read(ex_o); ids_w = lm._last[0]; u_o = lm.window_reuse(ex_o); ids_u = lm._last[0]; u_o0 = lm.window_reuse(dict(ex_o, history=[], turn=0)); p_o = lm.periscope(ex_o)
    ok = (w_o["pred"] == u_o["pred"] and ids_w == ids_u and u_o["fallback"] + u_o0["fallback"] == 0 and u_o0["shared"] == 1 and m_o["direct"] is None
          and isinstance(m_o["pred"], str) and isinstance(m_o["pred_store"], str) and m_m["pred_store"] in letters(4)[0] and m_m["direct"] in letters(4)[0] and isinstance(p_o["pred"], str))
    d_rs = max(abs(m_m["read_scores"][y] - m_m["store_scores"][y]) for y in m_m["read_scores"]); ok = ok and d_rs > 1e-4
    print(f"open arms (select both): masked select {m_o['pred']!a}, store {m_o['pred_store']!a} (reread_s {m_o['reread_s']:.3f}, store_s {m_o['store_s']:.3f}); window {w_o['pred']!a} == window_reuse "
          f"{u_o['pred']!a} from the same tokens {ids_w == ids_u}; next turn shares the cache {u_o0['shared']}; multiple choice with history: direct {m_m['direct']}, select {m_m['pred']}, "
          f"store {m_m['pred_store']} (letter scores differ from the reread's by {d_rs:.2e}, must NOT be 0); periscope {p_o['pred']!a} -> {'OK' if ok else 'FAIL'}")
    args.select_mode, args.max_new, args.prefill_chunk = saved; lm.release()


class _FakeTokenSparse:
    """Stand-in for the release's SparseCluster on the CPU: its compress and scatter logic, with a fixed selection per head in place of its Triton
    scores (keep_all keeps every token, otherwise each head keeps a different two thirds plus the last two tokens, in order)."""
    keep_all = False
    def get_index(s, q, k):
        import torch as t
        s.bsz, s.num_heads, s.q_len, s.head_dim = q.shape
        rows = [list(range(s.q_len)) if s.keep_all else [i for i in range(s.q_len) if (i + hh) % 3 != 0 or i >= s.q_len - 2] for hh in range(s.num_heads)]
        n = min(len(r) for r in rows); rows = [r[len(r) - n:] for r in rows]                       # equal counts per head, as the release's top-k gives
        return t.tensor(rows).unsqueeze(0)
    def compress_qkv(s, q, k, v, idx):
        e = idx.unsqueeze(-1).expand(-1, -1, -1, s.head_dim); return q.gather(2, e), k.gather(2, e), v.gather(2, e)
    def decompress_output(s, out, idx):
        import torch as t
        out = out.transpose(1, 2); full = t.zeros(s.bsz, s.num_heads, s.q_len, s.head_dim, dtype=out.dtype)
        full.scatter_(2, idx.unsqueeze(-1).expand(-1, -1, -1, s.head_dim), out); return full.transpose(1, 2).reshape(s.bsz, s.q_len, -1)


def sparse_checks(lm, ex):
    """The sparse-prefill hook's plumbing on the tiny model (the methods' own kernels need a GPU: --check_sparse). With stand-ins that compute what
    each method computes when it keeps everything, the hooked arms must equal the plain arms and only the prefill may reach the method; at the
    attention function, Token Sparse Attention's compress, causal attention and scatter must equal an explicit per-head mask over the kept tokens."""
    import torch as t
    if args.attn != "sdpa": print("sparse hook: skipped (the tiny check runs it on sdpa)"); return
    F = t.nn.functional; saved_pc = args.prefill_chunk; args.prefill_chunk = 300
    # 1) XAttention keeping every block is causal attention over the repeated heads
    fake_x = lambda q, k, v, stride, norm, threshold, use_triton: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    lm.release(); base_w, base_u = lm.window_read(ex), lm.window_reuse(ex); lm.release()
    sp = SparsePrefill("xattention", "sdpa", "tiny", lm.n_layers, fns={"xattention_prefill": fake_x}); lm.sparse = sp
    try:
        hw, hu = lm.window_read(ex), lm.window_reuse(ex); lm.release()
        saved = args.gen_tokens; args.gen_tokens = 3; hg = lm.window_read(ex); args.gen_tokens = saved
    finally: sp.remove(); lm.sparse = None
    dx = max(max(abs(base_w["scores"][y] - hw["scores"][y]) for y in base_w["scores"]), max(abs(base_u["scores"][y] - hu["scores"][y]) for y in base_u["scores"]))
    calls = (hw["sparse_calls"], hu["sparse_calls"], hg["sparse_calls"]); ok = dx < 1e-4 and calls == (lm.n_layers,) * 3 and hg["pred"] == base_w["pred"]
    print(f"sparse hook, XAttention keeping every block: window and window_reuse vs plain, max |dscore| {dx:.2e}; sparse calls {calls}, one per layer "
          f"(the prefill only: the request over the cache and the decode stay on the backend) -> {'OK' if ok else 'FAIL'}")
    # 2) Token Sparse Attention at the attention function: random heads with grouped keys, against an explicit mask per head
    H, Hkv, L, D = 4, 2, 23, 16; g = t.Generator().manual_seed(1)
    q, k, v = t.randn(1, H, L, D, generator=g), t.randn(1, Hkv, L, D, generator=g), t.randn(1, Hkv, L, D, generator=g)
    class _Mod: layer_idx, is_causal, num_key_value_groups, scaling, training = 0, True, H // Hkv, D ** -0.5, False
    sp = SparsePrefill("tokensparse", "sdpa", "tiny", 1, fns={"SparseCluster": _FakeTokenSparse})
    try:
        off, _ = sp(_Mod(), q, k, v, None, scaling=D ** -0.5); ref_off, _ = sp.orig(_Mod(), q, k, v, None, scaling=D ** -0.5)
        sp.on = True; out, _ = sp(_Mod(), q, k, v, None, scaling=D ** -0.5); sp.on = False
    finally: sp.remove()
    kr, vr = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1); idx = _FakeTokenSparse().get_index(q, kr); ref = t.zeros(1, L, H, D)
    for hh in range(H):
        S = idx[0, hh].tolist(); a = (q[0, hh, S] @ kr[0, hh, S].T) * D ** -0.5
        a = a.masked_fill(t.triu(t.ones(len(S), len(S), dtype=t.bool), 1), float("-inf")).softmax(-1); ref[0, S, hh] = a @ vr[0, hh, S]
    dt_, doff = float((out - ref).abs().max()), float((off - ref_off).abs().max())
    print(f"sparse hook, Token Sparse Attention at the attention function: kept-token attention vs explicit per-head mask, max |d| {dt_:.2e}; "
          f"switched off vs the backend, max |d| {doff:.2e} -> {'OK' if dt_ < 1e-5 and doff == 0 else 'FAIL'}")
    # 3) Token Sparse Attention in the model: keeping every token must equal the plain read, keeping two thirds must not
    res = {}
    for keep_all in (True, False):
        cls = type("Keep", (_FakeTokenSparse,), {"keep_all": keep_all}); sp = SparsePrefill("tokensparse", "sdpa", "tiny", lm.n_layers, fns={"SparseCluster": cls}); lm.sparse = sp
        try: res[keep_all] = lm.window_read(ex)
        finally: sp.remove(); lm.sparse = None
    d_all = max(abs(base_w["scores"][y] - res[True]["scores"][y]) for y in base_w["scores"]); d_sub = max(abs(base_w["scores"][y] - res[False]["scores"][y]) for y in base_w["scores"])
    print(f"sparse hook, Token Sparse Attention in the model (sparse layers {sp.layers}): keep every token vs plain max |dscore| {d_all:.2e}, keep two thirds {d_sub:.2e} "
          f"(must NOT be 0), kept share {res[False]['ts_kept']:.2f} -> {'OK' if d_all < 1e-4 and d_sub > 1e-4 else 'FAIL'}")
    args.prefill_chunk = saved_pc


def check_sparse():
    """On the GPU, before any run with --sparse_prefill: the released kernels through the hook on one real prompt (the first question of the dataset,
    its context cut to 16k tokens) against the dense backend. XAttention keeping every block (threshold 1.0) must equal the dense read to bf16 noise,
    Token Sparse Attention dropping nothing (coverage 0) must equal it exactly; then the published settings are reported (logit change, top token, kept share)."""
    assert args.sparse_prefill != "none", "set --sparse_prefill"
    lm = LM(args.model); sp = lm.sparse; ex = load_examples()[0]
    ctx = lm.tok.encode(ex["context"], add_special_tokens=False)
    kept = fit_prefix(lm.tok, ctx, lambda tx: qa_prompt(lm.tok, tx, ex["question"], ex["options"], False), 16384)
    ids = lm.tok.encode(qa_prompt(lm.tok, lm.tok.decode(kept, skip_special_tokens=True), ex["question"], ex["options"], False), add_special_tokens=False)
    def run(on):
        sp.on, sp.n_sparse, sp.kept = on, 0, []
        try: return lm.forward_last([ids])[0].float()
        finally: sp.on = False
    dense = run(False)
    def report(name, lg, bound):
        d = float((lg - dense).abs().max()); same = int(lg.argmax()) == int(dense.argmax())
        line = f"  {name}: {sp.n_sparse} sparse attention calls, max |dlogit| {d:.3f}, same top token {same}"
        if sp.kept: line += f", kept {np.mean(sp.kept):.3f} of the tokens"
        if bound is not None: line += f" -> {'OK' if same and d <= bound else 'FAIL'}"
        print(line, flush=True)
    print(f"check_sparse: {args.sparse_prefill} on one prompt of {len(ids)} tokens, {args.attn}", flush=True)
    if sp.method == "xattention":
        saved = sp.thr; sp.thr = [1.0] * len(saved); report("threshold 1.0 (every block kept)", run(True), 0.5); sp.thr = saved
        report(f"threshold {args.xattn_threshold}, stride {args.xattn_stride}", run(True), None)
    else:
        saved = [c.coverage for c in sp.clusters]
        for c in sp.clusters: c.coverage = 0.0
        report("coverage 0 (nothing dropped)", run(True), 0.0)
        for c, cv in zip(sp.clusters, saved): c.coverage = cv
        report(f"coverage {args.ts_coverage}", run(True), None)

# ============================================================== main
def summary():
    """The combined table from the saved rows: accuracy all / in window / beyond window, costs, and the cache-reuse split for the masked arms."""
    tag = args.model.split("/")[-1]; W = args.window
    def acc(rows, key, metric="main"):
        sc = [r for r in rows if r.get(key) is not None]; return (100 * sum(row_score(r, key, metric) for r in sc) / len(sc)) if sc else float("nan"), len(sc)
    print(f"{args.dataset} {SPLIT_TAG}, {tag}, window {W}. Accuracy (token F1 for open answers, then SCBench's match): all / in window (n_tokens <= window) / beyond window.")
    print(f"{'arm':11s}{'readout':8s}{'all':>7s}{'n':>5s}{'in-win':>8s}{'n':>5s}{'beyond':>8s}{'n':>5s}")
    costs = []
    for arm in ARMS:
        path = os.path.join(args.out_dir, f"{args.dataset}_{SPLIT_TAG}_{arm}_{tag}.jsonl")
        if not os.path.exists(path): print(f"{arm:11s}no file {path}"); continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if not rows: print(f"{arm:11s}empty file {path}"); continue
        inw = [r for r in rows if r["n_tokens"] <= W]; bey = [r for r in rows if r["n_tokens"] > W]
        mono = arm in ("window", "window_reuse"); opn = isinstance(rows[0]["gold"], list)
        for key in [x for x in READOUTS if any(r.get(x) is not None for r in rows)]:
            name = "window" if mono else "one read" if arm.startswith("mask_all") else ("store" if key == "pred" and rows[0].get("select_mode") == "store" else READOUTS[key])
            for metric in (("main", "sc") if opn else ("main",)):
                (a, n), (b, nb), (c, nc) = acc(rows, key, metric), acc(inw, key, metric), acc(bey, key, metric)
                print(f"{arm:11s}{(name if metric == 'main' else ' ' + name[:3] + ' sc'):8s}{a:7.1f}{n:5d}{b:8.1f}{nb:5d}{c:8.1f}{nc:5d}")
            if args.dataset == "needle":                                                          # the heatmap: RULER's string match by length and depth
                Ls, Ds = sorted({r["length"] for r in rows}), sorted({r["depth"] for r in rows})
                print(f"{'':11s}{name}: string match (%) by length (rows) and depth in % (columns)")
                print(f"{'':11s}{'length':>7s} " + " ".join(f"{d:>4d}" for d in Ds))
                for L in Ls:
                    cells = [[row_score(r, key, "sc") for r in rows if r["length"] == L and r["depth"] == d and r.get(key) is not None] for d in Ds]
                    print(f"{'':11s}{L // 1024:>6d}k " + " ".join(f"{100 * np.mean(c):4.0f}" if c else "   -" for c in cells))
        skipped = sum(1 for r in rows if r.get("skipped")); rows = [r for r in rows if not r.get("skipped")] or rows                  # costs over the rows that ran
        first = [r for r in rows if not r.get("shared")]; shared = [r for r in rows if r.get("shared")]
        line = (f"{arm:11s}cost    content tok/q {np.mean([r.get('content_tokens', r.get('tokens', 0)) for r in rows]):8.0f}  s/q {np.mean([r['latency'] for r in rows]):6.1f}"
                f"  peak GB {np.max([r.get('peak_gb', 0) for r in rows]):5.1f}  truncated {sum(r.get('truncated', 0) for r in rows)}  skipped {skipped}")
        if shared:
            line += f"  | first-of-book s/q {np.mean([r['latency'] for r in first]):6.1f} (n={len(first)}), shared s/q {np.mean([r['latency'] for r in shared]):6.1f} (n={len(shared)}), encode s {np.mean([r.get('encode_s', 0) for r in first]):6.1f}"
            if any("copies_s" in r for r in rows): line += f", copies s {np.mean([r.get('copies_s', 0) for r in rows]):5.1f}"
            if any("ask_s" in r for r in rows): line += f", ask s {np.mean([r.get('ask_s', 0) for r in rows]):5.2f} (tokenize s {np.mean([r.get('tok_s', 0) for r in rows]):4.2f} excluded, fallbacks {sum(r.get('fallback', 0) for r in rows)})"
        if any("reread_s" in r for r in rows): line += f"  | reread s {np.mean([r.get('reread_s', 0) for r in rows]):5.2f}"
        dec = [r for r in rows if "decode_ms_per_tok" in r]
        if dec: line += f"  | decode ms/tok {np.mean([r['decode_ms_per_tok'] for r in dec]):6.1f} ({dec[0]['gen_tokens']} tokens, n={len(dec)})"
        if any("copy_batch" in r for r in rows):
            line += (f"  | copy_batch {sorted({r.get('copy_batch', 1) for r in rows})}, offloaded books {sum(r.get('offloaded', 0) for r in first)}"
                     f" (page-locked {sum(r.get('pinned', 0) for r in first)}), copies s on them {np.mean([r.get('copies_s', 0) for r in rows if r.get('offloaded')] or [0]):5.1f}")
        print(line)
        # the same costs on the in-window rows only, where every arm reads the whole document: the rows of the main table
        rows = [r for r in rows if r["n_tokens"] <= W]
        if not rows: continue
        first = [r for r in rows if not r.get("shared")]; shared = [r for r in rows if r.get("shared")]; dec = [r for r in rows if "decode_ms_per_tok" in r]
        line = f"{arm:11s}in-win  content tok/q {np.mean([r.get('content_tokens', r.get('tokens', 0)) for r in rows]):8.0f}  s/q {np.mean([r['latency'] for r in rows]):6.1f}  peak GB {np.max([r.get('peak_gb', 0) for r in rows]):5.1f}  (n={len(rows)})"
        if shared: line += f"  | first-of-book s/q {np.mean([r['latency'] for r in first]):6.1f} (n={len(first)}), shared s/q {np.mean([r['latency'] for r in shared]):6.1f} (n={len(shared)}), encode s {np.mean([r.get('encode_s', 0) for r in first]):6.1f}"
        if any("reread_s" in r for r in rows): line += f"  | reread s {np.mean([r.get('reread_s', 0) for r in rows]):5.2f}"
        if any("store_s" in r for r in rows): line += f"  | store s {np.mean([r.get('store_s', 0) for r in rows]):5.2f}"
        if dec: line += f"  | decode ms/tok {np.mean([r['decode_ms_per_tok'] for r in dec]):6.1f}"
        print(line)
        # with both select read-outs on a row, latency holds both: each read-out's time to the first token leaves the other out; an open answer
        # adds its generation for the time to the full answer
        both = [r for r in rows if "pred_store" in r]
        if both:
            print(f"{arm:11s}in-win  s/q to first token: select reread {np.mean([r['latency'] - r.get('store_s', 0) for r in both]):6.2f}, store {np.mean([r['latency'] - r.get('reread_s', 0) for r in both]):6.2f}"
                  + (f"  | to the full answer: reread {np.mean([r['latency'] - r.get('store_s', 0) + r.get('gen_s', 0) for r in both]):6.2f}, "
                     f"store {np.mean([r['latency'] - r.get('reread_s', 0) + r.get('gen_s_store', 0) for r in both]):6.2f}" if any("gen_s" in r for r in both) else "") + f"  (n={len(both)})")
        elif any("gen_s" in r for r in rows):
            print(f"{arm:11s}in-win  s/q to the full answer {np.mean([r['latency'] + r.get('gen_s', 0) for r in rows]):6.2f}, generated tokens/q {np.mean([r.get('gen_n', 0) for r in rows]):5.1f}  (n={len(rows)})")

def main():
    if args.check: checks(); return
    if args.summary: summary(); return
    if args.check_sparse: check_sparse(); return
    if args.sparse_prefill == "tokensparse" and "window_reuse" in ARMS:
        raise SystemExit("Token Sparse Attention scores tokens with the question, so its prefill cannot be shared across questions: run it on the window arm")
    if args.yarn and any(a not in ("window", "window_reuse") for a in ARMS):
        raise SystemExit("--yarn changes the rotary frequencies of every position, so it is for the monolithic arms: FCA reads with the model's native positions")
    if args.sparse_prefill != "none" and any(a not in ("window", "window_reuse") for a in ARMS):
        print(f"note: --sparse_prefill {args.sparse_prefill} applies to the window and window_reuse arms only; the other arms run on {args.attn}", flush=True)
    import torch
    examples = load_examples(); print(f"{args.dataset} {SPLIT_TAG}: {len(examples)} questions", flush=True)
    order = {}
    for ex in examples: ex["book"] = hashlib.md5(ex["context"].encode("utf-8")).hexdigest(); order.setdefault(ex["book"], len(order))
    examples.sort(key=lambda e: order[e["book"]])                                                # questions of one text are consecutive, so the masked arms reuse the cache
    print(f"{len(order)} distinct texts; questions per text: min {min(sum(1 for e in examples if e['book'] == b) for b in order)}, max {max(sum(1 for e in examples if e['book'] == b) for b in order)}", flush=True)
    lm = LM(args.model); tag = args.model.split("/")[-1]
    limit, src = host_ram_bytes(); print(f"host memory limit for offloaded chunk caches: {limit / 1e9:.0f} GB ({src}); caches above {args.gpu_cache_gb:.0f} GB leave the GPU", flush=True)
    # Periscope's own sanity check on the prompt format
    ys, ab = letters(4); top = lm.top20(lm.forward_last([lm.tok.encode(qa_prompt(lm.tok, "The sky is blue.", "What colour is the sky?", ["green", "blue", "red", "black"], True), add_special_tokens=False)]))[0]
    print("trivial probe log-odds vs E:", {k: round(v, 2) for k, v in log_odds(top, ys, ab).items()}, "(B must win)", flush=True)
    assert args.dataset == "conversation" or not (set(ARMS) & {"fca_append", "mono_conv"}), "fca_append and mono_conv read conversations: --dataset conversation"
    for arm in ARMS:
        lm.release()                                                                                # no cache of the previous arm survives into this one
        path = os.path.join(args.out_dir, f"{args.dataset}_{SPLIT_TAG}_{arm}_{tag}.jsonl"); done = {}
        if os.path.exists(path): done = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
        with open(path, "a", encoding="utf-8") as f:
            for n_i, ex in enumerate(examples):
                if ex["id"] in done: continue
                if lm.dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                t0 = time.time(); rec = {"window": lm.window_read, "window_reuse": lm.window_reuse, "periscope": lm.periscope, "mask_doc": lm.mask_doc, "mask_chunk": lm.mask_chunk, "mask_span": lm.mask_span, "mask_stair": lm.mask_stair, "mask_stair_alone": lm.mask_stair_alone,
                                                  "mask_stair_both": lm.mask_stair_both, "mask_all": lm.mask_all, "mask_all_anchor": lm.mask_all_anchor,
                                                  "fca_append": lm.fca_append, "mono_conv": lm.mono_conv}[arm](ex)
                if lm.dev == "cuda": torch.cuda.synchronize()
                rec.update(id=ex["id"], gold=ex["gold"], split=ex["split"], domain=ex.get("domain"), book=ex["book"], n_tokens=len(lm.tok.encode(ex["context"], add_special_tokens=False)),
                           latency=time.time() - t0 - sum(rec.get(x, 0.0) for x in ("decode_s", "tok_s", "gen_s", "gen_s_store")),   # to the first answer token
                           peak_gb=(torch.cuda.max_memory_allocated() / 1e9) if lm.dev == "cuda" else 0.0)
                for f_ in ("turn", "length", "depth", "conv", "target_turn", "age"):
                    if f_ in ex: rec[f_] = ex[f_]
                if args.yarn: rec["yarn"] = args.yarn
                for key in ("pred", "pred_store"):                                                # open answers: both metrics stored with the row
                    if isinstance(ex["gold"], list) and rec.get(key) is not None:
                        sfx = key[4:]; rec["f1" + sfx], rec["sc" + sfx] = qa_f1(rec[key], ex["gold"]), sc_match(rec[key], ex["gold"])
                f.write(json.dumps(rec, default=float) + "\n"); f.flush(); done[ex["id"]] = rec
                if lm.dev == "cuda": torch.cuda.empty_cache()
                if (n_i + 1) % 10 == 0: print(f"    {arm} {n_i + 1}/{len(examples)} ({rec['latency']:.0f}s last, {rec['n_tokens']} tokens{', shared cache' if rec.get('shared') else ''})", flush=True)
        rows = list(done.values()); mono = arm in ("window", "window_reuse")
        for key in [x for x in READOUTS if any(r.get(x) is not None for r in rows)]:
            name = "window" if mono else "one read" if arm.startswith("mask_all") else ("store" if key == "pred" and rows[0].get("select_mode") == "store" else READOUTS[key])
            sc = [r for r in rows if r.get(key) is not None]; acc = 100 * sum(row_score(r, key) for r in sc) / len(sc)
            line = f"{arm:11s}{name:7s} all {acc:5.1f} (n={len(sc)})"
            if isinstance(sc[0]["gold"], list): line += f"  token F1; SCBench match {100 * np.mean([row_score(r, key, 'sc') for r in sc]):5.1f}"
            for sp in ("short", "medium", "long", "enmc"):
                sub = [r for r in sc if r["split"] == sp]
                if sub: line += f"  {sp} {100 * sum(row_score(r, key) for r in sub) / len(sub):5.1f}"
            print(line, flush=True)
        print(f"{arm:11s}cost   content tokens/q {np.mean([r.get('content_tokens', r.get('tokens', 0)) for r in rows]):9.0f}  all tokens/q {np.mean([r.get('tokens', 0) for r in rows]):9.0f}"
              f"  passes/q {np.mean([r.get('passes', 1) for r in rows]):6.1f}  s/q {np.mean([r['latency'] for r in rows]):6.1f}  peak GB {np.max([r['peak_gb'] for r in rows]):5.1f}", flush=True)
        if args.dataset == "conversation": conv_lines(arm, rows)

def conv_lines(arm, rows):
    """The conversation test of one arm: string match (%) and seconds per turn by the turn of the conversation, and match by the age of the asked needle."""
    R = [r for r in rows if r.get("sc") is not None and r.get("turn") is not None]
    if not R: return
    T = max(r["turn"] for r in R) + 1; cut = sorted({0, T // 4, T // 2, 3 * T // 4, T}); bins = [(a, b) for a, b in zip(cut, cut[1:]) if b > a]
    m = lambda S, f: f"{np.mean([f(r) for r in S]):6.1f}" if S else "   n/a"
    print(f"{arm:11s}conv   string match {100 * np.mean([r['sc'] for r in R]):5.1f} (n={len(R)}); by turn " +
          ", ".join(f"{a + 1}-{b}: {m([r for r in R if a <= r['turn'] < b], lambda r: 100 * r['sc'])}" for a, b in bins), flush=True)
    print(f"{arm:11s}conv   seconds per turn by turn " + ", ".join(f"{a + 1}-{b}: {m([r for r in R if a <= r['turn'] < b], lambda r: r['latency'])}" for a, b in bins) +
          ("; of which encoding " + ", ".join(f"{m([r for r in R if a <= r['turn'] < b and 'encode_s' in r], lambda r: r['encode_s'])}" for a, b in bins) if any('encode_s' in r for r in R) else ""), flush=True)
    ages = [(0, 1), (1, 4), (4, 10), (10, 10 ** 6)]
    print(f"{arm:11s}conv   string match by age of the needle (turns ago) " + ", ".join(f"{a}{'' if b == a + 1 else '+' if b > 100 else '-' + str(b - 1)}: {m([r for r in R if a <= r.get('age', 0) < b], lambda r: 100 * r['sc'])}" for a, b in ages), flush=True)
    if any("needle_in_context" in r for r in R): print(f"{arm:11s}conv   needle still in the kept turns {100 * np.mean([r['needle_in_context'] for r in R]):5.1f}%, turns truncated {100 * np.mean([r['truncated'] for r in R]):5.1f}%", flush=True)
    N = [r for r in R if r.get("needle_rank")]
    if N: print(f"{arm:11s}conv   map: median rank of the needle block {np.median([r['needle_rank'] for r in N]):.0f} of {np.median([r['map_cells'] for r in N]):.0f} cells, selected {100 * np.mean([r['needle_selected'] for r in N]):5.1f}% (n={len(N)}); store at the last turn {max(r['doc_tokens'] for r in R)} tokens", flush=True)

if __name__ == "__main__":
    main()
