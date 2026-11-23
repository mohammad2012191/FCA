#!/usr/bin/env python3
"""
demo.py: Factorized Context Attention on your own text.

The text is encoded once into the store. Every question after the first reuses that store, so only the first answer pays the encoding.

  # one multiple-choice question
  python demo.py --text book.txt --question "Who finds the letter?" --options "Anna" "Boris" "Clara" "David"

  # one open question
  python demo.py --text book.txt --question "Where does the story end?"

  # a conversation: type questions, an empty line ends it
  python demo.py --text book.txt

Other models: --model meta-llama/Llama-3.1-8B-Instruct --window 131072. Any model of fca_text.py works (Qwen3, Llama, Mistral).
--tiny runs a random two-layer model on the CPU. It answers nonsense and only shows that the code path runs.
"""
import argparse, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--text", required=True, help="a plain text file, of any length")
ap.add_argument("--question", default=None, help="one question; without it the demo opens a conversation")
ap.add_argument("--options", nargs="*", default=None, help="answer options of a multiple-choice question")
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--window", type=int, default=262144, help="the context length of the model")
ap.add_argument("--attn", default="flash_attention_2", help="attention backend: flash_attention_2 or sdpa")
ap.add_argument("--max_new", type=int, default=100, help="tokens generated for an open answer")
ap.add_argument("--tiny", action="store_true", help="random tiny model on the CPU, to test the code path")
a = ap.parse_args()

# fca_text.py reads its settings from the command line when it is imported
sys.argv = ["fca_text.py", "--model", a.model, "--window", str(a.window), "--attn", "sdpa" if a.tiny else a.attn,
            "--max_new", str(a.max_new), "--out_dir", "runs_demo"]
import fca_text as fca

lm = fca.LM(a.model, tiny=a.tiny)
text = open(a.text, encoding="utf-8").read()
history = []                                                        # earlier turns of the conversation, carried in the request


def ask(question, options=None):
    """One request over the store of the text. The first call encodes the text, later calls reuse the store."""
    ex = {"id": "demo", "context": text, "question": question, "options": options, "gold": "A" if options else [""],
          "split": "demo", "history": list(history)}
    t0 = time.time(); rec = lm.mask_stair(ex); seconds = time.time() - t0
    if rec.get("skipped"): raise SystemExit("the store does not fit this machine: " + rec["skipped"])
    k = rec["k"]; blocks = rec.get("selected") or []
    print(f"  [{rec['n_chunks']} blocks on a {k} x {k} grid, {2 * k} span sequences, "
          f"{'store reused' if rec['shared'] else 'text encoded in %.1f s' % rec['encode_s']}, {seconds:.1f} s in all, "
          f"evidence in blocks {sorted(blocks)[:8]}]")
    return rec


if a.question and a.options:                                        # multiple choice: the direct and the select read-out
    rec = ask(a.question, a.options); letters = "ABCDEFGH"
    for name in ("direct", "pred"):
        y = rec[name]; print(f"{'direct' if name == 'direct' else 'select'} read-out: {y}. {a.options[letters.index(y)]}")
elif a.question:                                                    # one open question
    print(ask(a.question)["pred"])
else:                                                               # a conversation over one store
    print("Ask about the text. An empty line ends the conversation.")
    while True:
        q = input("> ").strip()
        if not q: break
        answer = (ask(q)["pred"] or "").strip(); print(answer)
        history.append({"question": q, "options": None, "answer": answer})
