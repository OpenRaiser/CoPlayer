#!/usr/bin/env python3
"""Token-level check: LlamaFactory `qwen3_8` template (enable_thinking=False) vs the model's own chat template + processor.

For each sample (one of each kind by default) it builds the prompt input_ids two ways:
  LF : mm_plugin.process_messages (expands <image>) -> template.encode_oneturn -> prompt_ids   (what SFT trains on)
  HF : processor.apply_chat_template(openai-style messages, add_generation_prompt, enable_thinking=False)
       -> processor(text, images) -> input_ids                                              (what vLLM / evaluation feeds)
and reports whether they are identical; on mismatch prints the first differing region decoded.
CPU only. usage: check_template.py --model Qwen/Qwen3.8-27B [--data $COPLAY_DATA/training/sft_dev.json] [--n 6]
"""
import argparse
import json
import os
import sys

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(os.environ.get("COPLAY_DATA", "data/CoPlayBench"), "training", "sft_dev.json"))
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    margs = ModelArguments(model_name_or_path=a.model, image_max_pixels=1048576, image_min_pixels=1024, trust_remote_code=True)
    tm = load_tokenizer(margs)
    tok, proc = tm["tokenizer"], tm["processor"]
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_8", enable_thinking=False))

    root = os.path.dirname(os.path.abspath(a.data))
    data = json.load(open(a.data))
    pick, seen = [], set()
    for s in data:                                    # one of each kind first
        if s["kind"] not in seen:
            seen.add(s["kind"])
            pick.append(s)
    ok_all = True
    for s in pick[:a.n]:
        system, user, answer = (m["content"] for m in s["messages"])
        paths = [os.path.join(root, p) for p in s["images"]]
        images = [Image.open(p).convert("RGB") for p in paths]

        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
        msgs = template.mm_plugin.process_messages(msgs, paths, [], [], proc)
        prompt_ids, resp_ids = template.encode_oneturn(tok, msgs, system)

        parts = []
        chunks = user.split("<image>")
        for i, c in enumerate(chunks):
            if c:
                parts.append({"type": "text", "text": c})
            if i < len(chunks) - 1:
                parts.append({"type": "image"})
        oai = [{"role": "system", "content": system}, {"role": "user", "content": parts}]
        text = proc.apply_chat_template(oai, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        hf_ids = proc(text=[text], images=images, return_tensors="pt")["input_ids"][0].tolist()

        same = prompt_ids == hf_ids
        ok_all &= same
        print("%-22s %-28s LF %5d tok  HF %5d tok  -> %s" % (s["kind"], s["id"], len(prompt_ids), len(hf_ids),
                                                            "IDENTICAL" if same else "DIFFERENT"))
        print("    response:", repr(tok.decode(resp_ids))[:160])
        if not same:
            i = next((k for k, (x, y) in enumerate(zip(prompt_ids, hf_ids)) if x != y), min(len(prompt_ids), len(hf_ids)))
            print("    first diff at token", i)
            print("    LF:", repr(tok.decode(prompt_ids[max(0, i - 20):i + 40])))
            print("    HF:", repr(tok.decode(hf_ids[max(0, i - 20):i + 40])))
    print("ALL IDENTICAL" if ok_all else "MISMATCH FOUND")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
