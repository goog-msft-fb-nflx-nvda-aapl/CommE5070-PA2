#!/usr/bin/env python
"""CommE5070 HW2 inference entry point. Generates MIDI + WAV listening samples from the released checkpoints.

  python infer.py uncond   --out_dir OUT                              # Basic task a: 20 x 32-bar pieces
  python infer.py continue --prompt_dir POP909/prompts/basic-b --out_dir OUT   # Basic task b: 3 prompts x 3 configs
  python infer.py bonus    --prompt_dir POP909/prompts/bonus   --out_dir OUT   # Bonus: MELODY -> PIANO for the 10 prompts

Every mode calls the same scripts used for the experiments (src/generate.py, src/gen_bonus.py); --seed fixes sampling.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CK = lambda *p: os.path.join(HERE, "checkpoints", *p)
VOC = lambda *p: os.path.join(HERE, "vocab", *p)
SRC = lambda *p: os.path.join(HERE, "src", *p)

UNCOND_CKPT = CK("uncond_final.pt")        # GPT-2 8L x 512, pos_per_beat 16, dropout 0.3, trained on all 896 training songs (run F1)
UNCOND_VOCAB = VOC("tok2idx_p16.json")
POS_PER_BEAT = "16"
# Task a / b sampling configs (see README / report): name -> generate.py flags
CONFIGS = {
    "cfg1_T1.0_pure": ["--temperature", "1.0"],
    "cfg2_T1.0_barbias3": ["--temperature", "1.0", "--bar_bias", "3"],
    "cfg3_T0.9_top_p0.9": ["--temperature", "0.9", "--top_p", "0.9"],
}


def run(cmd, env_extra=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env_extra or {})})


def render(midi_dir, wav_dir=None):
    run([PY, SRC("render_wav.py"), "--midi_dir", midi_dir] + (["--out_dir", wav_dir] if wav_dir else []))


def uncond(a):
    out = os.path.join(a.out_dir, "taskA_" + a.config)
    run([PY, SRC("generate.py"), "uncond", "--ckpt", UNCOND_CKPT, "--tok2idx", UNCOND_VOCAB, "--pos_per_beat", POS_PER_BEAT,
         "--out_dir", out, "--n", str(a.n), "--n_bars", "32", "--seed", str(a.seed), "--tag", "g"] + CONFIGS[a.config])
    render(out)
    print("done:", out)


def cont(a):
    tmp = os.path.join(a.out_dir, "_prompt_tokens")
    os.makedirs(tmp, exist_ok=True)
    run([PY, SRC("tokenize_hw2.py"), "--dataset_path", a.prompt_dir, "--tok2idx_output", os.path.join(tmp, "tok2idx.json"),
         "--sequences_output", os.path.join(tmp, "prompts.json"), "--pos_per_beat", POS_PER_BEAT])
    for name, flags in CONFIGS.items():
        out = os.path.join(a.out_dir, "taskB_" + name)
        run([PY, SRC("generate.py"), "cont", "--ckpt", UNCOND_CKPT, "--tok2idx", UNCOND_VOCAB, "--pos_per_beat", POS_PER_BEAT,
             "--prompt_tokens", os.path.join(tmp, "prompts.json"), "--prompt_bars", "8", "--n_bars", "32", "--n", "1",
             "--out_dir", out, "--seed", str(a.seed), "--tag", name.split("_")[0]] + flags)
        render(out)
    print("done:", a.out_dir)


def bonus(a):
    cfg = json.load(open(VOC("bonus_final.json")))[a.mode]
    tmp = os.path.join(a.out_dir, "_bonus_prompts")
    os.makedirs(tmp, exist_ok=True)
    run([PY, SRC("bonus_data.py"), "--prompts_only", "1", "--prompt_dir", a.prompt_dir, "--base_tok2idx", VOC("tok2idx_p4.json"),
         "--out_dir", tmp, "--pos_per_beat", "4"])  # the bonus models use the p=4 grid
    out = os.path.join(a.out_dir, "bonus_" + a.mode)
    cmd = [PY, SRC("gen_bonus.py"), "--ckpt", CK(cfg["ckpt"]), "--tok2idx", os.path.join(tmp, "tok2idx_bonus.json"),
           "--prompts", os.path.join(tmp, "bonus_prompts.json"), "--prompt_dir", a.prompt_dir, "--out_dir", out,
           "--seed", str(a.seed)] + cfg["flags"]
    if cfg.get("stage1"):
        cmd += ["--chord_from", CK(cfg["stage1"])]
    run(cmd, {"PYTHONPATH": SRC()})
    render(out)
    print("done:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    u = sub.add_parser("uncond"); u.add_argument("--out_dir", required=True); u.add_argument("--n", type=int, default=20)
    u.add_argument("--seed", type=int, default=0); u.add_argument("--config", default="cfg2_T1.0_barbias3", choices=list(CONFIGS))
    c = sub.add_parser("continue"); c.add_argument("--prompt_dir", required=True); c.add_argument("--out_dir", required=True)
    c.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bonus"); b.add_argument("--prompt_dir", required=True); b.add_argument("--out_dir", required=True)
    b.add_argument("--seed", type=int, default=0); b.add_argument("--mode", default="melody_only", choices=["melody_only", "chord_given"])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    {"uncond": uncond, "continue": cont, "bonus": bonus}[a.task](a)
