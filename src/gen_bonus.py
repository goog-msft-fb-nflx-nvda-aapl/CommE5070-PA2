"""Generate PIANO accompaniment for the bonus MELODY prompts, write MELODY+PIANO MIDI on the prompt's own
beat grid (eval_bonus compares seconds against beat_midi.txt), score with the unmodified eval_bonus.py.

arch A (decoder-only): segment-injection loop following the training schedule: melody bars are fed, a <P> tag is
  fed, the model generates piano events until it emits a tag (<M>/<P>/<EOS>); that tag is replaced by the schedule's own.
arch B (T5): encoder = whole melody, decoder generates all piano bars (stops at the (n+1)-th Bar or EOS).
"""
import argparse, copy, hashlib, json, os, re, subprocess, sys
import numpy as np
import torch
import pretty_midi
from transformers import (GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration, StoppingCriteria,
                          StoppingCriteriaList, LogitsProcessorList)
import bonus_lib as bl
from generate import NoSpecial, TypedSampler, BarStop
from transformers import LogitsProcessor


class LastIn(StoppingCriteria):
    def __init__(self, ids):
        self.ids = torch.tensor(ids)
    def __call__(self, input_ids, scores, **kw):
        return torch.isin(input_ids[:, -1].cpu(), self.ids).to(input_ids.device)


def load(ckpt, dev, tok2idx=None):
    ck = torch.load(ckpt, map_location="cpu")
    if ck["arch"] == "A":
        if ck["args"].get("chord_emb"):
            vv = bl.Vocab(json.load(open(tok2idx)))
            m = bl.ChordEmbGPT2._from_config(GPT2Config(**ck["config"]), attn_implementation="sdpa"); m.setup_chord(vv.chord_ids + [vv.chord_N])
        else:
            m = GPT2LMHeadModel._from_config(GPT2Config(**ck["config"]), attn_implementation="sdpa")
    else:
        m = T5ForConditionalGeneration(T5Config(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck


def grid_time(beat_times, beat_info, P):
    idx = {}
    for j, (b, k) in enumerate(beat_info):
        idx.setdefault((b, k), j)
    bt = np.array(beat_times); nb = len(bt)
    def at(p):
        if p <= 0: return bt[0] + p * (bt[1] - bt[0])
        if p >= nb - 1: return bt[-1] + (p - (nb - 1)) * (bt[-1] - bt[-2])
        return float(np.interp(p, np.arange(nb), bt))
    def note(bar, pos, dur):
        k, fr = divmod(pos, P)
        j = idx.get((bar, k))
        if j is None:
            return None
        return at(j + fr / P), at(j + fr / P + dur / P)
    return note


def parse_piano(bar_tokens, note_fn, P):
    """bar_tokens: list (per bar) of token-string lists -> pretty_midi notes; returns (notes, n_malformed)"""
    notes, bad = [], 0
    for bar, toks in enumerate(bar_tokens):
        i = 0
        while i < len(toks):
            g = toks[i:i + 4]
            if len(g) == 4 and g[0].startswith("Pos_") and g[1].startswith("Pitch_") and g[2].startswith("Vel_") and g[3].startswith("Dur_"):
                t = note_fn(bar, int(g[0][4:]), int(g[3][4:]))
                if t is not None and t[1] > t[0]:
                    notes.append(pretty_midi.Note(velocity=int(min(127, max(1, int(g[2][4:])))), pitch=int(g[1][6:]), start=t[0], end=t[1]))
                i += 4
            else:
                bad += 1; i += 1
    return notes, bad


class ChordBias(LogitsProcessor):
    """inference-time harmony guidance: adds `bias` to the logit of every Pitch_x whose pitch class is in the triad of the current bar's
    conditioning chord (given or stage-1 predicted), applied when the next token is a pitch (previous token is a Pos_x). Set .cur per bar."""
    def __init__(self, v, bias):
        self.bias, self.cur = bias, -1
        self.pos = torch.tensor([i for t, i in v.t2i.items() if t.startswith("Pos_")])
        self.pitch = {int(t.split("_")[1]): i for t, i in v.t2i.items() if t.startswith("Pitch_")}
        self.v = v
    def __call__(self, input_ids, scores):
        if self.cur < 0:
            return scores
        root, minor = self.cur % 12, self.cur >= 12
        pcs = {root, (root + (3 if minor else 4)) % 12, (root + 7) % 12}
        m = torch.isin(input_ids[:, -1].cpu(), self.pos).to(scores.device)
        add = torch.zeros(scores.shape[-1], device=scores.device)
        for p, i in self.pitch.items():
            if p % 12 in pcs:
                add[i] = self.bias
        return scores + m[:, None].float() * add[None, :]


def rerank_score(per_bar, bars, note_fn, P):
    """fraction of bars (with a conditioning chord label) whose generated chord, estimated by eval_bonus.estimate_chord, equals it"""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("eval_bonus", os.path.join(here, "..", "scripts", "eval_bonus.py"))
    eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
    notes, _ = parse_piano(per_bar, note_fn, P)
    arr = eb.note_arrays(notes)
    starts = [note_fn(b, 0, 0)[0] for b in range(len(bars))]
    hits = n = 0
    for b in range(len(bars)):
        if bars[b]["c"] < 0:
            continue
        t0 = starts[b]; t1 = starts[b + 1] if b + 1 < len(bars) else t0 + (t0 - starts[b - 1])
        hits += int(eb.estimate_chord(arr, t0, t1) == bars[b]["c"]); n += 1
    return hits / max(n, 1)


def prefix_for(bars, gen, j, i, L, chord, v):
    """context (training-crop layout starting at bar j) right before piano bar i is generated"""
    ids = [v.BOS]
    for kind, k in bl.schedule(len(bars) - j, L):
        k += j
        if kind == "M":
            ids += bl.m_segment(bars[k], v, chord)
        else:
            ids.append(v.P)
            if chord in (2, 3):
                ids.append(v.chord_tok(bars[k]["c"]))
            if k == i:
                return ids
            ids += gen[k]
    raise AssertionError


@torch.no_grad()
def gen_A(model, v, bars, L, chord, ctx, gen_kw, dev, max_bar_tokens=500, typed=None, cbias=None):
    gen = [[] for _ in bars]; trunc = 0; slid = 0; pred = [None] * len(bars)
    chord_set = set(v.chord_ids + [v.chord_N])
    lp = LogitsProcessorList([NoSpecial([v.PAD, v.BOS])])
    if typed is not None:
        lp.append(typed)
    stop = StoppingCriteriaList([LastIn([v.M, v.P, v.EOS])])
    if cbias is not None:
        lp.append(cbias)
    for i in range(len(bars)):
        if cbias is not None:
            cbias.cur = bars[i]["c"]
        j = 0
        while True:  # sliding window: re-base the crop at bar j until the context fits
            ids = prefix_for(bars, gen, j, i, L, chord, v)
            max_new = min(max_bar_tokens, ctx - len(ids) - 1)
            if max_new >= 150 or j == i:
                break
            j += 1
        slid += j > 0
        x = torch.tensor([ids], device=dev)
        g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, max_new_tokens=max(max_new, 1),
                           logits_processor=lp, stopping_criteria=stop, pad_token_id=v.PAD, use_cache=not hasattr(model, 'chord_emb'), **gen_kw)
        new = g[0, len(ids):].tolist()
        if new and new[-1] in (v.M, v.P, v.EOS):
            new = new[:-1]
        else:
            trunc += 1
        gen[i] = new  # (mode 4: gen[i] starts with the model's own chord token, kept in the context)
        if chord == 4 and new and new[0] in chord_set:
            pred[i] = new[0]
    out = [[v.i2t[t] for t in g_ if t not in chord_set] for g_ in gen]
    return out, trunc, pred


@torch.no_grad()
def gen_B(model, v, bars, chord, ctx, gen_kw, dev, typed=None, cbias=None):
    enc, _ = bl.build_B([dict(b, p=np.zeros(0, dtype=np.int64)) for b in bars], v, chord)
    x = torch.tensor([enc], device=dev)
    att = torch.ones_like(x)
    lp = LogitsProcessorList([NoSpecial([v.PAD, v.BOS])])
    if typed is not None:
        lp.append(typed)
    if cbias is not None:
        lp.append(cbias)
    if chord in (2, 3):
        # chord token forced in the decoder after every Bar: generate bar by bar
        dec = [v.BOS, v.BAR, v.chord_tok(bars[0]["c"])]
        per_bar = []
        stop = StoppingCriteriaList([LastIn([v.BAR, v.EOS])])
        trunc = 0
        for i in range(len(bars)):
            if cbias is not None:
                cbias.cur = bars[i]["c"]
            room = ctx - len(dec) - 1
            if room <= 0:  # decoder context exhausted: the remaining bars stay empty (counted)
                trunc += len(bars) - i
                per_bar += [[] for _ in range(len(bars) - i)]
                break
            d = torch.tensor([dec], device=dev)
            g = model.generate(input_ids=x, attention_mask=att, decoder_input_ids=d, do_sample=True,
                               max_new_tokens=min(500, room), logits_processor=lp, pad_token_id=v.PAD,
                               eos_token_id=v.EOS, stopping_criteria=stop, **gen_kw)
            new = g[0, len(dec):].tolist()
            ended = bool(new) and new[-1] in (v.BAR, v.EOS)
            ev = new[:-1] if ended else new
            per_bar.append([v.i2t[t] for t in ev if t not in (v.PAD, v.BOS, v.EOS, v.BAR) and t not in v.chord_ids + [v.chord_N]])
            dec += ev
            if i + 1 < len(bars):
                dec += [v.BAR, v.chord_tok(bars[i + 1]["c"])]
        return per_bar, trunc, [None] * len(bars)
    g = model.generate(input_ids=x, attention_mask=att, decoder_start_token_id=v.BOS, do_sample=True,
                       max_new_tokens=ctx - 1, logits_processor=lp, pad_token_id=v.PAD, eos_token_id=v.EOS,
                       stopping_criteria=StoppingCriteriaList([BarStop(v.BAR, len(bars) + 1, torch.zeros(1, dtype=torch.long, device=dev))]),
                       **gen_kw)
    toks = [t for t in g[0].tolist()[1:] if t not in (v.PAD,)]
    per_bar, pred = [], []
    chord_set = set(v.chord_ids + [v.chord_N])
    for t in toks:
        if t == v.BAR:
            if len(per_bar) == len(bars):
                break
            per_bar.append([]); pred.append(None)
        elif t == v.EOS:
            break
        elif per_bar:
            if t in chord_set:  # mode 4: the model's own chord token (kept out of the notes)
                if pred[-1] is None:
                    pred[-1] = t
            else:
                per_bar[-1].append(v.i2t[t])
    per_bar += [[] for _ in range(len(bars) - len(per_bar))]; pred += [None] * (len(bars) - len(pred))
    return per_bar, 0, pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok2idx", required=True)
    ap.add_argument("--prompts", required=True, help="bonus_prompts.json")
    ap.add_argument("--prompt_dir", required=True, help="POP909/prompts/bonus (melody.mid, beat/chord files)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pos_per_beat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chord", type=int, default=None, help="default: what the checkpoint was trained with")
    ap.add_argument("--lookahead", type=int, default=None, help="A: default = training value")
    ap.add_argument("--bar_bias", type=float, default=0.0, help="B only: added to the Bar logit")
    ap.add_argument("--t_struct", type=float, default=None, help="per-token-type temperature (default: --temperature)")
    ap.add_argument("--t_pitch", type=float, default=None)
    ap.add_argument("--t_vel", type=float, default=None)
    ap.add_argument("--t_dur", type=float, default=None)
    ap.add_argument("--results", default=None, help="append one scored json row here")
    ap.add_argument("--tag", default="")
    ap.add_argument("--chord_from", default=None, help="stage-1 chord predictor ckpt (arch C): chords for the stage-2 model come from the melody only")
    ap.add_argument("--chord_from_sample", type=int, default=0, help="1: sample stage-1 chords (T=1) instead of greedy")
    ap.add_argument("--best_of", type=int, default=1, help="N candidates per prompt, keep the one whose chords (24-triad estimator of eval_bonus) agree most with the conditioning chords of the bars (given or stage-1 predicted); needs chord_input != 0")
    ap.add_argument("--chord_bias", type=float, default=0.0, help="add this to the logit of pitches in the current bar's conditioning triad (needs chord_input 2 or A)")
    ap.add_argument("--chord_shuffle", type=int, default=0, help="control: permute the chord labels across the bars of each prompt")
    a = ap.parse_args()

    dev = "cuda"
    v = bl.Vocab(json.load(open(a.tok2idx)))
    model, ck = load(a.ckpt, dev, a.tok2idx)
    arch, targs = ck["arch"], ck["args"]
    chord = targs["chord"] if a.chord is None else a.chord
    L = targs.get("lookahead", 1) if a.lookahead is None else a.lookahead
    ctx = targs["ctx"]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen_kw = dict(temperature=a.temperature, top_k=a.top_k if a.top_k > 0 else None, top_p=a.top_p if a.top_p < 1 else None)
    typed = None
    tt = [a.t_struct, a.t_pitch, a.t_vel, a.t_dur]
    if a.bar_bias != 0.0 or any(x is not None for x in tt):
        temps = [a.temperature if x is None else x for x in tt]
        typed = TypedSampler({**v.t2i}, *temps, a.bar_bias, a.top_k, a.top_p, dev)
        gen_kw = dict(temperature=1.0, top_k=None, top_p=None)  # TypedSampler does all the sampling transforms
    cbias = ChordBias(v, a.chord_bias) if a.chord_bias else None
    stage1 = load(a.chord_from, dev, a.tok2idx)[0] if a.chord_from else None
    os.makedirs(a.out_dir, exist_ok=True)
    records, tot_bad, tot_trunc, chord_hits, s1_hits, sel_scores = [], 0, 0, [], [], []
    for pr in json.load(open(a.prompts)):
        bars = bl.to_bars(pr, v)
        if stage1 is not None:  # chords predicted from the melody only (no chord file used)
            enc, _ = bl.build_B([dict(b, p=np.zeros(0, dtype=np.int64)) for b in bars], v, 0)
            xe = torch.tensor([enc], device=dev)
            g = stage1.generate(input_ids=xe, attention_mask=torch.ones_like(xe), decoder_start_token_id=v.BOS, do_sample=bool(a.chord_from_sample),
                                max_new_tokens=len(bars) + 1, pad_token_id=v.PAD, eos_token_id=v.EOS)
            toks = [t for t in g[0].tolist()[1:] if t in v.chord_ids or t == v.chord_N][:len(bars)]
            toks += [v.chord_N] * (len(bars) - len(toks))
            for b, t in zip(bars, toks):
                b["c"] = -1 if t == v.chord_N else v.chord_ids.index(t)
            hit = [int(b["c"] == pb["c"]) for b, pb in zip(bars, bl.to_bars(pr, v)) if pb["c"] >= 0]
            s1_hits += hit
        if a.chord_shuffle:
            perm = np.random.RandomState(a.seed + 1000).permutation(len(bars))
            labs = [b["c"] for b in bars]
            for b, k in zip(bars, perm):
                b["c"] = labs[k]
        note_fn = grid_time(pr["beat_times"], pr["beat_info"], a.pos_per_beat)
        best = None
        for _cand in range(max(1, a.best_of)):
            pb_, tr_, pd_ = (gen_A(model, v, bars, L, chord, ctx, gen_kw, dev, typed=typed, cbias=cbias) if arch == "A"
                             else gen_B(model, v, bars, chord, ctx, gen_kw, dev, typed, cbias))
            if a.best_of <= 1:
                best = (0, pb_, tr_, pd_); break
            sc = rerank_score(pb_, bars, note_fn, a.pos_per_beat)
            if best is None or sc > best[0]:
                best = (sc, pb_, tr_, pd_)
        per_bar, trunc, pred = best[1], best[2], best[3]
        if a.best_of > 1:
            sel_scores.append(best[0])
        if chord == 4:  # accuracy of the model's own chord prediction vs the annotation (bars with a usable label)
            for b, pc in zip(bars, pred):
                if b["c"] >= 0:
                    chord_hits.append(int(pc == v.chord_ids[b["c"]]))
        notes, bad = parse_piano(per_bar, note_fn, a.pos_per_beat)
        tot_bad += bad; tot_trunc += trunc
        mel = pretty_midi.PrettyMIDI(os.path.join(a.prompt_dir, pr["song"], "melody.mid"))
        out = pretty_midi.PrettyMIDI(initial_tempo=float(mel.get_tempo_changes()[1][0]))
        mi = copy.deepcopy(next((i for i in mel.instruments if i.name.strip().upper() == "MELODY"), mel.instruments[0]))
        mi.name = "MELODY"
        pi = pretty_midi.Instrument(program=0, name="PIANO"); pi.notes = notes
        out.instruments += [mi, pi]
        out.write(os.path.join(a.out_dir, pr["song"] + ".mid"))
        records.append({"song": pr["song"], "piano_tokens": per_bar, "n_notes": len(notes)})
    json.dump(records, open(os.path.join(a.out_dir, "piano_tokens.json"), "w"))
    json.dump(vars(a), open(os.path.join(a.out_dir, "gen_config.json"), "w"))
    print(f"generated {len(records)} pieces; arch {arch} chord {chord} L {L}; malformed {tot_bad}; truncated bars {tot_trunc}; notes/bar {np.mean([r['n_notes'] for r in records]) / 16:.1f}")
    if a.results:
        here = os.path.dirname(os.path.abspath(__file__))
        ev = subprocess.run([sys.executable, os.path.join(here, "..", "scripts", "eval_bonus.py"), "--prompt_dir", a.prompt_dir,
                             "--gen_midi_dir", a.out_dir, "--output_csv", os.path.join(a.out_dir, "metrics.csv")], capture_output=True, text=True)
        print(ev.stdout[-700:])
        m = {k: float(x) for k, x in re.findall(r"^Average ([^:\n]+): ([\d.]+)", ev.stdout, re.M)}
        sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
        row = {"ckpt": a.ckpt, "ckpt_sha": sha(a.ckpt), "arch": arch, "chord_input": chord, "lookahead": L if arch == "A" else None,
               "temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p, "bar_bias": a.bar_bias, "t_struct": a.t_struct, "t_pitch": a.t_pitch, "t_vel": a.t_vel, "t_dur": a.t_dur, "seed": a.seed, "tag": a.tag, "chord_shuffle": a.chord_shuffle,
               "pred_chord_acc": (float(np.mean(chord_hits)) if chord_hits else (float(np.mean(s1_hits)) if s1_hits else None)),
               "chord_from": a.chord_from, "best_of": a.best_of, "chord_bias": a.chord_bias, "selected_conditioning_chord_agreement": (float(np.mean(sel_scores)) if sel_scores else None),
               "metrics": m, "malformed": tot_bad, "truncated_bars": tot_trunc, "notes_per_bar": float(np.mean([r["n_notes"] for r in records]) / 16),
               "code_sha256": {f: sha(os.path.join(here, f)) for f in ["gen_bonus.py", "bonus_lib.py", "generate.py"]},
               "train_args": targs}
        open(a.results, "a").write(json.dumps(row) + "\n")
