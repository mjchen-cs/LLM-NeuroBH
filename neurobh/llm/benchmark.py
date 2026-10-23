"""Intent benchmark: natural-language service intents with ground-truth specs.

Each scenario has bilingual (zh/en) templates with numeric slots.  The ground
truth (priorities, network profile, DSL constraints) is derived from the same
slots, so labels are exact.  Category 'conflict' contains intents whose
explicit requirements cannot be satisfied together (the verifier must detect
them and the repair loop must produce a feasible relaxation).

Optionally, --paraphrase rewrites every intent with an LLM (labels unchanged)
to increase linguistic diversity.  Please also add 20-30 intents written by
real people (colleagues / operators) to data/intents_human.jsonl.
"""
import argparse
import json
import os
import random

SCEN = []


def S(name, profile, prio, zh, en, slots, cons, conflict=False):
    SCEN.append(dict(name=name, profile=profile, prio=dict(zip(
        ["quality", "smoothness", "stall_avoidance", "data_saving", "low_latency"], prio)),
        zh=zh, en=en, slots=slots, cons=cons, conflict=conflict))


S("live_commerce", "medium", [3, 3, 4, 1, 5],
  ["我们在做直播带货，主播要和观众实时互动，播放端缓存不要超过{X}秒，同时缓冲区至少保留{b}秒防止卡顿。",
   "直播卖货场景，延迟必须控制在{X}秒以内；另外缓冲区最低不能少于{b}秒。",
   "带货直播间对互动实时性要求很高：最大缓冲{X}秒，缓冲不得低于{b}秒，家用WiFi环境。"],
  ["Live-commerce stream with real-time host-viewer interaction: keep at most {X} seconds buffered and never let the buffer drop below {b} s.",
   "For our live shopping channel the end-to-end latency must stay within {X} s; the buffer must not go under {b} seconds.",
   "Live selling over home WiFi. Max buffer {X}s, minimum buffer {b}s, interaction matters more than picture quality."],
  {"X": [6, 8, 10], "b": [1, 2]},
  ["ALWAYS max_buffer <= {X}", "ALWAYS buffer >= {b}"])

S("esports_live", "fast", [4, 3, 4, 1, 5],
  ["5G环境下的电竞赛事直播，延迟不超过{X}秒；带宽高于{T}kbps时码率不能低于{F}kbps。",
   "电竞直播（5G）：最大缓冲{X}秒，预测带宽超过{Tm}Mbps时必须至少{F}kbps的码率。"],
  ["E-sports live over 5G: latency below {X} s, and whenever predicted bandwidth exceeds {T} kbps the bitrate must be at least {F} kbps.",
   "Esports broadcast on 5G. Cap the buffer at {X} seconds; if the predicted bandwidth is above {Tm} Mbps, never stream below {F} kbps."],
  {"X": [8, 10, 12], "F": [1850, 2850]},
  ["ALWAYS max_buffer <= {X}", "ALWAYS bitrate >= {F} WHEN pred_bw >= {T}"])

S("short_video_metro", "slow", [2, 3, 5, 3, 2],
  ["用户在地铁上刷短视频，网络很差，最重要的是不卡顿：缓冲区始终保持至少{b}秒，码率每次最多只升一档。",
   "地铁3G弱网短视频，缓冲不能低于{b}秒，升码率一次只能升一级，画质无所谓。"],
  ["Short videos watched on a crowded metro with poor 3G: stalls are the worst, keep at least {b} s in the buffer and only step the bitrate up one level at a time.",
   "Poor mobile network on the subway; buffer must stay above {b} seconds; up-switch at most one ladder step."],
  {"b": [2, 3]},
  ["ALWAYS buffer >= {b}", "ALWAYS switch <= 1"])

S("movie_home", "fast", [5, 4, 3, 1, 1],
  ["用户在家用千兆光纤看电影，画质优先，预测带宽超过{T}kbps时码率不得低于{F}kbps，尽量平滑。",
   "家庭光纤点播电影：带宽大于{Tm}Mbps时至少{F}kbps，追求最佳画质。"],
  ["Movie night on gigabit fibre: picture quality first; when predicted bandwidth is above {T} kbps the bitrate must be at least {F} kbps; keep it smooth.",
   "VoD movie at home over fibre. Whenever bandwidth > {Tm} Mbps stream at >= {F} kbps."],
  {"F": [2850, 4300]},
  ["ALWAYS bitrate >= {F} WHEN pred_bw >= {T}"])

S("data_plan", "medium", [2, 3, 3, 5, 2],
  ["用户流量套餐有限，码率不能超过{C}kbps，最大缓冲不超过{M}秒，避免中途退出浪费流量。",
   "省流量模式：码率上限{Cm}Mbps，缓冲最多{M}秒。"],
  ["Users on a limited data plan: never exceed {C} kbps and keep at most {M} s buffered so that abandoned videos waste little data.",
   "Data-saver mode on 4G: bitrate cap {Cm} Mbps, max buffer {M} seconds."],
  {"C": [750, 1200, 1850], "M": [10, 15]},
  ["ALWAYS bitrate <= {C}", "ALWAYS max_buffer <= {M}"])

S("online_education", "medium", [3, 4, 5, 2, 2],
  ["在线课堂回放，学生最怕卡顿和画面忽好忽坏：缓冲至少{b}秒，码率每次最多升一档。",
   "网课点播：缓冲区不得低于{b}秒，升码率每次不超过1级。"],
  ["Recorded online lectures: students hate stalls and quality oscillation; buffer at least {b} s and up-switch by at most one level.",
   "E-learning VoD over campus WiFi; keep buffer >= {b} s; at most one step up at a time."],
  {"b": [2, 3]},
  ["ALWAYS buffer >= {b}", "ALWAYS switch <= 1"])

S("telemedicine", "medium", [3, 3, 5, 1, 5],
  ["远程会诊视频流，医生需要低延迟：缓存不超过{X}秒，同时绝不能卡顿，缓冲区至少{b}秒。",
   "远程医疗直播，延迟上限{X}秒，最低缓冲{b}秒。"],
  ["Remote medical consultation stream: latency at most {X} seconds, and the buffer must never drop below {b} s.",
   "Telemedicine live video on hospital WiFi: max buffer {X} s, min buffer {b} s."],
  {"X": [8, 10], "b": [2, 3]},
  ["ALWAYS max_buffer <= {X}", "ALWAYS buffer >= {b}"])

S("sports_highlights", "fast", [5, 3, 4, 2, 2],
  ["5G体育集锦，运动画面多，画质要高：预测带宽超过{T}kbps时至少{F}kbps，缓冲区不低于{b}秒。",
   "体育高光点播(5G)，带宽高于{Tm}Mbps时码率不低于{F}kbps，缓冲最少{b}秒。"],
  ["Sports highlights on 5G with lots of motion: when predicted bandwidth exceeds {T} kbps use at least {F} kbps; buffer never below {b} s.",
   "High-motion sports clips, 5G: >= {F} kbps whenever bandwidth > {Tm} Mbps, buffer floor {b} s."],
  {"F": [2850, 4300], "b": [2]},
  ["ALWAYS bitrate >= {F} WHEN pred_bw >= {T}", "ALWAYS buffer >= {b}"])

S("operator_sla", "unknown", [3, 3, 5, 3, 2],
  ["CDN运营商SLA：播放缓冲区任何时候不得低于{b}秒，最大缓冲不超过{M}秒以控制带宽成本。",
   "根据SLA，缓冲下限{b}秒，缓冲上限{M}秒。"],
  ["Operator SLA: the playback buffer must never fall below {b} seconds, and the maximum buffer is limited to {M} s to control egress cost.",
   "Per our CDN SLA: buffer floor {b} s, buffer ceiling {M} s."],
  {"b": [3, 4], "M": [20, 30]},
  ["ALWAYS buffer >= {b}", "ALWAYS max_buffer <= {M}"])

S("kiosk_ads", "medium", [2, 2, 2, 4, 1],
  ["商场广告屏循环播放，画质要求不高，码率不要超过{C}kbps，节省带宽。",
   "广告屏后台播放，码率上限{Cm}Mbps。"],
  ["Looping ads on mall kiosks: quality is not important, cap the bitrate at {C} kbps to save bandwidth.",
   "Background signage playback; bitrate must not exceed {Cm} Mbps."],
  {"C": [1200, 1850]},
  ["ALWAYS bitrate <= {C}"])

S("news_5g", "fast", [4, 3, 3, 2, 3],
  ["5G网络下的新闻短视频，希望画质好一些，没有其他特殊要求。",
   "新闻资讯类视频，5G用户，常规体验即可，画质稍微优先。"],
  ["News clips for 5G users; good picture quality is nice to have, no other special requirements.",
   "Regular news VoD over 5G, slightly prefer quality."],
  {}, [])

# ---------------------------------------------------------------- conflicts
S("conflict_latency_buffer", "medium", [3, 3, 5, 1, 5],
  ["互动直播，缓存最多{X}秒，但缓冲区任何时候都不能低于{b}秒。"],
  ["Interactive live show: at most {X} s of buffer, yet the buffer must never be below {b} s."],
  {"X": [4, 5], "b": [3, 4]},
  ["ALWAYS max_buffer <= {X}", "ALWAYS buffer >= {b}"], conflict=True)

S("conflict_cap_floor", "medium", [5, 3, 3, 5, 2],
  ["码率始终不低于{F}kbps保证高清，同时为了省流量码率不能超过{C}kbps。"],
  ["Always stream at least {F} kbps for HD, but to save data never exceed {C} kbps."],
  {"F": [2850, 4300], "C": [750, 1200]},
  ["ALWAYS bitrate >= {F}", "ALWAYS bitrate <= {C}"], conflict=True)

S("conflict_slow_hd", "slow", [5, 3, 5, 1, 2],
  ["3G弱网用户也必须一直看{F}kbps以上的高清，并且不能卡顿。"],
  ["Even our 3G users must always get at least {F} kbps HD with no stalls."],
  {"F": [2850, 4300]},
  ["ALWAYS bitrate >= {F}"], conflict=True)


def _fill(s, slots):
    v = dict(slots)
    if "F" in v:
        v["T"] = int(v["F"] * 2); v["Tm"] = v["T"] / 1000
    if "C" in v:
        v["Cm"] = v["C"] / 1000
    for k, x in list(v.items()):
        if isinstance(x, float) and x.is_integer():
            v[k] = int(x)
    return s.format(**v)


def generate(n_per_scenario=20, seed=0):
    rng = random.Random(seed)
    items = []
    for sc in SCEN:
        for i in range(n_per_scenario):
            slots = {k: rng.choice(vals) for k, vals in sc["slots"].items()}
            lang = "zh" if i % 2 == 0 else "en"
            tmpl = rng.choice(sc[lang])
            text = _fill(tmpl, slots)
            gt = {"scenario": sc["name"], "network_profile": sc["profile"], "priorities": sc["prio"],
                  "constraints": [_fill(c, slots) for c in sc["cons"]]}
            items.append({"id": f"{sc['name']}_{i:03d}", "scenario": sc["name"], "lang": lang,
                          "text": text, "gt": gt, "conflict": sc["conflict"]})
    # de-duplicate identical texts
    seen, out = set(), []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"]); out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="intents per scenario (before de-dup)")
    ap.add_argument("--out", default="data/intents.jsonl")
    ap.add_argument("--paraphrase", default=None, help="model name in configs/llm.yaml used to paraphrase")
    ap.add_argument("--cfg", default="configs/llm.yaml")
    a = ap.parse_args()
    items = generate(a.n)
    if a.paraphrase:
        from .client import load_clients
        from .prompts import PARAPHRASE_PROMPT
        llm = load_clients(a.cfg, [a.paraphrase])[a.paraphrase]
        roles = ["product manager", "network operator", "content creator", "customer-support agent"]
        for k, it in enumerate(items):
            msg = PARAPHRASE_PROMPT.format(role=roles[k % 4], lang="Chinese" if it["lang"] == "zh" else "English",
                                           text=it["text"])
            r = llm.chat([{"role": "user", "content": msg}])
            it["text_original"] = it["text"]; it["text"] = r["text"].strip()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} intents to {a.out} "
          f"({sum(i['conflict'] for i in items)} conflicting)")


if __name__ == "__main__":
    main()
