"""Encode a source video into a DASH stream with the NeuroBH bitrate ladder.

Produces in <out>/:
    manifest.mpd, init-*.m4s, chunk-*.m4s      the DASH stream (4 s segments, aligned key frames)
    sizes.json                                  per-segment sizes (and VMAF) read by deploy/player.html
    video_size_0 .. video_size_5                Pensieve-format sizes (copy to data/videos/test/<name>/)
    vmaf.csv                                    chunk,level,vmaf  (only with --vmaf; needs ffmpeg with libvmaf)

Recommended sources (open licence, can be named in the paper):
    Big Buck Bunny   https://download.blender.org/peach/bigbuckbunny_movies/  (big_buck_bunny_1080p_h264.mov)
    Tears of Steel   https://mango.blender.org/download/                      (tears_of_steel_1080p.mov)

Examples
    python tools/encode_dash.py --src big_buck_bunny_1080p_h264.mov --out dash/bbb --duration 300
    python tools/encode_dash.py --src tears_of_steel_1080p.mov --out dash/tos --duration 300 --vmaf

Windows: install ffmpeg first (https://www.gyan.dev/ffmpeg/builds/ "full" build includes libvmaf)
and make sure `ffmpeg -version` works in the terminal.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

# kbps, height  (same ladder as training: configs/exp.yaml ladder_kbps)
LADDER = [(300, 240), (750, 360), (1200, 480), (1850, 720), (2850, 1080), (4300, 1080)]


def run(cmd, out=None):
    """Run ffmpeg; if `out` is given, write to a temporary name first and rename on success,
    so an interrupted run never leaves a truncated file that a re-run would reuse."""
    if out:
        root, ext = os.path.splitext(out)
        tmp = root + ".part" + ext
        cmd = [tmp if c == out else c for c in cmd]
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode})")
    if out:
        os.replace(tmp, out)


def has_libvmaf():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return re.search(r"\blibvmaf\b", out) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="start offset in the source (s)")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds to encode (sessions last 180 s)")
    ap.add_argument("--seg", type=float, default=4.0, help="segment duration (s)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--vmaf", action="store_true", help="also compute per-segment VMAF (slow)")
    ap.add_argument("--codec", default="h264", choices=["h264", "vp9"],
                    help="h264 (default, plays in Chrome/Edge) or vp9 (open codec, e.g. for Chromium)")
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found in PATH")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_levels")
    os.makedirs(tmp, exist_ok=True)
    gop = int(round(a.fps * a.seg))
    trim = ["-ss", str(a.start), "-t", str(a.duration)]

    # 1) reference clip (1080p, fixed fps) -- also used for VMAF
    ref = os.path.join(tmp, "ref.mp4")
    if not os.path.exists(ref):
        print("[1/4] reference clip")
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *trim, "-i", a.src,
             "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={a.fps}", "-c:v", "libx264", "-crf", "12", "-preset", "fast",
             "-c:a", "aac", "-b:a", "128k", "-ac", "2", ref], out=ref)

    # 2) one constrained-VBR encode per level, key frame exactly every segment.
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", ref], capture_output=True, text=True).stdout
    # some ffprobe builds print the stream more than once (e.g. "1920,1080\n\n1920,1080"): use the first line
    line = next(l_ for l_ in wh.splitlines() if l_.strip())
    W, H = [int(v) for v in re.findall(r"\d+", line)[:2]]

    def exact_size(h):
        # the ladder is defined for 16:9; wider content (e.g. Tears of Steel, 1920x800) keeps the
        # 16:9 WIDTH of each level, i.e. it is fitted into the 16:9 box and never upscaled
        h = min(h, int(round(h * 16 / 9 * H / W)), H)
        h -= h % 2
        """even (w, h) close to height h with exactly the source aspect ratio (identical DAR on all
        levels, required by the DASH muxer); falls back to the rounded size."""
        for d in range(0, 61, 2):
            for hh in (h - d, h + d):
                if hh > 0 and hh % 2 == 0 and (hh * W) % H == 0 and (hh * W // H) % 2 == 0:
                    return hh * W // H, hh
        return int(round(h * W / H / 2)) * 2, h
    print("[2/4] encoding ladder", [exact_size(h) for _, h in LADDER])
    lv_files = []
    ext = "mp4" if a.codec == "h264" else "webm"
    if a.codec == "h264":
        venc = lambda kbps: ["-c:v", "libx264", "-preset", a.preset, "-profile:v", "high"]
    else:
        venc = lambda kbps: ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1"]
    for k, (kbps, h) in enumerate(LADDER):
        f = os.path.join(tmp, f"v{k}.{ext}")
        lv_files.append(f)
        if os.path.exists(f):
            continue
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", ref, "-an", "-dn", "-map", "0:v:0",
             "-vf", "scale={}:{},setsar=1".format(*exact_size(h)), *venc(kbps),
             "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.1)}k", "-bufsize", f"{kbps * 2}k",
             "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
             "-force_key_frames", f"expr:gte(t,n_forced*{a.seg})", "-pix_fmt", "yuv420p", f], out=f)
    aud = os.path.join(tmp, "a.m4a" if a.codec == "h264" else "a.webm")
    if not os.path.exists(aud):
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", ref, "-vn", "-dn", "-map", "0:a:0",
             "-c:a", "aac" if a.codec == "h264" else "libopus", "-b:a", "64k", "-ac", "2", aud], out=aud)

    # 3) DASH packaging without re-encoding
    print("[3/4] DASH packaging")
    seg_ext = "m4s" if a.codec == "h264" else "webm"
    for old in glob.glob(os.path.join(a.out, "chunk-*")) + glob.glob(os.path.join(a.out, "init-*")) + \
            glob.glob(os.path.join(a.out, "*.mpd")):
        os.remove(old)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for f in lv_files + [aud]:
        cmd += ["-i", f]
    # map only the video / audio stream of each file: sources such as the Blender .mov files carry an
    # extra timecode/data track, which the DASH muxer rejects ("Stream 1 is not mapped to an AdaptationSet")
    for i in range(len(lv_files)):
        cmd += ["-map", f"{i}:v:0"]
    cmd += ["-map", f"{len(lv_files)}:a:0"]
    cmd += ["-c", "copy", "-f", "dash", "-seg_duration", str(a.seg), "-use_template", "1",
            "-use_timeline", "0", "-adaptation_sets", f"id=0,streams=v id=1,streams=a",
            *(["-dash_segment_type", "webm"] if a.codec == "vp9" else []),
            "-init_seg_name", f"init-$RepresentationID$.{seg_ext}",
            "-media_seg_name", f"chunk-$RepresentationID$-$Number%05d$.{seg_ext}",
            # forward slashes: on Windows the DASH muxer derives the segment folder from the last '/'
            # only, so "dash/bbb\\manifest.mpd" would write the segments into "dash/"
            os.path.join(a.out, "manifest.mpd").replace("\\", "/")]
    run(cmd)

    # per-segment sizes (representation ids 0..5 are the video levels in ladder order)
    sizes = []
    for k in range(len(LADDER)):
        segs = sorted(glob.glob(os.path.join(a.out, f"chunk-{k}-*.{seg_ext}")))
        sizes.append([os.path.getsize(s) for s in segs])
    n = min(len(s) for s in sizes)
    sizes = [s[:n] for s in sizes]
    for k, s in enumerate(sizes):
        with open(os.path.join(a.out, f"video_size_{k}"), "w") as f:
            f.write("\n".join(str(x) for x in s) + "\n")
    print(f"  {n} segments x {len(LADDER)} levels")

    # 4) optional per-segment VMAF (model vmaf_v0.6.1, distorted upscaled to the reference size)
    vm = None
    if a.vmaf:
        if not has_libvmaf():
            print("[4/4] WARNING: this ffmpeg has no libvmaf -> skipped (use a 'full' build)")
        else:
            print("[4/4] VMAF (this takes a while)")
            vm = []
            for k, f in enumerate(lv_files):
                log = os.path.join(tmp, f"vmaf{k}.json")
                if not os.path.exists(log):
                    # ffmpeg filter-option escaping: use forward slashes and escape ':' (Windows drive letters)
                    lp = log.replace("\\", "/").replace(":", "\\\\:")
                    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", f, "-i", ref, "-lavfi",
                         f"[0:v]scale={W}:{H}:flags=bicubic,setpts=PTS-STARTPTS[d];"
                         f"[1:v]setpts=PTS-STARTPTS[r];"
                         f"[d][r]libvmaf=log_fmt=json:log_path={lp}:n_threads=8", "-f", "null", "-"])
                    if not os.path.exists(log):
                        sys.exit("libvmaf wrote no log (check the path escaping above)")
                frames = json.load(open(log))["frames"]
                per = [[] for _ in range(n)]
                for fr in frames:
                    c = int(fr["frameNum"] // gop)
                    if c < n:
                        per[c].append(fr["metrics"]["vmaf"])
                vm.append([round(sum(p) / len(p), 3) if p else 0.0 for p in per])
            with open(os.path.join(a.out, "vmaf.csv"), "w") as f:
                f.write("chunk,level,vmaf\n")
                for k in range(len(LADDER)):
                    for c in range(n):
                        f.write(f"{c},{k},{vm[k][c]}\n")
            print("  mean VMAF per level:", [round(sum(v) / len(v), 1) for v in vm])
    json.dump({"seg_dur": a.seg, "ladder_kbps": [b for b, _ in LADDER], "n_chunks": n,
               "sizes": sizes, "vmaf": vm}, open(os.path.join(a.out, "sizes.json"), "w"))
    print(f"done -> {a.out}/manifest.mpd  (upload the whole folder to the media server;"
          f" _levels/ is not needed there)")


if __name__ == "__main__":
    main()
