"""Emergence - run the world.

    python run.py                          # live viewer (pygame), default 384x384
    python run.py --size 512               # bigger world
    python run.py --headless --steps 200000 --out runs/night1 --video
    python run.py --load runs/night1/ckpt_200000.pt     # continue a saved world
    python run.py --set mutation_rate=0.05 --set regulation=0   # change a law

Viewer keys
    1..8     view: life, bodies, clades, food, signals, state, energy, age
    SPACE    pause            + / -   steps per frame
    left drag     wipe cells (damage)      mouse wheel  brush size
    right click   inspect the genome under the cursor
    F + left drag pour food
    R        abiogenesis: 500 random new cells
    S        save checkpoint           P   screenshot          V  start/stop video
    H        hide/show panel           Q / ESC  quit
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from emergence.config import Config
from emergence.world import World
from emergence.render import render, VIEWS


def build(args):
    cfg = Config()
    if args.size:
        cfg.H = cfg.W = args.size
    cfg.seed = args.seed
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true"))
    if args.load:
        d = torch.load(args.load, map_location="cpu", weights_only=False)
        saved = Config.from_dict(d["config"])
        for kv in args.set or []:            # explicit overrides still win
            k, v = kv.split("=", 1)
            setattr(saved, k, type(getattr(saved, k))(v))
        saved.device = cfg.device
        cfg = saved
        saved_seed = cfg.seed_cells
        cfg.seed_cells = 0
        w = World(cfg)
        cfg.seed_cells = saved_seed
        w.load_state_dict(d)
        print(f"loaded {args.load} at t={w.t}")
    else:
        w = World(cfg)
    print(f"device={w.device}  world={cfg.H}x{cfg.W}  "
          f"cell net {w.n_percept}->{cfg.hidden}->{w.n_actions}  "
          f"translator d={cfg.d_model} x{cfg.n_layers} layers")
    return w


def fmt(m):
    keys = ["t", "population", "bodies", "body_size_cellweighted", "genomes", "clades",
            "mean_depth", "max_depth",
            "mean_genome_len", "kin_neighbours", "top_genome_share", "reseeds"]
    return "  ".join(f"{k}={m[k]:.3g}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                     for k in keys if k in m)


# ---------------------------------------------------------------------- headless
def headless(w, args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(w.cfg.to_json())
    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(os.path.join(args.out, "life.mp4"), fps=30, quality=8,
                                    macro_block_size=1)
    csv_path = os.path.join(args.out, "metrics.csv")
    fh = open(csv_path, "a", newline="")
    cw = None
    t0, s0 = time.time(), w.t
    end = w.t + args.steps
    while w.t < end:
        w.step()
        if w.t % args.log_every == 0:
            m = w.metrics()
            m["steps_per_sec"] = (w.t - s0) / (time.time() - t0)
            if cw is None:
                cw = csv.DictWriter(fh, fieldnames=list(m.keys()), extrasaction="ignore")
                if fh.tell() == 0:
                    cw.writeheader()
            cw.writerow(m)
            fh.flush()
            print(fmt(m) + f"  ({m['steps_per_sec']:.0f} steps/s)", flush=True)
        if writer is not None and w.t % args.video_every == 0:
            writer.append_data(render(w, "life"))
        if w.t % args.snap_every == 0:
            from PIL import Image
            tiles = np.concatenate([render(w, v) for v in ["life", "bodies", "food", "signals"]], 1)
            Image.fromarray(tiles).save(os.path.join(args.out, f"snap_{w.t:07d}.png"))
        if w.t % args.ckpt_every == 0:
            torch.save(w.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    torch.save(w.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    if writer is not None:
        writer.close()
    fh.close()


# ---------------------------------------------------------------------- viewer
LABELS = [("population", "cells"), ("bodies", "organisms"),
          ("body_size_cellweighted", "body size (per cell)"), ("max_body", "largest body"),
          ("multicell_fraction", "in bodies >= 8"), ("genomes", "genomes alive"),
          ("clades", "founding clades"), ("mean_depth", "mutations (mean)"),
          ("max_depth", "mutations (max)"), ("mean_genome_len", "genome length"),
          ("kin_neighbours", "body-mate neighbours"), ("stranger_neighbours", "stranger neighbours"),
          ("top_genome_share", "top genome share"), ("mean_energy", "energy / cell"),
          ("mean_age", "age / cell"), ("mutations", "mutations total"), ("reseeds", "reseeds")]
def viewer(w, args):
    import pygame
    pygame.init()
    cfg = w.cfg
    info = pygame.display.Info()
    scale = args.scale or max(1, min((info.current_h - 120) // cfg.H, (info.current_w - 420) // cfg.W))
    panel = 380
    screen = pygame.display.set_mode((cfg.W * scale + panel, max(cfg.H * scale, 640)))
    pygame.display.set_caption("Emergence")
    font = pygame.font.SysFont("consolas,menlo,monospace", 15)
    small = pygame.font.SysFont("consolas,menlo,monospace", 12)
    clock = pygame.time.Clock()

    view, paused, spf, brush, show_panel = "life", False, 2, 10, True
    feeding, inspect, rec = False, None, None
    hist = {k: [] for k in ["population", "genomes", "mean_depth", "body_size_cellweighted",
                            "mean_genome_len"]}
    last_m, t_rate, s_rate, rate = w.metrics(), time.time(), w.t, 0.0
    t_metrics = 0.0
    os.makedirs(args.out, exist_ok=True)

    def to_world(pos):
        return pos[1] // scale, pos[0] // scale

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_1 <= k <= pygame.K_8:
                    view = VIEWS[k - pygame.K_1]
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    spf = min(64, spf * 2)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    spf = max(1, spf // 2)
                elif k == pygame.K_r:
                    w.seed_life(500)
                elif k == pygame.K_s:
                    p = os.path.join(args.out, f"ckpt_{w.t:07d}.pt")
                    torch.save(w.state_dict(), p)
                    print("saved", p)
                elif k == pygame.K_p:
                    p = os.path.join(args.out, f"shot_{w.t:07d}.png")
                    pygame.image.save(screen, p)
                    print("saved", p)
                elif k == pygame.K_f:
                    feeding = not feeding
                elif k == pygame.K_h:
                    show_panel = not show_panel
                elif k == pygame.K_v:
                    if rec is None:
                        import imageio
                        p = os.path.join(args.out, f"video_{w.t:07d}.mp4")
                        rec = imageio.get_writer(p, fps=30, quality=8, macro_block_size=1)
                        print("recording", p)
                    else:
                        rec.close(); rec = None
                        print("recording stopped")
            elif ev.type == pygame.MOUSEWHEEL:
                brush = int(np.clip(brush + ev.y * 2, 2, 80))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                y, x = to_world(ev.pos)
                if 0 <= y < cfg.H and 0 <= x < cfg.W and bool(w.alive[y, x]):
                    inspect = int(w.gid[y, x])
        buttons = pygame.mouse.get_pressed()
        if buttons[0]:
            y, x = to_world(pygame.mouse.get_pos())
            if 0 <= y < cfg.H and 0 <= x < cfg.W:
                (w.feed if feeding else w.wipe)(y, x, brush)

        if not paused:
            for _ in range(spf):
                w.step()
        if time.time() - t_metrics > 0.4:
            t_metrics = time.time()
            last_m = w.metrics()
            for k in hist:
                hist[k].append(last_m.get(k, 0))
                hist[k] = hist[k][-300:]
        now = time.time()
        if now - t_rate > 1:
            rate = (w.t - s_rate) / (now - t_rate)
            t_rate, s_rate = now, w.t

        img = render(w, view)
        if rec is not None:
            rec.append_data(img)
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        if scale != 1:
            surf = pygame.transform.scale(surf, (cfg.W * scale, cfg.H * scale))
        screen.fill((14, 14, 18))
        screen.blit(surf, (0, 0))
        mx, my = pygame.mouse.get_pos()
        if mx < cfg.W * scale:
            pygame.draw.circle(screen, (255, 255, 255) if not feeding else (120, 255, 120),
                               (mx, my), brush * scale, 1)

        if show_panel:
            x0, y = cfg.W * scale + 14, 10
            lines = [f"EMERGENCE   t={w.t}", f"view [{VIEWS.index(view)+1}] {view}",
                     f"{rate:.0f} steps/s   x{spf}/frame" + ("   PAUSED" if paused else ""),
                     f"brush {brush}  {'FOOD' if feeding else 'wipe'}" + ("  REC" if rec else ""), ""]
            for k, lab in LABELS:
                v = last_m.get(k)
                if v is not None:
                    lines.append(f"{lab:<22}{v:.3g}" if isinstance(v, float) else f"{lab:<22}{v}")
            for ln in lines:
                screen.blit(font.render(ln, True, (220, 220, 225)), (x0, y)); y += 18
            y += 6
            for k, series in hist.items():
                if len(series) > 1:
                    lo, hi = min(series), max(series)
                    pts = [(x0 + i * 340 / 300, y + 30 - 30 * (v - lo) / (hi - lo + 1e-9))
                           for i, v in enumerate(series)]
                    pygame.draw.lines(screen, (120, 200, 255), False, pts, 1)
                    screen.blit(small.render(f"{k}  {lo:.3g}..{hi:.3g}", True, (150, 150, 160)),
                                (x0, y + 32))
                y += 50
            if inspect is not None and int(w.pool.count[inspect]) > 0:
                p = w.pool
                screen.blit(font.render(f"genome #{inspect}  cells={int(p.count[inspect])}",
                                        True, (255, 230, 150)), (x0, y)); y += 18
                screen.blit(small.render(f"clade {int(p.clade[inspect])}  depth {int(p.depth[inspect])}"
                                         f"  len {int(p.length[inspect])}  born t={int(p.born[inspect])}",
                                         True, (200, 200, 200)), (x0, y)); y += 16
                toks = p.tokens[inspect, :int(p.length[inspect])].tolist()
                for i, tk in enumerate(toks):
                    c = [int(255 * v) for v in torch.tensor(
                        [0.5 + 0.5 * np.cos(6.28 * (tk / 32 + o)) for o in (0, .33, .66)]).tolist()]
                    pygame.draw.rect(screen, c, (x0 + (i % 24) * 14, y + (i // 24) * 14, 12, 12))
                y += 34
                ob = w.obias[inspect].tolist()
                names = ["eat", "share", "divide", "attack", "detach"]
                C, Q = cfg.n_state, cfg.n_signal
                vals = [ob[C + Q], ob[C + Q + 1], ob[C + Q + 2], ob[C + Q + 7], ob[C + Q + 8]]
                for nm, v in zip(names, vals):
                    pygame.draw.rect(screen, (80, 80, 90), (x0 + 60, y + 3, 200, 8))
                    pygame.draw.rect(screen, (255, 180, 90), (x0 + 160, y + 3, int(v / 1.5 * 100), 8))
                    screen.blit(small.render(nm, True, (200, 200, 200)), (x0, y)); y += 14
        pygame.display.flip()
        clock.tick(120)
    if rec is not None:
        rec.close()
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="override a law, e.g. --set regulation=0")
    ap.add_argument("--load", type=str, default="")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--out", type=str, default="runs/default")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=50000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-every", type=int, default=20)
    ap.add_argument("--scale", type=int, default=0)
    args = ap.parse_args()
    w = build(args)
    if args.headless:
        headless(w, args)
    else:
        viewer(w, args)


if __name__ == "__main__":
    main()
