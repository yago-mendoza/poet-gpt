# Bird's-eye view of the whole run: the data once, a batch every step, the forward pass, and the loop that learns.
# One still: PYTHONPATH=figures/manim/stub .venv-manim/Scripts/python -X utf8 figures/manim/overview.py
# (the stub stands in for PyAV, whose DLLs Windows Application Control blocks here; stills never use video)
import math
import re
from pathlib import Path

import manimpango
from manim import *

HERE = Path(__file__).parent
for f in (HERE / 'fonts').glob('*.otf'):
    manimpango.register_font(str(f))

SERIF, MONO, GREEK = 'Latin Modern Roman 10', 'Latin Modern Mono 10', 'Cambria'
BLUE, BLUE_L, BLUE_D, FILL = '#3987e5', '#9cc6f7', '#1c3d66', '#08111d'
INK, GREY, DIM = '#e9eef4', '#9aa4af', '#5d6670'
COPPER, COPPER_FILL = '#e08a4a', '#160c04'

config.pixel_width, config.pixel_height = 3840, 2560
config.frame_width, config.frame_height = 24, 16
config.background_color = '#000000'
config.save_last_frame = True
config.write_to_movie = False
config.output_file = 'overview'
config.media_dir = str(HERE / 'out')

X0, X1 = -11.6, 11.6


def M(s, size=30, color=INK, slant=NORMAL):
    """Text with a tiny markup: <b>bold</b>, <i>italic</i>, <m>mono</m>; Greek letters fall back to Cambria Math."""
    plain, t2w, t2s, t2f = '', {}, {}, {}
    for tag, inner, bare in re.findall(r'<(b|i|m)>(.*?)</\1>|([^<]+)', s):
        if bare:
            plain += bare
            continue
        plain += inner
        {'b': t2w, 'i': t2s, 'm': t2f}[tag][inner] = {'b': BOLD, 'i': ITALIC, 'm': MONO}[tag]
    for g in re.findall(r'[α-ω][₀-₉]*', plain):
        t2f[g] = GREEK
    return Text(plain, font=SERIF, font_size=size, color=color, slant=slant, t2w=t2w, t2s=t2s, t2f=t2f)


def box(w, h, color=BLUE, fill=FILL, sw=3):
    return RoundedRectangle(width=w, height=h, corner_radius=0.14, stroke_color=color, stroke_width=sw,
                            fill_color=fill, fill_opacity=1)


def dashed(mob, n=120):
    return DashedVMobject(mob, num_dashes=n, dashed_ratio=0.55)


def card(w, h, title, lines, x, y, color=BLUE, fill=FILL, title_color=INK, size=25, title_size=34, title_font=SERIF):
    b = box(w, h, color, fill).move_to([x, y, 0])
    t = Text(title, font=title_font, font_size=title_size, color=title_color).next_to(b.get_top(), DOWN, buff=0.3)
    body = VGroup(*[M(l, size, GREY) for l in lines]).arrange(DOWN, buff=0.12).next_to(t, DOWN, buff=0.24)
    return VGroup(b, t, body)


def row(widths, x0=X0, x1=X1):
    gap = (x1 - x0 - sum(widths)) / (len(widths) - 1)
    xs, x = [], x0
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    return xs


def arrow(a, b, color=GREY, sw=4, buff=0.1):
    return Arrow(a, b, buff=buff, stroke_width=sw, color=color, tip_length=0.2, max_tip_length_to_length_ratio=0.45)


def header(n, title, what, y, rule_from=None):
    num = M(n, 34, BLUE).move_to([-10.85, y, 0])
    head = M(title, 34, BLUE_L).next_to(num, RIGHT, buff=0.25)
    g = VGroup(num, head)
    if what:
        g.add(M(what, 27, DIM, slant=ITALIC).next_to(head, RIGHT, buff=0.4).align_to(head, DOWN))
    start = rule_from if rule_from is not None else g.get_right()[0] + 0.35
    g.add(Line([start, y, 0], [X1, y, 0], stroke_width=1.5, color=BLUE_D))
    return g


def cells(chars, size=0.5, color=BLUE, font_size=30, fill=FILL):
    g = VGroup()
    for c in chars:
        sq = Square(size, stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1)
        g.add(VGroup(sq, Text(c, font=MONO, font_size=font_size, color=INK).move_to(sq)))
    return g.arrange(RIGHT, buff=0)


class Overview(Scene):
    def construct(self):
        # the frame: the image's own outer contour, in the accent
        self.add(Rectangle(width=24 - 0.05, height=16 - 0.05, stroke_color=BLUE, stroke_width=7))

        # 1. DATA, once ---------------------------------------------------------------------------------------
        self.add(header('1', 'DATA', 'prepare.py, run once before training', 7.45))
        yA, hA = 5.85, 2.75
        wA = [4.0, 5.5, 5.5, 5.6]
        xA = row(wA)
        a1 = card(wA[0], hA, '1,976 posts', ['3,629,723 characters', '205 distinct ones'], xA[0], yA)
        a2 = card(wA[1], hA, 'clean', ['@mentions and #hashtags out,', 'and the 80 characters seen', 'fewer than 5 times'], xA[1], yA)
        vb = box(wA[2], hA).move_to([xA[2], yA, 0])
        vt = Text('one integer per character', font=SERIF, font_size=32, color=INK).next_to(vb.get_top(), DOWN, buff=0.3)
        ch = cells(list('Porque'), 0.56, BLUE, 32)
        ids = VGroup(*[Text(str(i), font=MONO, font_size=28, color=BLUE_L) for i in (43, 71, 74, 73, 77, 61)])
        for c, i in zip(ch, ids):
            i.next_to(c, DOWN, buff=0.13)
        grid = VGroup(ch, ids).next_to(vt, DOWN, buff=0.22).shift(RIGHT * 0.35)
        lab = Text('stoi', font=MONO, font_size=26, color=GREY).next_to(grid, LEFT, buff=0.3).align_to(ids, DOWN)
        vn = M('122 characters, so 122 integers', 23, DIM, slant=ITALIC).next_to(grid, DOWN, buff=0.16).set_x(xA[2])
        a3 = VGroup(vb, vt, grid, lab, vn)
        sb = box(wA[3], hA).move_to([xA[3], yA, 0])
        st = Text('the text as integers', font=SERIF, font_size=32, color=INK).next_to(sb.get_top(), DOWN, buff=0.3)
        tr = Rectangle(width=4.3, height=0.36, stroke_width=0, fill_color=BLUE, fill_opacity=1)
        va = Rectangle(width=0.48, height=0.36, stroke_width=0, fill_color=BLUE_L, fill_opacity=1)
        bars = VGroup(tr, va).arrange(RIGHT, buff=0.07).next_to(st, DOWN, buff=0.3)
        tl = M('<m>train.bin</m>  3,240,816 (90%)', 24, GREY).next_to(bars, DOWN, buff=0.16).align_to(bars, LEFT)
        vl = M('<m>val.bin</m>  370,539, 197 whole posts', 24, GREY).next_to(tl, DOWN, buff=0.08).align_to(bars, LEFT)
        a4 = VGroup(sb, st, bars, tl, vl)
        self.add(a1, a2, a3, a4)
        for l, r in ((a1, a2), (a2, a3), (a3, a4)):
            self.add(arrow(l[0].get_right(), r[0].get_left()))

        # 2. BATCH, every step --------------------------------------------------------------------------------
        self.add(header('2', 'BATCH', 'get_batch, drawn again at every step', 3.95))
        yB = 2.2
        tape = VGroup(*[Rectangle(width=0.16, height=0.62, stroke_color=BLUE_D, stroke_width=1.2, fill_color=FILL,
                                  fill_opacity=1) for _ in range(44)]).arrange(RIGHT, buff=0).move_to([-7.45, yB + 0.25, 0])
        tape_lbl = M('<m>train.bin</m>: 3.24 million integers in a row', 25, GREY).next_to(tape, UP, buff=0.22).align_to(tape, LEFT)
        dots = VGroup(Text('...', font=SERIF, font_size=30, color=DIM).next_to(tape, LEFT, buff=0.1),
                      Text('...', font=SERIF, font_size=30, color=DIM).next_to(tape, RIGHT, buff=0.1))
        picks = [3, 10, 17, 25, 31, 39]
        for k in picks:
            tape[k].set_fill(BLUE, 1).set_stroke(BLUE_L, 1.5)
        ticks = VGroup(*[Triangle(fill_color=BLUE_L, fill_opacity=1, stroke_width=0).scale(0.09).rotate(PI)
                         .next_to(tape[k], DOWN, buff=0.08) for k in picks])
        tick_lbl = M('32 random starting points i', 24, GREY, slant=ITALIC).next_to(ticks, DOWN, buff=0.14).align_to(tape, LEFT)
        self.add(tape_lbl, tape, dots, ticks, tick_lbl)
        # one chunk, zoomed: x and y are the same characters, shifted by one
        xs = cells(list('Porque·t'), 0.5, BLUE, 30)
        ys = cells(list('orque·te'), 0.5, BLUE_L, 30, fill='#0d1f36')
        xs.move_to([-0.45, yB + 0.85, 0])
        ys.next_to(xs, DOWN, buff=0.36).shift(RIGHT * 0.5)
        xl = Text('x', font=MONO, font_size=34, color=BLUE).next_to(xs, LEFT, buff=0.25)
        yl = Text('y', font=MONO, font_size=34, color=BLUE_L).next_to(ys, LEFT, buff=0.75)
        pairs = VGroup(*[Line(xs[j].get_bottom(), ys[j].get_top(), stroke_width=2, color=DIM) for j in range(8)])
        note = M('drawn with 8 of the 256', 22, DIM, slant=ITALIC).next_to(xs, UP, buff=0.12).align_to(xs, LEFT)
        code = VGroup(Text('x = train[i : i+256]', font=MONO, font_size=24, color=GREY),
                      Text('y = train[i+1 : i+257]', font=MONO, font_size=24, color=GREY),
                      M('each position predicts the next one', 22, DIM, slant=ITALIC)
                      ).arrange(DOWN, buff=0.08, aligned_edge=LEFT).next_to(ys, DOWN, buff=0.22).align_to(xl, LEFT)
        k = picks[3]
        zoom = VGroup(DashedLine(tape[k].get_corner(UR), xs.get_corner(UL) + LEFT * 0.62, stroke_width=1.5, color=DIM, dash_length=0.07),
                      DashedLine(tape[k].get_corner(DR), ys.get_corner(DL) + LEFT * 1.12, stroke_width=1.5, color=DIM, dash_length=0.07))
        self.add(zoom, xs, ys, xl, yl, pairs, note, code)
        # the batch tensor
        rows_, cols_ = 8, 18
        bt = VGroup(*[Square(0.2, stroke_color=BLUE_D, stroke_width=1, fill_color=BLUE if (r + c) % 7 else BLUE_L,
                             fill_opacity=0.35 + 0.5 * ((r * 5 + c * 3) % 4) / 4) for r in range(rows_) for c in range(cols_)])
        bt.arrange_in_grid(rows_, cols_, buff=0.025).move_to([7.4, yB - 0.12, 0])
        bt_c = M('256 characters each', 25, GREY).next_to(bt, UP, buff=0.16)
        bt_n = M('<b>8,192</b> predictions per step', 26, INK).next_to(bt_c, UP, buff=0.12)
        bt_r = M('32 rows', 25, GREY).next_to(bt, RIGHT, buff=0.2)
        self.add(bt, bt_c, bt_n, bt_r, arrow([3.05, yB + 0.4, 0], bt.get_left() + LEFT * 0.05))

        # 3. MODEL ---------------------------------------------------------------------------------------------
        self.add(header('3', 'MODEL', 'the forward pass', 0.12))
        yC, hC = -1.95, 3.0
        wC = [3.5, 6.5, 3.4, 2.8, 4.0]
        xC = row(wC, -10.95)
        c1 = card(wC[0], hC, 'embedding', ['token  122 × 256', '+ position  256 × 256', 'each character', 'becomes 256 numbers'], xC[0], yC, size=24, title_size=32)
        stack = VGroup(*[box(wC[1], hC, BLUE if k == 0 else BLUE_D).move_to([xC[1] + 0.1 * k, yC + 0.08 * k, 0]) for k in range(6)][::-1])
        front = stack[-1]
        bt_t = M('block × 6', 32).next_to(front.get_top(), DOWN, buff=0.28)
        rows3 = VGroup(
            M('LayerNorm, then <b>attention</b>: 8 heads of 32', 24, GREY),
            M('each position reads the ones before it', 22, DIM, slant=ITALIC),
            M('LayerNorm, then <b>MLP</b>: 256, 1,024, 256', 24, GREY),
            M('each position on its own', 22, DIM, slant=ITALIC),
            M('both <i>add</i> to what came in: the residual', 24, GREY),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(bt_t, DOWN, buff=0.22)
        rows3[1].shift(RIGHT * 0.3)
        rows3[3].shift(RIGHT * 0.3)
        c2 = VGroup(stack, bt_t, rows3)
        c3 = card(wC[2], hC, 'unembedding', ['LayerNorm, then', '256 × 122:', "the embedding's", 'matrix, reused (tied)'], xC[2], yC, size=24, title_size=32)
        pb = box(wC[3], hC).move_to([xC[3], yC, 0])
        pt = Text('softmax', font=SERIF, font_size=32, color=INK).next_to(pb.get_top(), DOWN, buff=0.28)
        bars3 = VGroup()
        for c, p in [('e', .34), ('a', .17), ('o', .11), ('·', .08), ('n', .05)]:
            bars3.add(VGroup(Text(c, font=MONO, font_size=24, color=GREY),
                             Rectangle(width=4.0 * p, height=0.2, stroke_width=0, fill_color=BLUE_L, fill_opacity=1)).arrange(RIGHT, buff=0.15))
        bars3.arrange(DOWN, buff=0.09, aligned_edge=LEFT).next_to(pt, DOWN, buff=0.2)
        pn = M('122 probabilities', 22, DIM, slant=ITALIC).next_to(bars3, DOWN, buff=0.12)
        c4 = VGroup(pb, pt, bars3, pn)
        lb = box(wC[4], hC, COPPER, COPPER_FILL).move_to([xC[4], yC, 0])
        lt = Text('loss', font=SERIF, font_size=34, color=COPPER).next_to(lb.get_top(), DOWN, buff=0.28)
        lbody = VGroup(M('how surprised the model is', 22, GREY), M('by the right character,', 22, GREY),
                       M('averaged over 8,192', 22, GREY), M('step 0: <b>4.88</b>', 23, INK),
                       M('a uniform guess: ln 122 = 4.80', 20, DIM, slant=ITALIC),
                       M('step 3,000: <b>1.19</b>', 23, INK)).arrange(DOWN, buff=0.06).next_to(lt, DOWN, buff=0.16)
        c5 = VGroup(lb, lt, lbody)
        self.add(c1, c2, c3, c4, c5)
        self.add(arrow(c1[0].get_right(), front.get_left()), arrow(front.get_right(), c3[0].get_left()),
                 arrow(c3[0].get_right(), pb.get_left()), arrow(pb.get_right(), lb.get_left(), COPPER))
        # the rails: x feeds the embedding, y only meets the model at the loss
        xr = 6.0
        x_path = VMobject(stroke_color=BLUE, stroke_width=4).set_points_as_corners(
            [[xr, bt.get_bottom()[1], 0], [xr, 0.55, 0], [-11.4, 0.55, 0], [-11.4, yC, 0]])
        self.add(x_path, arrow([-11.4, yC, 0], c1[0].get_left(), BLUE, buff=0.0))
        self.add(M('<m>x</m>: 32 × 256 integers', 24, BLUE).next_to([4.1, 0.55, 0], UP, buff=0.1))
        yr = lb.get_center()[0] - 0.9
        self.add(arrow([yr, bt.get_bottom()[1], 0], [yr, lb.get_top()[1], 0], BLUE_L, buff=0.03))
        self.add(M('<m>y</m>: the answers', 24, BLUE_L).next_to([yr, 0.55, 0], RIGHT, buff=0.15))
        theta = dashed(RoundedRectangle(width=c3[0].get_right()[0] - c1[0].get_left()[0] + 0.5, height=hC + 0.32,
                                        corner_radius=0.2, stroke_color=BLUE_L, stroke_width=2), 150)
        theta.move_to([(c3[0].get_right()[0] + c1[0].get_left()[0]) / 2 + 0.05, yC + 0.06, 0])
        theta_l = M('θ: the 4.75M numbers training changes', 24, BLUE_L).next_to(theta, DOWN, buff=0.1).align_to(theta, LEFT).shift(RIGHT * 0.2)
        self.add(theta, theta_l)

        # 4. LEARNING ------------------------------------------------------------------------------------------
        yD, hD = -6.42, 2.45
        wD = [5.0, 4.3, 7.5, 4.3]
        xD = row(wD)
        loop = M('<b>3,000 times</b>: 24.6 million predictions, about 7.6 passes over train.bin', 25, COPPER)
        loop.move_to([0, -4.68, 0]).align_to([xD[1] + 0.35, 0, 0], LEFT)
        h4 = header('4', 'LEARNING', None, -4.68, rule_from=loop.get_right()[0] + 0.35)
        self.add(h4, loop, Line([h4[1].get_right()[0] + 0.35, -4.68, 0], [loop.get_left()[0] - 0.35, -4.68, 0], stroke_width=1.5, color=BLUE_D))
        ev = dashed(box(wD[0], hD, GREY), 110).move_to([xD[0], yD, 0])
        evt = Text('every 250 steps', font=SERIF, font_size=30, color=INK).next_to(ev.get_top(), DOWN, buff=0.3)
        evb = VGroup(M('loss on 50 batches of train and', 23, GREY), M('of val, dropout off, no update;', 23, GREY),
                     M('the checkpoint is kept only', 23, GREY), M('if val improved', 23, GREY)).arrange(DOWN, buff=0.07).next_to(evt, DOWN, buff=0.18)
        self.add(ev, evt, evb)
        d3 = card(wD[1], hD, 'update', ['each of the 4.75M numbers', 'moves a little against', 'its gradient, then', 'the next batch'], xD[1], yD,
                  COPPER, COPPER_FILL, COPPER, 23, 32)
        ab = box(wD[2], hD, COPPER, COPPER_FILL).move_to([xD[2], yD, 0])
        at = Text('AdamW', font=SERIF, font_size=32, color=COPPER).next_to(ab.get_corner(UL), DR, buff=0.3)
        ad = VGroup(M('β₁ 0.9,  β₂ 0.99', 23, GREY), M('weight decay 0.1', 23, GREY), M('gradients clipped at 1.0', 23, GREY)
                    ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(at, DOWN, buff=0.22).align_to(at, LEFT)
        ax = Axes(x_range=[0, 3000, 1000], y_range=[0, 1.1e-3, 1e-3], x_length=3.0, y_length=1.1,
                  axis_config={'stroke_width': 1.5, 'color': DIM, 'include_ticks': False, 'include_tip': False})
        ax.move_to([ab.get_right()[0] - 1.85, yD + 0.15, 0])
        lr = lambda s: 1e-3 * s / 100 if s < 100 else 1e-4 + 0.5 * (1 + math.cos(math.pi * (s - 100) / 2900)) * 9e-4
        curve = ax.plot(lr, x_range=[0, 3000, 10], color=COPPER, stroke_width=3.5)
        ax_t = M('learning rate', 22, GREY, slant=ITALIC).next_to(ax, UP, buff=0.1)
        ax_w = VGroup(M('1e-3 after 100 warm-up steps,', 20, DIM), M('cosine down to 1e-4', 20, DIM)).arrange(DOWN, buff=0.05).next_to(ax, DOWN, buff=0.1)
        d2 = VGroup(ab, at, ad, ax, curve, ax_t, ax_w)
        d1 = card(wD[3], hD, 'loss.backward()', ['one gradient per number:', 'how the loss would move', 'if it moved'], xD[3], yD,
                  COPPER, COPPER_FILL, COPPER, 23, 30, MONO)
        self.add(d1, d2, d3)
        self.add(arrow(lb.get_bottom(), [lb.get_center()[0], d1[0].get_top()[1], 0], COPPER, buff=0.04),
                 arrow(d1[0].get_left(), ab.get_right(), COPPER), arrow(ab.get_left(), d3[0].get_right(), COPPER))
        self.add(Arrow(d3[0].get_top(), [xD[1], theta.get_bottom()[1], 0], buff=0.04, stroke_width=5, color=COPPER,
                       tip_length=0.24, max_tip_length_to_length_ratio=0.6))


if __name__ == '__main__':
    Overview().render()
    print(next((HERE / 'out').rglob('overview*.png')))
