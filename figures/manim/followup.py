# The two option figures, in the manner of the overview: a title band, filled boxes, large type, matrices to scale.
# PYTHONPATH="figures/manim/stub;figures/manim" .venv-manim/Scripts/python -X utf8 figures/manim/followup.py a|b
import sys

from manim import *

import overview as ov
from overview import BLUE, BLUE_L, BLUE_D, FILL, INK, GREY, DIM, COPPER, COPPER_FILL, MONO, SERIF, M, arrow, cells

RAND_FILL, RAND_STROKE = '#4a2c14', COPPER        # learned in this run, from random numbers
TUNED_FILL = '#1f4f86'                            # learned before (pretrained); a copper outline means it moves now
FROZEN_FILL, FROZEN_STROKE = '#1f4f86', BLUE_L    # learned before and left as it was


def frame_border(scene):
    scene.add(Rectangle(width=config.frame_width - 0.05, height=config.frame_height - 0.05, stroke_color=BLUE, stroke_width=7))


def band(letter, title, sub, y):
    num = M(letter, 40, BLUE).move_to([-11.0, y, 0])
    head = M(title, 40, BLUE_L).next_to(num, RIGHT, buff=0.3)
    s = M(sub, 28, GREY, slant=ITALIC).next_to(head, RIGHT, buff=0.45).align_to(head, DOWN)
    rule = Line([s.get_right()[0] + 0.35, y, 0], [11.6, y, 0], stroke_width=1.5, color=BLUE_D)
    return VGroup(num, head, s, rule)


def mat(w, h, kind):
    fill, stroke, sw = {'random': (RAND_FILL, RAND_STROKE, 4), 'tuned': (TUNED_FILL, COPPER, 4),
                        'frozen': (FROZEN_FILL, FROZEN_STROKE, 2)}[kind]
    return Rectangle(width=w, height=h, fill_color=fill, fill_opacity=1, stroke_color=stroke, stroke_width=sw)


def chips(words, size=30):
    g = VGroup()
    for w in words:
        t = Text(w.replace(' ', '·'), font=MONO, font_size=size, color=INK)
        b = RoundedRectangle(width=t.width + 0.3, height=0.62, corner_radius=0.08, stroke_color=BLUE, stroke_width=2.5,
                             fill_color=FILL, fill_opacity=1)
        g.add(VGroup(b, t.move_to(b)))
    return g.arrange(RIGHT, buff=0.08)


def key(items, y):
    g = VGroup()
    for kind, text in items:
        sw = mat(0.6, 0.36, kind) if kind != 'lora' else Rectangle(width=0.6, height=0.36, fill_color=COPPER, fill_opacity=1, stroke_width=0)
        g.add(VGroup(sw, M(text, 26, GREY)).arrange(RIGHT, buff=0.2))
    return g.arrange(RIGHT, buff=0.9).move_to([0, y, 0])


class OptionA(Scene):
    def construct(self):
        frame_border(self)
        self.add(band('A', 'GPT2-SPANISH, TUNED WHOLE', 'the same anatomy, wider, deeper, and it has already read Spanish', 6.0))
        L, Rx = -5.95, 5.95
        self.add(DashedLine([0, 5.2, 0], [0, -5.0, 0], stroke_width=1.5, color=BLUE_D, dash_length=0.12))
        # column heads
        self.add(M('the character model', 36, INK).move_to([L, 4.75, 0]), M('every number starts random', 27, COPPER, slant=ITALIC).move_to([L, 4.2, 0]),
                 M('GPT2-Spanish', 36, INK).move_to([Rx, 4.75, 0]), M('every number starts from 11.5 GB of Spanish', 27, BLUE_L, slant=ITALIC).move_to([Rx, 4.2, 0]))
        # one sentence, tokenized by each
        lt = cells(list('Porque·te·quiero'), 0.48, BLUE, 26).move_to([L, 3.2, 0])
        rt = chips(['Porque', ' te', ' quiero']).move_to([Rx, 3.2, 0])
        self.add(lt, rt, M('16 tokens: one per character', 26, GREY).next_to(lt, DOWN, buff=0.18),
                 M('3 tokens: one per word', 26, GREY).next_to(rt, DOWN, buff=0.18))
        # embeddings
        el = VGroup(mat(9.6, 0.62, 'random'), M('embedding  122 × 256  =  31K numbers', 27, INK)); el[1].move_to(el[0])
        er = VGroup(mat(10.4, 0.62, 'tuned'), M('embedding  50,257 × 768  =  38.6M numbers', 27, INK)); er[1].move_to(er[0])
        self.add(el.move_to([L, 1.75, 0]), er.move_to([Rx, 1.75, 0]))

        # one block each, its four matrices to the same scale (1 unit = 600 dimensions)
        S = 1 / 600

        def block(d, kind, cx, top, inside):
            def m(w, h, name):
                r = mat(w * S, h * S, kind)
                lab = M(name, 24, INK)
                if inside and r.width > lab.width + 0.3:
                    lab.move_to(r)
                    return VGroup(r, lab)
                lab.next_to(r, DOWN, buff=0.1)
                return VGroup(r, lab)
            att = VGroup(m(3 * d, d, f'W_qkv {d:,}×{3 * d:,}'), m(d, d, 'W_o')).arrange(RIGHT, buff=0.3, aligned_edge=UP)
            mlp = VGroup(m(4 * d, d, f'W_1 {d:,}×{4 * d:,}'), m(4 * d, d, f'W_2 {4 * d:,}×{d:,}')).arrange(RIGHT, buff=0.3, aligned_edge=UP)
            g = VGroup(M('attention', 26, BLUE_L), att, M('MLP', 26, BLUE_L), mlp).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
            return g.move_to([cx, 0, 0]).align_to([0, top, 0], UP)

        bl = block(256, 'random', L, 1.05, False)
        br = block(768, 'tuned', Rx, 1.05, True)
        self.add(bl, br)
        self.add(M('one block of 6: 0.79M numbers', 30, INK).move_to([L, -3.55, 0]),
                 M('one block of 12: 7.1M numbers, 9× as many', 30, INK).move_to([Rx, -3.55, 0]),
                 M('4.75M in all, every one trained', 28, COPPER).move_to([L, -4.2, 0]),
                 M('124M in all, every one tuned', 28, COPPER).move_to([Rx, -4.2, 0]))
        self.add(M('the matrices are drawn to one scale in both columns', 24, DIM, slant=ITALIC).move_to([0, -5.15, 0]))
        self.add(key([('random', 'learned in this run, from random'), ('tuned', 'learned before, and moved in this run')], -6.05))


class OptionB(Scene):
    def construct(self):
        frame_border(self)
        self.add(band('B', 'QWEN2.5-1.5B, FROZEN, WITH LoRA', '1.2% of its numbers learn; the rest stay as they were', 6.0))
        # left: the whole model
        xL = -9.1
        tk = chips(['Por', 'que', ' te', ' quiero'], 28).move_to([xL, 4.85, 0])
        self.add(tk, M('4 tokens, from 151,936 pieces', 24, GREY).next_to(tk, DOWN, buff=0.14))
        emb = VGroup(mat(4.6, 0.62, 'frozen'), M('embedding · 233M', 26, INK)); emb[1].move_to(emb[0])
        emb.move_to([xL, 3.45, 0])
        stack = VGroup(*[Rectangle(width=4.6, height=0.13, stroke_width=1.2, stroke_color=BLUE_L, fill_color=FROZEN_FILL, fill_opacity=1)
                         for _ in range(28)]).arrange(DOWN, buff=0.055).next_to(emb, DOWN, buff=0.45)
        stack[8].set_stroke(COPPER, 4)
        out = VGroup(mat(4.6, 0.62, 'frozen'), M('output, tied', 26, INK)); out[1].move_to(out[0])
        out.next_to(stack, DOWN, buff=0.45)
        self.add(emb, stack, out, M('28 blocks', 28, INK).rotate(PI / 2).next_to(stack, LEFT, buff=0.15))
        self.add(arrow(tk.get_bottom() + DOWN * 0.42, emb.get_top(), GREY, buff=0.04), arrow(emb.get_bottom(), stack.get_top(), GREY, buff=0.04),
                 arrow(stack.get_bottom(), out.get_top(), GREY, buff=0.04))
        self.add(M('1,544M frozen', 32, BLUE_L).next_to(out, DOWN, buff=0.45), M('18.5M trained', 32, COPPER).next_to(out, DOWN, buff=1.0))

        # right: one block to scale (1 unit = 900 dimensions), a LoRA pair on each matrix
        S, x0, t = 1 / 900, -5.6, 0.15
        self.add(DashedLine(stack[8].get_right(), [x0 - 0.15, 4.95, 0], stroke_width=2, color=COPPER, dash_length=0.12))
        self.add(M('one block, to scale', 30, BLUE_L).move_to([0, 5.15, 0]).align_to([x0, 0, 0], LEFT))

        def pair(w, h):
            W = mat(w, h, 'frozen')
            A = Rectangle(width=w, height=t, fill_color=COPPER, fill_opacity=1, stroke_width=0).next_to(W, DOWN, buff=0.06).align_to(W, LEFT)
            B = Rectangle(width=t, height=h, fill_color=COPPER, fill_opacity=1, stroke_width=0).next_to(W, RIGHT, buff=0.06).align_to(W, UP)
            return VGroup(W, A, B)

        def nums(name, din, dout):
            lora = 16 * (din + dout)
            return VGroup(M(f'{name}  {din:,} × {dout:,}', 26, INK),
                          M(f'+{lora / 1000:.0f}K trained · {100 * lora / (din * dout):.1f}%', 24, COPPER)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)

        self.add(M('attention', 28, BLUE_L).move_to([0, 4.45, 0]).align_to([x0, 0, 0], LEFT))
        for i, (n, din, dout) in enumerate([('W_q', 1536, 1536), ('W_k', 1536, 256), ('W_v', 1536, 256), ('W_o', 1536, 1536)]):
            cx = x0 + 1.9 + i * 4.35
            g = pair(dout * S, din * S).move_to([cx, 0, 0]).align_to([0, 4.0, 0], UP)
            lab = nums(n, din, dout).next_to(g, DOWN, buff=0.18)
            lab.set_x(cx)
            self.add(g, lab)
        self.add(M('W_k and W_v are six times narrower: 12 query heads share 2 key and value heads', 24, DIM, slant=ITALIC)
                 .move_to([0, 0.95, 0]).align_to([x0, 0, 0], LEFT))
        self.add(M('MLP', 28, BLUE_L).move_to([0, 0.3, 0]).align_to([x0, 0, 0], LEFT))
        for i, (n, din, dout) in enumerate([('gate', 1536, 8960), ('up', 1536, 8960), ('down', 8960, 1536)]):
            g = pair(8960 * S, 1536 * S).move_to([0, 0, 0]).align_to([x0, 0, 0], LEFT).align_to([0, -0.05 - 2.05 * i, 0], UP)
            lab = nums(n if n != 'down' else 'down, turned', din, dout).next_to(g, RIGHT, buff=0.35).align_to(g, UP)
            self.add(g, lab)
        self.add(M('W x  +  2 · B (A x)', 38, INK).move_to([8.15, -5.5, 0]))
        self.add(M('A and B: the copper strips,', 24, DIM, slant=ITALIC).move_to([8.15, -6.02, 0]),
                 M('thicker than scale; B starts at zero', 24, DIM, slant=ITALIC).move_to([8.15, -6.38, 0]))


if __name__ == '__main__':
    config.pixel_width, config.pixel_height, config.frame_width, config.frame_height = 3840, 2160, 24, 13.5
    if sys.argv[1] == 'a':
        config.output_file = 'option-a'
        OptionA().render()
    else:
        config.output_file = 'option-b'
        OptionB().render()
