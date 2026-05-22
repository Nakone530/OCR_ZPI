import cv2, numpy as np, sys, io, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, ".")
from ocr.bbox_annotator import clamp_box
import ocr.bbox_annotator as ba

COLORS_BGR = [(255, 80, 0), (30, 160, 30), (0, 100, 255), (180, 0, 180)]
STEP_COLORS = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#6a0dad']


def compute_split_steps(img_bgr, box, median_h):
    H_img, W_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, W_img, H_img)
    w = x2 - x1
    h = y2 - y1
    ref_h = float(h) if h >= 4 else float(median_h or 20)
    expected_word_w = max(1.0, 2.5 * ref_h)

    crop_bgr = img_bgr[y1:y2, x1:x2].copy()
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    bk = max(3, int(round(0.05 * ref_h)) * 2 + 1)
    blurred = cv2.GaussianBlur(crop_gray, (bk, bk), 0)
    block_size = max(21, (min(crop_gray.shape[:2]) // 8) | 1)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5)

    dil_w = max(1, int(round(0.45 * ref_h)))
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, 1))
    mask_proj = cv2.dilate(mask, dil_kernel, iterations=1)

    proj = np.sum(mask_proj > 0, axis=0).astype(np.float32)
    p20 = float(np.percentile(proj, 20))
    p85 = float(np.percentile(proj, 85))
    valley_thr = max(0.0, p20 + 0.48 * max(0.0, p85 - p20))
    empty = proj <= valley_thr
    min_gap = max(int(round(0.35 * ref_h)), 5)

    split_points = []
    run_start = None
    for idx, ie in enumerate(empty):
        if ie and run_start is None:
            run_start = idx
        elif not ie and run_start is not None:
            run_len = idx - run_start
            if run_len >= min_gap:
                split_points.append(run_start + run_len // 2)
            run_start = None
    if run_start is not None and (len(empty) - run_start) >= min_gap:
        split_points.append(run_start + (len(empty) - run_start) // 2)

    min_piece_w = max(int(round(1.2 * ref_h)), 8)
    pieces = []
    left = 0
    for sp in split_points:
        if sp - left >= min_piece_w:
            pieces.append((left, sp))
        left = sp
    if w - left >= min_piece_w:
        pieces.append((left, w))

    def naturalness(pw):
        return 1.0 / (1.0 + abs(np.log(max(1e-6, pw) / expected_word_w)))

    piece_widths = [float(pe - ps) for ps, pe in pieces]
    min_natural_w = max(8.0, 1.2 * ref_h)
    too_narrow = any(pw < min_natural_w for pw in piece_widths)
    score_before = naturalness(float(w))
    score_after = float(np.mean([naturalness(pw) for pw in piece_widths])) if piece_widths else 0
    split_ok = len(pieces) > 1 and not too_narrow and score_after > score_before

    return dict(
        x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h,
        ref_h=ref_h, expected_word_w=expected_word_w,
        crop_bgr=crop_bgr, crop_gray=crop_gray,
        bk=bk, block_size=block_size, mask=mask,
        dil_w=dil_w, mask_proj=mask_proj,
        proj=proj, valley_thr=valley_thr, empty=empty,
        min_gap=min_gap, split_points=split_points,
        pieces=pieces, piece_widths=piece_widths,
        min_natural_w=min_natural_w, too_narrow=too_narrow,
        score_before=score_before, score_after=score_after,
        split_ok=split_ok,
    )


def make_steps_figure(img_bgr, box, median_h, label=""):
    H_img, W_img = img_bgr.shape[:2]
    s = compute_split_steps(img_bgr, box, median_h)
    w, h = s['w'], s['h']

    scale = min(1.0, 500 / max(w, 1), 160 / max(h, 1))
    cw = max(1, int(w * scale))
    ch = max(1, int(h * scale))

    def rs(arr, interp=cv2.INTER_NEAREST):
        return cv2.resize(arr, (cw, ch), interpolation=interp)

    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor('#12121f')
    gs = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.55, wspace=0.22,
                           left=0.02, right=0.98, top=0.87, bottom=0.07)

    step_titles = [
        "Krok 1 — szeroki bbox",
        "Krok 2a — maska binarna",
        "Krok 2b — dylatacja pozioma",
        "Krok 3 — projekcja + doliny",
        "Krok 4 — weryfikacja",
    ]

    def make_ax(row, col, title, sc):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0a18')
        for sp in ax.spines.values():
            sp.set_edgecolor(sc); sp.set_linewidth(2)
        ax.set_title(title, color=sc, fontsize=9, pad=5, fontweight='bold')
        return ax

    # ── Krok 1: miniatura obrazu z zaznaczonym bboxem ─────────────────────
    ax1 = make_ax(0, 0, step_titles[0], STEP_COLORS[0])
    thumb_h = 160
    thumb_w = max(1, int(W_img * thumb_h / H_img))
    thumb = cv2.resize(img_bgr, (thumb_w, thumb_h))
    tx1 = int(s['x1'] * thumb_w / W_img); tx2 = int(s['x2'] * thumb_w / W_img)
    ty1 = int(s['y1'] * thumb_h / H_img); ty2 = int(s['y2'] * thumb_h / H_img)
    cv2.rectangle(thumb, (tx1, ty1), (tx2, ty2), (50, 100, 255), 2)
    ax1.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.text(0.5, -0.06, f"w={w}px  h={h}px  w/h={w/h:.1f}  >  min_split={max(110,int(4*s['ref_h']))}px",
             ha='center', va='top', transform=ax1.transAxes,
             color=STEP_COLORS[0], fontsize=7.5)

    # Krok 1 dolny: powiększony crop
    ax1b = make_ax(1, 0, f"Crop  ref_h={s['ref_h']:.0f}px", STEP_COLORS[0])
    ax1b.imshow(cv2.cvtColor(rs(s['crop_bgr'], cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    ax1b.axis('off')

    # ── Krok 2a: maska binarna ────────────────────────────────────────────
    ax2 = make_ax(slice(0, 2), 1, step_titles[1], STEP_COLORS[1])
    ax2.imshow(rs(s['mask']), cmap='gray', aspect='auto')
    ax2.axis('off')
    ax2.text(0.5, -0.04,
             f"GaussianBlur k={s['bk']}px\nAdaptiveThreshold block={s['block_size']}px",
             ha='center', va='top', transform=ax2.transAxes,
             color=STEP_COLORS[1], fontsize=7)

    # ── Krok 2b: dylatacja — overlay (cyan = strefa rozszerzona) ─────────
    ax3 = make_ax(slice(0, 2), 2, step_titles[2], STEP_COLORS[2])
    mask_s = rs(s['mask']); mdil_s = rs(s['mask_proj'])
    overlay = np.zeros((ch, cw, 3), dtype=np.uint8)
    overlay[mdil_s > 0] = [0, 190, 170]
    overlay[mask_s > 0] = [240, 240, 240]
    ax3.imshow(overlay, aspect='auto')
    ax3.axis('off')
    ax3.text(0.5, -0.04,
             f"dil_w={s['dil_w']}px  (bialy=oryg. maska, cyan=po dylatacji)",
             ha='center', va='top', transform=ax3.transAxes,
             color=STEP_COLORS[2], fontsize=7)

    # ── Krok 3: projekcja z progiem i strefami ───────────────────────────
    ax4 = make_ax(slice(0, 2), 3, step_titles[3], STEP_COLORS[3])
    xs = np.arange(len(s['proj']))
    valley_mask = s['proj'] <= s['valley_thr']
    ax4.fill_between(xs, s['proj'], where=~valley_mask,
                     color='#457b9d', alpha=0.85, label='tekst')
    ax4.fill_between(xs, s['proj'], where=valley_mask,
                     color='#888888', alpha=0.55, label='dolina')
    ax4.axhline(s['valley_thr'], color='#f4a261', lw=1.8,
                ls='--', label=f"prog={s['valley_thr']:.1f}")
    for sp in s['split_points']:
        ax4.axvline(sp, color='#e63946', lw=2.5, alpha=0.9, zorder=5)
    ax4.set_facecolor('#0a0a18')
    ax4.tick_params(colors='#aaaaaa', labelsize=7)
    ax4.set_xlim(0, len(s['proj']))
    ax4.set_xlabel("kolumna [px]", color='#aaaaaa', fontsize=7)
    ax4.legend(fontsize=6.5, labelcolor='white',
               facecolor='#0a0a18', edgecolor='#457b9d')
    ax4.text(0.5, -0.04,
             f"min_gap={s['min_gap']}px  |  {len(s['split_points'])} kandydatow podzialu",
             ha='center', va='top', transform=ax4.transAxes,
             color=STEP_COLORS[3], fontsize=7)

    # ── Krok 4: wynik końcowy ─────────────────────────────────────────────
    ax5 = make_ax(slice(0, 2), 4, step_titles[4], STEP_COLORS[4])
    result_img = cv2.cvtColor(rs(s['crop_bgr'], cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB).copy()
    if s['split_ok'] and s['pieces']:
        for i, (ps, pe) in enumerate(s['pieces']):
            col = COLORS_BGR[i % len(COLORS_BGR)]
            rps = int(ps * scale); rpe = min(int(pe * scale), cw - 1)
            blend = result_img[:, rps:rpe].copy()
            tint = np.array(col[::-1], dtype=np.float32)
            blend = (blend * 0.45 + tint * 0.55).clip(0, 255).astype(np.uint8)
            result_img[:, rps:rpe] = blend
            cv2.rectangle(result_img, (rps, 1), (rpe, ch - 2), col[::-1], 2)
            mid = (rps + rpe) // 2
            txt = f"{int(s['piece_widths'][i])}px"
            cv2.putText(result_img, txt,
                        (max(0, mid - 14), ch // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(result_img, (0, 0), (cw - 1, ch - 1), (160, 160, 160), 2)
    ax5.imshow(result_img, aspect='auto')
    ax5.axis('off')
    verdict = f"PODZIELONO na {len(s['pieces'])} czesci" if s['split_ok'] else "ZOSTAWIONO"
    vcol = '#5eff5e' if s['split_ok'] else '#ff7070'
    ax5.text(0.5, -0.05, verdict,
             ha='center', va='top', transform=ax5.transAxes,
             color=vcol, fontsize=8.5, fontweight='bold')
    delta = s['score_after'] - s['score_before']
    sign = "+" if delta >= 0 else ""
    ax5.text(0.5, -0.13,
             f"naturalnosc: {s['score_before']:.2f}  ->  {s['score_after']:.2f}  ({sign}{delta:.2f})",
             ha='center', va='top', transform=ax5.transAxes,
             color='#cccccc', fontsize=7.5)

    # strzalki
    for xf in [0.218, 0.412, 0.606, 0.800]:
        fig.text(xf, 0.48, "→", fontsize=26, color='#444466',
                 ha='center', va='center')

    fig.suptitle(label, color='white', fontsize=11, y=0.95)
    return fig


# ── Zbierz kandydatów ──────────────────────────────────────────────────────
candidates = []
for path in ["Chopin/chopin 1.jpg", "Chopin/chopin 4.jpg", "Chopin/chopin 9.jpg"]:
    img = cv2.imread(path)
    if img is None:
        continue
    log = []
    _orig = ba.split_wide_box_by_projection

    def _spy(image, box, median_h=None):
        r = _orig(image, box, median_h=median_h)
        log.append((box, median_h, r))
        return r

    ba.split_wide_box_by_projection = _spy
    ba.detect_word_boxes_auto(img)
    ba.split_wide_box_by_projection = _orig

    for box, mh, r in log:
        bw = box[2] - box[0]; bh = box[3] - box[1]
        ref = float(bh) if bh >= 4 else float(mh or 1)
        if bw >= max(110, 4 * ref):
            candidates.append((path, img, box, mh, r))

split_ones = [c for c in candidates if len(c[4]) > 1]
nosplit_ones = [c for c in candidates if len(c[4]) == 1]
chosen = (split_ones[:2] + nosplit_ones[:1])[:3]

for i, (path, img, box, mh, result) in enumerate(chosen):
    bw = box[2] - box[0]; bh = box[3] - box[1]
    tag = "SPLIT" if len(result) > 1 else "brak_podzialu"
    name = path.split("/")[-1].replace(".", "_")
    lbl = (f"{name}   bbox={bw}x{bh}px   "
           f"expected_word_w={max(1.0,2.5*(bh if bh>=4 else mh or 1)):.0f}px")
    fig = make_steps_figure(img, box, mh, label=lbl)
    out = f"steps_{i+1}_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#12121f')
    plt.close(fig)
    print(f"Zapisano: {out}")
