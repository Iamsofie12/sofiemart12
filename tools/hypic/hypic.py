#!/usr/bin/env python3
"""
Hypic maison — retouche portrait locale, gratuite et illimitée.

Chaque zone se règle de 0 (rien) à 4 (maximum, toujours naturel), par pas de 0,5.

Exemple :
    python3 hypic.py photo.jpg --bronzage 1.5 --acne 4 --lissage 1 --cils 1 --sourcils 0.5 --cheveux 1

Le programme :
  1. repère le visage (478 points : yeux, sourcils, lèvres…) et sépare l'image
     en zones (peau du visage, peau du corps, cheveux, vêtements, fond) ;
  2. applique chaque retouche UNIQUEMENT dans sa zone, avec des bords fondus ;
  3. garde la résolution d'origine et le grain de peau (pas d'effet plastique).
"""
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

MODELS = {
    "face": ("face_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
    "seg": ("selfie_multiclass_256x256.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"),
}

# Indices des points du visage (MediaPipe Face Mesh)
EYE_R_UP = [33, 246, 161, 160, 159, 158, 157, 173, 133]      # œil à gauche de l'image
EYE_R_LO = [33, 7, 163, 144, 145, 153, 154, 155, 133]
EYE_L_UP = [263, 466, 388, 387, 386, 385, 384, 398, 362]     # œil à droite de l'image
EYE_L_LO = [263, 249, 390, 373, 374, 380, 381, 382, 362]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSTRILS = [[79, 166, 59, 75, 60, 20, 238], [309, 392, 289, 305, 290, 250, 458]]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# ---------------------------------------------------------------- utilitaires
def model_path(key):
    name, url = MODELS[key]
    folder = os.path.join(os.path.expanduser("~"), ".cache", "hypic")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"Téléchargement du modèle {name} (une seule fois)…")
        urllib.request.urlretrieve(url, path)
    return path


def gauss(x, sigma):
    sigma = max(float(sigma), 0.3)
    return cv2.GaussianBlur(x, (0, 0), sigma)


def guided_filter(guide, src, r, eps=1e-3):
    """Recolle un masque flou sur les vrais contours de l'image (He et al.)."""
    r = max(int(r), 1)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mi, mp_ = box(guide), box(src)
    cov = box(guide * src) - mi * mp_
    var = box(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = mp_ - a * mi
    return np.clip(box(a) * guide + box(b), 0, 1)


def blend(base, other, mask):
    m = mask[..., None] if other.ndim == 3 else mask
    return base * (1 - m) + other * m


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(np.float32)


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)


def u8(img):
    return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- analyse
def analyse(img, zone=None):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    rgb = u8(img)
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    h, w = img.shape[:2]

    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path("seg")),
        output_confidence_masks=True, output_category_mask=False))
    conf = [cv2.resize(m.numpy_view().astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            for m in seg.segment(mpi).confidence_masks]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    r = max(2, int(min(h, w) / 150))
    conf = [guided_filter(gray, c, r) for c in conf]
    # 0 fond, 1 cheveux, 2 peau du corps, 3 peau du visage, 4 vêtements, 5 accessoires

    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("face")), num_faces=4))

    def detect(x0, y0, x1, y1):
        """Cherche les visages dans un morceau de l'image (zoom), renvoie les points en coordonnées globales."""
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return []
        r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        cw, ch = x1 - x0, y1 - y0
        return [np.array([[p.x * cw + x0, p.y * ch + y0] for p in f], np.float32) for f in r.face_landmarks]

    def candidates():
        # 1) image entière (selfies, portraits serrés)
        found = detect(0, 0, w, h)
        if found:
            return found
        # 2) zoom autour de chaque zone de "peau du visage" repérée par la segmentation
        fs = (conf[3] > 0.4).astype(np.uint8)
        if zone is not None:
            fs &= (zone > 0.5).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(fs)
        order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1 if n > 1 else []
        for i in list(order)[:4]:
            if st[i, cv2.CC_STAT_AREA] < 150:
                continue
            x, y, bw, bh = st[i, :4]
            side = int(max(bw, bh) * 2.6)
            cx, cy = x + bw // 2, y + bh // 2
            found += detect(max(0, cx - side // 2), max(0, cy - side // 2), min(w, cx + side // 2), min(h, cy + side // 2))
        if found:
            return found
        # 3) secours : balayage de l'image par morceaux qui se chevauchent
        for n_ in (2, 3, 4):
            step_w, step_h = w / (n_ + 1), h / (n_ + 1)
            for i in range(n_):
                for j in range(n_):
                    found += detect(int(j * step_w), int(i * step_h), int((j + 2) * step_w), int((i + 2) * step_h))
            if found:
                return found
        return found

    # Plusieurs visages possibles : on garde le plus grand, situé dans la zone de Sofie.
    lm, best = None, 0
    for pts in candidates():
        cx, cy = pts.mean(0).astype(int)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        if zone is not None and zone[cy, cx] < 0.5:
            continue
        size = np.linalg.norm(pts[454] - pts[234])
        if size > best:
            lm, best = pts, size
    seg.close()
    fl.close()
    return conf, lm


# --------------------------------------------------------------- retouches
def features_mask(shape, lm, fw):
    """Yeux, sourcils, lèvres, narines : zones qu'on ne traite jamais comme de la peau."""
    m = np.zeros(shape[:2], np.float32)
    for up, lo in ((EYE_R_UP, EYE_R_LO), (EYE_L_UP, EYE_L_LO)):
        m = np.maximum(m, poly_mask(shape, np.vstack([lm[up], lm[lo][::-1]])))
    for idx in (BROW_R, BROW_L, LIPS, *NOSTRILS):
        m = np.maximum(m, poly_mask(shape, lm[idx]))
    k = max(3, int(fw / 40)) | 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return gauss(m, fw / 120)


def bronzage(img, skin, level):
    if level <= 0:
        return img
    k = level / 4.0
    lab = to_lab(img)
    L, a, b = cv2.split(lab)
    # Les reflets très clairs (brillance) sont moins assombris, pour garder le relief.
    hl = np.clip((L - 80) / 18, 0, 1)
    L2 = L * (1 - 0.15 * k * (1 - 0.5 * hl))
    a2 = a + 3.5 * k
    b2 = b + 11.0 * k
    out = from_lab(cv2.merge([L2, a2, b2]))
    return blend(img, out, skin)


def acne(img, face, feat, fw, level, protect=None):
    if level <= 0:
        return img
    lab = to_lab(img)
    L, a, b = cv2.split(lab)
    zone = face > 0.5
    if zone.sum() < 100:
        return img

    # 1) Unifier les rougeurs diffuses (joues, menton) sans toucher aux taches de rousseur.
    a_low = gauss(a, fw / 30)
    med_a = np.median(a[zone])
    excess = np.clip(a_low - med_a, 0, None)
    a = a - excess * (0.13 * level) * face

    # 2) Repérer chaque bouton : petite zone plus rouge (et souvent plus sombre ou plus claire) que la peau autour.
    big = fw / 22
    a_hp = a - gauss(a, big)
    L_hp = L - gauss(L, big)
    score = a_hp + 0.45 * np.clip(-L_hp, 0, None) * (a_hp > -0.5) + 0.15 * np.clip(L_hp, 0, None) * (a_hp > 0)
    vals = score[zone]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    thr = med + (3.2 - 0.5 * level) * mad
    e = max(3, int(fw / 22)) | 1
    core = cv2.erode(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))) > 0
    f = max(3, int(fw / 14)) | 1
    near_feat = cv2.dilate((feat > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (f, f))) > 0
    core &= ~near_feat
    spots = ((score > thr) & core).astype(np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # on ne garde que des taches de taille "bouton" (pas de grandes zones)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(spots)
    max_area = (fw / 10) ** 2
    keep = np.zeros(n, bool)
    bw, bh = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    elong = np.maximum(bw, bh) / np.maximum(1, np.minimum(bw, bh))
    fill = stats[1:, cv2.CC_STAT_AREA] / np.maximum(1, bw * bh)
    small = np.maximum(bw, bh) < fw / 25
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] < max_area) & (((elong < 4) & (fill > 0.3)) | small)
    if protect is not None:
        # zone des fossettes : on n'enlève que les petits boutons ronds, jamais le creux de la fossette
        in_dimple = np.zeros(n, bool)
        if n > 1:
            pv = np.bincount(lab_cc.ravel(), weights=(protect > 0.3).ravel().astype(np.float64), minlength=n)
            in_dimple = pv > 0
        small_round = np.zeros(n, bool)
        small_round[1:] = (stats[1:, cv2.CC_STAT_AREA] < (fw / 28) ** 2) & (elong < 2.2)
        keep &= ~in_dimple | small_round
    spots = keep[lab_cc].astype(np.uint8)
    d = max(3, int(fw / 110)) | 1
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    if os.environ.get('HYPIC_DEBUG'):
        Image.fromarray(spots * 255).save(os.environ['HYPIC_DEBUG'])
    work = from_lab(cv2.merge([L, a, b]))
    if spots.any():
        # 3) Reboucher avec la peau autour, puis remettre le grain fin d'origine.
        filled = cv2.inpaint(u8(work), spots, max(3, int(fw / 60)), cv2.INPAINT_TELEA).astype(np.float32) / 255
        g = max(1.0, fw / 350)
        filled = gauss(filled, g) + (work - gauss(work, g)) * 0.7
        m = gauss(spots.astype(np.float32), max(1.0, fw / 300)) * min(1.0, 0.35 + 0.18 * level)
        work = blend(work, np.clip(filled, 0, 1), m * face)
    return work, int(spots.max() and cv2.connectedComponents(spots)[0] - 1)


def lissage(img, face, fw, level):
    if level <= 0:
        return img
    g = max(1.0, fw / 380)
    base = gauss(img, g)
    fine = img - base                      # grain et pores : conservés
    smooth = cv2.bilateralFilter(base, 15, 0.07, fw / 45)
    s = min(1.0, 0.22 * level)
    out = base + (smooth - base) * s + fine
    return blend(img, np.clip(out, 0, 1), face)


def visible_eyes(lm):
    """Visage de profil : l'œil du fond paraît beaucoup plus étroit. On ne retouche que les yeux bien visibles."""
    wr = np.linalg.norm(lm[133] - lm[33])
    wl = np.linalg.norm(lm[362] - lm[263])
    m = max(wr, wl, 1e-6)
    return wr / m > 0.62, wl / m > 0.62


def _eye_frame(lm, up):
    p0, p1 = lm[up[0]], lm[up[-1]]
    ux = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-6)
    vy = np.array([-ux[1], ux[0]])
    # "haut" de l'œil = côté du sourcil (repère fiable même de profil)
    mid = lm[up].mean(0)
    brows = [lm[BROW_R].mean(0), lm[BROW_L].mean(0)]
    brow = min(brows, key=lambda b: np.linalg.norm(b - mid))
    if np.dot(brow - mid, vy) < 0:
        vy = -vy
    return p0, ux, vy


def cils(img, lm, fw, level):
    """Allonge légèrement les cils du haut (étirement depuis la paupière) + mascara discret."""
    if level <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    vis = visible_eyes(lm)
    for (up, lo), ok in zip(((EYE_R_UP, EYE_R_LO), (EYE_L_UP, EYE_L_LO)), vis):
        if not ok:
            continue
        p0, ux, vy = _eye_frame(lm, up)
        width = np.linalg.norm(lm[up[-1]] - lm[up[0]])
        eye_h = np.linalg.norm(lm[up[4]] - lm[lo[4]])
        pts_u = (lm[up] - p0) @ ux
        pts_v = (lm[up] - p0) @ vy
        order = np.argsort(pts_u)
        pts_u, pts_v = pts_u[order], pts_v[order]

        Z = max(eye_h * 1.3, width * 0.28)         # zone d'influence au-dessus de la paupière
        x0, x1 = int(max(0, min(lm[up][:, 0].min(), lm[lo][:, 0].min()) - Z)), int(min(w, max(lm[up][:, 0].max(), lm[lo][:, 0].max()) + Z))
        y0, y1 = int(max(0, lm[up][:, 1].min() - 2 * Z)), int(min(h, lm[lo][:, 1].max() + Z))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rel = np.stack([xx - p0[0], yy - p0[1]], -1)
        U = rel @ ux
        V = rel @ vy
        Vlid = np.interp(U, pts_u, pts_v)
        d = V - Vlid                                 # >0 : au-dessus de la paupière
        t = np.clip(U / (width + 1e-6), 0, 1)
        # plus d'effet vers l'extérieur de l'œil, rien aux coins
        inner_is_start = np.linalg.norm(lm[up[0]] - lm[1]) < np.linalg.norm(lm[up[-1]] - lm[1])
        tout = 1 - t if inner_is_start else t
        along = np.clip(np.sin(np.pi * np.clip((U + 0.05 * width) / (1.1 * width), 0, 1)), 0, 1) * (0.7 + 0.5 * tout)
        g = min(0.6, 0.2 * level) * along
        inside = (d > 0) & (d < Z)
        dsrc = np.where(inside, d / (1 + g * (1 - d / Z)), d)
        Vsrc = Vlid + dsrc
        src = (p0 + U[..., None] * ux + Vsrc[..., None] * vy).astype(np.float32)
        patch = cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        patch = np.clip(patch, 0, 1)

        # mascara : on fonce uniquement ce qui est déjà un cil (pixels plus sombres que la peau voisine)
        lab = to_lab(patch)
        L = lab[..., 0]
        Lloc = gauss(L, max(2.0, eye_h / 3))
        lash = np.clip((Lloc - L) / 9, 0, 1)
        band = np.clip(1 - np.abs(d - 0.25 * Z) / (0.45 * Z), 0, 1) * (along > 0.05)
        line = np.clip(1 - np.abs(d) / max(1.5, eye_h * 0.09), 0, 1) * np.clip(along, 0, 1)
        dark = (0.36 * level) * lash * band + (0.12 * level) * line
        dark = np.clip(dark, 0, 0.7)
        lab[..., 0] = L * (1 - dark)
        lab[..., 1] *= 1 - dark
        lab[..., 2] *= 1 - dark
        patch = from_lab(lab)

        fade = np.clip(1 - np.abs(d - 0.2 * Z) / (1.0 * Z), 0, 1)
        fade = np.clip(fade * 3, 0, 1) * np.clip(along * 3, 0, 1)
        out[y0:y1, x0:x1] = blend(out[y0:y1, x0:x1], patch, fade.astype(np.float32))
    return out


def sourcils(img, lm, fw, level):
    """Nettoie les poils qui dépassent autour du sourcil (affinage très léger)."""
    if level <= 0:
        return img
    out = img
    L = to_lab(img)[..., 0]
    for idx, ok in zip((BROW_R, BROW_L), visible_eyes(lm)):
        if not ok:
            continue
        pts = lm[idx]
        c = pts.mean(0)
        core_pts = c + (pts - c) * np.array([1.0, 1 - 0.06 * level])
        core = poly_mask(img.shape, core_pts)
        thick = max(3.0, np.ptp(pts[:, 1]))
        k = max(3, int(thick * (0.6 + 0.2 * level))) | 1
        around = cv2.dilate(poly_mask(img.shape, pts), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ring = np.clip(around - cv2.dilate(core, np.ones((3, 3), np.uint8)), 0, 1)
        skinL = cv2.medianBlur(u8(L / 100), max(5, int(thick) | 1)).astype(np.float32) / 255 * 100
        stray = (((skinL - L) > 3) & (ring > 0)).astype(np.uint8)
        # on n'efface que les petits poils isolés, jamais un morceau du sourcil lui-même
        core_d = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8))
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(stray)
        keep = np.zeros(n_, bool)
        if n_ > 1:
            big = st_[1:, cv2.CC_STAT_AREA] > thick * thick * 0.35
            keep[1:] = ~big
        stray = keep[lab_] & (stray > 0)
        stray &= ~(cv2.dilate(core_d, np.ones((3, 3), np.uint8)) > 0) | (st_[lab_, cv2.CC_STAT_AREA] < thick * 0.8)
        if not stray.any():
            continue
        filled = cv2.inpaint(u8(out), stray.astype(np.uint8), max(3, int(thick / 3)), cv2.INPAINT_TELEA).astype(np.float32) / 255
        g = max(1.0, fw / 400)
        filled = gauss(filled, g) + (out - gauss(out, g)) * 0.6
        m = gauss(stray.astype(np.float32), 1.0) * min(0.85, 0.3 + 0.25 * level)
        out = blend(out, np.clip(filled, 0, 1), m)
    return out


def cheveux(img, conf, fw, level):
    """Cheveux plus lisses + suppression des petits cheveux qui dépassent sur le fond."""
    if level <= 0:
        return img
    hair = conf[1]
    out = img
    # 1) lissage doux de la masse de cheveux (on garde les mèches)
    smooth = cv2.bilateralFilter(img, 9, 0.08, max(3.0, fw / 60))
    inner = gauss(cv2.erode((hair > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), 2)
    out = blend(out, smooth, inner * min(0.6, 0.15 * level))

    # 2) frisottis : fins cheveux isolés autour de la silhouette, sur le fond
    hb = (hair > 0.4).astype(np.uint8)
    r_open = max(3, int(fw * (0.04 + 0.02 * min(level, 3)))) | 1
    main = cv2.morphologyEx(hb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_open, r_open)))
    main = cv2.dilate(main, np.ones((3, 3), np.uint8))
    reach = max(5, int(fw * (0.12 + 0.03 * level))) | 1
    zone = cv2.dilate(hb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) & (1 - main)
    protect = (conf[2] + conf[3] + conf[4] + conf[5]) > 0.3
    zone = (zone > 0) & ~protect
    if not zone.any():
        return out
    k = max(5, int(fw / 35)) | 1
    med = cv2.medianBlur(u8(out), min(k, 255) if k <= 5 else 5)
    for _ in range(max(1, k // 5 - 1)):
        med = cv2.medianBlur(med, 5)
    med = med.astype(np.float32) / 255
    diff = np.abs(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) - cv2.cvtColor(med, cv2.COLOR_RGB2GRAY))
    strands = (diff > max(0.012, 0.026 - 0.003 * level)) & zone
    m = gauss(cv2.dilate(strands.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), 1.0)
    m *= min(1.0, 0.55 + 0.2 * level)
    return blend(out, med, m)


LIP_OUT_UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIP_OUT_LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIP_IN_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_IN_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def levres(img, lm, fw, level):
    """Lèvres plus pulpeuses : chaque lèvre est épaissie vers l'extérieur, depuis sa ligne intérieure
    (la bouche, les dents et le sourire ne bougent pas)."""
    if level <= 0:
        return img
    h, w = img.shape[:2]
    p0, p1 = lm[61], lm[291]
    width = np.linalg.norm(p1 - p0)
    if width < 8:
        return img
    ux = (p1 - p0) / width
    vy = np.array([ux[1], -ux[0]])                       # vers le haut (vers le nez)
    if np.dot(lm[1] - p0, vy) < 0:
        vy = -vy

    def curve(idx):
        pu = (lm[idx] - p0) @ ux
        pv = (lm[idx] - p0) @ vy
        o = np.argsort(pu)
        return pu[o], pv[o]

    ou_u, ou_v = curve(LIP_OUT_UP); iu_u, iu_v = curve(LIP_IN_UP)
    ol_u, ol_v = curve(LIP_OUT_LO); il_u, il_v = curve(LIP_IN_LO)
    pad = width * 0.6
    allp = lm[LIP_OUT_UP + LIP_OUT_LO]
    x0, y0 = [int(max(0, v - pad)) for v in allp.min(0)]
    x1, y1 = int(min(w, allp[:, 0].max() + pad)), int(min(h, allp[:, 1].max() + pad))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rel = np.stack([xx - p0[0], yy - p0[1]], -1)
    U, V = rel @ ux, rel @ vy
    Vi_u, Vo_u = np.interp(U, iu_u, iu_v), np.interp(U, ou_u, ou_v)
    Vi_l, Vo_l = np.interp(U, il_u, il_v), np.interp(U, ol_u, ol_v)
    t_u = np.maximum(Vo_u - Vi_u, width * 0.03)
    t_l = np.maximum(Vi_l - Vo_l, width * 0.03)
    along = np.clip(np.sin(np.pi * np.clip((U + 0.04 * width) / (1.08 * width), 0, 1)), 0, 1) ** 0.6
    g = 0.075 * level * along

    Vsrc = V.copy()
    du = V - Vi_u                                         # au-dessus de la ligne intérieure du haut
    Zu = t_u * 2.2
    m_u = (du > 0) & (du < Zu)
    Vsrc = np.where(m_u, Vi_u + du / (1 + g * (1 - du / Zu)), Vsrc)
    dl = Vi_l - V                                         # en dessous de la ligne intérieure du bas
    Zl = t_l * 2.2
    m_l = (dl > 0) & (dl < Zl)
    Vsrc = np.where(m_l, Vi_l - dl / (1 + g * (1 - dl / Zl)), Vsrc)
    src = (p0 + U[..., None] * ux + Vsrc[..., None] * vy).astype(np.float32)
    patch = cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(patch, 0, 1)
    return out


def cernes(img, lm, fw, face, level):
    """Atténue poches et cernes : éclaircit l'ombre sous l'œil vers le teint de la pommette et lisse le relief."""
    if level <= 0:
        return img
    h, w = img.shape[:2]
    k = min(1.0, 0.3 * level)
    lab = to_lab(img)
    out_lab = lab.copy()
    total = np.zeros((h, w), np.float32)
    vis = visible_eyes(lm)
    for (up, lo), ok in zip(((EYE_R_UP, EYE_R_LO), (EYE_L_UP, EYE_L_LO)), vis):
        if not ok:
            continue
        p0, ux, vy = _eye_frame(lm, up)
        width = np.linalg.norm(lm[up[-1]] - lm[up[0]])
        if width < 6:
            continue
        lu = (lm[lo] - p0) @ ux
        lv = (lm[lo] - p0) @ vy
        o = np.argsort(lu)
        lu, lv = lu[o], lv[o]
        pad = width * 0.8
        x0, y0 = [int(max(0, v - pad)) for v in lm[lo].min(0)]
        x1, y1 = int(min(w, lm[lo][:, 0].max() + pad)), int(min(h, lm[lo][:, 1].max() + pad * 1.2))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        rel = np.stack([xx - p0[0], yy - p0[1]], -1)
        U, V = rel @ ux, rel @ vy
        d = np.interp(U, lu, lv) - V                      # >0 : sous la paupière inférieure
        tu = U / width
        # bande sous l'œil : commence juste sous les cils, s'arrête au haut de la pommette
        across = np.clip(np.sin(np.pi * np.clip((tu + 0.05) / 1.15, 0, 1)), 0, 1)
        band = np.clip((d - 0.04 * width) / (0.06 * width), 0, 1) * np.clip((0.5 * width - d) / (0.18 * width), 0, 1)
        m = gauss((band * across).astype(np.float32), width * 0.06)
        total[y0:y1, x0:x1] = np.maximum(total[y0:y1, x0:x1], m)
    total *= np.clip(face * 1.5, 0, 1)
    if total.max() <= 0:
        return img
    # teint de référence : la peau juste en dessous (pommette), en basse fréquence
    eyew = max(np.linalg.norm(lm[133] - lm[33]), np.linalg.norm(lm[362] - lm[263]))
    big = gauss(lab, eyew * 0.5)
    ref = np.roll(big, int(eyew * 0.55), axis=0)          # la pommette est sous la cerne
    low = gauss(lab, eyew * 0.12)
    dL = np.clip(ref[..., 0] - low[..., 0], 0, None)
    out_lab[..., 0] = lab[..., 0] + dL * k * total
    # teinte : on corrige seulement un peu (jamais plus de 3 points), pour ne pas créer de reflets bleus ou verts
    out_lab[..., 1] = lab[..., 1] + np.clip(ref[..., 1] - low[..., 1], -3, 3) * 0.6 * k * total
    out_lab[..., 2] = lab[..., 2] + np.clip(ref[..., 2] - low[..., 2], -3, 3) * 0.6 * k * total
    out = from_lab(out_lab)
    # lisser le relief de la poche (moyennes fréquences), grain fin conservé
    g = max(1.0, fw / 380)
    base, fine = gauss(out, g), out - gauss(out, g)
    smooth = cv2.bilateralFilter(base, 9, 0.06, eyew * 0.15)
    out = np.clip(base + (smooth - base) * min(1.0, 0.32 * level) + fine, 0, 1)
    return blend(img, out, total)


def dents(img, lm, fw, level):
    """Blanchit les dents (uniquement les zones claires dans la bouche ouverte, pas les gencives ni la langue)."""
    if level <= 0:
        return img
    inner = poly_mask(img.shape, np.vstack([lm[LIP_IN_UP], lm[LIP_IN_LO][::-1]]))
    if inner.sum() < 20:
        return img
    lab = to_lab(img)
    L, a, b = cv2.split(lab)
    mouth = inner > 0
    Lm = L[mouth]
    thr = max(35.0, np.percentile(Lm, 40))
    teeth = (inner > 0) & (L > thr) & (a < 14)
    teeth = cv2.morphologyEx(teeth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if teeth.sum() < 10:
        return img
    m = gauss(teeth.astype(np.float32), max(0.8, fw / 400)) * inner
    k = level / 4.0
    b2 = b - np.clip(b, 0, None) * min(0.75, 0.6 * k + 0.1)
    a2 = a - np.clip(a, 0, None) * 0.35 * k
    L2 = L + (100 - L) * 0.12 * k
    out = from_lab(cv2.merge([L2, a2, b2]))
    return blend(img, out, m)


def fossettes_mask(shape, lm, fw):
    """Zone des fossettes (à côté des coins de la bouche) : on les PROTÈGE, jamais lissées ni rebouchées."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    center = (lm[61] + lm[291]) / 2
    for corner, ok in zip((61, 291), visible_eyes(lm)):
        if not ok:
            continue
        out_dir = lm[corner] - center
        c = lm[corner] + out_dir * 0.55 + (lm[1] - lm[152]) * 0.03
        cv2.ellipse(m, (int(c[0]), int(c[1])), (max(1, int(fw * 0.09)), max(1, int(fw * 0.12))), 0, 0, 360, 1.0, -1)
    return gauss(m, fw * 0.03)


def gloss(img, lm, fw, level):
    """Effet gloss : lèvres plus brillantes (on renforce leurs reflets) + petit reflet doux au centre de la lèvre du bas."""
    if level <= 0:
        return img
    k = level / 4.0
    lips = np.clip(poly_mask(img.shape, lm[LIPS]) - poly_mask(img.shape, np.vstack([lm[LIP_IN_UP], lm[LIP_IN_LO][::-1]])), 0, 1)
    if lips.sum() < 20:
        return img
    lips = gauss(lips, max(0.8, fw / 300))
    lab = to_lab(img)
    L, a, b = cv2.split(lab)
    # reflets existants amplifiés
    hp = np.clip(L - gauss(L, fw / 40), 0, None)
    hp = np.clip(hp / (np.percentile(hp[lips > 0.5], 97) + 1e-6), 0, 1) if (lips > 0.5).any() else hp
    # reflet doux au milieu de la lèvre du bas (là où la lumière tombe naturellement)
    lo_mid = lm[14] + (lm[17] - lm[14]) * 0.4
    spot = np.zeros(L.shape, np.float32)
    mw = np.linalg.norm(lm[291] - lm[61])
    cv2.ellipse(spot, (int(lo_mid[0]), int(lo_mid[1])), (max(1, int(mw * 0.16)), max(1, int(mw * 0.045))), 0, 0, 360, 1.0, -1)
    spot = gauss(spot, mw * 0.04)
    shine = np.clip(hp * 0.8 + spot * 0.6, 0, 1) * lips
    L2 = L + (100 - L) * 0.45 * k * shine
    # lèvres un peu plus "mouillées" : couleur légèrement plus riche
    a2 = a * (1 + 0.12 * k * lips)
    b2 = b * (1 + 0.05 * k * lips)
    return from_lab(cv2.merge([L2, a2, b2]))


def liner(img, lm, fw, level):
    """Contour des lèvres légèrement marqué, couleur naturelle (un ton plus profond que la lèvre)."""
    if level <= 0:
        return img
    k = level / 4.0
    h, w = img.shape[:2]
    mw = np.linalg.norm(lm[291] - lm[61])
    if mw < 10:
        return img
    m = np.zeros((h, w), np.float32)
    th = max(1, int(round(mw * 0.03)))
    for idx in (LIP_OUT_UP, LIP_OUT_LO):
        cv2.polylines(m, [np.round(lm[idx]).astype(np.int32)], False, 1.0, th, cv2.LINE_AA)
    # plus discret aux coins de la bouche
    corners = np.zeros((h, w), np.float32)
    for c in (61, 291):
        cv2.circle(corners, (int(lm[c][0]), int(lm[c][1])), max(1, int(mw * 0.08)), 1.0, -1)
    m = gauss(m, th * 0.7) * (1 - 0.7 * gauss(corners, mw * 0.04))
    lab = to_lab(img)
    L, a, b = cv2.split(lab)
    L2 = L * (1 - 0.24 * k * m)
    a2 = a + 5.0 * k * m                                  # rosé-brun naturel
    b2 = b + 1.5 * k * m
    return from_lab(cv2.merge([L2, a2, b2]))


def maquillage(img, lm, fw, face, level):
    """Maquillage très naturel : blush sur les pommettes, contouring léger sous les pommettes et sur les côtés du nez,
    petit éclat sur l'arête du nez."""
    if level <= 0:
        return img
    h, w = img.shape[:2]
    k = level / 4.0
    across = lm[263] - lm[33]
    across /= np.linalg.norm(across) + 1e-6
    lab = to_lab(img)
    L, a, b = cv2.split(lab)

    def blob(center, rx, ry, angle):
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(center[0]), int(center[1])), (max(1, int(rx)), max(1, int(ry))), angle, 0, 360, 1.0, -1)
        return gauss(m, max(rx, ry) * 0.45)

    def stroke(p, q, width_):
        m = np.zeros((h, w), np.float32)
        cv2.line(m, tuple(int(v) for v in p), tuple(int(v) for v in q), 1.0, max(1, int(width_)))
        return gauss(m, width_ * 0.8)

    ang = float(np.degrees(np.arctan2(across[1], across[0])))
    # blush : pommes des joues (entre l'œil et le coin de la bouche, vers l'extérieur)
    vis = visible_eyes(lm)
    blush = np.zeros((h, w), np.float32)
    for (eye_out, mouth_c, side), ok in zip(((33, 61, 234), (263, 291, 454)), vis):
        if not ok:
            continue
        c = 0.45 * lm[eye_out] + 0.30 * lm[mouth_c] + 0.25 * lm[side]
        blush = np.maximum(blush, blob(c, fw * 0.13, fw * 0.09, ang - 15 if side == 234 else ang + 15))
    # contour : sous l'os de la pommette, de l'oreille vers le coin de la bouche (s'arrête à mi-chemin)
    contour = np.zeros((h, w), np.float32)
    for (side, mouth_c, eye_out), ok in zip(((234, 61, 33), (454, 291, 263)), vis):
        if not ok:
            continue
        start = lm[side] + (lm[mouth_c] - lm[side]) * 0.05 + (lm[mouth_c] - lm[eye_out]) * 0.15
        end = lm[side] + (lm[mouth_c] - lm[side]) * 0.5
        contour = np.maximum(contour, stroke(start, end, fw * 0.07))
    # nez : deux traits fins le long des côtés de l'arête + petit éclat au centre
    top, tip = lm[168], lm[195]
    off = across * fw * 0.075
    light = np.zeros((h, w), np.float32)
    if all(vis):                                          # nez : seulement visage de face ou de 3/4
        for sgn in (-1, 1):
            contour = np.maximum(contour, stroke(top + sgn * off, tip + sgn * off * 0.85, fw * 0.035) * 0.8)
        light = stroke(lm[6], lm[195], fw * 0.03)
    fm = np.clip(face * 1.2, 0, 1)
    blush *= fm; contour *= fm; light *= fm
    a = a + 7.0 * k * blush
    b = b + 1.0 * k * blush
    L = L * (1 - 0.03 * k * blush)
    L = L * (1 - 0.09 * k * contour)
    b = b + 2.0 * k * contour
    L = L + (100 - L) * 0.06 * k * light
    return from_lab(cv2.merge([L, a, b]))


IRIS_R, IRIS_L = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]


def iris_eclat(img, lm, fw, level):
    """Éclat des yeux : iris plus lumineux, plus colorés et plus nets ; blanc de l'œil légèrement éclairci."""
    if level <= 0 or lm.shape[0] < 478:
        return img
    k = level / 4.0
    h, w = img.shape[:2]
    out = img.copy()
    vis = visible_eyes(lm)
    if not all(vis):                                      # de profil, l'iris est mal repéré : on ne touche pas
        return img
    for (up, lo, ir), ok in zip(((EYE_R_UP, EYE_R_LO, IRIS_R), (EYE_L_UP, EYE_L_LO, IRIS_L)), vis):
        if not ok:
            continue
        opening = poly_mask(img.shape, np.vstack([lm[up], lm[lo][::-1]]))
        if opening.sum() < 15:
            continue
        c = lm[ir[0]]
        r = float(np.mean(np.linalg.norm(lm[ir[1:]] - c, axis=1)))
        cy, cx = int(round(c[1])), int(round(c[0]))
        if r < 2 or not (0 <= cy < h and 0 <= cx < w) or opening[cy, cx] < 0.5:
            continue                                      # iris pas clairement dans l'œil : on s'abstient
        x0, y0 = int(max(0, c[0] - 3 * r)), int(max(0, c[1] - 3 * r))
        x1, y1 = int(min(w, c[0] + 3 * r)), int(min(h, c[1] + 3 * r))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dist = np.hypot(xx - c[0], yy - c[1])
        op = gauss(opening[y0:y1, x0:x1], max(0.6, r * 0.08))
        ring = np.clip((r * 1.02 - dist) / max(1.0, r * 0.12), 0, 1)          # iris
        ring *= np.clip((dist - r * 0.32) / max(1.0, r * 0.12), 0, 1)         # sans la pupille
        ring *= op
        sclera = np.clip((dist - r * 1.05) / max(1.0, r * 0.2), 0, 1) * op
        patch = img[y0:y1, x0:x1]
        lab = to_lab(patch)
        L, a, b = cv2.split(lab)
        # iris : plus clair, plus de couleur, plus de détail
        detail = L - gauss(L, max(0.8, r * 0.15))
        L_i = L * (1 + 0.3 * k) + 7 * k + detail * 1.3 * k
        a_i, b_i = a * (1 + 0.55 * k), b * (1 + 0.55 * k)
        # blanc de l'œil : un peu plus clair, moins rouge/jaune
        L_s = L + (100 - L) * 0.16 * k
        a_s, b_s = a * (1 - 0.35 * k), b * (1 - 0.25 * k)
        Lr = L + (L_i - L) * ring + (L_s - L) * sclera
        ar = a + (a_i - a) * ring + (a_s - a) * sclera
        br = b + (b_i - b) * ring + (b_s - b) * sclera
        out[y0:y1, x0:x1] = from_lab(cv2.merge([Lr.astype(np.float32), ar.astype(np.float32), br.astype(np.float32)]))
    return out


def glow(img, face, skin, fw, level):
    """Glow : lumière douce et diffuse sur la peau (effet peau lumineuse), sans perdre les détails."""
    if level <= 0:
        return img
    k = level / 4.0
    soft = gauss(img, fw / 22)
    lum = cv2.cvtColor(soft, cv2.COLOR_RGB2GRAY)[..., None]
    # écran doux : les zones déjà claires diffusent un peu de lumière autour d'elles
    screen = 1 - (1 - img) * (1 - soft * (0.35 * k) * np.clip(lum * 1.3, 0, 1))
    warm = screen * np.array([1.0 + 0.012 * k, 1.0, 1.0 - 0.015 * k], np.float32)
    mask = np.clip(face + 0.5 * np.clip(skin - face, 0, 1), 0, 1)
    return blend(img, np.clip(warm, 0, 1), mask * min(1.0, 0.5 + 0.25 * level))


def highlights(img, lm, fw, face, level):
    """Brillance / highlights : renforce les points qui captent déjà la lumière (pommettes, arête du nez,
    arc de Cupidon, menton, arcade) — on amplifie la lumière existante au lieu de peindre des taches."""
    if level <= 0:
        return img
    h, w = img.shape[:2]
    k = level / 4.0
    lab = to_lab(img)
    L = lab[..., 0]
    base = gauss(L, fw / 60)
    hp = np.clip(base - gauss(L, fw / 10), 0, None)       # zones plus claires que leur entourage
    hp = hp / (np.percentile(hp[face > 0.5], 99) + 1e-6) if (face > 0.5).any() else hp
    hp = np.clip(hp, 0, 1)

    def blob(p, rx, ry):
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(p[0]), int(p[1])), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 1.0, -1)
        return gauss(m, max(rx, ry) * 0.6)

    zones = np.zeros((h, w), np.float32)
    vis = visible_eyes(lm)
    for (eye_out, side), ok in zip(((33, 234), (263, 454)), vis):
        if ok:                                            # haut des pommettes
            zones = np.maximum(zones, blob(0.55 * lm[eye_out] + 0.25 * lm[side] + 0.2 * lm[1] + (lm[1] - lm[168]) * 0.15, fw * 0.1, fw * 0.06))
    zones = np.maximum(zones, blob((lm[6] + lm[195]) / 2, fw * 0.03, fw * 0.12))     # arête du nez
    zones = np.maximum(zones, blob(lm[0] + (lm[0] - lm[13]) * 0.6, fw * 0.04, fw * 0.02))  # arc de Cupidon
    zones = np.maximum(zones, blob(lm[152] + (lm[17] - lm[152]) * 0.4, fw * 0.06, fw * 0.04))  # menton
    weight = np.clip(hp * (0.4 + 0.6 * zones) + 0.25 * zones, 0, 1) * np.clip(face * 1.3, 0, 1)
    lab[..., 0] = L + (100 - L) * (0.22 * k) * weight
    lab[..., 1] *= 1 - 0.08 * k * weight
    lab[..., 2] *= 1 - 0.05 * k * weight
    return from_lab(lab)


# --------------------------------------------------------------- programme
def retouche(path, out_path, s):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    orig = np.asarray(pil).astype(np.float32) / 255
    img = orig
    h, w = img.shape[:2]
    zone = None
    if s.zone:
        # zones "x0,y0,x1,y1" en fractions de l'image : on ne retouche QUE la personne dedans
        zone = np.zeros((h, w), np.float32)
        for z in s.zone:
            x0, y0, x1, y1 = [float(v) for v in z.split(",")]
            zone[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1
        zone = gauss(zone, min(h, w) / 200)
    conf, lm = analyse(img, zone)
    report = []

    if lm is not None:
        fw = float(np.linalg.norm(lm[454] - lm[234]))
        feat = features_mask(img.shape, lm, fw)
        oval = gauss(poly_mask(img.shape, lm[FACE_OVAL]), fw / 60)
    else:
        fw = min(h, w) / 4
        feat = np.zeros((h, w), np.float32)
        oval = np.ones((h, w), np.float32)
        report.append("⚠️ Visage non détecté : seuls bronzage et cheveux appliqués.")

    face = np.clip(conf[3] * (1 - feat) * np.maximum(oval, 0.0), 0, 1)
    skin = np.clip((conf[2] + conf[3]) * (1 - feat), 0, 1)

    if lm is not None:
        # fossettes : protégées (ni acné, ni lissage, ni contouring dessus)
        dimples = fossettes_mask(img.shape, lm, fw)
        face_nd = face * (1 - 0.9 * dimples)
        r = acne(img, face, feat, fw, s.acne, protect=dimples)
        if isinstance(r, tuple):
            img, n = r
            report.append(f"Acné {s.acne} : {n} imperfections traitées + rougeurs unifiées")
        img = lissage(img, face_nd, fw, s.lissage)
        report.append(f"Lissage {s.lissage}")
        img = cernes(img, lm, fw, face, s.cernes)
        report.append(f"Cernes {s.cernes}")
        img = sourcils(img, lm, fw, s.sourcils)
        report.append(f"Sourcils {s.sourcils}")
        img = cils(img, lm, fw, s.cils)
        report.append(f"Cils + mascara {s.cils}")
        img = dents(img, lm, fw, s.dents)
        report.append(f"Dents {s.dents}")
        img = gloss(img, lm, fw, s.gloss)
        report.append(f"Gloss {s.gloss}")
        img = liner(img, lm, fw, s.liner)
        report.append(f"Liner lèvres {s.liner}")
        img = levres(img, lm, fw, s.levres)
        report.append(f"Lèvres {s.levres}")
        img = maquillage(img, lm, fw, face_nd, s.maquillage)
        report.append(f"Maquillage {s.maquillage}")
        img = iris_eclat(img, lm, fw, s.iris)
        report.append(f"Iris / éclat des yeux {s.iris}")
    img = cheveux(img, conf, fw, s.cheveux)
    report.append(f"Cheveux {s.cheveux}")
    img = bronzage(img, skin, s.bronzage)
    report.append(f"Bronzage {s.bronzage}")
    if lm is not None:
        img = highlights(img, lm, fw, face, s.highlights)
        report.append(f"Highlights {s.highlights}")
        img = glow(img, face, skin, fw, s.glow)
        report.append(f"Glow {s.glow}")

    if zone is not None:
        img = blend(orig, img, zone)
        # Les autres personnes : seulement un léger bronzage (par défaut la moitié de celui de Sofie),
        # pour que l'écart de teint ne se remarque pas. Aucune autre retouche sur elles.
        autres = s.bronzage / 2 if s.autres is None else s.autres
        if autres > 0:
            others_skin = np.clip((conf[2] + conf[3]) * (1 - zone), 0, 1)
            img = bronzage(img, others_skin, autres)
            report.append(f"Autres personnes : bronzage {autres} uniquement")
    Image.fromarray(u8(img)).save(out_path, quality=96, subsampling=0)
    if s.comparer:
        before = np.asarray(pil)
        sep = np.full((h, max(4, w // 100), 3), 255, np.uint8)
        cmp_path = os.path.splitext(out_path)[0] + "_avant-apres.jpg"
        Image.fromarray(np.hstack([before, sep, u8(img)])).save(cmp_path, quality=92)
        report.append(f"Comparaison : {cmp_path}")
    return report


def main():
    p = argparse.ArgumentParser(description="Hypic maison : retouche portrait (zones notées de 0 à 4).")
    p.add_argument("image")
    p.add_argument("-o", "--sortie")
    for name, default in (("bronzage", 1.5), ("acne", 4), ("lissage", 1), ("cils", 1), ("sourcils", 0.5), ("cheveux", 1),
                          ("levres", 0), ("cernes", 0), ("dents", 0), ("maquillage", 0),
                          ("glow", 0), ("iris", 0), ("highlights", 0), ("gloss", 0), ("liner", 0)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--zone", action="append", help="x0,y0,x1,y1 en fractions (0-1) : retoucher seulement cette zone (répétable)")
    p.add_argument("--autres", type=float, default=None,
                   help="bronzage des autres personnes (hors zone), 0-4 ; par défaut la moitié du bronzage de Sofie")
    p.add_argument("--comparer", action="store_true", help="crée aussi une image avant/après")
    s = p.parse_args()
    for name in ("bronzage", "acne", "lissage", "cils", "sourcils", "cheveux", "autres"):
        v = getattr(s, name)
        if v is None:
            continue
        if not 0 <= v <= 4:
            sys.exit(f"{name} doit être entre 0 et 4")
    out = s.sortie or os.path.splitext(s.image)[0] + "_hypic.jpg"
    for line in retouche(s.image, out, s):
        print("•", line)
    print("✅", out)


if __name__ == "__main__":
    main()
