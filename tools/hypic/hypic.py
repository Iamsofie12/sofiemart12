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
def analyse(img):
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
        base_options=BaseOptions(model_asset_path=model_path("face")), num_faces=1))
    res = fl.detect(mpi)
    seg.close()
    fl.close()
    lm = None
    if res.face_landmarks:
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
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


def acne(img, face, feat, fw, level):
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


def _eye_frame(lm, up):
    p0, p1 = lm[up[0]], lm[up[-1]]
    ux = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-6)
    vy = np.array([-ux[1], ux[0]])
    # "haut" de l'œil = côté où se trouve la paupière supérieure
    mid = lm[up].mean(0)
    if np.dot(mid - (p0 + p1) / 2, vy) > 0:
        vy = -vy
    return p0, ux, vy


def cils(img, lm, fw, level):
    """Allonge légèrement les cils du haut (étirement depuis la paupière) + mascara discret."""
    if level <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    for up, lo in ((EYE_R_UP, EYE_R_LO), (EYE_L_UP, EYE_L_LO)):
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
        g = min(0.45, 0.13 * level) * along
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
        lash = np.clip((Lloc - L) / 12, 0, 1)
        band = np.clip(1 - np.abs(d - 0.25 * Z) / (0.45 * Z), 0, 1) * (along > 0.05)
        line = np.clip(1 - np.abs(d) / max(1.5, eye_h * 0.09), 0, 1) * np.clip(along, 0, 1)
        dark = (0.24 * level) * lash * band + (0.08 * level) * line
        dark = np.clip(dark, 0, 0.6)
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
    for idx in (BROW_R, BROW_L):
        pts = lm[idx]
        c = pts.mean(0)
        core_pts = c + (pts - c) * np.array([1.0, 1 - 0.07 * level])
        core = poly_mask(img.shape, core_pts)
        thick = max(3.0, np.ptp(pts[:, 1]))
        k = max(3, int(thick * (0.5 + 0.15 * level))) | 1
        around = cv2.dilate(poly_mask(img.shape, pts), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ring = np.clip(around - cv2.dilate(core, np.ones((3, 3), np.uint8)), 0, 1)
        skinL = cv2.medianBlur(u8(L / 100), max(5, int(thick) | 1)).astype(np.float32) / 255 * 100
        stray = ((skinL - L) > 4) & (ring > 0)
        if not stray.any():
            continue
        filled = cv2.inpaint(u8(out), stray.astype(np.uint8), max(3, int(thick / 3)), cv2.INPAINT_TELEA).astype(np.float32) / 255
        g = max(1.0, fw / 400)
        filled = gauss(filled, g) + (out - gauss(out, g)) * 0.6
        m = gauss(stray.astype(np.float32), 1.0) * min(1.0, 0.45 * level)
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
    r_open = max(3, int(fw * 0.035 * min(level, 3))) | 1
    main = cv2.morphologyEx(hb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_open, r_open)))
    main = cv2.dilate(main, np.ones((3, 3), np.uint8))
    reach = max(5, int(fw * 0.10)) | 1
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
    strands = (diff > 0.03) & zone
    m = gauss(cv2.dilate(strands.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), 1.0)
    m *= min(1.0, 0.4 * level)
    return blend(out, med, m)


# --------------------------------------------------------------- programme
def retouche(path, out_path, s):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = np.asarray(pil).astype(np.float32) / 255
    conf, lm = analyse(img)
    h, w = img.shape[:2]
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
        r = acne(img, face, feat, fw, s.acne)
        if isinstance(r, tuple):
            img, n = r
            report.append(f"Acné {s.acne} : {n} imperfections traitées + rougeurs unifiées")
        img = lissage(img, face, fw, s.lissage)
        report.append(f"Lissage {s.lissage}")
        img = sourcils(img, lm, fw, s.sourcils)
        report.append(f"Sourcils {s.sourcils}")
        img = cils(img, lm, fw, s.cils)
        report.append(f"Cils + mascara {s.cils}")
    img = cheveux(img, conf, fw, s.cheveux)
    report.append(f"Cheveux {s.cheveux}")
    img = bronzage(img, skin, s.bronzage)
    report.append(f"Bronzage {s.bronzage}")

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
    for name, default in (("bronzage", 1.5), ("acne", 4), ("lissage", 1), ("cils", 1), ("sourcils", 0.5), ("cheveux", 1)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.add_argument("--comparer", action="store_true", help="crée aussi une image avant/après")
    s = p.parse_args()
    for name in ("bronzage", "acne", "lissage", "cils", "sourcils", "cheveux"):
        v = getattr(s, name)
        if not 0 <= v <= 4:
            sys.exit(f"{name} doit être entre 0 et 4")
    out = s.sortie or os.path.splitext(s.image)[0] + "_hypic.jpg"
    for line in retouche(s.image, out, s):
        print("•", line)
    print("✅", out)


if __name__ == "__main__":
    main()
