---
name: hypic
description: Retouche photo portrait de Sofie selon ses critères standards (bronzage par niveau, acné effacée, peau lissée légèrement, mascara/cils discrets, sourcils affinés très légèrement, cheveux lisses sans frisottis), en gardant la qualité et le naturel de l'image. Utiliser quand Sofie tape /hypic ou demande de retoucher/améliorer une photo d'elle.
---

# /hypic — Retouche photo "standard Sofie"

## Principe
Retouche **naturelle et invisible** : la photo doit rester spontanée, on doit la reconnaître à 100 %.
Toujours partir de l'image originale (jamais d'une version déjà retouchée, sauf demande explicite).

## Échelle d'intensité (la même pour TOUTES les zones)
De `0` à `4`, par pas de 0,5 :
| Note | Intensité |
|---|---|
| `0` | aucune retouche |
| `1` | léger |
| `1.5` | léger-moyen |
| `2` | moyen |
| `2.5` | moyen-fort |
| `3` | fort |
| `3.5` | fort+ |
| `4` | maximum (toujours naturel) |

## Réglages par défaut (profil standard Sofie)
| Zone | Défaut | Ce que fait la note |
|---|---|---|
| Bronzage | `1.5` | 1 = à peine doré · 2 = doré-caramel · 3 = bronzé caramel · 4 = très bronzé, jamais orange |
| Acné | `4` | 4 = plus aucun bouton, rougeur ni marque visible · plus bas = on garde quelques marques légères |
| Lissage peau | `1` | toujours grain et pores conservés ; 4 = peau très unifiée mais pas plastique |
| Cils + mascara | `1` | 1 = cils un peu plus longs, mascara discret · 4 = cils nettement allongés, mascara marqué |
| Sourcils | `0.5` | 0.5 = à peine nettoyés · 4 = nettement affinés, forme gardée |
| Cheveux | `1` | 1 = frisottis supprimés, effet doux · 4 = très lisses et brillants, coiffure inchangée |

## Utilisation
`/hypic [zone note] [zone note] ... [consignes libres]`
- `/hypic` → profil standard
- `/hypic bronzage 2.5 cheveux 2` → change seulement ces zones
- `/hypic bronzage 3 mascara 0 "garde la mèche sur le front"`
Les zones non citées gardent leur valeur par défaut. Toujours rappeler en une ligne les réglages appliqués.

## Règles permanentes
1. Bronzage homogène sur TOUTE la peau visible (visage, cou, décolleté, bras, mains, jambes), sans démarcation, sous-tons naturels, ombres et reflets cohérents avec la lumière.
2. Acné traitée méticuleusement zone par zone (front, joues, menton, mâchoire, tempes).
3. Interdits : modifier les traits, la forme des yeux, la morphologie, l'expression, les vêtements, le décor, le cadrage, la lumière générale. Aucun maquillage en dehors du mascara.
4. Partir toujours de l'original, sauf demande explicite.

## Outil
Higgsfield (`mcp__higgsfield_ia__*`), modèle par défaut `gpt_image_2_6` (édition à partir d'une image de référence). Vérifier les rôles de médias avec `models_explore` si besoin.

## Déroulé
1. **Récupérer la photo** : si le fichier est lisible dans l'environnement → `media_upload` + PUT + `media_confirm`. Sinon → `media_upload_widget` (seul outil du tour) et attendre l'upload.
2. **Lire les dimensions** de l'image originale (garder le même ratio d'aspect en sortie).
3. **Construire le prompt** à partir du modèle ci-dessous, en traduisant chaque note (0-4) en intensité : 0.5-1 = "very subtle", 1.5-2 = "subtle to moderate", 2.5-3 = "noticeable", 3.5-4 = "strong but natural". Une zone à 0 = la retirer du prompt.
4. **Coût** : `estimate_image_cost` → annoncer le coût en crédits. Lancer directement si c'est l'usage habituel (1 image) ; demander validation si plusieurs variantes ou upscale.
5. **Générer** (`generate_image`, count 1 par défaut ; 2 si Sofie veut comparer).
6. **Vérifier** le résultat avant de le présenter : ressemblance, acné vraiment partie, bronzage homogène, pas d'effet plastique, cils discrets. Si un point rate, le dire honnêtement et proposer une correction ciblée.
7. **Option qualité** : proposer `upscale_image` (2k/4k) si la sortie est moins définie que l'originale.
8. Résumer en 3 lignes ce qui a été fait + demander s'il faut ajuster (plus/moins de bronzage, cheveux, etc.).

## Modèle de prompt (anglais, plus fiable pour le modèle)
```
Professional, subtle beauty retouch of this exact photo. Keep the person's identity, facial features,
face shape, body shape, expression, pose, clothing, background, framing, lighting and photo grain
exactly the same. Photorealistic, natural, high resolution, no filter look.
- Skin tone: apply a {TAN_DESCRIPTION} evenly on ALL visible skin (face, neck, chest, arms, hands, legs),
  consistent everywhere, no tan lines, no orange cast, natural undertones, shadows and highlights preserved.
- Acne: carefully remove every pimple, blemish, redness and acne mark, especially on the forehead,
  cheeks, chin, jawline and temples.
- Skin texture: very light smoothing only; keep pores and natural skin texture, no plastic or airbrushed look.
- Eyes: slightly longer and fuller eyelashes with a subtle touch of black mascara; do not change eye shape,
  size or color.
- Eyebrows: very slightly refined and cleaned up, keep their natural shape and thickness almost identical.
- Hair: smoother and sleeker, remove flyaways and frizz, soft subtle shine; keep hairstyle, volume,
  length and color unchanged.
- Do not add any other makeup. Do not alter anything else.
{EXTRA_ADJUSTMENTS}
```
Descriptions de bronzage : 1 = "very light, barely sun-kissed golden tan" · 1.5 = "light-to-medium natural golden tan, like after one week of vacation" · 2 = "medium golden-caramel tan" · 2.5 = "medium-to-deep golden caramel tan" · 3 = "deep natural caramel tan" · 4 = "very deep, rich bronze tan, natural, never orange".

## Limites à rappeler (une fois, brièvement)
- L'IA régénère l'image : de très légères différences de détails sont possibles. Toujours garder l'original.
- Chaque génération consomme des crédits Higgsfield.
