---
name: hypic
description: Retouche photo portrait de Sofie selon ses critères standards (bronzage par niveau, acné effacée, peau lissée légèrement, mascara/cils discrets, sourcils affinés très légèrement, cheveux lisses sans frisottis), en gardant la qualité et le naturel de l'image. Utiliser quand Sofie tape /hypic ou demande de retoucher/améliorer une photo d'elle.
---

# /hypic — Retouche photo "standard Sofie"

## Principe
Retouche **naturelle et invisible** : la photo doit rester spontanée, on doit la reconnaître à 100 %.
Toujours partir de l'image originale (jamais d'une version déjà retouchée, sauf demande explicite).

## Arguments (optionnels)
`/hypic [niveau de bronzage] [ajustements libres]`
- Niveau de bronzage :
  | Code | Niveau | Consigne |
  |---|---|---|
  | `0` | aucun | teint d'origine |
  | `1` | léger | peau à peine dorée, effet "quelques jours au soleil" |
  | `1.5` | **léger-moyen (DÉFAUT actuel)** | doré chaud et uniforme, effet "une semaine de vacances" |
  | `2` | moyen | hâle bien marqué, doré-caramel |
  | `3` | bronzé | hâle profond, caramel |
  | `4` | très bronzé | hâle très soutenu, sans virer orange |
- Ajustements libres : ex. "cheveux plus lisses", "pas de mascara", "plus de lissage".

## Critères standards (appliqués par défaut)
1. **Bronzage** au niveau demandé (défaut 1.5), homogène sur TOUTES les zones de peau visibles (visage, cou, décolleté, bras, mains, jambes) — même teinte partout, aucune démarcation, pas d'orange, sous-tons naturels conservés, ombres et reflets cohérents avec la lumière de la photo.
2. **Acné** : supprimer chaque bouton, rougeur et marque d'acné, surtout sur le visage (front, joues, menton, mâchoire, tempes). Méticuleux, zone par zone.
3. **Lissage de peau très léger** : garder le grain et les pores, aucun effet plastique / filtre.
4. **Yeux** : cils légèrement allongés et densifiés + une touche de mascara noir discrète. Ne pas changer la forme, la taille ou la couleur des yeux.
5. **Sourcils** : affinés très très légèrement (nettoyer les poils épars), forme naturelle conservée.
6. **Cheveux** : lisses et doux, suppression des petits cheveux qui partent dans tous les sens (frisottis), brillance légère. Volume, longueur, couleur et coiffure inchangés. Intensité légère, ajustable selon l'image.
7. **Interdits** : ne pas modifier les traits du visage, la morphologie, le corps, l'expression, les vêtements, le décor, le cadrage ni la lumière générale. Pas de maquillage ajouté en dehors du mascara.

## Outil
Higgsfield (`mcp__higgsfield_ia__*`), modèle par défaut `gpt_image_2_6` (édition à partir d'une image de référence). Vérifier les rôles de médias avec `models_explore` si besoin.

## Déroulé
1. **Récupérer la photo** : si le fichier est lisible dans l'environnement → `media_upload` + PUT + `media_confirm`. Sinon → `media_upload_widget` (seul outil du tour) et attendre l'upload.
2. **Lire les dimensions** de l'image originale (garder le même ratio d'aspect en sortie).
3. **Construire le prompt** à partir du modèle ci-dessous (niveau de bronzage + ajustements).
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
Descriptions de bronzage : 1 = "very light, barely sun-kissed golden tan" · 1.5 = "light-to-medium natural golden tan, like after one week of vacation" · 2 = "medium golden-caramel tan" · 3 = "deep natural caramel tan" · 4 = "very deep, rich bronze tan, natural, never orange".

## Limites à rappeler (une fois, brièvement)
- L'IA régénère l'image : de très légères différences de détails sont possibles. Toujours garder l'original.
- Chaque génération consomme des crédits Higgsfield.
