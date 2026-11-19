---
name: hypic
description: Retouche photo portrait de Sofie (bronzage, acné, lissage, cils/mascara, sourcils, cheveux), chaque zone notée de 0 à 4, avec le programme de retouche maison gratuit et illimité (tools/hypic/hypic.py), Canva en complément. Utiliser quand Sofie tape /hypic ou demande de retoucher/améliorer une photo d'elle.
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

## Outil : retouche maison (gratuite, illimitée) — JAMAIS Higgsfield
Programme : `tools/hypic/hypic.py` (Python, 100 % local, aucun crédit).
Installation (une fois) : `pip install -r tools/hypic/requirements.txt` (sous Linux, ajouter `libegl1`).
Les modèles de détection (visage + zones) se téléchargent tout seuls la première fois dans `~/.cache/hypic`.

Commande :
```
python3 tools/hypic/hypic.py PHOTO --bronzage 1.5 --acne 4 --lissage 1 --cils 1 --sourcils 0.5 --cheveux 1 --comparer
```
→ crée `PHOTO_hypic.jpg` (même résolution, JPEG qualité 96) et `PHOTO_hypic_avant-apres.jpg`.

Ce que fait le programme, zone par zone :
- repère le visage (478 points) et sépare peau du visage / peau du corps / cheveux / vêtements / fond ;
- acné : unifie les rougeurs diffuses, détecte chaque bouton (petite zone plus rouge ou plus sombre que la peau voisine), le rebouche avec la peau autour et remet le grain d'origine. Les taches de rousseur sont gardées ;
- lissage : agit sur les moyennes fréquences, garde pores et grain ;
- cils : étire légèrement les cils du haut depuis la paupière + fonce uniquement les cils existants (mascara) ;
- sourcils : nettoie les poils qui dépassent autour ;
- cheveux : lissage doux de la masse + suppression des frisottis qui dépassent sur le fond ;
- bronzage : sur toute la peau détectée, reflets préservés.

## Choisir l'outil selon la photo
| Demande / photo | Outil |
|---|---|
| Cas standard (bronzage, acné, peau, cils, sourcils, cheveux) | **Programme maison** |
| Recadrage, luminosité/couleurs globales, détourage, mise en page (story, post) | **Canva** (connecteur) |
| Retouche impossible proprement en local (ex. objet à effacer du décor, cheveux qui traversent le visage, changement de forme) | Le dire honnêtement à Sofie et proposer Canva ; ne jamais passer par Higgsfield sans son accord explicite |

## Déroulé
1. Récupérer la photo (fichier envoyé). Toujours garder l'original intact.
2. Lancer le programme avec les notes demandées (profil standard sinon) et `--comparer`.
3. **Vérifier soi-même** le résultat en zoomant sur : joues/front/menton (acné), yeux (cils), sourcils, contour des cheveux, bras/jambes (bronzage homogène, pas de démarcation avec les vêtements).
4. Si un défaut se voit (bouton restant, halo, zone ratée), ajuster une note et relancer, ou corriger le programme.
5. Envoyer la photo finale + l'avant/après, résumer les réglages en une ligne, demander s'il faut ajuster.

## Limites honnêtes du programme maison
- Cils : effet "mascara + légèrement plus longs", pas de faux cils dessinés.
- Frisottis : supprimés surtout sur fond uni ; sur fond chargé, l'effet est plus léger.
- Visage non détecté (de dos, très petit, très flou) : seuls bronzage et cheveux s'appliquent.

