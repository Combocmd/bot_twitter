# Rapport de comparaison - Modèle actuel vs Modèle de comparaison

- Modèle actuel : `qwen2.5:7b (test)`
- Modèle de comparaison : `qwen3:8b (test)`
- Nombre de tweets testés : 1

> **Note importante** : Le modèle actuel bénéficie d'une auto-correction
> (jusqu'à 3 tentatives dans `rewrite()`), tandis que le modèle de comparaison
> n'en fait qu'une seule. Les résultats ne sont donc pas strictement comparables
> sur la dimension « nombre de tentatives », mais le sont sur la qualité d'une seule passe.

## Résumé global

| Métrique | Modèle actuel | Modèle comparaison |
|----------|---------------|--------------------|
| Succès technique | 1/1 | 1/1 |
| Succès factuel | 1/1 | 1/1 |
| Succès global | 1/1 | 1/1 |
| Erreurs faits critiques | 0 | 0 |
| Erreurs termes protégés | 0 | 0 |
| Relecture humaine nécessaire | 0 | 0 |
| Temps total (s) | 1.5 | 2.0 |
| Temps moyen par tweet (s) | 1.50 | 2.00 |

## Détail par catégorie

| Catégorie | Nb | Actuel tech | Actuel factuel | Actuel global | Comp tech | Comp factuel | Comp global |
|-----------|----|-------------|--------------|---------------|-----------|--------------|-------------|
| test | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Détails par tweet (premiers 20)

### #1 - test (en)
- Source : Test source
- Attendu : Test attendu
- Actuel (qwen2.5:7b (test)) : Test réécrit
- Comparaison (qwen3:8b (test)) : Test comparé
