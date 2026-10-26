# Changelog

Toutes les évolutions notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et
[Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.1] - 2026-10-26

Durcissement éditorial après l'analyse de tweets publiés de qualité insuffisante.

### Corrigé

- **Notation brute de remplacement** : les flèches et pastilles copiées de la source
  (`↑🟢 Adeyemi, ↓🔴 Gordon`) sont désormais rejetées. Le tweet doit être une phrase
  française : « Double changement pour le Barça à la 84e : Adeyemi et Gabriel Jesus
  entrent, Gordon et Raphinha sortent. »
- **Fait déformé** : un « deuxième but » de la source ne peut plus devenir une
  « égalisation » dans la réécriture. Contrôle ajouté pour `second/2nd goal`,
  `third/3rd goal` et `equaliser / levels the match`.
- **Traductions douteuses** : « levelle le match » → « égalise », « But n°2 » →
  « deuxième but », avec rejet si le modèle réintroduit la formulation fautive.

### Ajouté

- Rappels de correction enrichis lors de la deuxième tentative (flèches, faits exacts).
- Exemples de référence dans le prompt système (changements, buts).

## [1.0.0] - 2026-10-26

Première version publique du bot @UniversCule.

### Ajouté

- Veille de 7 comptes sources X (polling 1 min, sources < 2 min), filtre Barça.
- Réécriture en français via Ollama local (`qwen2.5:7b`) avec style éditorial humain.
- Dédoublonnage sémantique par embeddings (sentence-transformers) sur 48 h.
- Téléchargement et re-publication des médias (images/vidéos) des tweets sources.
- Calendrier des matchs du Barça (source ESPN, cache 6 h, fuseau `Europe/Madrid`)
  et **mode match** : seuls les tweets avec média sont publiés pendant un match.
- File de publication persistante (SQLite) avec créneaux humains, plafond
  journalier et réponses différées (20–30 min).
- Session X persistante : profil navigateur, cookies sauvegardés, vérification du
  compte avant publication ; pause et arrêt de l'ingestion si la session est invalide.
- Commande `setup` : vérification complète (config, `.env`, dépendances, Ollama,
  Chromium, sessions, base).
- `InstanceLock` avec détection de PID mort : redémarrage après crash sans double instance.

### Qualité éditoriale

- Glossaire football français imposé (`égalise`, `remplacement`, `entre en jeu`, `sort`).
- Rejet des anglicismes résiduels (`the`, `goal`, `substitution`, `enters`…).
- Contrôle des faits : tout chiffre, score ou minute absent de la source est rejeté
  (y compris les scores inversés).
- Rejet des élisions cassées (`l' 2e`, `l'2e`), sans confondre le symbole minute (`84'`).
- Typographie française (espace avant `: ; ! ?`) sans casser les heures (`21:30`).
- Deuxième tentative de réécriture puis rejet définitif, motif écrit dans les logs.
- Dédoublonnage d'**événement** après réécriture : même événement = un seul tweet ;
  une version avec média remplace la version sans média encore en file.
- Média annoncé mais non téléchargeable → tweet ignoré (jamais de texte seul à la place).

### Fiabilité

- Limite quotidienne X (`daily limit`, 501) : la file est reportée à la
  réinitialisation (minuit local + 5 min) sans perdre les tweets ni consommer d'essai.
- Plancher d'un post par minute pour éviter les rafales qui déclenchent la limite X.
- `SafeConsole` : une console fermée ou redirigée n'interrompt plus les cycles.

### Sécurité

- `.env`, sessions X, profils navigateur, base SQLite et logs exclus du dépôt.

[1.0.0]: https://github.com/Combocmd/bot_twitter/releases/tag/v1.0.0
[1.0.1]: https://github.com/Combocmd/bot_twitter/releases/tag/v1.0.1
