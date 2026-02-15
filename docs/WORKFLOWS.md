# Workflows GitHub Actions (AstroOracle)

Objectif
- Avoir des boutons "Run workflow" partout ou c'est utile.
- Garder un CI rapide sur push/PR.
- Ajouter 2 workflows bases sur un petit echantillon de donnees reelles (Gaia-derived), sans rendre le repo lourd.

Workflows a conserver
- CI: tests + lint + coverage sur push/PR + bouton manual.
- Docs: build Sphinx sur push/PR + bouton manual.
- Release: build + PyPI Trusted Publishing sur tags v* + bouton manual.
- demo-visuals: generation d'artefacts de demo (donnees synthetiques) sur bouton manual.
- AstroOracle Batch HTML (commit): genere docs/batch_out et le pousse sur une branche de publication, bouton + schedule.
- Real data smoke (offline coords): batch-html sur un echantillon reel, coords pseudo deterministes (0 dependance reseau).
- Real data smoke (Gaia coords via astroquery): coords reelles via Gaia DR3, bouton manual.
- Real data batch (publish branch): genere docs/batch_out a partir du sample reel et pousse sur une branche, bouton + schedule.

Nettoyage
- Si tu as d'autres fichiers YAML dans .github/workflows, supprime-les (ils se rajoutent dans le menu Actions et brouillent la lisibilite).
- "Dependabot Updates" apparait dans Actions a cause de .github/dependabot.yml. Si tu ne veux pas de dependabot, supprime ce fichier.

Donnees reelles
- test_data/real/vari_summary_sample.csv.gz est un extrait d'une table Gaia de variabilite (2000 lignes).
- tools/real_data/build_candidates_from_ecsv.py construit candidates.parquet a partir de ce fichier, avec deux modes coords:
  - pseudo (offline)
  - gaia (astroquery, reseau, Gaia DR3)
