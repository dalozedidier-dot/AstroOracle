Workflows à conserver
=====================

Dans `.github/workflows/`, garde uniquement :

- `ci.yml` (CI)
- `docs.yml` (Docs)
- `release.yml` (Release)
- `batch_html_commit.yml` (rafraîchit le rapport offline versionné)
- `real_data_offline.yml` (données réelles, offline)
- `real_data_gaia.yml` (données réelles, RA/Dec Gaia)
- `real_data_publish.yml` (publie un rapport real_demo dans docs/)

Supprime le reste des YAML si tu en as d'autres, sinon tu vas continuer à voir des workflows en double dans l'onglet Actions.

Note Dependabot
---------------

Si tu veux faire disparaitre “Dependabot Updates” du menu Actions, supprime `.github/dependabot.yml` et désactive Dependabot dans les settings du repo.
