# Høringsdashboard

En Python-app som henter publiserte høringssvar fra `regjeringen.no`, leser både HTML-svar og PDF-vedlegg, grupperer svarene etter språklig likhet og viser et dashbord over hvem som mener hva.

## Hva appen gjør

- Leser en høringslenke fra `regjeringen.no`
- Finner publiserte svar automatisk
- Henter enkeltstående HTML-svar og PDF-svar
- Trekker ut tekst og metadata
- Lager likhetsgrupper med TF-IDF + k-means
- Viser et dashbord med grupper, mest like svar og komplett svarliste

## Lokal kjøring

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Åpne deretter [http://localhost:8000](http://localhost:8000).

## Railway

Appen er klargjort for Railway med `Procfile` og `railway.json`.

1. Opprett et nytt GitHub-repo
2. Push dette prosjektet
3. Opprett et Railway-prosjekt fra repoet
4. Railway vil installere `requirements.txt` og starte `gunicorn app:app`

### Anbefalt deploy-oppsett

- Bruk GitHub-autodeploy i Railway for vanlig publisering.
- Slå på `Wait for CI` i Railway etter at repoet er koblet til. Repoet har en ferdig workflow i `.github/workflows/ci.yml`.
- Hvis du heller vil trigge deploy manuelt fra GitHub Actions, bruk `.github/workflows/railway-deploy.yml` og legg inn `RAILWAY_TOKEN` som repository secret.

### Eksakte Railway-kommandoer

Installer CLI:

```bash
npm install -g @railway/cli
```

Logg inn:

```bash
railway login
```

Opprett prosjekt og første deploy fra lokal kode:

```bash
railway init
railway up
```

Hvis du vil deploye mot et eksisterende prosjekt fra CI/CD, bruk et prosjekt-token:

```bash
RAILWAY_TOKEN=<token> railway up --ci
```

## GitHub

```bash
git init
git add .
git commit -m "Initial commit: hearing dashboard"
git branch -M main
git remote add origin <din-github-url>
git push -u origin main
```

### Neste GitHub-steg

Når repoet finnes på GitHub, kobler du denne mappen til repoet slik:

```bash
git remote add origin https://github.com/<bruker>/<repo>.git
git push -u origin main
```

Hvis `origin` allerede finnes:

```bash
git remote set-url origin https://github.com/<bruker>/<repo>.git
git push -u origin main
```

## Viktige begrensninger

- Grupperingen er heuristisk og ikke juridisk fasit.
- Noen enkeltkilder kan være midlertidig utilgjengelige fra `regjeringen.no`. Appen forsøker likevel å hente resten og viser importmerknader.
- Standskategoriene (`støtter`, `kritiske`, `betinget`, `uklar`) er regelbaserte og bør sees som førsteutkast.
