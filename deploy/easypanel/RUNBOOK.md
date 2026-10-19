# Runbook Deploy — EasyPanel (programatik via MCP/CLI)

Deploy aplikasi ke EasyPanel **tanpa klik UI**: build image di CI → EasyPanel pull image → provisioning lewat **MCP EasyPanel** (utama) atau CLI `easypanel` (fallback). Detail: ADR proyek + [`docs/runbooks/deploy.md`](../../../docs/runbooks/deploy.md) di repo portofolio.

> Prinsip: EasyPanel hanya **pull** image. **Jangan build di server.** Build & push dari CI (mis. GHCR saat push ke `main`).

## 0. Prasyarat

- Server EasyPanel + akun pemilik.
- **API key EasyPanel**: Settings → Server → Users → **Generate API Key**.
- Image sudah terbit di registry (mis. `ghcr.io/<owner>/<app>:latest`). Paket **public**, atau sediakan kredensial registry (username + token) bila privat.
- Domain otomatis server (mis. `*.easypanel.host`) atau domain sendiri yang DNS-nya mengarah ke VPS.

## 1. Sambungkan MCP EasyPanel (sekali per mesin)

Tambahkan ke config opencode global (`~/.config/opencode/opencode.json(c)`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "easypanel": {
      "type": "remote",
      "url": "https://<panel-domain>/api/mcp",
      "enabled": true,
      "oauth": false,
      "headers": { "Authorization": "Bearer {env:EASYPANEL_API_KEY}" }
    }
  }
}
```

- Simpan API key di env `EASYPANEL_API_KEY` (jangan di config/repo).
- **Restart opencode** — config tidak di-hot-reload.

Alternatif CLI (tanpa MCP):

```sh
# Windows: irm https://get.easypanel.io/cli.ps1 | iex
easypanel server add prod https://<panel-domain>   # tempel API key saat diminta
easypanel projects list
```

## 2. Protokol MCP (WAJIB dipatuhi)

MCP EasyPanel mengekspos 4 tool: `search_procedures`, `execute_query`, `execute_mutation`, `execute_destructive`.

1. **Jangan menebak nama prosedur.** Panggil `search_procedures` dengan frasa aksi+sumber (mis. `"create postgres service"`, `"create app service docker image"`, `"update app environment variables"`, `"create service domain"`, `"deploy app service"`), lalu baca `name` + `inputSchema` persis dari hasilnya.
2. Jalankan sesuai risiko: `execute_query` (baca), `execute_mutation` (ubah non-destruktif), `execute_destructive` (mengganti/menghapus — mis. `updateAppEnv`, `deployAppService`). Konfirmasi ke pengguna untuk yang destruktif.
3. Jika tersedia skrip `deploy.mjs`, ia sudah membungkus langkah-langkah ini (lihat bagian 4).

## 3. Alur provisioning (MCP)

Tanyakan nama **project EasyPanel** tujuan (jangan asumsikan). Ganti `<project>`, `<service>`, `<app>` sesuai proyek.

1. **Cek yang ada** — `listProjectsAndServices` (query).
2. **Project** — `createProject` bila belum ada.
3. **Database** — `createPostgresService` `{ projectName, serviceName:"db", user, databaseName, password, resources:0 }`.
4. **Cache** (bila perlu) — `createRedisService` `{ projectName, serviceName:"redis", password, resources:0 }`.
5. **Kredensial & host** — `inspectPostgresService` / `inspectRedisService`. Hostname internal = `<projectName>_<serviceName>` (mis. `m_db:5432`, `m_redis:6379`).
6. **App** — `createAppService`:
   ```jsonc
   {
     "projectName": "<project>",
     "serviceName": "<service>",
     "source": { "type": "image", "image": "ghcr.io/<owner>/<app>:latest" },
     "env": "NODE_ENV=production\nSITE_URL=https://<domain>\nDATABASE_URL=postgres://<user>:<pw>@<project>_db:5432/<db>\nREDIS_URL=redis://:<pw>@<project>_redis:6379",
     "resources": { "cpuLimit": 0, "cpuReservation": 0, "memoryLimit": 0, "memoryReservation": 0 }
   }
   ```
   Untuk registry privat, tambahkan `source.username` + `source.password`.
7. **Domain** — `createDomain` (pola mengikuti service yang sudah ada; cek lewat `listDomains`):
   ```jsonc
   {
     "certificateResolver": "", "host": "<project>-<service>.<wildcard-domain>",
     "https": true, "id": "", "middlewares": [], "path": "/", "wildcard": false,
     "destinationType": "service",
     "serviceDestination": { "path": "/", "port": 3000, "projectName": "<project>", "protocol": "http", "serviceName": "<service>" }
   }
   ```
8. **Env rahasia** — minta pengguna mengisi token/secret (mis. `GITHUB_TOKEN`, `HUB_*`, `VAPID_*`, `SNAPSHOT_SECRET`) di panel; atau set lewat `updateAppEnv` bila nilainya diberikan.
9. **Deploy** — `deployAppService` `{ projectName, serviceName, forceRebuild:false }`.
10. **Verifikasi** — `inspectAppService` untuk primary domain; `GET https://<domain>/api/health` → `200`.

## 4. Skrip `deploy.mjs`

`templates/deploy/easypanel/deploy.mjs` — tanpa dependensi, memanggil MCP via JSON-RPC memakai `EASYPANEL_API_KEY`.

```sh
export EASYPANEL_API_KEY=...        # PowerShell: $env:EASYPANEL_API_KEY="..."
export EASYPANEL_BASE_URL=https://<panel-domain>

node deploy.mjs list
node deploy.mjs search "create app service docker image"
node deploy.mjs create-postgres <project> db --user app --db app
node deploy.mjs create-redis <project> redis
node deploy.mjs create-app <project> <service> --image ghcr.io/<owner>/<app>:latest --env-file .env.production
node deploy.mjs create-domain <project> <service> <host> --port 3000
node deploy.mjs set-env <project> <service> --env-file .env.production
node deploy.mjs deploy <project> <service>
node deploy.mjs inspect <project> <service> --type app
```

## 5. Rollback

Ubah tag image ke versi sebelumnya (via `updateAppSourceImage`) lalu `deployAppService`.

## 6. Keamanan

- API key EasyPanel = akses penuh; **revoke** di Settings → Server → Users bila sudah tidak dipakai atau bocor.
- Jangan menulis API key atau secret ke repo; gunakan env / `.env` yang di-ignore.
- `listProjectsAndServices` mengembalikan env/secret secara plaintext — perlakukan output sebagai sensitif.
