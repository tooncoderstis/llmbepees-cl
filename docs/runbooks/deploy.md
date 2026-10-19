# Runbook: Deploy EasyPanel (programatik)

Deploy prototipe asisten (FastAPI, Fase 5) ke EasyPanel **tanpa klik UI**:
build & push image → EasyPanel pull image → provisioning via **MCP EasyPanel** (utama) / CLI (fallback).

> Prinsip: EasyPanel hanya **pull** image. **Jangan build di server.**

Status saat ini: **belum diprovision** (menunggu Fase 5). Dokumen ini disiapkan lebih awal (ADR-0002).

## 0. Prasyarat
- Server EasyPanel + API key (Settings → Server → Users → Generate API Key).
- Image terbit di registry (mis. `ghcr.io/tooncoderstis/llmbepees-cl:latest`). Paket publik, atau sediakan kredensial registry.
- Keputusan **nama project EasyPanel** tujuan (jangan diasumsikan; tanyakan saat eksekusi).

## 1. Build & push image
```powershell
powershell -File deploy/easypanel-docker/build-push.ps1 -Image ghcr.io/tooncoderstis/llmbepees-cl -Tag 0.1.0 -Push
```
Disarankan lewat CI (push ke `main`) alih-alih lokal.

## 2. Provision (MCP EasyPanel)
Ikuti `deploy/easypanel/RUNBOOK.md` (protokol MCP wajib: `search_procedures` dulu, jangan menebak nama prosedur).
Ringkas alur: `listProjectsAndServices` → `createAppService` (`source: {type:"image", image:...}`) →
`createDomain` (port target, mis. 8000) → env non-rahasia → `deployAppService`.

## 3. Verifikasi
- `GET https://<domain>/health` → `200`.
- Halaman/API utama memuat; HTTPS aktif.

## 4. Rollback
`updateAppSourceImage` ke tag sebelumnya → `deployAppService`.

## Keamanan
- API key EasyPanel = akses penuh; simpan di env (`.env`, tidak di-commit), revoke bila bocor.
- `listProjectsAndServices` mengembalikan secret secara plaintext → perlakukan output sebagai sensitif.
