# Runbook Deploy — EasyPanel (Docker Image)

Fokus bagian ini: **build & publish image**. Untuk provisioning & deploy di EasyPanel
(project/service/domain/env), gunakan runbook programatik: [`../easypanel/RUNBOOK.md`](../easypanel/RUNBOOK.md).
EasyPanel hanya **pull** image — jangan build di server.

> Status: diaktifkan pada **Fase 5** (asisten FastAPI belum ada di Fase 0).

1. **Build & push** (lokal, bila tidak lewat CI):
   `powershell -File deploy/easypanel-docker/build-push.ps1 -Image <registry>/llmbepees-cl -Tag <versi> -Push`
   Disarankan via CI (mis. GHCR saat push ke `main`).
2. **Sambungkan MCP EasyPanel** (sekali): lihat [`../easypanel/RUNBOOK.md`](../easypanel/RUNBOOK.md) §1.
3. **Provision & deploy** via MCP: `createAppService` `source: {type:"image", image:"<registry>/llmbepees-cl:<tag>"}`
   → `createDomain` (port 8000) → env → `deployAppService`. Skrip bantu: `deploy/easypanel/deploy.mjs`.
4. **Verifikasi**: `GET https://<domain>/health` → 200; HTTPS aktif.
5. **Rollback**: `updateAppSourceImage` ke tag sebelumnya → `deployAppService`.

Volume/DB/env rahasia diisi lewat panel atau `updateAppEnv`. Keamanan: rotasi kredensial yang pernah terekspos; backup terjadwal.
