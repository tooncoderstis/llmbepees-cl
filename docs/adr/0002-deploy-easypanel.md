# ADR-0002: Deploy EasyPanel via Docker image (programatik MCP/CLI)

- **Status**: Diterima
- **Tanggal**: 2026-10-17

## Konteks
Prototipe asisten BPS akan dibungkus API (FastAPI) yang perlu dilayankan. Server mudah yang tersedia
adalah **EasyPanel** (dipakai proyek Aasa lain, MCP `easypanel` sudah terpasang di config opencode global).
Asisten baru lahir di Fase 5 (M9–10), sehingga belum ada artefak untuk dideploy di Fase 0. Sumber daya
server terbatas (pembuatan project baru pernah ditolak batas paket pada proyek lain).

## Keputusan
Deploy memakai pola **EasyPanel "Source: Docker Image"** dan dikendalikan **programatik** (MCP EasyPanel
utama, CLI `easypanel` sebagai fallback) — **bukan** klik UI. Build & push image dilakukan di mesin lokal
/ CI (mis. GHCR); EasyPanel hanya **pull** image. Pada Fase 0 hanya di-*scaffold* (Dockerfile + runbook);
**provisioning aktual ditunda sampai Fase 5** ketika asisten FastAPI sudah ada.

## Alasan
- Programatik = dapat direproduksi & didokumentasikan; tidak bergantung pada langkah manual di UI.
- Image-based menghindari build di server (server ringan, build sebaiknya di CI).
- Menunda provisioning menghindari layanan kosong/placeholder yang tidak berguna dan menghemat resource.

## Konsekuensi
- **Positif**: deploy terdefinisi di repo; rollback mudah (ganti tag image).
- **Negatif**: butuh registry image & API key; domain/EasyPanel project harus diputuskan saat Fase 5.
- **Lain-lain**: API key EasyPanel = akses penuh → disimpan di env (`.env`, tidak di-commit), dirotasi bila bocor.

## Alternatif yang dipertimbangkan
- **Deploy via Git (build di server)** — ditolak; server harus ringan dan build di CI/lokal.
- **Platform lain (Vercel/Railway/Fly)** — bukan standar Aasa; EasyPanel sudah tersedia & terotomasi.
- **Tanpa deploy (lokal saja)** — ditolak untuk jangka panjang; asisten perlu dapat diakses pemilik dokumen.
