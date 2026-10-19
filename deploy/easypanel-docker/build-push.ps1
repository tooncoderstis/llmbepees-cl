param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$Tag = "latest",

    [switch]$Push
)

$ErrorActionPreference = "Stop"
$full = "${Image}:${Tag}"

Write-Host "==> Building $full (linux/amd64)" -ForegroundColor Cyan
docker build `
    --platform linux/amd64 `
    -f deploy/easypanel-docker/Dockerfile `
    --build-arg "APP_VERSION=$Tag" `
    -t $full `
    .

if ($LASTEXITCODE -ne 0) { Write-Error "Build gagal."; exit 1 }

if ($Push) {
    Write-Host "==> docker login (jika belum), lalu push $full" -ForegroundColor Cyan
    docker push $full
    if ($LASTEXITCODE -ne 0) { Write-Error "Push gagal."; exit 1 }
}

Write-Host "==> Selesai: $full" -ForegroundColor Green
