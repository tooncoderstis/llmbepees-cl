# Kebijakan Keamanan

## Melaporkan kerentanan
Jangan buka issue publik. Kirim email ke `<email-keamanan>` dengan langkah reproduksi.
Kami akan menindaklanjuti dalam `<SLA>`.

## Praktik
- Jangan commit rahasia (`.env` di-ignore).
- Rotasi kredensial yang pernah terekspos.
- Jalankan dependency audit & secret scan di CI.
- Hanya memakai dokumen publik; jangan commit data internal/sensitif.
- Ikuti OWASP ASVS untuk verifikasi keamanan.
