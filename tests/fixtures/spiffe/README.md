# SPIFFE X509-SVID samples (Phase 0 evidence)

Minted locally with a real SPIRE server (`ghcr.io/spiffe/spire-server:1.13.2`,
`spire-server x509 mint -spiffeID spiffe://example.org/<name>`) on 2026-09-08,
trust domain `example.org` (a placeholder, not any client's). Certificates,
not keys, are committed; they expired one hour after minting, so they are
evidence of the SVID *shape* (one `URI:spiffe://...` SAN, `CA:FALSE`, EC P-256,
issuer `O=SPIFFE`) for parsing tests, never for a live TLS handshake. The
end-to-end mTLS test mints its own long-lived test CA and client certs.
