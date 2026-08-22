from harness.recovery.patterns import ErrorPatterns


def test_error_patterns_classify_ssl_certificate_failures():
    patterns = ErrorPatterns()

    matches = patterns.classify(
        [
            "pip failed: SSL: CERTIFICATE_VERIFY_FAILED",
            "requests.exceptions.SSLError: certificate verify failed",
        ]
    )

    assert "ssl_certificate_verification" in matches
    assert patterns.validate() == []


def test_error_patterns_render_matching_recovery_guidance():
    rendered = ErrorPatterns().render(
        {
            "raw_errors": [
                "requests.exceptions.SSLError: unable to get local issuer certificate"
            ]
        }
    )

    assert "Known Recovery Patterns" in rendered
    assert "ssl_certificate_verification" in rendered
    assert "SSL_CERT_FILE" in rendered
    assert "do not disable TLS verification" in rendered


def test_error_patterns_falls_back_to_all_patterns_without_error_context():
    rendered = ErrorPatterns().raw_content()

    assert "command_not_found" in rendered
    assert "ssl_certificate_verification" in rendered
