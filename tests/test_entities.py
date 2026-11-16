"""Named entity recognition tests."""

from chatgangs.core.entities import EntityRecognizer, get_entity_recognizer


def labels(text):
    return {(entity.label, entity.text) for entity in get_entity_recognizer().extract(text)}


def test_email_phone_url():
    found = labels("Email budi@mail.com, telepon 081234567890, situs https://chatgangs.dev")
    assert ("EMAIL", "budi@mail.com") in found
    assert any(label == "PHONE" for label, _ in found)
    assert any(label == "URL" for label, _ in found)


def test_money_and_percent():
    found = labels("Harganya Rp 5.000.000 dengan diskon 25%")
    assert any(label == "MONEY" for label, _ in found)
    assert any(label == "PERCENT" for label, _ in found)


def test_dates_and_times():
    found = labels("Kirim besok jam 14:30 atau tanggal 12/03/2025")
    assert any(label == "DATE" for label, _ in found)
    assert any(label == "TIME" for label, _ in found)


def test_organization_patterns():
    found = labels("PT Maju Jaya bekerja sama dengan Acme Corporation")
    organizations = {text for label, text in found if label == "ORGANIZATION"}
    assert "PT Maju Jaya" in organizations
    assert any("Acme" in org for org in organizations)


def test_known_person_and_places():
    found = labels("Budi dan Siti pergi ke Bandung lalu ke Jakarta")
    assert any(label == "PERSON" for label, _ in found)
    assert any(label == "GPE" and text in ("Bandung", "Jakarta") for label, text in found)


def test_quantity_with_unit():
    found = labels("Saya memesan 5 unit laptop")
    assert any(label == "QUANTITY" for label, _ in found)


def test_offsets_match_text():
    text = "Hubungi budi@mail.com besok"
    for entity in get_entity_recognizer().extract(text):
        assert text[entity.start:entity.end] == entity.text


def test_summary_groups_entities():
    entities = get_entity_recognizer().extract("Google dan Microsoft ada di Jakarta")
    summary = get_entity_recognizer().summarize(entities)
    assert "ORGANIZATION" in summary
    assert "Google" in summary["ORGANIZATION"]


def test_label_filtering():
    recognizer = EntityRecognizer()
    entities = recognizer.extract(
        "Email budi@mail.com di Jakarta", include_default_labels=["EMAIL"]
    )
    assert all(entity.label == "EMAIL" for entity in entities)


def test_empty_text_returns_no_entities():
    assert get_entity_recognizer().extract("") == []
