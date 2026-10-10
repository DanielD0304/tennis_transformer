import pytest

from src.names import display_name, player_key


@pytest.mark.parametrize("variants", [
    ("Medvedev D.", "D. Medvedev RUS", "Daniil Medvedev", "Medvedev, D."),
    ("Carreno Busta P.", "Carreno-Busta, P.", "P. Carreno Busta ESP"),
    ("Struff J.L.", "J.-L. Struff GER", "Struff, J."),
    ("Auger-Aliassime F.", "F. Auger-Aliassime CAN", "Auger-Aliassime, F."),
    ("De Minaur A.", "A. de Minaur AUS", "de Minaur, A."),
    ("Cerúndolo F.", "Cerundolo, F.", "F. Cerúndolo ARG"),
])
def test_same_player_same_key(variants):
    keys = {player_key(v) for v in variants}
    assert len(keys) == 1
    assert None not in keys


def test_different_initial_different_key():
    assert player_key("Cerundolo F.") != player_key("Cerundolo J.M.")


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_names(name):
    assert player_key(name) is None


def test_display_name_strips_country():
    assert display_name("D. Medvedev RUS") == "D. Medvedev"
    assert display_name("Tabilo, A.") == "Tabilo, A."
