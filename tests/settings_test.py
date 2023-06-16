import os
from bonn.settings import settings

def test_load_settings():
    assert settings.load_dotenv is True
    assert settings.ENVVAR_PREFIX_FOR_DYNACONF == 'BONN'

    os.environ["BONN_STOPWORDS_LANGUAGE"] = "welsh"
    settings.reload()
    assert settings.STOPWORDS_LANGUAGE == "welsh"
