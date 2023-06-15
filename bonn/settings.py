from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="BONN",
    load_dotenv=True, 
)

settings.reload()
