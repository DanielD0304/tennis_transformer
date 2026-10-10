import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 'src' liegt im Repo-Root, die Django-Module (betObjekts, dataProcessing, scraper) in betscraper/
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "betscraper")):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "betscraper.settings")
