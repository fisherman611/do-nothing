#!/usr/bin/env bash
cd /code
/opt/app/bin/jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser
