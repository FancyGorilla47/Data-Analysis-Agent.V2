# 1. Install Microsoft ODBC Driver for SQL Server
#    This section is crucial for the 'pyodbc' package to install and run correctly.
#    It adds the Microsoft package repository and installs the necessary drivers.
apt-get update
apt-get install -y curl apt-transport-https
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

# 2. Start the FastAPI application using Gunicorn
#    Gunicorn is a production-ready web server that's more robust than Uvicorn's development server.
#    -w 4: Starts 4 worker processes to handle requests.
#    -k uvicorn.workers.UvicornWorker: Tells Gunicorn to use Uvicorn's high-performance worker class.
#    main:app: Points to the 'app' instance in your 'main.py' file.
#    --bind 0.0.0.0:$PORT: Binds to all available network interfaces on the port provided by Azure.
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT