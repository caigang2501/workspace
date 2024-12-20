from app import create_app

# gunicorn --workers 3 wsgi:app
app = create_app()
if __name__ == "__main__":
    app.run()