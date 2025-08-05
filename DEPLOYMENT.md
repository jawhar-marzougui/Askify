# Deployment Guide for Askify

This document provides instructions for deploying the Askify Streamlit app using Docker and on free cloud platforms like Streamlit Cloud and Render.

## Docker Deployment

1. Build the Docker image:

```bash
docker build -t askify .
```

2. Run the Docker container with environment variables:

```bash
docker run -p 8501:8501 --env-file .env askify
```

3. Access the app at `http://localhost:8501`

## Deploying on Streamlit Cloud

1. Create a GitHub repository with the project files including:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `.env.example`
- `README.md`
- Other source files

2. Sign up or log in to [Streamlit Cloud](https://streamlit.io/cloud).

3. Connect your GitHub repository and deploy the app.

4. Set environment variables in the Streamlit Cloud app settings (use the values from your `.env` file).

5. The app will be available at the provided Streamlit Cloud URL.

## Deploying on Render

1. Create a GitHub repository with the project files as above.

2. Sign up or log in to [Render](https://render.com).

3. Create a new Web Service and connect your GitHub repository.

4. Use the following build and start commands:

- Build Command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
- Start Command: `streamlit run app.py --server.port $PORT`

5. Add environment variables in the Render dashboard (matching your `.env` file).

6. Deploy the service and access the app at the Render URL.

## Notes

- Ensure your Neo4j database and API keys are accessible from the cloud environment.
- Keep your `.env` file secure and do not commit it to public repositories.
- Update the README with any platform-specific instructions as needed.

---

For any questions or issues, please refer to the official documentation of Streamlit, Docker, and your chosen cloud platform.
