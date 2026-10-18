from pathlib import Path
import joblib
import zipfile
from google.colab import files

source = Path("/content/movie_rating_outputs/movie_rating_model.pkl")
if not source.exists():
    print("Upload your movie_rating_model.pkl file")
    files.upload()
    source = Path("/content/movie_rating_model.pkl")

regression_pipeline = joblib.load(source)
regression_pipeline.named_steps["model"].save_model("/content/rating_xgboost.json")
joblib.dump(regression_pipeline.named_steps["preprocess"], "/content/rating_preprocessor.pkl")

with zipfile.ZipFile("/content/movie_model_portable_fix.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for filename in ["rating_xgboost.json", "rating_preprocessor.pkl"]:
        z.write("/content/" + filename, arcname=filename)

print("Portable model files exported successfully")
files.download("/content/movie_model_portable_fix.zip")
