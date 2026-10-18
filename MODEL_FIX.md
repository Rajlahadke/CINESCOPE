# Windows XGBoost model loading fix

Your model was saved in Google Colab (Linux) as an XGBoost-containing Joblib pickle. XGBoost's internal serialized state sometimes cannot be loaded on Windows, even if both machines have the same XGBoost version.

## 1. Export portable model files from Google Colab

Open your **original SI Lab notebook**. If the trained model outputs still exist in the Colab runtime, simply add a new code cell containing the entire code from `colab_portable_export.py`, then run it.

If Colab has reset since training, run the following in a new runtime *before* executing the export cell:

```python
!pip -q install scikit-learn==1.6.1 xgboost==3.4.1 joblib
```

The export code automatically offers a file uploader if the old model file is no longer in `/content/movie_rating_outputs/`. In that case, upload **`movie_rating_model.pkl`** from this VS Code folder (just that one file); it will convert the existing trained model without retraining.

When the conversion succeeds, Colab downloads `movie_model_portable_fix.zip`.

## 2. Copy the two exported files into the VS Code project

Extract the ZIP and place both files **next to `app.py`**:

- `rating_xgboost.json`
- `rating_preprocessor.pkl`

Do not replace `movie_classification_model.pkl`: it is used for the SVM classification and remains unchanged.

## 3. Start Streamlit

In the VS Code terminal (with the project folder open):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You don't need to reinstall anything if `xgboost==3.4.1` and `scikit-learn==1.6.1` are already installed. The new app first loads the native XGBoost JSON when it is present and combines it with the scikit-learn preprocessing pipeline.
