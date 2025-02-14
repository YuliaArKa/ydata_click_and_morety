import joblib

model = joblib.load(r"C:\Users\Golds\.cookiecutters\serj-ds-skeleton\{{cookiecutter.project_name}}\ydata_click_and_morety\models\model_xgboost.joblib")
params = model.get_params()

for key, value in params.items():
    print(f"{key}: {value}")