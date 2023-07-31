from flask import Flask, request, render_template, jsonify
from src.pipeline.predection_pipeline import CustomData, PredictPipeline


application = Flask(__name__, template_folder="templets")
app = application


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            CPU=(request.form.get("CPU")),
            RAM=int(request.form.get("RAM")),
            Storage=int(request.form.get("Storage")),
            Storage_type=(request.form.get("Storage type")),
            GPU=(request.form.get("GPU")),
            Screen=float(request.form.get("Screen")),
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity"),
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template("results.html", final_result=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
