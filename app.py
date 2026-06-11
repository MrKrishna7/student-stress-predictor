from flask import Flask, request, render_template
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        
        data = CustomData(
            study_hours = float(request.form.get("study_hours")),
            sleep_hours= float(request.form.get("sleep_hours")),
            social_hours = float(request.form.get("social_hours")),
            physical_activity = float(request.form.get("physical_activity")),
            extracurricular = float(request.form.get("extracurricular")),
            gpa  = float(request.form.get("gpa"))
        )

        pred_df  = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        result , confidence = pipeline.predict(pred_df)

        return render_template("home.html", result=result, confidence=confidence, study_hours=data.study_hours, sleep_hours=data.sleep_hours, social_hours=data.social_hours, physical_activity=data.physical_activity, extracurricular=data.extracurricular, gpa=data.gpa)
    
    return render_template("home.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)