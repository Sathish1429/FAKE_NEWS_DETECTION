from flask import Flask, render_template, request
from predict import predict_article, predict_url, predict_csv
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    csv_results = []

    if request.method == "POST":
        input_type = request.form.get("input_type")

        text = request.form.get("text_input", "").strip()
        url = request.form.get("url_input", "").strip()
        file = request.files.get("csv_file")

        if input_type == "text" and text:
            result = predict_article(text)
            csv_results = []  # clear previous CSV results

        elif input_type == "url" and url:
            result = predict_url(url)
            csv_results = []  # clear previous CSV results

        elif input_type == "csv" and file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = ""  # clear single result
            csv_results = predict_csv(filepath)

    return render_template("index.html", result=result, csv_results=csv_results)



if __name__ == "__main__":
    app.run(debug=True)
