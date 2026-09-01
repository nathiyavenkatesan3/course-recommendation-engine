from flask import Flask, render_template, request
from model import recommend_courses

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    # Get student information
    name = request.form["name"]
    domain = request.form["domain"]
    level = request.form["level"]
    skill = request.form["skills"]
    duration = request.form["duration"]

    # Get recommendations
    recommendations = recommend_courses(
        domain,
        level,
        skill,
        duration
    )

    # Convert to dictionary
    courses = recommendations.to_dict("records")

    # Display result
    return render_template(
        "recommendations.html",
        courses=courses,
        name=name
    )


if __name__ == "__main__":
    app.run()