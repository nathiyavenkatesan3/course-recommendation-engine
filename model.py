import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------------------------------
# 1. Load course dataset
# ---------------------------------------

courses = pd.read_csv("courses.csv")

courses["skills"] = courses["skills"].fillna("").str.lower()


# ---------------------------------------
# 2. Prepare K-NN features
# ---------------------------------------

features = courses[["domain", "level", "skills"]]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            ["domain", "level"]
        ),
        (
            "skills",
            OneHotEncoder(handle_unknown="ignore"),
            ["skills"]
        )
    ]
)


# ---------------------------------------
# 3. Create K-NN model
# ---------------------------------------

knn = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        NearestNeighbors(
            n_neighbors=5,
            metric="cosine"
        )
    )
])

knn.fit(features)

print("K-NN model trained successfully!")


# ---------------------------------------
# 4. Recommendation function
# ---------------------------------------

def recommend_courses(domain, level, skill, duration):

    duration = int(duration)
    skill = skill.lower()

    # -----------------------------------
    # STEP 1: Duration filtering
    # -----------------------------------

    filtered_courses = courses[
        courses["duration"] <= duration
    ].copy()

    # If fewer than 5 courses are available,
    # use all courses
    if len(filtered_courses) < 5:
        filtered_courses = courses.copy()


    # -----------------------------------
    # STEP 2: Calculate preference scores
    # -----------------------------------

    def calculate_score(row):

        score = 0

        # Domain match
        if row["domain"].lower() == domain.lower():
            score += 5

        # Level match
        if row["level"].lower() == level.lower():
            score += 3

        # Skill match
        user_skills = skill.split()

        for user_skill in user_skills:

            if user_skill in row["skills"]:
                score += 4

        # Rating contribution
        score += row["rating"]

        return score


    filtered_courses["preference_score"] = filtered_courses.apply(
        calculate_score,
        axis=1
    )


    # -----------------------------------
    # STEP 3: Sort courses
    # -----------------------------------

    filtered_courses = filtered_courses.sort_values(
        by="preference_score",
        ascending=False
    )


    # -----------------------------------
    # STEP 4: Return top 5
    # -----------------------------------

    return filtered_courses.head(5)


# ---------------------------------------
# 5. Test the system
# ---------------------------------------

if __name__ == "__main__":

    result = recommend_courses(
        "Artificial Intelligence",
        "Beginner",
        "python",
        3
    )

    print("\nRecommended Courses:")

    print(
        result[
            [
                "course",
                "domain",
                "level",
                "duration",
                "rating"
            ]
        ]
    )