"""Script to generate synthetic data for dietary intake analysis.

This script creates three CSV files with synthetic data about dietary intake:
1. daily_food_intake.csv - specific foods consumed by individuals daily
2. nutrient_summary.csv - nutritional breakdown per day per person
3. food_categories_intake.csv - consumption by food category in servings
"""

# %%
import csv
import os
import random

import pandas as pd
from pathway.agent_for_forecast.forecaster import generate_ts


# Create tables directory if it doesn't exist
os.makedirs("pathway/dataframes_QA/tables", exist_ok=True)

# Define constants
NUM_PEOPLE = 10
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack"]
GENDERS = ["Male", "Female", "Non-binary"]

# Food data by category
FOOD_ITEMS = {
    "Fruits": [
        "Apple",
        "Banana",
        "Orange",
        "Strawberries",
        "Grapes",
        "Blueberries",
        "Watermelon",
    ],
    "Vegetables": [
        "Broccoli",
        "Spinach",
        "Carrots",
        "Tomatoes",
        "Cucumber",
        "Lettuce",
        "Bell Peppers",
    ],
    "Grains": [
        "Bread",
        "Rice",
        "Pasta",
        "Cereal",
        "Oatmeal",
        "Quinoa",
        "Tortilla",
    ],
    "Dairy": ["Milk", "Cheese", "Yogurt", "Cottage Cheese", "Ice Cream"],
    "Protein_Foods": [
        "Chicken",
        "Beef",
        "Eggs",
        "Fish",
        "Tofu",
        "Beans",
        "Nuts",
    ],
    "Sweets": ["Chocolate", "Cookies", "Cake", "Candy", "Donut", "Ice Cream"],
    "Beverages": ["Water", "Coffee", "Tea", "Soda", "Juice", "Energy Drink"],
}

# Units for food items
UNITS = {
    "Fruits": ["piece", "cup", "serving"],
    "Vegetables": ["cup", "serving", "oz"],
    "Grains": ["cup", "serving", "slice", "oz"],
    "Dairy": ["cup", "oz", "serving", "slice"],
    "Protein_Foods": ["oz", "serving", "cup"],
    "Sweets": ["piece", "serving", "oz"],
    "Beverages": ["cup", "oz", "bottle", "can"],
}

# Nutritional content estimates per food item (approximate values per serving)
# Format: (calories, protein_g, carbs_g, fat_g, sugar_g, fiber_g)
NUTRITION = {
    "Apple": (95, 0.5, 25, 0.3, 19, 4),
    "Banana": (105, 1.3, 27, 0.4, 14, 3),
    "Orange": (62, 1.2, 15, 0.2, 12, 3),
    "Strawberries": (32, 0.7, 7.7, 0.3, 4.9, 2),
    "Grapes": (104, 1.1, 27, 0.2, 23, 1.4),
    "Blueberries": (84, 1.1, 21, 0.5, 15, 3.6),
    "Watermelon": (46, 0.9, 11.5, 0.2, 9.4, 0.6),
    "Broccoli": (55, 3.7, 11, 0.6, 2.6, 5.2),
    "Spinach": (23, 2.9, 3.6, 0.4, 0.4, 2.2),
    "Carrots": (50, 1.2, 12, 0.3, 6, 3.4),
    "Tomatoes": (22, 1.1, 4.8, 0.2, 3.2, 1.5),
    "Cucumber": (16, 0.7, 3.6, 0.1, 1.7, 0.5),
    "Lettuce": (5, 0.5, 1, 0.1, 0.4, 0.5),
    "Bell Peppers": (31, 1, 7.6, 0.3, 4.2, 2.5),
    "Bread": (77, 3, 14, 1, 1.5, 1.1),
    "Rice": (206, 4.3, 45, 0.4, 0.1, 0.6),
    "Pasta": (221, 8.1, 43, 1.3, 0.8, 2.5),
    "Cereal": (110, 3, 23, 1, 8, 3),
    "Oatmeal": (166, 5.9, 28, 3.6, 0.4, 4),
    "Quinoa": (222, 8.1, 39, 3.6, 1.6, 5.2),
    "Tortilla": (120, 3.3, 22, 2.5, 0.5, 1.6),
    "Milk": (122, 8.1, 11.7, 4.8, 12.3, 0),
    "Cheese": (113, 7, 0.9, 9, 0.5, 0),
    "Yogurt": (154, 12, 17, 3.8, 14, 0),
    "Cottage Cheese": (120, 14, 3.5, 5, 3, 0),
    "Chicken": (165, 31, 0, 3.6, 0, 0),
    "Beef": (213, 22, 0, 13, 0, 0),
    "Eggs": (78, 6.3, 0.6, 5.3, 0.6, 0),
    "Fish": (175, 26, 0, 8, 0, 0),
    "Tofu": (94, 10, 2.3, 5.9, 0.7, 1.4),
    "Beans": (127, 8.7, 22.8, 0.5, 0.4, 7.4),
    "Nuts": (172, 5, 6, 15, 1.4, 2),
    "Chocolate": (235, 3.4, 26, 13, 21, 2.8),
    "Cookies": (148, 1.8, 21, 7, 10, 0.5),
    "Cake": (260, 3, 40, 10, 28, 0.5),
    "Candy": (140, 0, 36, 0, 24, 0),
    "Donut": (195, 2.1, 22, 11, 10, 0.7),
    "Ice Cream": (207, 3.5, 24, 11, 21, 0.5),
    "Water": (0, 0, 0, 0, 0, 0),
    "Coffee": (2, 0.3, 0, 0, 0, 0),
    "Tea": (2, 0, 0.5, 0, 0, 0),
    "Soda": (140, 0, 39, 0, 39, 0),
    "Juice": (112, 0.5, 26, 0.3, 22, 0.5),
    "Energy Drink": (160, 0, 40, 0, 38, 0),
}

# Length of each synthetic time-series (e.g. one day at one-minute resolution)
TS_LENGTH = 24 * 60  # 1 day in minutes


# Helper function to get random food item from a category
def get_random_food(category):
    return random.choice(FOOD_ITEMS[category])


# Helper function to get random unit for a food item's category
def get_random_unit(category):
    return random.choice(UNITS[category])


# Helper function to get a category for a food item
def get_category_for_food(food_item):
    for category, foods in FOOD_ITEMS.items():
        if food_item in foods:
            return category
    return None


# Helper function to calculate BMI from height and weight
def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)
    return round(bmi, 1)


# Generate demographic data
def generate_demographics():
    filename = "pathway/dataframes_QA/tables/demographics.csv"

    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "person_id",
            "age",
            "gender",
            "height_cm",
            "weight_kg",
            "bmi",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for person_id in range(1, NUM_PEOPLE + 1):
            # Generate random demographic data
            age = random.randint(18, 65)
            gender = random.choice(GENDERS)

            # Height and weight based loosely on gender (just for data variation)
            if gender == "Male":
                height_cm = random.randint(165, 190)
                weight_kg = random.randint(60, 100)
            elif gender == "Female":
                height_cm = random.randint(155, 180)
                weight_kg = random.randint(50, 85)
            else:
                height_cm = random.randint(160, 185)
                weight_kg = random.randint(55, 90)

            bmi = calculate_bmi(height_cm, weight_kg)

            writer.writerow(
                {
                    "person_id": person_id,
                    "age": age,
                    "gender": gender,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "bmi": bmi,
                },
            )

    print(f"Created {filename}")


# Generate daily_food_intake.csv
def generate_daily_food_intake():
    filename = "pathway/dataframes_QA/tables/daily_food_intake.csv"

    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "person_id",
            "day_of_week",
            "meal_type",
            "food_item",
            "quantity",
            "unit",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for person_id in range(1, NUM_PEOPLE + 1):
            for day in DAYS_OF_WEEK:
                # Each person has 3-5 meal entries per day
                num_entries = random.randint(3, 5)

                for _ in range(num_entries):
                    meal_type = random.choice(MEAL_TYPES)

                    # Select a random food category with different probabilities
                    categories = list(FOOD_ITEMS.keys())
                    weights = [
                        0.15,
                        0.15,
                        0.2,
                        0.15,
                        0.15,
                        0.1,
                        0.1,
                    ]  # Probabilities for each category
                    category = random.choices(
                        categories,
                        weights=weights,
                        k=1,
                    )[0]

                    food_item = get_random_food(category)
                    quantity = round(random.uniform(0.5, 3.0), 1)
                    unit = get_random_unit(category)

                    writer.writerow(
                        {
                            "person_id": person_id,
                            "day_of_week": day,
                            "meal_type": meal_type,
                            "food_item": food_item,
                            "quantity": quantity,
                            "unit": unit,
                        },
                    )

    print(f"Created {filename}")


# Generate nutrient_summary.csv with total daily calories
def generate_nutrient_summary():
    # First, read the daily food intake to calculate nutrient totals
    daily_food_df = pd.read_csv(
        "pathway/dataframes_QA/tables/daily_food_intake.csv",
    )

    # Load demographic data to include in the summary
    demographics_df = pd.read_csv(
        "pathway/dataframes_QA/tables/demographics.csv",
    )

    # Initialize the nutrient summary dataframe
    nutrient_summary = []

    for person_id in range(1, NUM_PEOPLE + 1):
        # Get demographic data for this person
        person_demo = demographics_df[demographics_df["person_id"] == person_id].iloc[0]

        for day in DAYS_OF_WEEK:
            # Filter daily food intake for this person and day
            person_day_df = daily_food_df[
                (daily_food_df["person_id"] == person_id)
                & (daily_food_df["day_of_week"] == day)
            ]

            # Initialize nutrient totals
            calories = 0
            protein = 0
            carbs = 0
            fat = 0
            sugar = 0
            fiber = 0

            # Calculate totals based on food items consumed
            for _, row in person_day_df.iterrows():
                food_item = row["food_item"]
                quantity = row["quantity"]

                if food_item in NUTRITION:
                    food_nutrition = NUTRITION[food_item]
                    calories += food_nutrition[0] * quantity
                    protein += food_nutrition[1] * quantity
                    carbs += food_nutrition[2] * quantity
                    fat += food_nutrition[3] * quantity
                    sugar += food_nutrition[4] * quantity
                    fiber += food_nutrition[5] * quantity

            # Add some random variation
            variation = random.uniform(0.9, 1.1)
            calories *= variation
            protein *= variation
            carbs *= variation
            fat *= variation
            sugar *= variation
            fiber *= variation

            # Add entry to summary with demographic info
            nutrient_summary.append(
                {
                    "person_id": person_id,
                    "day_of_week": day,
                    "age": person_demo["age"],
                    "gender": person_demo["gender"],
                    "bmi": person_demo["bmi"],
                    "total_calories": round(calories, 1),
                    "protein_g": round(protein, 1),
                    "carbs_g": round(carbs, 1),
                    "fat_g": round(fat, 1),
                    "sugar_g": round(sugar, 1),
                    "fiber_g": round(fiber, 1),
                },
            )

    # Convert to dataframe and save to CSV
    nutrient_df = pd.DataFrame(nutrient_summary)
    filename = "pathway/dataframes_QA/tables/nutrient_summary.csv"
    nutrient_df.to_csv(filename, index=False)
    print(f"Created {filename}")


# Generate food_categories_intake.csv
def generate_food_categories_intake():
    # First, read the daily food intake to calculate category totals
    daily_food_df = pd.read_csv(
        "pathway/dataframes_QA/tables/daily_food_intake.csv",
    )

    # Load demographic data to include in the summary
    demographics_df = pd.read_csv(
        "pathway/dataframes_QA/tables/demographics.csv",
    )

    # Initialize the category intake dataframe
    category_intake = []

    for person_id in range(1, NUM_PEOPLE + 1):
        # Get demographic data for this person
        person_demo = demographics_df[demographics_df["person_id"] == person_id].iloc[0]

        for day in DAYS_OF_WEEK:
            # Filter daily food intake for this person and day
            person_day_df = daily_food_df[
                (daily_food_df["person_id"] == person_id)
                & (daily_food_df["day_of_week"] == day)
            ]

            # Initialize category totals
            category_servings = dict.fromkeys(FOOD_ITEMS.keys(), 0)

            # Calculate servings per category
            for _, row in person_day_df.iterrows():
                food_item = row["food_item"]
                quantity = row["quantity"]

                category = get_category_for_food(food_item)
                if category:
                    category_servings[category] += quantity

            # Add some random variation
            for category in category_servings:
                variation = random.uniform(0.9, 1.1)
                category_servings[category] *= variation

            # Convert dictionary to lowercase keys for CSV columns
            entry = {
                "person_id": person_id,
                "day_of_week": day,
                "age": person_demo["age"],
                "gender": person_demo["gender"],
                "bmi": person_demo["bmi"],
            }

            # Calculate total calories for each day by category
            total_calories = 0
            for category, foods in FOOD_ITEMS.items():
                # Estimate average calories per serving for this category
                category_calories = 0
                for food in foods:
                    if food in NUTRITION:
                        category_calories += NUTRITION[food][0]  # Index 0 is calories

                if foods:  # Avoid division by zero
                    avg_calories_per_serving = category_calories / len(foods)
                    # Calculate calories for this category
                    calories_from_category = (
                        avg_calories_per_serving * category_servings[category]
                    )
                    total_calories += calories_from_category

            entry["total_calories"] = round(total_calories, 1)

            # Add food category servings to entry
            for category, servings in category_servings.items():
                entry[category.lower()] = round(servings, 1)

            category_intake.append(entry)

    # Convert to dataframe and save to CSV
    category_df = pd.DataFrame(category_intake)
    filename = "pathway/dataframes_QA/tables/food_categories_intake.csv"
    category_df.to_csv(filename, index=False)
    print(f"Created {filename}")


def generate_patient_time_series(ts_length: int = TS_LENGTH) -> None:
    """Generate a CSV containing a synthetic time-series per patient.

    A new file named ``patient_time_series.csv`` is written to the standard
    tables directory.  Each row contains three fields:

    * ``person_id`` – unique identifier of the patient.
    * ``timestamp`` – ISO-formatted timestamp for the sample.
    * ``biomarcador`` – synthetic biomarker value generated with
      :func:`generate_ts`.

    The length of the series can be controlled through *ts_length* (defaults
    to ``TS_LENGTH``).  A single series is generated per ``person_id``.
    """
    filename = "pathway/dataframes_QA/tables/patient_time_series.csv"

    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["person_id", "timestamp", "biomarcador"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for person_id in range(1, NUM_PEOPLE + 1):
            # ``generate_ts`` returns a DataFrame where the index is the
            # timestamp and the single column (named "simulation") contains
            # the numeric values.  We map it to the required output schema.
            ts_df = generate_ts(ts_length, add_trend=True, add_noise=True).reset_index(
                drop=False, names=["timestamp"]
            )

            for _, row in ts_df.iterrows():
                writer.writerow(
                    {
                        "person_id": person_id,
                        "timestamp": row["timestamp"],
                        "biomarcador": round(float(row["simulation"]), 4),
                    },
                )

    print(f"Created {filename}")


# Generate all CSV files
def generate_all_files():
    print("Generating synthetic dietary data and time-series...")
    generate_demographics()
    generate_daily_food_intake()
    generate_nutrient_summary()
    generate_food_categories_intake()
    generate_patient_time_series()
    print("Data generation complete.")


if __name__ == "__main__":
    generate_all_files()
