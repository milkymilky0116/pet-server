import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np


def parse_range(value):
    try:
        low, high = map(float, value.split("-"))
        return (low + high) / 2
    except:
        return 0


def prepare_vectors(user_df, df, mode="bi"):
    data = df.copy()

    # Map English column names to Korean for processing
    column_mapping = {
        "color": "애견 모색",
        "personality": "애견 성격",
        "region": "애견 지역",
        "vaccinated": "예방접종",
        "preference_color": "preference_모색",
        "preference_personality": "preference_성격",
        "preference_region": "preference_지역",
        "preference_vaccine": "preference_예방접종",
    }

    # Rename columns in user_df to match data format
    user_df = user_df.rename(columns=column_mapping)

    # Basic preprocessing
    for col in ["애견 모색", "애견 성격", "애견 지역", "예방접종"]:
        le = LabelEncoder()
        # Combine both datasets to fit the encoder
        combined_data = pd.concat([data[col], user_df[col]], ignore_index=True)
        le.fit(combined_data.astype(str))
        # Transform both datasets
        data[col] = le.transform(data[col].astype(str))
        user_df[col] = le.transform(user_df[col].astype(str))

    personalities = df["애견 성격"].values
    if mode == "bi":
        for col in [
            "preference_모색",
            "preference_성격",
            "preference_지역",
            "preference_예방접종",
        ]:
            le = LabelEncoder()
            # Combine both datasets to fit the encoder
            combined_data = pd.concat([data[col], user_df[col]], ignore_index=True)
            le.fit(combined_data.astype(str))
            # Transform both datasets
            data[col] = le.transform(data[col].astype(str))
            user_df[col] = le.transform(user_df[col].astype(str))

        data["preference_나이"] = df["preference 연령 range (개월)"].map(parse_range)
        data["preference_체중"] = df["preference_몸무게 range (kg)"].map(parse_range)

        feature_cols = [
            "애견 나이(개월)",
            "애견 몸무게(kg)",
            "애견 모색",
            "애견 성격",
            "애견 지역",
            "예방접종",
            "preference_모색",
            "preference_성격",
            "preference_지역",
            "preference_예방접종",
            "preference_나이",
            "preference_체중",
        ]
    else:
        feature_cols = [
            "애견 나이(개월)",
            "애견 몸무게(kg)",
            "애견 모색",
            "애견 성격",
            "애견 지역",
            "예방접종",
        ]

    # Rename user_df columns back to match feature_cols
    user_df = user_df.rename(
        columns={"age_month": "애견 나이(개월)", "weight": "애견 몸무게(kg)"}
    )

    scaler = MinMaxScaler()
    features = scaler.fit_transform(data[feature_cols])
    gps = data[["lat", "lon"]].values
    ids = data["ID"].values
    names = data["애견 이름"].values
    return user_df, features, gps, ids, names, personalities
