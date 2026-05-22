import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Flight Price Dashboard",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    section[data-testid="stSidebar"] {
        width: 370px !important;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%);
    }

    div.stButton > button {
        width: 100%;
        height: 3.6rem;
        border-radius: 16px;
        font-size: 1.05rem;
        font-weight: 800;
        border: 1px solid rgba(96, 165, 250, 0.8);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
    }

    div.stButton > button:hover {
        border: 1px solid white;
        transform: scale(1.01);
    }

    .hero {
        padding: 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #020617 0%, #1e293b 45%, #312e81 100%);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 3.1rem;
        font-weight: 900;
        line-height: 1.1;
    }

    .hero-subtitle {
        margin-top: 0.8rem;
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 900px;
    }

    .mini-card {
        padding: 1.25rem;
        border-radius: 22px;
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 128px;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
    }

    .mini-value {
        color: white;
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 0.3rem;
    }

    .mini-note {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }

    .price-card {
        padding: 1.7rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
        border: 1px solid rgba(74,222,128,0.6);
        min-height: 230px;
    }

    .price-value {
        font-size: 3.3rem;
        font-weight: 950;
        color: #bbf7d0;
        margin-top: 0.2rem;
    }

    .clean-card {
        padding: 1.5rem;
        border-radius: 24px;
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.7rem;
        font-weight: 900;
        margin-bottom: 0.7rem;
    }

    .big-center-text {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 900;
    }

    .muted {
        color: #94a3b8;
    }

    .pill {
        display: inline-block;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background-color: rgba(37,99,235,0.2);
        border: 1px solid rgba(96,165,250,0.35);
        color: #bfdbfe;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():
    data = pd.read_csv("airlines_flights_data.csv")

    if "index" in data.columns:
        data = data.drop(columns=["index"])

    data["duration"] = pd.to_numeric(data["duration"], errors="coerce")
    data["days_left"] = pd.to_numeric(data["days_left"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna()

    return data


data = load_data()


# =========================================================
# DATA FEATURE GROUPING
# =========================================================

def time_group(value):
    if value in ["Early_Morning", "Morning"]:
        return "Morning"
    if value in ["Afternoon", "Evening"]:
        return "Afternoon"
    return "Night"


def week_group(days):
    if days <= 7:
        return "This Week"
    if days <= 14:
        return "2 Weeks"
    if days <= 21:
        return "3 Weeks"
    if days <= 30:
        return "1 Month"
    return "More Than 1 Month"


def duration_group(hours):
    if hours <= 6:
        return "Short Flight"
    if hours <= 12:
        return "Medium Flight"
    if hours <= 24:
        return "Long Flight"
    return "Very Long Flight"


data["departure_group"] = data["departure_time"].apply(time_group)
data["arrival_group"] = data["arrival_time"].apply(time_group)
data["week_group"] = data["days_left"].apply(week_group)
data["duration_group"] = data["duration"].apply(duration_group)


# =========================================================
# MODEL TRAINING
# =========================================================

FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class",
    "duration",
    "days_left"
]

TARGET = "price"

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "stops",
    "class"
]

NUMERIC_FEATURES = [
    "duration",
    "days_left"
]


@st.cache_resource
def train_prediction_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=42,
        n_jobs=-1,
        max_depth=18,
        min_samples_leaf=2
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    return pipeline


@st.cache_data
def compare_models(data):
    sample_size = min(70000, len(data))
    sample_data = data.sample(sample_size, random_state=42)

    X = sample_data[FEATURES]
    y = sample_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES)
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=18,
            min_samples_leaf=3
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            max_depth=18,
            min_samples_leaf=2
        )
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })

    return pd.DataFrame(results)


model = train_prediction_model(data)
model_results = compare_models(data)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_price(value):
    return f"₹ {value:,.0f}"


def show_mini_card(label, value, note=""):
    st.markdown(f"""
    <div class="mini-card">
        <div class="mini-label">{label}</div>
        <div class="mini-value">{value}</div>
        <div class="mini-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_price_card(price, note):
    st.markdown(f"""
    <div class="price-card">
        <div class="mini-label">Predicted Ticket Price</div>
        <div class="price-value">{format_price(price)}</div>
        <div class="mini-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def mode_or_default(series, default_value):
    if len(series) == 0:
        return default_value
    mode_value = series.mode()
    if len(mode_value) == 0:
        return default_value
    return mode_value.iloc[0]


def median_or_default(series, default_value):
    if len(series) == 0:
        return default_value
    return float(series.median())


def filter_dataset(
    data,
    airline,
    flight_class,
    source_city,
    destination_city,
    stops,
    departure_group_choice=None,
    arrival_group_choice=None,
    week_group_choice=None,
    duration_group_choice=None
):
    filtered = data.copy()

    filtered = filtered[filtered["airline"] == airline]
    filtered = filtered[filtered["class"] == flight_class]
    filtered = filtered[filtered["source_city"] == source_city]
    filtered = filtered[filtered["destination_city"] == destination_city]
    filtered = filtered[filtered["stops"] == stops]

    if departure_group_choice != "Any":
        filtered = filtered[filtered["departure_group"] == departure_group_choice]

    if arrival_group_choice != "Any":
        filtered = filtered[filtered["arrival_group"] == arrival_group_choice]

    if week_group_choice != "Any":
        filtered = filtered[filtered["week_group"] == week_group_choice]

    if duration_group_choice != "Any":
        filtered = filtered[filtered["duration_group"] == duration_group_choice]

    return filtered


def build_model_input(
    filtered_data,
    airline,
    flight_class,
    source_city,
    destination_city,
    stops,
    departure_group_choice,
    arrival_group_choice,
    week_group_choice,
    duration_group_choice
):
    fallback_data = data[
        (data["airline"] == airline) &
        (data["class"] == flight_class) &
        (data["source_city"] == source_city) &
        (data["destination_city"] == destination_city) &
        (data["stops"] == stops)
    ]

    if len(fallback_data) == 0:
        fallback_data = data.copy()

    if departure_group_choice == "Any":
        departure_time = mode_or_default(fallback_data["departure_time"], data["departure_time"].mode().iloc[0])
    else:
        temp = fallback_data[fallback_data["departure_group"] == departure_group_choice]
        departure_time = mode_or_default(temp["departure_time"], fallback_data["departure_time"].mode().iloc[0])

    if arrival_group_choice == "Any":
        arrival_time = mode_or_default(fallback_data["arrival_time"], data["arrival_time"].mode().iloc[0])
    else:
        temp = fallback_data[fallback_data["arrival_group"] == arrival_group_choice]
        arrival_time = mode_or_default(temp["arrival_time"], fallback_data["arrival_time"].mode().iloc[0])

    if week_group_choice == "Any":
        days_left = median_or_default(fallback_data["days_left"], data["days_left"].median())
    else:
        temp = fallback_data[fallback_data["week_group"] == week_group_choice]
        days_left = median_or_default(temp["days_left"], fallback_data["days_left"].median())

    if duration_group_choice == "Any":
        duration = median_or_default(fallback_data["duration"], data["duration"].median())
    else:
        temp = fallback_data[fallback_data["duration_group"] == duration_group_choice]
        duration = median_or_default(temp["duration"], fallback_data["duration"].median())

    user_input = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "destination_city": [destination_city],
        "departure_time": [departure_time],
        "arrival_time": [arrival_time],
        "stops": [stops],
        "class": [flight_class],
        "duration": [duration],
        "days_left": [days_left]
    })

    return user_input


def get_match_score(filtered_data, user_input):
    if len(filtered_data) == 0:
        return filtered_data

    working = filtered_data.copy()

    working["duration_gap"] = abs(working["duration"] - user_input["duration"].iloc[0])
    working["days_gap"] = abs(working["days_left"] - user_input["days_left"].iloc[0])
    working["match_score"] = 100 - (working["duration_gap"] * 1.5) - (working["days_gap"] * 0.8)

    working["match_score"] = working["match_score"].clip(lower=0)

    return working.sort_values(
        by=["match_score", "price"],
        ascending=[False, True]
    )


def make_price_distribution_chart(filtered_data):
    fig = px.histogram(
        filtered_data,
        x="price",
        nbins=30,
        title="Ticket Price Distribution",
        labels={"price": "Ticket Price"}
    )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def make_airline_price_chart(data):
    airline_summary = data.groupby("airline", as_index=False)["price"].mean()
    airline_summary = airline_summary.sort_values("price", ascending=False)

    fig = px.bar(
        airline_summary,
        x="airline",
        y="price",
        title="Average Ticket Price by Airline",
        labels={
            "airline": "Airline",
            "price": "Average Price"
        }
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def make_class_price_chart(data):
    fig = px.box(
        data,
        x="class",
        y="price",
        title="Price Range by Class",
        labels={
            "class": "Class",
            "price": "Ticket Price"
        }
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def make_days_price_chart(data):
    day_summary = data.groupby("week_group", as_index=False)["price"].mean()

    order = [
        "This Week",
        "2 Weeks",
        "3 Weeks",
        "1 Month",
        "More Than 1 Month"
    ]

    day_summary["week_group"] = pd.Categorical(
        day_summary["week_group"],
        categories=order,
        ordered=True
    )

    day_summary = day_summary.sort_values("week_group")

    fig = px.line(
        day_summary,
        x="week_group",
        y="price",
        markers=True,
        title="Average Price by Booking Time",
        labels={
            "week_group": "Booking Time",
            "price": "Average Price"
        }
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def make_model_comparison_chart(model_results, metric):
    sorted_results = model_results.sort_values(metric, ascending=True if metric in ["MAE", "RMSE"] else False)

    fig = px.bar(
        sorted_results,
        x="Model",
        y=metric,
        title=f"Model Comparison by {metric}",
        text=metric
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("✈️ Flight Price App")
st.sidebar.caption("Visual Presentation Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Main Dashboard",
        "🎫 Flight Explorer",
        "📊 Visual Insights",
        "🧠 Model Comparison",
        "🔒 Advanced Notes"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Main Selectors")

airline = st.sidebar.selectbox(
    "Airline",
    sorted(data["airline"].unique())
)

flight_class = st.sidebar.selectbox(
    "Class",
    sorted(data["class"].unique())
)

source_city = st.sidebar.selectbox(
    "Departure City",
    sorted(data["source_city"].unique())
)

destination_options = sorted([city for city in data["destination_city"].unique() if city != source_city])

destination_city = st.sidebar.selectbox(
    "Destination City",
    destination_options
)

stops = st.sidebar.selectbox(
    "Transit / Stops",
    sorted(data["stops"].unique())
)

with st.sidebar.expander("Show Additional Options", expanded=False):
    departure_group_choice = st.selectbox(
        "Departure Time",
        ["Any", "Morning", "Afternoon", "Night"]
    )

    arrival_group_choice = st.selectbox(
        "Arrival Time",
        ["Any", "Morning", "Afternoon", "Night"]
    )

    week_group_choice = st.selectbox(
        "Weeks Before Departure",
        ["Any", "This Week", "2 Weeks", "3 Weeks", "1 Month", "More Than 1 Month"]
    )

    duration_group_choice = st.selectbox(
        "Duration Category",
        ["Any", "Short Flight", "Medium Flight", "Long Flight", "Very Long Flight"]
    )

filtered_flights = filter_dataset(
    data=data,
    airline=airline,
    flight_class=flight_class,
    source_city=source_city,
    destination_city=destination_city,
    stops=stops,
    departure_group_choice=departure_group_choice,
    arrival_group_choice=arrival_group_choice,
    week_group_choice=week_group_choice,
    duration_group_choice=duration_group_choice
)

model_input = build_model_input(
    filtered_data=filtered_flights,
    airline=airline,
    flight_class=flight_class,
    source_city=source_city,
    destination_city=destination_city,
    stops=stops,
    departure_group_choice=departure_group_choice,
    arrival_group_choice=arrival_group_choice,
    week_group_choice=week_group_choice,
    duration_group_choice=duration_group_choice
)

predicted_price = model.predict(model_input)[0]

scored_flights = get_match_score(filtered_flights, model_input)


# =========================================================
# PAGE 1: MAIN DASHBOARD
# =========================================================

if page == "🏠 Main Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Flight Price Prediction Dashboard</div>
        <div class="hero-subtitle">
            A visual machine learning dashboard that predicts flight ticket prices from historical airline data.
            Built for quick presentation, simple user input, and clear regression output.
        </div>
    </div>
    """, unsafe_allow_html=True)

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)

    with top_col1:
        show_mini_card("Dataset Rows", f"{len(data):,}", "Historical flight records")

    with top_col2:
        show_mini_card("Problem Type", "Regression", "Predicting a number")

    with top_col3:
        show_mini_card("Target", "Price", "Ticket price estimate")

    with top_col4:
        show_mini_card("Main Model", "Random Forest", "Tree-based ML model")

    st.write("")

    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        st.markdown('<div class="section-title">Current Selection</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="clean-card">
            <span class="pill">Airline: {airline}</span>
            <span class="pill">Class: {flight_class}</span>
            <span class="pill">Route: {source_city} → {destination_city}</span>
            <span class="pill">Transit: {stops}</span>
            <br><br>
            <span class="pill">Departure: {departure_group_choice}</span>
            <span class="pill">Arrival: {arrival_group_choice}</span>
            <span class="pill">Booking: {week_group_choice}</span>
            <span class="pill">Duration: {duration_group_choice}</span>
        </div>
        """, unsafe_allow_html=True)

        search_button = st.button("🔍 Predict Ticket Price", type="primary")

    with right_col:
        if search_button:
            show_price_card(
                predicted_price,
                "Estimated using selected flight information and historical Kaggle flight data."
            )
        else:
            st.markdown("""
            <div class="price-card">
                <div class="mini-label">Ready to Predict</div>
                <div class="price-value">Click Search</div>
                <div class="mini-note">Use the sidebar to select flight details, then generate the prediction.</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">Quick Visual Evidence</div>', unsafe_allow_html=True)

    evidence_col1, evidence_col2, evidence_col3 = st.columns(3)

    with evidence_col1:
        show_mini_card(
            "Matching Flights",
            f"{len(filtered_flights):,}",
            "Dataset records matching selected filters"
        )

    with evidence_col2:
        if len(filtered_flights) > 0:
            show_mini_card(
                "Average Similar Price",
                format_price(filtered_flights["price"].mean()),
                "Average from matching dataset rows"
            )
        else:
            show_mini_card(
                "Average Similar Price",
                "No Match",
                "Try fewer optional filters"
            )

    with evidence_col3:
        if len(filtered_flights) > 0:
            show_mini_card(
                "Cheapest Similar Price",
                format_price(filtered_flights["price"].min()),
                "Lowest matching dataset price"
            )
        else:
            show_mini_card(
                "Cheapest Similar Price",
                "No Match",
                "Try broader search"
            )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(make_airline_price_chart(data), use_container_width=True)

    with chart_col2:
        st.plotly_chart(make_class_price_chart(data), use_container_width=True)


# =========================================================
# PAGE 2: FLIGHT EXPLORER
# =========================================================

elif page == "🎫 Flight Explorer":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Dataset Flight Explorer</div>
        <div class="hero-subtitle">
            Search ticket-like records from the Kaggle dataset. This page is for visual demonstration,
            not live airline booking.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("This page pretends the Kaggle records are available ticket options for demo purposes.")

    explorer_col1, explorer_col2, explorer_col3, explorer_col4 = st.columns(4)

    with explorer_col1:
        show_mini_card("Found Tickets", f"{len(filtered_flights):,}", "Based on selected filters")

    with explorer_col2:
        if len(filtered_flights) > 0:
            show_mini_card("Lowest Price", format_price(filtered_flights["price"].min()), "Cheapest matching record")
        else:
            show_mini_card("Lowest Price", "-", "No records")

    with explorer_col3:
        if len(filtered_flights) > 0:
            show_mini_card("Average Price", format_price(filtered_flights["price"].mean()), "Average matching record")
        else:
            show_mini_card("Average Price", "-", "No records")

    with explorer_col4:
        if len(filtered_flights) > 0:
            show_mini_card("Highest Price", format_price(filtered_flights["price"].max()), "Most expensive matching record")
        else:
            show_mini_card("Highest Price", "-", "No records")

    st.write("")

    if len(filtered_flights) > 0:
        chart_col1, chart_col2 = st.columns([1.2, 1])

        with chart_col1:
            st.plotly_chart(make_price_distribution_chart(filtered_flights), use_container_width=True)

        with chart_col2:
            top_airline = filtered_flights.groupby("airline", as_index=False)["price"].mean()
            fig = px.bar(
                top_airline,
                x="airline",
                y="price",
                title="Average Price in Current Search"
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Closest Ticket Options</div>', unsafe_allow_html=True)

        table_columns = [
            "airline",
            "flight",
            "source_city",
            "destination_city",
            "departure_time",
            "arrival_time",
            "stops",
            "class",
            "duration",
            "days_left",
            "price"
        ]

        if "match_score" in scored_flights.columns:
            table_columns.append("match_score")

        st.dataframe(
            scored_flights[table_columns].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No matching ticket records found. Try changing optional filters to 'Any'.")


# =========================================================
# PAGE 3: VISUAL INSIGHTS
# =========================================================

elif page == "📊 Visual Insights":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Visual Insights</div>
        <div class="hero-subtitle">
            Key dataset patterns shown as simple visuals for presentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.plotly_chart(make_airline_price_chart(data), use_container_width=True)

    with insight_col2:
        st.plotly_chart(make_class_price_chart(data), use_container_width=True)

    insight_col3, insight_col4 = st.columns(2)

    with insight_col3:
        st.plotly_chart(make_days_price_chart(data), use_container_width=True)

    with insight_col4:
        route_summary = data.groupby(
            ["source_city", "destination_city"],
            as_index=False
        )["price"].mean()

        route_summary["route"] = route_summary["source_city"] + " → " + route_summary["destination_city"]
        route_summary = route_summary.sort_values("price", ascending=False).head(12)

        fig = px.bar(
            route_summary,
            x="price",
            y="route",
            orientation="h",
            title="Top Routes by Average Price",
            labels={
                "price": "Average Price",
                "route": "Route"
            }
        )

        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">Presentation Keywords</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        show_mini_card("Class Impact", "High", "Business class increases price")

    with k2:
        show_mini_card("Route Impact", "Strong", "Different cities have different pricing")

    with k3:
        show_mini_card("Booking Time", "Important", "Days left affects price")

    with k4:
        show_mini_card("Duration", "Useful", "Long flights often cost more")


# =========================================================
# PAGE 4: MODEL COMPARISON
# =========================================================

elif page == "🧠 Model Comparison":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Model Comparison</div>
        <div class="hero-subtitle">
            Compare regression models and explain why we choose the strongest model for prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    best_mae_model = model_results.sort_values("MAE").iloc[0]
    best_r2_model = model_results.sort_values("R2 Score", ascending=False).iloc[0]

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        show_mini_card("Best MAE", best_mae_model["Model"], format_price(best_mae_model["MAE"]))

    with model_col2:
        show_mini_card("Best R²", best_r2_model["Model"], f"{best_r2_model['R2 Score']:.3f}")

    with model_col3:
        show_mini_card("Selected Model", "Random Forest", "Balanced and strong")

    tab1, tab2, tab3 = st.tabs(["MAE", "RMSE", "R² Score"])

    with tab1:
        st.plotly_chart(make_model_comparison_chart(model_results, "MAE"), use_container_width=True)
        st.caption("Lower MAE is better. It means the model prediction is closer to actual price.")

    with tab2:
        st.plotly_chart(make_model_comparison_chart(model_results, "RMSE"), use_container_width=True)
        st.caption("Lower RMSE is better. It penalizes bigger errors more strongly.")

    with tab3:
        st.plotly_chart(make_model_comparison_chart(model_results, "R2 Score"), use_container_width=True)
        st.caption("Higher R² is better. It shows how well the model explains price variation.")

    st.markdown('<div class="section-title">Model Result Table</div>', unsafe_allow_html=True)

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Presentation point: We compare multiple regression models, then select the model with strong error performance."
    )


# =========================================================
# PAGE 5: ADVANCED NOTES
# =========================================================

elif page == "🔒 Advanced Notes":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Advanced Group Notes</div>
        <div class="hero-subtitle">
            This page is mainly for group members. Use it to answer tutor questions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What is the purpose of this Streamlit app?", expanded=True):
        st.write("""
        The purpose is to deploy a trained regression model so users can input new flight information
        and receive a predicted ticket price.
        """)

        st.write("""
        This is not a real booking system. It does not connect to airline APIs.
        It uses Kaggle historical flight data.
        """)

    with st.expander("Why is this a regression problem?"):
        st.write("""
        Regression predicts a continuous number.
        In this project, the target number is ticket price.
        """)

        st.write("""
        Classification would predict a category, such as cheap/expensive or yes/no.
        But our project predicts the actual price value, so it is regression.
        """)

    with st.expander("How does the model work?"):
        st.write("""
        The app uses flight features such as airline, class, route, transit, duration, and days left.
        Categorical features are converted using OneHotEncoder.
        Then Random Forest Regressor learns patterns from the dataset.
        """)

        st.code("""
CSV Dataset
→ Clean Data
→ Feature Selection
→ OneHotEncoder for categorical columns
→ RandomForestRegressor
→ Predicted Ticket Price
        """)

    with st.expander("Why use Random Forest?"):
        st.write("""
        Random Forest works well for tabular datasets because it combines many decision trees.
        It can capture non-linear relationships, such as how class, route, transit, and booking time affect price.
        """)

    with st.expander("Simple presentation script"):
        st.write("""
        1. Our group gets a regression problem.
        2. We use airline flight data to predict ticket price.
        3. The user selects flight information from the sidebar.
        4. The model predicts the estimated price.
        5. We also show visual insights and similar dataset records.
        6. This is not live ticket booking. It is a Data Science model deployment prototype.
        """)

    with st.expander("Limitations"):
        st.write("""
        - No live ticket availability.
        - No real Traveloka or Tiket.com API.
        - No real-time promotion data.
        - No exact calendar date or holiday feature.
        - Prediction depends only on the Kaggle dataset.
        """)

    st.divider()

    st.markdown('<div class="section-title">Raw Model Input Used for Prediction</div>', unsafe_allow_html=True)

    st.dataframe(
        model_input,
        use_container_width=True,
        hide_index=True
    )