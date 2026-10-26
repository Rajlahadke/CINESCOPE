from pathlib import Path
from html import escape
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent
BG = "#090b15"
PANEL = "#111625"
TEXT = "#e9eef8"
MUTED = "#93a1ba"
VIOLET = "#a78bfa"
CYAN = "#56d4db"
CORAL = "#fa8b80"
GOLD = "#f8c879"
COLORS = [VIOLET, CYAN, CORAL, GOLD, "#7ea5ff", "#86dba7", "#d6a4e8"]

# Extra inputs used by the upgraded model (post-release signals + one simulated feature)
EXTRA_FEATURES = ["vote_count", "popularity", "revenue", "simulated_critic_score"]

st.set_page_config(
    page_title="Cinescope | Movie Rating Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
:root {color-scheme: dark;}
html, body, [class*="css"], [data-testid="stApp"] {font-family: 'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse 800px 360px at 76% -100px, #252044 0%, transparent 72%), #090b15;}
[data-testid="stSidebar"] {background: #0e1220; border-right: 1px solid #242a3b;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #acb8cc;}
.block-container {max-width: 1450px; padding-top: 2.1rem; padding-bottom: 3rem;}
header[data-testid="stHeader"] {background: transparent;}
h1,h2,h3 {font-family:'Outfit',sans-serif !important; letter-spacing:-.035em; color:#edf1fa;}
h1 {font-weight: 750 !important;}
[data-testid="stMetric"] {background: linear-gradient(145deg,#151b2c,#111625); border:1px solid #293044; border-radius: 16px; padding:17px 19px; min-height:116px;}
[data-testid="stMetricLabel"] {color:#99a8c2; font-size:.85rem;}
[data-testid="stMetricValue"] {color:#f4f6fb; font-family:Outfit,sans-serif; font-weight:700;}
[data-testid="stMetricDelta"] {font-size:.76rem;}
[data-testid="stVerticalBlock"] > div {scroll-margin-top: 60px;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:10px;}
button[data-baseweb="tab"] {border-radius:9px; padding:0 15px;}
button[data-baseweb="tab"][aria-selected="true"] {color:#c9b5ff;}
[data-testid="stPlotlyChart"] {border:1px solid #242b3b; border-radius:16px; background:#101522; padding:8px; overflow:hidden;}
[data-testid="stDataFrame"] {border:1px solid #293043; border-radius:12px; overflow:hidden;}
[data-testid="stForm"] {background:#111624; border:1px solid #293043; border-radius:16px; padding:19px;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {background:linear-gradient(110deg,#8062dd,#624abc); color:white; border:0; border-radius:10px; font-weight:700;}
.stButton > button:hover, .stFormSubmitButton > button:hover {border-color:#a78bfa;}
.hero {background:radial-gradient(ellipse at 90% 12%,#473063aa,transparent 53%),linear-gradient(115deg,#19182f,#10172b 58%,#142132); border:1px solid #37324e; border-radius:20px; padding:31px 36px; margin-bottom:22px; overflow:hidden;}
.eyebrow {display:inline-block; font-size:.7rem; letter-spacing:.18em; font-weight:800; color:#bda7ff; text-transform:uppercase; margin-bottom:11px;}
.hero h1 {font-size:clamp(2.1rem,3.3vw,3.4rem); line-height:1.05; margin:0 0 13px; letter-spacing:-.055em;}
.hero p {color:#adb9d1; max-width:750px; margin:0; line-height:1.7; font-size:.96rem;}
.section-title {font-family:Outfit,sans-serif; font-weight:700; color:#f4f5ff; font-size:1.27rem; margin:19px 0 4px;}
.section-sub {color:#97a4bb; font-size:.88rem; margin-bottom:12px;}
.brand {font-family:Outfit,sans-serif; font-size:1.5rem; font-weight:800; letter-spacing:-.055em; color:#f6f4ff; padding:5px 0 0;}
.brand span {color:#ad94ff;}
.brand-caption {color:#818da9; letter-spacing:.14em; text-transform:uppercase; font-weight:700; font-size:.66rem; margin-bottom:24px;}
.info-box {padding:15px 18px; border:1px solid #363652; background:#191a30; border-radius:12px; color:#c6cbe2; line-height:1.55; font-size:.88rem; margin:12px 0;}
.movie-panel {background:linear-gradient(145deg,#1c2035,#121b2a); border:1px solid #30364c; border-radius:17px; padding:24px 26px; margin:7px 0 19px;}
.movie-title {font-family:Outfit,sans-serif; font-size:1.75rem; font-weight:750; color:#f4f6fc; line-height:1.2;}
.movie-detail {font-size:.9rem; color:#aeb9cc; line-height:1.9;}
.result-score {font-family:Outfit,sans-serif; font-size:3.55rem; font-weight:800; letter-spacing:-.08em; color:#f1edff; line-height:1.2;}
.result-tag {display:inline-block; padding:7px 12px; background:#243b45; border:1px solid #396676; color:#93f0e4; border-radius:99px; font-weight:700; font-size:.79rem;}
.caption-small {color:#8693ac; font-size:.8rem; line-height:1.55;}
hr {border-color:#293044;}
@media (max-width: 800px) {.hero{padding:24px 19px;} .block-container{padding-top:1.3rem;} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(ROOT / "movie_dashboard_data.csv")
    numeric = ["movie_id", "release_year", "release_decade", "genre_count", "budget", "revenue", "runtime", "popularity", "vote_average", "vote_count", "high_rating"]
    for column in numeric:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")
    for column in ("primary_genre", "original_language", "primary_country", "title"):
        movies[column] = movies[column].fillna("Unknown").astype(str)
    movies = movies.dropna(subset=["vote_average"])
    regression = pd.read_csv(ROOT / "regression_comparison.csv")
    classification = pd.read_csv(ROOT / "classification_comparison.csv")
    with open(ROOT / "model_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return movies, regression, classification, info


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    # Prefer the "_2" files; fall back to the un-suffixed names exported from Colab.
    native_path = ROOT / "rating_xgboost_2.json"
    preprocess_path = ROOT / "rating_preprocessor_2.pkl"
    if not (native_path.exists() and preprocess_path.exists()):
        native_path = ROOT / "rating_xgboost.json"
        preprocess_path = ROOT / "rating_preprocessor.pkl"
    if native_path.exists() and preprocess_path.exists():
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor

        regressor = XGBRegressor()
        regressor.load_model(str(native_path))
        regression_model = Pipeline([
            ("preprocess", joblib.load(preprocess_path)),
            ("model", regressor),
        ])
    else:
        try:
            regression_model = joblib.load(ROOT / "movie_rating_model.pkl")
        except Exception as error:
            raise RuntimeError(
                "Cannot load the original cross-platform XGBoost pickle. "
                "Export rating_xgboost.json and rating_preprocessor.pkl from Colab "
                "and place them beside app.py. Follow MODEL_FIX.md."
            ) from error
    classification_model = joblib.load(ROOT / "movie_classification_model.pkl")
    return regression_model, classification_model


def align_to_model(frame, model):
    """Return `frame` with exactly the columns (and order) the fitted model expects.
    Columns the model wants but we do not have are filled with NaN (the saved imputer handles them);
    columns the model does not know are dropped. This lets old and new model files both work."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return frame
    frame = frame.copy()
    for col in expected:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[list(expected)]


def header(label, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{escape(label)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title, description=""):
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-sub">{escape(description)}</div>', unsafe_allow_html=True)


def note(content):
    st.markdown(f'<div class="info-box">{escape(content)}</div>', unsafe_allow_html=True)


def chart(fig, height=345):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial", color=MUTED, size=12),
        title=None,
        margin=dict(l=16, r=16, t=22, b=22),
        height=height,
        xaxis=dict(gridcolor="#252b3b", zerolinecolor="#252b3b", linecolor="#363e52"),
        yaxis=dict(gridcolor="#252b3b", zerolinecolor="#252b3b", linecolor="#363e52"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1c2233", font_color="#f5f5ff"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def top_movies(data, count=10):
    x = data[data["vote_count"] >= 100].nlargest(count, "vote_average").copy()
    return x[["title", "release_year", "primary_genre", "vote_average", "vote_count"]].rename(columns={
        "title": "Movie", "release_year": "Year", "primary_genre": "Genre",
        "vote_average": "Rating", "vote_count": "Votes"
    })


def overview(df, reg, clf, info):
    header("01 / Overview", "The story behind the ratings.", "Explore the people-powered signals inside thousands of films: scores, genres, release years, and the patterns that connect them.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Movies analyzed", f"{len(df):,}")
    k2.metric("Average rating", f"{df['vote_average'].mean():.2f} / 10")
    k3.metric("Rated ≥ 7", f"{(df['vote_average'] >= 7).mean():.1%}")
    k4.metric("Genres", f"{df['primary_genre'].nunique()}")

    a, b = st.columns([1.18, 1], gap="large")
    with a:
        section("Rating distribution", "How audience scores are spread across the catalogue")
        fig = px.histogram(df, x="vote_average", nbins=32, color_discrete_sequence=[VIOLET], labels={"vote_average": "Movie rating (0–10)", "count": "Movies"})
        fig.add_vline(x=df["vote_average"].mean(), line_dash="dash", line_color=GOLD, annotation_text="Mean", annotation_font_color=GOLD)
        fig.update_traces(marker_line_width=0, opacity=.94)
        chart(fig)
    with b:
        section("Most common genres", "Based on each movie's primary genre")
        genres = df["primary_genre"].value_counts().head(9).sort_values()
        fig = px.bar(x=genres.values, y=genres.index, orientation="h", color_discrete_sequence=[CYAN], labels={"x": "Movies", "y": ""})
        fig.update_traces(marker_line_width=0)
        chart(fig)

    a, b = st.columns([1.15, 1], gap="large")
    with a:
        section("Ratings across decades", "Average rating by decade, at least 30 movies")
        decades = df.dropna(subset=["release_decade"]).groupby("release_decade").agg(rating=("vote_average", "mean"), movies=("title", "size")).reset_index()
        decades = decades[decades["movies"] >= 30]
        fig = px.line(decades, x="release_decade", y="rating", markers=True, color_discrete_sequence=[VIOLET], hover_data=["movies"], labels={"release_decade": "Decade", "rating": "Average rating"})
        fig.update_traces(line_width=3, marker_size=8)
        chart(fig)
    with b:
        section("Top rated movies", "Minimum 100 recorded votes")
        st.dataframe(top_movies(df, 10), use_container_width=True, hide_index=True, height=370,
                     column_config={"Rating": st.column_config.NumberColumn(format="%.1f"), "Year": st.column_config.NumberColumn(format="%d")})
    note("Dataset: The Movies Dataset (historical, through 2017). Dashboard charts use movies with at least 20 votes, as prepared in Colab. Scores here are observed TMDB vote averages, not our model's predictions.")


def explore(df):
    header("02 / Explorer", "Find the signal in every film.", "Filter the movie collection, spot standouts, and inspect individual records without leaving the dashboard.")
    a, b, c = st.columns([1.2, 1, 1])
    genres = ["All genres"] + sorted(df["primary_genre"].unique().tolist())
    languages = ["All languages"] + sorted(df["original_language"].unique().tolist())
    with a:
        genre = st.selectbox("Genre", genres)
    with b:
        language = st.selectbox("Original language", languages)
    with c:
        minimum_votes = st.select_slider("Minimum votes", options=[20, 50, 100, 250, 500, 1000], value=50)
    d, e = st.columns([1.8, 1])
    years = df["release_year"].dropna()
    with d:
        year_window = st.slider("Release years", int(years.min()), int(years.max()), (1980, int(years.max())))
    with e:
        search = st.text_input("Search title", placeholder="e.g. Interstellar")

    view = df[(df["vote_count"] >= minimum_votes) & df["release_year"].between(*year_window)].copy()
    if genre != "All genres":
        view = view[view["primary_genre"] == genre]
    if language != "All languages":
        view = view[view["original_language"] == language]
    if search.strip():
        view = view[view["title"].str.contains(search.strip(), case=False, regex=False, na=False)]
    x1, x2, x3 = st.columns(3)
    x1.metric("Matching movies", f"{len(view):,}")
    x2.metric("Average score", f"{view['vote_average'].mean():.2f}" if len(view) else "—")
    x3.metric("Highly rated", f"{(view['vote_average'] >= 7).mean():.1%}" if len(view) else "—")
    if view.empty:
        st.info("No movies match these filters. Try widening the year range or lowering the vote threshold.")
        return

    section("Filtered catalogue", "Sort columns, search, and download your subset")
    show = view.sort_values(["vote_average", "vote_count"], ascending=False)[["title", "release_year", "primary_genre", "original_language", "runtime", "vote_average", "vote_count", "budget"]]
    st.dataframe(show.rename(columns={"title": "Title", "release_year": "Year", "primary_genre": "Genre", "original_language": "Language", "runtime": "Runtime", "vote_average": "Rating", "vote_count": "Votes", "budget": "Budget ($)"}), use_container_width=True, hide_index=True, height=370)
    st.download_button("↓ Download filtered CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="movies_filtered.csv", mime="text/csv")

    if len(view) > 1:
        a, b = st.columns([1.2, 1], gap="large")
        with a:
            section("Votes vs rating", "Dot size represents number of audience votes")
            plot_data = view.sample(min(2500, len(view)), random_state=42)
            fig = px.scatter(plot_data, x="vote_count", y="vote_average", color="primary_genre" if plot_data["primary_genre"].nunique() <= 8 else None,
                             hover_name="title", opacity=.6, log_x=True, labels={"vote_count": "Votes (log scale)", "vote_average": "Rating"}, color_discrete_sequence=COLORS)
            fig.update_traces(marker=dict(size=6))
            chart(fig)
        with b:
            section("Top genre averages", "Only genres with at least 10 matching movies")
            group = view.groupby("primary_genre").agg(rating=("vote_average", "mean"), count=("title", "size")).query("count >= 10").nlargest(10, "rating").sort_values("rating")
            if len(group):
                fig = px.bar(x=group["rating"], y=group.index, orientation="h", color_discrete_sequence=[CORAL], labels={"x": "Mean rating", "y": ""})
                chart(fig)
            else:
                st.info("Not enough movies for a genre comparison.")

    section("Movie spotlight")
    choices = view.sort_values("vote_count", ascending=False).head(800).copy()
    choices["display"] = choices.apply(lambda r: f"{r['title']} ({int(r['release_year']) if pd.notna(r['release_year']) else 'N/A'}) · ID {int(r['movie_id'])}", axis=1)
    selected = st.selectbox("Select a movie (top 800 by vote count in this filter)", choices["display"].tolist())
    record = choices.loc[choices["display"] == selected].iloc[0]
    st.markdown(
        '<div class="movie-panel">'
        f'<div class="eyebrow">Movie profile</div><div class="movie-title">{escape(record["title"])}</div>'
        f'<div class="movie-detail">{escape(record["primary_genre"])} · {int(record["release_year"]) if pd.notna(record["release_year"]) else "Unknown year"} · {escape(record["original_language"])}</div>'
        f'<div class="movie-detail">Rating: <b>{record["vote_average"]:.1f}/10</b> &nbsp;|&nbsp; Votes: <b>{int(record["vote_count"]):,}</b> &nbsp;|&nbsp; Runtime: <b>{record["runtime"]:.0f} min</b></div>'
        '</div>', unsafe_allow_html=True,
    )


def statistics(df):
    header("03 / Statistics lab", "From observations to evidence.", "Descriptive statistics, hypothesis tests, confidence intervals, and sampling experiments from the SI Lab syllabus.")
    tab1, tab2, tab3 = st.tabs(["Descriptive statistics", "Hypothesis testing", "Sampling & CLT"])
    with tab1:
        labels = {"vote_average": "Movie rating", "runtime": "Runtime (minutes)", "vote_count": "Vote count", "budget": "Budget (USD)", "popularity": "Popularity", "revenue": "Revenue (USD)"}
        field = st.selectbox("Numeric variable", list(labels), format_func=lambda k: labels[k], key="stat_field")
        values = df[field].dropna()
        values = values[np.isfinite(values)]
        q1, med, q3 = values.quantile([.25, .5, .75])
        iqr = q3-q1
        low, high = q1-1.5*iqr, q3+1.5*iqr
        outliers = int(((values < low) | (values > high)).sum())
        k = st.columns(4)
        k[0].metric("Mean", f"{values.mean():,.2f}")
        k[1].metric("Median", f"{med:,.2f}")
        k[2].metric("Std. deviation", f"{values.std():,.2f}")
        k[3].metric("IQR outliers", f"{outliers:,}")
        a, b = st.columns([1.15, 1], gap="large")
        with a:
            section("Distribution")
            fig = px.histogram(x=values, nbins=45, color_discrete_sequence=[VIOLET], labels={"x": labels[field], "y": "Count"})
            chart(fig)
        with b:
            section("Box plot & outlier detection")
            fig = px.box(x=values, color_discrete_sequence=[CYAN], points="outliers", labels={"x": labels[field]})
            chart(fig)
        st.dataframe(pd.DataFrame({"Measure": ["Records", "Mean", "Median", "Mode", "Variance", "Standard deviation", "Min", "Q1", "Q2", "Q3", "Max", "IQR", "Range", "Outliers (1.5×IQR)"],
                                   "Value": [len(values), values.mean(), med, values.mode().iloc[0] if not values.mode().empty else np.nan, values.var(), values.std(), values.min(), q1, med, q3, values.max(), iqr, values.max()-values.min(), outliers]}), use_container_width=True, hide_index=True)
        confidence = st.radio("Confidence level", ["95%", "99%"], horizontal=True)
        conf = .95 if confidence == "95%" else .99
        if len(values) >= 2:
            left, right = stats.t.interval(conf, len(values)-1, loc=values.mean(), scale=stats.sem(values))
            note(f"{confidence} t-based confidence interval for the mean: [{left:,.3f}, {right:,.3f}]. Standard error: {stats.sem(values):,.4f}. This treats the observed movies as a sample; it does not guarantee representativeness of all films.")

    with tab2:
        section("Hypothesis test explorer", "Alpha = 0.05 unless changed below")
        alpha = st.select_slider("Significance level (α)", options=[.01, .05, .10], value=.05)
        test = st.selectbox("Test", ["One-sample t-test", "Independent Welch t-test", "ANOVA: differences between genres", "Chi-square: genre vs high rating"])
        if test == "One-sample t-test":
            baseline = st.number_input("Hypothesized average movie rating", min_value=0.0, max_value=10.0, value=6.0, step=.1)
            result = stats.ttest_1samp(df["vote_average"].dropna(), popmean=baseline)
            description = f"H₀: the mean movie rating is {baseline:.1f}. H₁: the mean differs from {baseline:.1f}."
            statistic, p = result.statistic, result.pvalue
        elif test == "Independent Welch t-test":
            genres = df["primary_genre"].value_counts().head(12).index.tolist()
            a, b = st.columns(2)
            g1 = a.selectbox("First genre", genres, index=0)
            g2 = b.selectbox("Second genre", genres, index=1)
            v1 = df.loc[df["primary_genre"] == g1, "vote_average"]
            v2 = df.loc[df["primary_genre"] == g2, "vote_average"]
            if g1 == g2:
                st.warning("Choose two different genres.")
                return
            result = stats.ttest_ind(v1, v2, equal_var=False)
            statistic, p = result.statistic, result.pvalue
            description = f"H₀: {g1} and {g2} have equal mean ratings. H₁: the means differ. Samples: {len(v1):,} and {len(v2):,}."
        elif test == "ANOVA: differences between genres":
            genres = df["primary_genre"].value_counts().head(6).index.tolist()
            groups = [df.loc[df["primary_genre"] == genre, "vote_average"] for genre in genres]
            result = stats.f_oneway(*groups)
            statistic, p = result.statistic, result.pvalue
            description = "H₀: the six most common genres have equal mean ratings. H₁: at least one mean differs."
        else:
            top = df["primary_genre"].value_counts().head(8).index
            table = pd.crosstab(df.loc[df["primary_genre"].isin(top), "primary_genre"], df.loc[df["primary_genre"].isin(top), "high_rating"])
            st.dataframe(table.rename(columns={0: "Below 7", 1: "7 or above"}), use_container_width=True)
            statistic, p, dof, expected = stats.chi2_contingency(table)
            description = "H₀: primary genre and high-rating status are independent. H₁: they are associated."
        st.markdown(f"**{description}**")
        a, b, c = st.columns(3)
        a.metric("Test statistic", f"{statistic:,.3f}")
        b.metric("P-value", f"{p:.3g}")
        c.metric("Decision", "Reject H₀" if p < alpha else "Fail to reject H₀")
        note("An observed p-value below α is evidence against the null hypothesis, not proof of a causal relationship. T-tests and ANOVA assume suitable sampling and data distributions; treat these findings as exploratory.")

    with tab3:
        section("Central Limit Theorem simulator", "Sample with replacement from recorded movie ratings")
        a, b = st.columns(2)
        n = a.select_slider("Movies per sample", options=[10, 30, 50, 100, 200], value=30)
        repetitions = b.slider("Number of samples", 100, 2000, 800, 100)
        rng = np.random.default_rng(42)
        population = df["vote_average"].dropna().to_numpy()
        means = rng.choice(population, size=(repetitions, n), replace=True).mean(axis=1)
        fig = px.histogram(x=means, nbins=38, color_discrete_sequence=[CYAN], labels={"x": "Sample mean rating", "y": "Samples"})
        fig.add_vline(x=population.mean(), line_color=GOLD, line_dash="dash", annotation_text="Population mean", annotation_font_color=GOLD)
        chart(fig, 365)
        a, b, c = st.columns(3)
        a.metric("Mean of sample means", f"{means.mean():.3f}")
        b.metric("Simulated standard error", f"{means.std(ddof=1):.4f}")
        c.metric("Theoretical standard error", f"{population.std(ddof=1)/np.sqrt(n):.4f}")
        section("Bootstrap confidence interval", "Resampling a selected group of movie ratings")
        sample_size = st.select_slider("Original sample size", options=[30, 50, 100, 200], value=50)
        sample = rng.choice(population, size=sample_size, replace=False)
        boot_means = rng.choice(sample, size=(1200, sample_size), replace=True).mean(axis=1)
        ci = np.percentile(boot_means, [2.5, 97.5])
        fig = px.histogram(x=boot_means, nbins=36, color_discrete_sequence=[VIOLET], labels={"x": "Bootstrap sample mean", "y": "Frequency"})
        fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor=VIOLET, opacity=.13, line_width=0)
        chart(fig, 330)
        note(f"Original sample mean: {sample.mean():.3f}. Bootstrap mean: {boot_means.mean():.3f}. Percentile-based 95% confidence interval: [{ci[0]:.3f}, {ci[1]:.3f}].")


def correlations(df):
    header("04 / Relationships", "Do movie features move together?", "Study correlation, covariance, heatmaps, and scatter plots. Relationships are descriptive — not causal claims.")
    fields = ["vote_average", "vote_count", "budget", "revenue", "runtime", "popularity", "genre_count", "release_year"]
    section("Correlation matrix", "Pearson correlations calculated using available numeric pairs")
    corr = df[fields].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=[[0, CORAL], [.5, "#1b2435"], [1, CYAN]], aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    chart(fig, 455)
    st.markdown(" ")
    a, b = st.columns(2)
    xcol = a.selectbox("Feature X", [c for c in fields if c != "vote_average"], index=4)
    ycol = b.selectbox("Feature Y", fields, index=0)
    pair = df[[xcol, ycol, "title"]].replace([np.inf, -np.inf], np.nan).dropna()
    if xcol == ycol:
        st.info("Choose two different variables to compare.")
        return
    if len(pair) >= 3 and pair[xcol].nunique() > 1 and pair[ycol].nunique() > 1:
        pearson, pp = stats.pearsonr(pair[xcol], pair[ycol])
        spearman, sp = stats.spearmanr(pair[xcol], pair[ycol])
        covariance = pair[xcol].cov(pair[ycol])
        a, b, c = st.columns(3)
        a.metric("Pearson r", f"{pearson:+.3f}")
        b.metric("Spearman ρ", f"{spearman:+.3f}")
        c.metric("Covariance", f"{covariance:,.2f}")
        section("Relationship explorer", f"{len(pair):,} complete paired observations; showing at most 3,000 points")
        points = pair.sample(min(3000, len(pair)), random_state=42)
        fig = px.scatter(points, x=xcol, y=ycol, hover_name="title", opacity=.53, color_discrete_sequence=[VIOLET], labels={xcol: xcol.replace('_', ' ').title(), ycol: ycol.replace('_', ' ').title()})
        fig.update_traces(marker=dict(size=6))
        chart(fig, 395)
        note(f"Pearson p-value: {pp:.3g}; Spearman p-value: {sp:.3g}. A near-zero Pearson correlation means weak linear association, not necessarily no relationship.")
    else:
        st.info("Not enough varying numeric observations for this comparison.")


def predictor(df, info):
    header("05 / Prediction studio", "What rating might this movie earn?", "Set the characteristics of a movie. Our trained XGBoost regression pipeline estimates its rating; the SVM separately predicts whether it belongs to the 7+ category.")
    try:
        import sklearn
        if sklearn.__version__ != "1.6.1":
            st.error(f"The saved models require scikit-learn 1.6.1. Your active environment has {sklearn.__version__}.")
            st.code("python -m pip install -r requirements.txt", language="bash")
            st.stop()
        regression_model, classification_model = load_models()
    except Exception as exc:
        st.error("The trained models could not be loaded. Install the packages in requirements.txt and restart Streamlit.")
        st.exception(exc)
        st.stop()

    # Which input columns does the loaded regression model actually expect?
    expected = list(getattr(regression_model, "feature_names_in_", info["features"]))
    uses_extras = [f for f in EXTRA_FEATURES if f in expected]

    genres = df["primary_genre"].value_counts().index.tolist()
    languages = df["original_language"].value_counts().index.tolist()
    countries = df["primary_country"].value_counts().index.tolist()
    extras = {}
    with st.form("predict_form"):
        section("Movie details", "Enter film information available before its release")
        a, b, c = st.columns(3, gap="large")
        with a:
            genre = st.selectbox("Primary genre", genres, index=genres.index("Drama") if "Drama" in genres else 0)
            genre_count = st.number_input("Number of genres", min_value=0, max_value=8, value=2, step=1)
            language = st.selectbox("Original language", languages, index=0, format_func=lambda k: f"{k.upper()}" if k != "Unknown" else "Unknown")
        with b:
            year = st.number_input("Release year", min_value=1878, max_value=2035, value=2015, step=1)
            runtime_unknown = st.checkbox("Runtime unknown", value=False)
            runtime = st.number_input("Runtime (minutes)", min_value=1, max_value=500, value=110, step=5, disabled=runtime_unknown)
        with c:
            country = st.selectbox("Primary production country", countries, index=0)
            budget_unknown = st.checkbox("Budget unknown", value=False)
            budget_millions = st.number_input("Production budget (USD millions)", min_value=0.0, max_value=1000.0, value=30.0, step=5.0, disabled=budget_unknown)

        if uses_extras:
            section("Extra signals (upgraded model)", "These inputs are only used by the upgraded model. Leave 'unknown' ticked to let the model fill typical values.")
            d, e, f_, g = st.columns(4, gap="large")
            if "vote_count" in uses_extras:
                with d:
                    votes_unknown = st.checkbox("Vote count unknown", value=False)
                    extras["vote_count"] = np.nan if votes_unknown else float(st.number_input("Vote count", min_value=0, max_value=2_000_000, value=500, step=50, disabled=votes_unknown))
            if "popularity" in uses_extras:
                with e:
                    pop_unknown = st.checkbox("Popularity unknown", value=False)
                    extras["popularity"] = np.nan if pop_unknown else float(st.number_input("Popularity score", min_value=0.0, max_value=600.0, value=10.0, step=1.0, disabled=pop_unknown))
            if "revenue" in uses_extras:
                with f_:
                    rev_unknown = st.checkbox("Revenue unknown", value=True)
                    revenue_millions = st.number_input("Revenue (USD millions)", min_value=0.0, max_value=3000.0, value=50.0, step=10.0, disabled=rev_unknown)
                    extras["revenue"] = np.nan if rev_unknown else revenue_millions * 1_000_000
            if "simulated_critic_score" in uses_extras:
                with g:
                    extras["simulated_critic_score"] = float(st.slider("Simulated critic score (experimental)", 0.0, 10.0, 6.5, 0.1))
        submitted = st.form_submit_button("✦ Predict movie rating", type="primary", use_container_width=True)

    if submitted:
        row = {
            "budget": np.nan if budget_unknown else budget_millions * 1_000_000,
            "runtime": np.nan if runtime_unknown else float(runtime),
            "release_year": int(year),
            "genre_count": int(genre_count),
            "primary_genre": genre,
            "original_language": language,
            "primary_country": country,
        }
        row.update(extras)
        movie = pd.DataFrame([row])
        try:
            score = float(regression_model.predict(align_to_model(movie, regression_model))[0])
            classification = int(classification_model.predict(align_to_model(movie, classification_model))[0])
        except Exception as exc:
            st.error("Prediction failed. The models may have been exported with incompatible dependencies.")
            st.exception(exc)
            return
        display_score = np.clip(score, 0, 10)
        section("Your predicted result")
        a, b = st.columns([1.05, 1], gap="large")
        with a:
            kind = "pre-release" if not uses_extras else "input"
            st.markdown(
                f'<div class="movie-panel"><div class="eyebrow">XGBoost predicted rating</div>'
                f'<div class="result-score">{display_score:.2f}<span style="font-size:1.3rem;color:#9faacf"> / 10</span></div>'
                f'<div class="caption-small">Regression estimate from {len(expected)} {kind} features</div></div>',
                unsafe_allow_html=True,
            )
        with b:
            label = "Likely 7+ rated" if classification == 1 else "Likely below 7"
            st.markdown(
                f'<div class="movie-panel"><div class="eyebrow">SVM classification</div>'
                f'<div class="result-tag">{label}</div>'
                f'<div class="movie-detail" style="margin-top:12px">High-rating threshold: 7.0 / 10</div>'
                f'<div class="caption-small">The SVM was trained without calibrated probabilities; this label is not a confidence percentage.</div></div>',
                unsafe_allow_html=True,
            )
        if score < 0 or score > 10:
            st.warning(f"Raw regression output was {score:.2f}. The displayed score is limited to 0–10 for readability.")
        if year > 2017:
            st.warning("This release year is beyond the training dataset (which ends in 2017). Treat the prediction as an extrapolation.")
        if classification != int(score >= info["classification_threshold"]):
            st.info("The regression and classification outputs disagree about the 7+ threshold. They are separately trained models and can legitimately differ.")
        section("Comparable movies in the dataset", "Same primary genre; historical examples, not model recommendations")
        peers = df[(df["primary_genre"] == genre) & (df["vote_count"] >= 100)]
        if len(peers):
            peers = peers.assign(year_gap=(peers["release_year"]-year).abs()).sort_values(["year_gap", "vote_count"], ascending=[True, False]).head(6)
            st.dataframe(peers[["title", "release_year", "vote_average", "vote_count"]].rename(columns={"title": "Title", "release_year": "Year", "vote_average": "Actual rating", "vote_count": "Votes"}), use_container_width=True, hide_index=True)

        rmse = None
        metrics = info.get("regression_metrics") or []
        if metrics:
            rmse = min(m["RMSE"] for m in metrics if "RMSE" in m)
        rmse_text = f"A test RMSE around {rmse:.3f} points" if rmse is not None else "The test RMSE"
        if uses_extras:
            note(f"{rmse_text} applies to this upgraded model. It also uses post-release signals (votes, popularity, revenue) and a simulated critic score that is derived from the true rating plus noise, so its accuracy reflects those inputs and is not a genuine pre-release forecast. Predictions are educational estimates.")
        else:
            note(f"Predictions are educational estimates, not guaranteed future ratings. {rmse_text} indicates substantial uncertainty for individual films. No votes, revenue, or popularity are fed into this model.")
    else:
        if uses_extras:
            note("Tip: this upgraded model also reads vote count, popularity, revenue and a simulated critic score. The critic score strongly drives the prediction. Tick 'unknown' to let the saved preprocessing fill typical values. Press the purple button when ready.")
        else:
            note("Tip: Missing budgets and runtimes are handled by the model's saved preprocessing pipeline. Predictions reflect historical training patterns up to 2017. Press the purple button when ready.")



@st.cache_resource(show_spinner="Loading review sentiment model...")
def load_sentiment_model():
    return joblib.load(ROOT / "sentiment_model.pkl")


@st.cache_data(show_spinner=False)
def load_sentiment_results():
    with open(ROOT / "sentiment_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    comparison = pd.read_csv(ROOT / "sentiment_comparison.csv")
    confusion = pd.read_csv(ROOT / "sentiment_confusion_matrix.csv", index_col=0)
    roc = pd.read_csv(ROOT / "sentiment_roc.csv")
    terms = pd.read_csv(ROOT / "sentiment_terms.csv")
    return metrics, comparison, confusion, roc, terms


def review_sentiment():
    header("06 / Review intelligence", "What does the audience really think?", "Analyze the sentiment of a written movie review using a real IMDb-trained TF-IDF text classifier. This predicts positive or negative feedback, not a 0–10 movie score.")
    expected = ["sentiment_model.pkl", "sentiment_metrics.json", "sentiment_comparison.csv",
                "sentiment_confusion_matrix.csv", "sentiment_roc.csv", "sentiment_terms.csv"]
    missing = [name for name in expected if not (ROOT / name).is_file()]
    if missing:
        st.info("The review sentiment model has not been exported from Colab yet.")
        st.markdown("Run **CINESCOPE_IMDb_Sentiment_Training.ipynb** in Colab, download **cinescope_sentiment_assets.zip**, and extract the six files beside `app.py`. Then refresh this page.")
        with st.expander("Files required"):
            st.code("\n".join(expected), language="text")
        return
    try:
        model = load_sentiment_model()
        metrics, comparison, confusion, roc, terms = load_sentiment_results()
    except Exception as exc:
        st.error("Unable to load the sentiment artifacts. Check that all six exported files are present and scikit-learn==1.6.1 is installed.")
        st.exception(exc)
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
    k2.metric("F1 score", f"{metrics['f1']:.3f}")
    k3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    k4.metric("Reviews tested", f"{metrics['test_size']:,}")
    note(f"Model: {metrics['selected_model']}. Accuracy was measured on an untouched ~20% held-out test split, not guessed or hardcoded. Majority-class baseline: {metrics['majority_baseline_accuracy']:.0%}.")

    tab1, tab2, tab3 = st.tabs(["Try a review", "Test-set results", "Model insights"])
    with tab1:
        section("Your review, analyzed", "Write a movie review in English. Positive or negative is the target; there is no numeric star rating prediction on this page.")
        with st.form("review_form"):
            review = st.text_area("Movie review", placeholder="Write a few sentences about the movie's story, acting, pacing and overall experience...", height=185)
            submitted = st.form_submit_button("✦ Analyze review", type="primary", use_container_width=True)
        if submitted:
            if len(review.split()) < 4:
                st.warning("Enter at least four words so the review contains enough information to analyze.")
            else:
                try:
                    label = int(model.predict([review])[0])
                    margin = float(model.decision_function([review])[0])
                    name = "Positive review" if label == 1 else "Negative review"
                    chip = "#93f0e4" if label == 1 else "#fa8b80"
                    st.markdown(
                        f'<div class="movie-panel"><div class="eyebrow">Model prediction</div>'
                        f'<div class="movie-title" style="color:{chip};font-size:2rem">{name}</div>'
                        f'<div class="movie-detail" style="margin-top:12px">Decision margin: {margin:+.3f}</div>'
                        f'<div class="caption-small">This margin measures distance from the decision boundary; it is not a percentage or calibrated confidence.</div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error("Could not predict the sentiment of this review.")
                    st.exception(exc)
        section("Test several reviews at once", "Optional: upload a CSV with a review column to generate predictions.")
        csv_file = st.file_uploader("Review CSV", type=["csv"], key="reviews_csv")
        if csv_file is not None:
            try:
                batch = pd.read_csv(csv_file)
                if "review" not in batch.columns:
                    st.error("Your CSV must contain a column named review.")
                elif len(batch) == 0 or len(batch) > 2000:
                    st.warning("Upload between 1 and 2,000 rows per batch.")
                else:
                    texts = batch["review"].fillna("").astype(str)
                    batch["predicted_sentiment"] = np.where(model.predict(texts) == 1, "positive", "negative")
                    st.dataframe(batch.head(40), use_container_width=True, hide_index=True)
                    st.download_button("↓ Download predictions", batch.to_csv(index=False).encode("utf-8"),
                                       file_name="sentiment_predictions.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Could not process CSV: {exc}")
    with tab2:
        section("Validation comparison", "The final model was chosen using validation data, not the test set")
        st.dataframe(comparison.style.format({"Validation Accuracy": "{:.2%}", "Validation F1": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
        a, b = st.columns(2, gap="large")
        with a:
            section("Confusion matrix", "Unseen test reviews")
            fig = px.imshow(confusion.values, text_auto=True, color_continuous_scale=["#16192d", "#7c64cc", "#c3a9fa"],
                            labels={"x": "Predicted sentiment", "y": "Actual sentiment", "color": "Reviews"},
                            x=["Negative", "Positive"], y=["Negative", "Positive"], aspect="equal")
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, 370)
        with b:
            section("ROC curve", "Area under the curve evaluates ranking across decision thresholds")
            fig = px.line(roc, x="fpr", y="tpr", color_discrete_sequence=[VIOLET], labels={"fpr": "False positive rate", "tpr": "True positive rate"})
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GOLD, dash="dash"), showlegend=False))
            chart(fig, 370)
        note("Accuracy measures correct sentiment predictions, not correctness of numerical movie ratings. These results refer to the IMDb 50K review dataset; the original TMDB movie-rating dataset is a different task.")
    with tab3:
        section("Interpretable word signals", "Terms with the strongest linear model weights")
        a, b = st.columns(2, gap="large")
        with a:
            positive = terms.loc[terms["sentiment"] == "positive"].sort_values("weight").tail(12)
            fig = px.bar(positive, x="weight", y="term", orientation="h", color_discrete_sequence=[CYAN], labels={"term": "", "weight": "Positive weight"})
            chart(fig, 440)
        with b:
            negative = terms.loc[terms["sentiment"] == "negative"].sort_values("weight", ascending=False).tail(12)
            fig = px.bar(negative, x="weight", y="term", orientation="h", color_discrete_sequence=[CORAL], labels={"term": "", "weight": "Negative weight"})
            chart(fig, 440)
        note("TF-IDF converts word and two-word phrase frequencies into numeric features. Linear SVM / Logistic Regression then learns weights. This model may struggle with sarcasm, mixed opinions, and unusually short reviews.")


def models(df, regression, classification, info):
    header("06 / Model lab", "Benchmark every approach.", "Side-by-side performance from your Colab experiments, with separate evaluation metrics for regression and classification.")
    st.markdown("**Trained pipelines:** XGBoost regressor and SVM classifier · **Test split:** 20% · **Training filter:** ≥20 votes")
    tab1, tab2 = st.tabs(["Regression performance", "Classification performance"])
    with tab1:
        best = regression.sort_values("RMSE").iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Best regressor", str(best["Model"]))
        b.metric("RMSE", f"{best['RMSE']:.3f}")
        c.metric("MAE", f"{best['MAE']:.3f}")
        d.metric("R² score", f"{best['R2']:.3f}")
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            section("RMSE by model", "Lower is better")
            display = regression.sort_values("RMSE", ascending=False)
            fig = px.bar(display, x="RMSE", y="Model", orientation="h", color_discrete_sequence=[VIOLET], text="RMSE")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            chart(fig, 385)
        with b:
            section("Full regression comparison")
            st.dataframe(regression.style.format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), use_container_width=True, hide_index=True, height=388)
        if best["R2"] >= 0.6:
            note(f"{best['Model']} has the lowest test RMSE ({best['RMSE']:.3f}) and an R² of {best['R2']:.3f}. This upgraded benchmark includes post-release signals (votes, popularity, revenue) and a simulated critic-score feature built from the true rating plus noise, so it should not be read as pre-release prediction accuracy.")
        else:
            note(f"{best['Model']} has the lowest test RMSE ({best['RMSE']:.3f}), but an R² of {best['R2']:.3f} means it explains only about {best['R2']:.0%} of test-set rating variance. This is useful as a demonstration, not a precision-rating guarantee.")
    with tab2:
        chosen = classification.loc[classification["Model"] == info["best_classifier"]].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Selected model", str(chosen["Model"]))
        b.metric("ROC-AUC", f"{chosen['ROC_AUC']:.3f}")
        c.metric("F1 score", f"{chosen['F1']:.3f}")
        d.metric("Accuracy", f"{chosen['Accuracy']:.1%}")
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            section("F1 score by model", "Balances precision and recall")
            display = classification.sort_values("F1", ascending=True)
            fig = px.bar(display, x="F1", y="Model", orientation="h", color_discrete_sequence=[CYAN], text="F1")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            chart(fig, 385)
        with b:
            section("Full classification comparison")
            st.dataframe(classification.style.format({col: "{:.3f}" for col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}), use_container_width=True, hide_index=True, height=388)
        top_acc = classification.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        if top_acc != chosen["Model"]:
            note(f"{top_acc} has the highest accuracy here, but the saved {chosen['Model']} was selected for its stronger F1 and ROC-AUC. The classes were defined using a 7/10 threshold; high accuracy alone can be misleading when classes are imbalanced.")
        else:
            note(f"{chosen['Model']} has the highest accuracy and was selected as the saved classifier. The classes were defined using a 7/10 threshold; high accuracy alone can be misleading when classes are imbalanced.")
    section("Model input schema", "Features stored in both fitted preprocessing pipelines")
    st.dataframe(pd.DataFrame({"Feature": info["features"], "Type": ["Numeric" if f in info["numeric_features"] else "Categorical" for f in info["features"]]}), hide_index=True, use_container_width=True)


REQUIRED = ["movie_dashboard_data.csv", "regression_comparison.csv", "classification_comparison.csv", "model_info.json"]
missing = [name for name in REQUIRED if not (ROOT / name).exists()]
if missing:
    st.error("Missing project files: " + ", ".join(missing))
    st.info("Keep app.py and all exported model/data files together in the same folder.")
    st.stop()

movies, regression_results, classification_results, model_info = load_data()
with st.sidebar:
    st.markdown('<div class="brand">◉ CINE<span>SCOPE</span></div><div class="brand-caption">Movie rating intelligence</div>', unsafe_allow_html=True)
    st.caption("EXPLORE THE PROJECT")
    page = st.radio("Navigation", ["Overview", "Movie explorer", "Statistics lab", "Correlations", "Rating predictor", "Review sentiment", "Model comparisons"], label_visibility="collapsed", index=0)
    st.divider()
    st.caption("YOUR TRAINED MODELS")
    st.markdown(f"**Regression**  · {model_info['best_regressor']}")
    st.markdown(f"**Classification**  · {model_info['best_classifier']}")
    st.divider()
    st.caption(f"{len(movies):,} movies • Original data through 2017")
    st.caption("SI Lab · Python · Streamlit")

if page == "Overview":
    overview(movies, regression_results, classification_results, model_info)
elif page == "Movie explorer":
    explore(movies)
elif page == "Correlations":
    correlations(movies)
elif page == "Statistics lab":
    statistics(movies)
elif page == "Rating predictor":
    predictor(movies, model_info)
elif page == "Review sentiment":
    review_sentiment()
else:
    models(movies, regression_results, classification_results, model_info)
