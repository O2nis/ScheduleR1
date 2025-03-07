import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import io

# ====================
# 1) Matplotlib Style
# ====================
plt.style.use("fivethirtyeight")

# =====================
# 2) Streamlit Settings
# =====================
st.set_page_config(page_title="Gantt Chart App", layout="wide")
st.title("Project Schedule & Man-Hours Visualization")

# =========================
# 3) Global Constants/Funcs
# =========================
def get_period_start_end_dates(
    period_number: int,
    project_start: datetime,
    time_period: str
):
    """
    Given an integer period_number (1, 2, 3, ...),
    return (start_date, end_date) for either a month or a week,
    based on time_period value: 'Months' or 'Weeks'.
    """
    if time_period == "Months":
        # For months
        start_date = project_start + relativedelta(months=period_number - 1)
        end_date = start_date + relativedelta(months=1) - timedelta(days=1)
    else:
        # For weeks
        start_date = project_start + timedelta(weeks=period_number - 1)
        end_date = start_date + timedelta(days=6)  # 7-day span
    return start_date, end_date

def add_working_days(
    start_date: datetime,
    duration_days: int,
    work_days_per_week: int
) -> datetime:
    """
    Add working days (Mon-Fri by default, or as set by user) to a start date.
    """
    current_date = start_date
    days_added = 0
    while days_added < duration_days:
        if current_date.weekday() < work_days_per_week:
            days_added += 1
        current_date += timedelta(days=1)
    return current_date - timedelta(days=1)

def calculate_duration_days(
    man_hours: float,
    hours_per_day: float,
    work_days_per_week: int
) -> int:
    """
    Calculate calendar days needed for man_hours,
    given 'hours_per_day' and 'work_days_per_week'.
    """
    total_working_days = man_hours / hours_per_day
    total_calendar_days = total_working_days / work_days_per_week * 7
    return int(np.ceil(total_calendar_days))

def map_discipline(role, discipline_keywords):
    """
    Map each role to a discipline using keywords.
    """
    role_lower = str(role).lower()
    for discipline, keywords in discipline_keywords.items():
        for keyword in keywords["keywords"]:
            if keyword.lower() in role_lower:
                return discipline
    return 'Other'

def generate_random_color():
    """
    Generate a random color that is not too dark.
    """
    while True:
        r = random.random()
        g = random.random()
        b = random.random()
        # Ensure the color is not too dark
        if (r + g + b) / 3 > 0.5:
            return mcolors.to_hex((r, g, b))

# =============================
# 4) Streamlit Sidebar Settings
# =============================
st.sidebar.header("Configuration")

# -- Time Period Mode --
time_period = st.sidebar.radio(
    "Select Time Period",
    ["Weeks", "Months"],
    index=1
)

# -- Basic date and hours/days definitions --
project_start_date = st.sidebar.date_input(
    "Project Start Date",
    value=datetime(2025, 1, 1)
)

hours_per_day = st.sidebar.number_input(
    "Hours per Day",
    value=9,
    min_value=1,
    max_value=24
)

work_days_per_week = st.sidebar.number_input(
    "Working Days per Week",
    value=5,
    min_value=1,
    max_value=7
)

work_days_per_month = st.sidebar.number_input(
    "Working Days per Month",
    value=22,
    min_value=1,
    max_value=31
)

# Decide how many hours are in "one period" based on the chosen time period
if time_period == "Months":
    hours_per_period = hours_per_day * work_days_per_month
else:
    hours_per_period = hours_per_day * work_days_per_week

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your Schedule CSV",
    type=["csv"]
)

# ============================
# 5) Discipline Configuration
# ============================
st.sidebar.header("Discipline Configuration")

# Default disciplines and keywords
default_disciplines = {
    "Civil": {"keywords": ["Civil", "GIS", "Geotechnical"], "color": "#3498db"},
    "Electrical": {"keywords": ["Electrical", "PV"], "color": "#abebc6"},
    "HSE": {"keywords": ["HSE", "Safety"], "color": "#E60000"},
    "Instrument": {"keywords": ["Instrument", "SCADA", "I&C", "Automation"], "color": "#d2b4de"},
    "Management": {"keywords": ["Project Manager", "DCC", "Project Engineer", "Manager"], "color": "#2ecc71"},
    "Mechanical": {"keywords": ["Mechanical"], "color": "#ec7063"},
    "Site Supervision": {"keywords": ["Supervisor", "Commissioning"], "color": "#aed6f1"},
    "Other": {"keywords": [], "color": "#f8c471"}
}

# Initialize session state for disciplines
if "disciplines" not in st.session_state:
    st.session_state.disciplines = default_disciplines

# Add new discipline
new_discipline = st.sidebar.text_input("Add New Discipline")
new_keywords = st.sidebar.text_input("Keywords for New Discipline (comma-separated)")
if st.sidebar.button("Add Discipline"):
    if new_discipline and new_keywords:
        # Generate a random color for the new discipline
        new_color = generate_random_color()
        st.session_state.disciplines[new_discipline] = {
            "keywords": [k.strip() for k in new_keywords.split(",")],
            "color": new_color
        }
        st.sidebar.success(f"Added {new_discipline} with keywords: {new_keywords} and color: {new_color}")
    else:
        st.sidebar.error("Please provide both a discipline name and keywords.")

# Remove discipline
remove_discipline = st.sidebar.selectbox(
    "Select Discipline to Remove",
    list(st.session_state.disciplines.keys())
)
if st.sidebar.button("Remove Discipline"):
    if remove_discipline in st.session_state.disciplines:
        del st.session_state.disciplines[remove_discipline]
        st.sidebar.success(f"Removed {remove_discipline}")
    else:
        st.sidebar.error("Discipline not found.")

# Modify discipline color
modify_discipline = st.sidebar.selectbox(
    "Select Discipline to Modify Color",
    list(st.session_state.disciplines.keys())
)
new_color = st.sidebar.color_picker(f"Choose a new color for {modify_discipline}", st.session_state.disciplines[modify_discipline]["color"])
if st.sidebar.button("Update Color"):
    st.session_state.disciplines[modify_discipline]["color"] = new_color
    st.sidebar.success(f"Updated color for {modify_discipline} to {new_color}")

# Display current disciplines
st.sidebar.subheader("Current Disciplines")
for discipline, data in st.session_state.disciplines.items():
    st.sidebar.write(f"- **{discipline}**: {', '.join(data['keywords'])} (Color: {data['color']})")

# ============================
# 6) If File Uploaded, Process
# ============================
if uploaded_file is not None:
    # ---------- Read CSV ----------
    df = pd.read_csv(uploaded_file, na_values=['', ' ', 'NA', 'NaN'])
    df.columns = df.columns.str.strip()

    # Identify columns (adjust if your CSV has different headers)
    service_col = "Service"
    role_col = "Role"

    # The schedule columns are assumed to start from the 4th column onward
    period_columns = df.columns[3:]

    # Convert those columns to integers (e.g., "1", "2", etc.)
    period_numbers = [int(col) for col in period_columns]

    # Map each 'period' column to a start date based on Weeks or Months
    period_col_date_mapping = {}
    for col in period_columns:
        p_num = int(col)
        start_d, _ = get_period_start_end_dates(p_num, project_start_date, time_period)
        period_col_date_mapping[col] = start_d

    # ---------- Clean & Convert Period Data ----------
    for col in period_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({',': '.', '%': ''}, regex=True)  # transform decimal & remove "%"
            .replace('', np.nan)
        )
    df[period_columns] = df[period_columns].apply(pd.to_numeric, errors='coerce')

    # Remove columns that are completely empty or zero
    df_periods = df[period_columns]
    non_empty_periods = df_periods.columns[
        (df_periods.notna() & (df_periods != 0)).any()
    ]
    period_columns = non_empty_periods
    # Update the mapping to only include non-empty periods
    period_col_date_mapping = {
        c: period_col_date_mapping[c] for c in period_columns
    }

    # =============================
    # Calculate Total Man-Hours
    # =============================
    st.subheader("Total Man-Hours")

    # Total man-hours across all periods
    total_man_hours = df[period_columns].sum().sum() * hours_per_period
    st.write(f"**Total Man-Hours (All Periods):** {total_man_hours:.2f}")

    # Total man-hours by week or month
    st.subheader(f"Total Man-Hours by {time_period}")
    man_hours_by_period = df[period_columns].sum() * hours_per_period
    man_hours_by_period_df = pd.DataFrame({
        "Period": period_columns,
        "Start Date": [period_col_date_mapping[col] for col in period_columns],
        "Man-Hours": man_hours_by_period
    })
    st.dataframe(man_hours_by_period_df)

    # Total man-hours by discipline
    st.subheader("Total Man-Hours by Discipline")
    df["Discipline"] = df[role_col].apply(lambda x: map_discipline(x, st.session_state.disciplines))
    man_hours_by_discipline = df.groupby("Discipline")[period_columns].sum().sum(axis=1) * hours_per_period
    st.dataframe(man_hours_by_discipline.reset_index().rename(columns={0: "Man-Hours"}))

    # Total man-hours by role
    st.subheader("Total Man-Hours by Role")
    man_hours_by_role = df.groupby(role_col)[period_columns].sum().sum(axis=1) * hours_per_period
    st.dataframe(man_hours_by_role.reset_index().rename(columns={0: "Man-Hours"}))

    # =============================
    # 6) Gantt Chart by "Service"
    # =============================
    st.subheader(f"Gantt Chart by Service ({time_period})")

    gantt_data_services = []
    for task in df[service_col].unique():
        task_data = df[df[service_col] == task]
        for col in period_columns:
            period_sum = task_data[col].sum()
            if pd.notna(period_sum) and period_sum > 0:
                man_hours = period_sum * hours_per_period
                duration_days = calculate_duration_days(
                    man_hours, hours_per_day, work_days_per_week
                )

                p_num = int(col)
                start_date, end_date = get_period_start_end_dates(
                    p_num, project_start_date, time_period
                )

                # Adjust start_date to the first working day (if needed)
                while start_date.weekday() >= work_days_per_week:
                    start_date += timedelta(days=1)

                finish_date = add_working_days(
                    start_date,
                    duration_days - 1,
                    work_days_per_week
                )

                # If the finish date goes beyond the "nominal" period end_date, clamp it
                if finish_date > end_date:
                    finish_date = end_date

                gantt_data_services.append({
                    "Task": task,
                    "Start": start_date,
                    "Finish": finish_date,
                    "Duration": (finish_date - start_date).days + 1,
                    "Man_Hours": man_hours
                })

    gantt_df_services = pd.DataFrame(gantt_data_services)

    if not gantt_df_services.empty:
        tasks_ordered = (
            gantt_df_services.groupby("Task")["Start"]
            .min()
            .sort_values()
            .index
            .tolist()
        )
        num_tasks = len(tasks_ordered)
        task_positions = {
            task: num_tasks - i - 1 for i, task in enumerate(tasks_ordered)
        }

        fig_services, ax_services = plt.subplots(figsize=(14, 7))

        gantt_df_services["Start_num"] = mdates.date2num(gantt_df_services["Start"])
        gantt_df_services["End_num"] = mdates.date2num(gantt_df_services["Finish"])
        gantt_df_services["Duration"] = (
            gantt_df_services["End_num"] - gantt_df_services["Start_num"]
        )

        norm = mcolors.Normalize(
            vmin=gantt_df_services["Man_Hours"].min(),
            vmax=gantt_df_services["Man_Hours"].max()
        )
        custom_cmap_service = LinearSegmentedColormap.from_list("CyanGreen", ["cyan", "darkgreen"])

        for _, row in gantt_df_services.iterrows():
            task = row["Task"]
            y_pos = task_positions[task]
            man_hours_norm = norm(row["Man_Hours"])
            color = custom_cmap_service(man_hours_norm)

            ax_services.barh(
                y=y_pos,
                width=row["Duration"],
                left=row["Start_num"],
                height=0.4,
                align="center",
                color=color,
                edgecolor="black"
            )

            r, g, b, a = color
            hsv = colorsys.rgb_to_hsv(r, g, b)
            brightness = hsv[2]
            text_color = "black" if brightness > 0.5 else "white"
            x_center = row["Start_num"] + row["Duration"] / 2

            ax_services.text(
                x_center,
                y_pos,
                f"{row['Man_Hours']:.1f}h",
                va="center",
                ha="center",
                color=text_color,
                fontsize=7
            )

        ax_services.set_yticks(range(num_tasks))
        ax_services.set_yticklabels(
            [tasks_ordered[num_tasks - i - 1] for i in range(num_tasks)],
            fontsize=8
        )

        ax_services.xaxis_date()
        date_fmt = mdates.DateFormatter("%Y-%m-%d")
        ax_services.xaxis.set_major_formatter(date_fmt)
        plt.xticks(rotation=45, fontsize=8)

        ax_services.set_ylabel("Tasks (Ordered by Start Date)", fontsize=9)
        ax_services.set_xlabel("Date", fontsize=9)
        ax_services.set_title(f"Project Schedule Gantt Chart by Service ({time_period})", fontsize=10)

        sm = ScalarMappable(cmap=custom_cmap_service, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_services)
        cbar.set_label("Man-Hours", fontsize=9)

        plt.tight_layout()

        # Save the chart as a high-resolution PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        # Display the chart
        st.pyplot(fig_services)

        # Add a download button for the chart
        st.download_button(
            label="Download Gantt Chart by Service (PNG)",
            data=buf,
            file_name="gantt_chart_by_service.png",
            mime="image/png"
        )
    else:
        st.info("No data found for 'Service' to plot Gantt Chart.")

    # =========================
    # 8) Gantt Chart by "Role"
    # =========================
    st.subheader(f"Gantt Chart by Role ({time_period})")

    gantt_data_roles = []
    for role in df[role_col].unique():
        role_data = df[df[role_col] == role]
        for col in period_columns:
            period_sum = role_data[col].sum()
            if pd.notna(period_sum) and period_sum > 0:
                man_hours = period_sum * hours_per_period
                duration_days = calculate_duration_days(
                    man_hours, hours_per_day, work_days_per_week
                )

                p_num = int(col)
                start_date, end_date = get_period_start_end_dates(
                    p_num, project_start_date, time_period
                )

                while start_date.weekday() >= work_days_per_week:
                    start_date += timedelta(days=1)

                finish_date = add_working_days(
                    start_date,
                    duration_days - 1,
                    work_days_per_week
                )
                if finish_date > end_date:
                    finish_date = end_date

                gantt_data_roles.append({
                    "Role": role,
                    "Start": start_date,
                    "Finish": finish_date,
                    "Duration": (finish_date - start_date).days + 1,
                    "Man_Hours": man_hours
                })

    gantt_df_roles = pd.DataFrame(gantt_data_roles)

    if not gantt_df_roles.empty:
        roles_ordered = (
            gantt_df_roles.groupby("Role")["Start"]
            .min()
            .sort_values()
            .index
            .tolist()
        )
        num_roles = len(roles_ordered)
        role_positions = {r: num_roles - i - 1 for i, r in enumerate(roles_ordered)}

        fig_roles, ax_roles = plt.subplots(figsize=(14, 7))
        gantt_df_roles["Start_num"] = mdates.date2num(gantt_df_roles["Start"])
        gantt_df_roles["End_num"] = mdates.date2num(gantt_df_roles["Finish"])
        gantt_df_roles["Duration"] = (
            gantt_df_roles["End_num"] - gantt_df_roles["Start_num"]
        )

        norm_roles = mcolors.Normalize(
            vmin=gantt_df_roles["Man_Hours"].min(),
            vmax=gantt_df_roles["Man_Hours"].max()
        )
        custom_cmap_role = LinearSegmentedColormap.from_list("YellowRed", ["yellow", "red"])

        for _, row in gantt_df_roles.iterrows():
            role = row["Role"]
            y_pos = role_positions[role]
            man_hours_norm = norm_roles(row["Man_Hours"])
            color = custom_cmap_role(man_hours_norm)

            ax_roles.barh(
                y=y_pos,
                width=row["Duration"],
                left=row["Start_num"],
                height=0.4,
                align="center",
                color=color,
                edgecolor="black"
            )

            r, g, b, a = color
            hsv = colorsys.rgb_to_hsv(r, g, b)
            brightness = hsv[2]
            text_color = "black" if brightness > 0.5 else "white"
            x_center = row["Start_num"] + row["Duration"] / 2

            ax_roles.text(
                x_center,
                y_pos,
                f"{row['Man_Hours']:.1f}h",
                va="center",
                ha="center",
                color=text_color,
                fontsize=7
            )

        ax_roles.set_yticks(range(num_roles))
        ax_roles.set_yticklabels(
            [roles_ordered[num_roles - i - 1] for i in range(num_roles)],
            fontsize=8
        )

        ax_roles.xaxis_date()
        date_fmt = mdates.DateFormatter("%Y-%m-%d")
        ax_roles.xaxis.set_major_formatter(date_fmt)
        plt.xticks(rotation=45, fontsize=8)

        ax_roles.set_ylabel("Roles (Ordered by Start Date)", fontsize=9)
        ax_roles.set_xlabel("Date", fontsize=9)
        ax_roles.set_title(f"Project Schedule Gantt Chart by Role ({time_period})", fontsize=10)

        sm_roles = ScalarMappable(cmap=custom_cmap_role, norm=norm_roles)
        sm_roles.set_array([])
        cbar_roles = plt.colorbar(sm_roles, ax=ax_roles)
        cbar_roles.set_label("Man-Hours", fontsize=9)

        plt.tight_layout()

        # Save the chart as a high-resolution PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        # Display the chart
        st.pyplot(fig_roles)

        # Add a download button for the chart
        st.download_button(
            label="Download Gantt Chart by Role (PNG)",
            data=buf,
            file_name="gantt_chart_by_role.png",
            mime="image/png"
        )
    else:
        st.info("No data found for 'Role' to plot Gantt Chart.")

    # =================================
    # 9) Histogram: Man-Hours per Period
    # =================================
    st.subheader(f"Man-Hours per Discipline Over {time_period} (Stacked Histogram)")

    # Map role -> discipline using user-defined disciplines
    df["Discipline"] = df[role_col].apply(lambda x: map_discipline(x, st.session_state.disciplines))

    # Prepare an empty DataFrame
    disciplines = sorted(df["Discipline"].unique())
    discipline_data = pd.DataFrame(index=disciplines, columns=period_columns)

    # Fill the discipline_data with man-hours
    for disc in disciplines:
        disc_rows = df[df["Discipline"] == disc]
        for col in period_columns:
            total_man_hours = 0.0
            for _, row in disc_rows.iterrows():
                percentage = row[col]
                if pd.notna(percentage) and 0 <= percentage <= 1:
                    total_man_hours += percentage * hours_per_period
            discipline_data.loc[disc, col] = total_man_hours

    # Convert to float, fill NaN with 0
    discipline_data = discipline_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Transpose for plotting
    discipline_data_t = discipline_data.transpose()

    # Convert the index from strings ("1", "2", ...) to actual start dates
    discipline_data_t.index = [period_col_date_mapping[col] for col in discipline_data_t.index]
    discipline_data_t.sort_index(inplace=True)

    # Choose a smaller bar width if Weeks; a larger if Months
    width_weeks = 3
    width_months = 20

    # Define colors for disciplines
    discipline_colors = {disc: st.session_state.disciplines[disc]["color"] for disc in disciplines}

    fig_hist, ax_hist = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(discipline_data_t))

    for disc in discipline_data_t.columns:
        values = discipline_data_t[disc].values.astype(float)
        color = discipline_colors.get(disc, "#000000")

        # Decide bar width
        bar_width = width_weeks if time_period == "Weeks" else width_months

        bars = ax_hist.bar(
            discipline_data_t.index,
            values,
            bottom=bottom,
            label=disc,
            color=color,
            width=bar_width
        )
        # Annotate each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_hist.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{height:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                    rotation=90
                )
        bottom += values

    ax_hist.xaxis_date()
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax_hist.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45, fontsize=9)

    ax_hist.set_xlabel(time_period, fontsize=10)
    ax_hist.set_ylabel("Man-Hours", fontsize=10)
    plt.title(f"Man-Hours per Discipline Over {time_period}", fontsize=11)
    plt.legend(title="Discipline", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the chart as a high-resolution PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # Display the chart
    st.pyplot(fig_hist)

    # Add a download button for the chart
    st.download_button(
        label="Download Man-Hours Histogram (PNG)",
        data=buf,
        file_name="man_hours_histogram.png",
        mime="image/png"
    )

    # ========================================
    # 10) Donut Chart: Total Man-Hours by Disc.
    # ========================================
    st.subheader("Donut Chart: Total Man-Hours per Discipline")

    total_man_hours_per_discipline = discipline_data.sum(axis=1)
    percentages = (
        total_man_hours_per_discipline
        / total_man_hours_per_discipline.sum()
    ) * 100

    labels = total_man_hours_per_discipline.index
    sizes = percentages.values
    colors = [discipline_colors.get(d, "#000000") for d in labels]

    fig_donut, ax_donut = plt.subplots(figsize=(8, 8))
    center_circle = plt.Circle((0, 0), 0.70, fc="white")

    wedges, texts, autotexts = ax_donut.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=0.85,
        labeldistance=1.05,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"color": "black", "fontsize": 12}
    )
    ax_donut.add_artist(center_circle)
    ax_donut.axis("equal")
    plt.title("Percentage Distribution of Total Man-Hours per Discipline", fontsize=11)
    plt.tight_layout()

    # Save the chart as a high-resolution PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # Display the chart
    st.pyplot(fig_donut)

    # Add a download button for the chart
    st.download_button(
        label="Download Donut Chart (PNG)",
        data=buf,
        file_name="donut_chart.png",
        mime="image/png"
    )

else:
    # If no file uploaded, prompt user
    st.info("Please upload a CSV file to continue.")
