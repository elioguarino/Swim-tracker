import os
import time
import base64
import bcrypt
import requests
import numpy as np
import streamlit as st
from io import BytesIO
from datetime import date, timedelta

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go

from PIL import Image, ImageDraw, ImageOps
from supabase import create_client, Client
import streamlit.components.v1 as components


# ============================================================
# SUPABASE CONNECTION
# ============================================================
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "logged_in": False,
    "current_user": None,
    "current_user_id": None,
    "current_day": None,
    "show_register": False,
    "login_status": None,
    "show_settings": False,
    "rgba": None,
    "last_uploaded": None,
    "show_change_username": False,
    "show_change_password": False,
    "line_colour": None,
    "profile_pic": None,
    "show_lc_change": False,
    "compare_group_ids": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PAGE CONFIG + STYLING
# ============================================================
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #00d2ff 0%, #2edaff 50%, #9bedff 100%);
    }
    .stApp * {
        color: #00008B !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2edaff 0%, #73e6ff 100%);
        border-right: 1px solid rgba(255,255,255,0.3);
    }

    header[data-testid="stHeader"] {
        background-color: #00d2ff;
    }

    h1, h2, h3, h4, .stMarkdown p {
        color: #06304a;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    div[data-testid="stToolbarActions"] {display: none;}
    .stAppDeployButton {display: none;}

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea,
    input[data-testid="stDateInputField"] {
        background-color: #ffffff !important;
        color: #06304a !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
        outline: none !important;
        padding: 8px 12px !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }

    @media (max-width: 640px) {
        div[class*="st-key-log_swim_btn"],
        div[class*="st-key-delete_swim_btn"] {
            position: static !important;
            top: 0 !important;
            left: 0 !important;
        }
    }

    /* keep title + pfp side-by-side even on mobile */
    div[class*="st-key-top_bar_container"] div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 8px !important;
    }
    div[class*="st-key-top_bar_container"] div[data-testid="stColumn"] {
        width: auto !important;
        min-width: 0 !important;
    }
    
    div[class*="st-key-pfp_btn"] {
        text-align: center !important;
    }
    div[class*="st-key-pfp_btn"] button {
        display: inline-block !important;
    }

    @media (max-width: 640px) {
        div[class*="st-key-pfp_btn"] button {
            width: 56px !important;
            height: 56px !important;
            min-width: 56px !important;
        }
        div[class*="st-key-top_bar_container"] .stMarkdown div {
            font-size: 16px !important;
        }
    }

    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"] {
        display: none !important;
    }

    .stButton button {
        background-color: #ffffff;
        color: #06304a;
        border: none;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        font-weight: 600;
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        color: #4dd8ff;
    }

    div[data-testid="stExpander"] {
        background-color: rgba(255,255,255,0.35);
        border-radius: 12px;
        border: none;
    }
    </style>
    
""", unsafe_allow_html=True)


# ============================================================
# IMAGE HELPERS  (module-level + cached)
# ============================================================
def img_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@st.cache_data(ttl=300, show_spinner=False)
def make_circular(image_source, border_color, size=200, border_width=10, padding=0):
    canvas_size = size + (padding * 2)

    if image_source.startswith("http"):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        img = Image.open(image_source).convert("RGBA")

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")
    img = img.resize((size, size))

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.paste(img, (padding, padding), img)

    # circular mask
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((padding, padding, padding + size, padding + size), fill=255)
    canvas.putalpha(mask)

    # ring outline
    draw_ring = ImageDraw.Draw(canvas)
    draw_ring.ellipse(
        (border_width // 2, border_width // 2,
         canvas_size - border_width // 2, canvas_size - border_width // 2),
        outline=border_color, width=border_width
    )

    return np.array(canvas)

def _dim_colour(hex_colour, factor=0.55):
    """Lightens a hex colour toward white — used to fade out non-selected pie slices."""
    hex_colour = (hex_colour or "#000000").lstrip('#')
    try:
        r, g, b = int(hex_colour[0:2], 16), int(hex_colour[2:4], 16), int(hex_colour[4:6], 16)
    except ValueError:
        return "#cccccc"
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
# ============================================================
# PROFILE PICTURE FETCH  (module-level + cached)
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def get_prof_pic(user_id):
    default_path = "assets/default_prof.png"
    try:
        profile = supabase.table("profiles").select("avatar_path").eq("id", user_id).single().execute()
        avatar_path = profile.data.get("avatar_path")

        if not avatar_path:
            return default_path

        public_url = supabase.storage.from_("avatars").get_public_url(avatar_path)
        return public_url
    except Exception:
        return default_path


# ============================================================
# GROUP MEMBERSHIP FETCH  (module-level + cached, shared by sidebar + compare list)
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def get_my_memberships(user_id):
    result = supabase.table("group_members")\
        .select("group_id, groups(name)")\
        .eq("user_id", user_id)\
        .execute()
    return result.data or []


@st.cache_data(ttl=30, show_spinner=False)
def get_group_members(group_id):
    result = supabase.table("group_members")\
        .select("profiles(username)")\
        .eq("group_id", group_id)\
        .execute()
    return result.data or []


# ============================================================
# SWIM DATA FETCH  (module-level + cached, cleared on insert/delete)
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def get_last_7_days_totals(user_id):
    today = date.today()
    start_date = today - timedelta(days=6)

    response = supabase.table("swims") \
        .select("swim_date, distance_m") \
        .eq("user_id", user_id) \
        .gte("swim_date", start_date.isoformat()) \
        .lte("swim_date", today.isoformat()) \
        .execute()

    totals = {}
    for row in response.data:
        d = row["swim_date"]
        totals[d] = totals.get(d, 0) + row["distance_m"]

    labels, values = [], []
    for i in range(7):
        d = start_date + timedelta(days=i)
        labels.append(d.strftime("%A"))
        values.append(totals.get(d.isoformat(), 0))

    return labels, values


@st.cache_data(ttl=30, show_spinner=False)
def get_group_totals(group_ids):
    try:
        all_member_ids = set()
        member_info = {}

        for group_id in group_ids:
            members = supabase.table("group_members")\
                .select("user_id, profiles(username, line_colour)")\
                .eq("group_id", group_id)\
                .execute()

            for member in members.data:
                member_id = member["user_id"]
                all_member_ids.add(member_id)
                member_info[member_id] = {
                    "username": member["profiles"]["username"],
                    "colour": member["profiles"]["line_colour"] or "#000000"
                }

        today = date.today()
        start_date = today - timedelta(days=6)
        all_members_data = []

        for member_id in all_member_ids:
            response = supabase.table("swims") \
                .select("swim_date, distance_m") \
                .eq("user_id", member_id) \
                .gte("swim_date", start_date.isoformat()) \
                .lte("swim_date", today.isoformat()) \
                .execute()

            totals = {}
            for row in response.data:
                d = row["swim_date"]
                totals[d] = totals.get(d, 0) + row["distance_m"]

            values = []
            for i in range(7):
                d = start_date + timedelta(days=i)
                values.append(totals.get(d.isoformat(), 0))

            all_members_data.append({
                "user_id": member_id,
                "username": member_info[member_id]["username"],
                "colour": member_info[member_id]["colour"],
                "values": values
            })

        return all_members_data
    except Exception as e:
        st.error(f"error loading group data: {e}")
        return []


def clear_swim_caches():
    """Call right after any insert/delete of a swim so the graph shows fresh data."""
    get_last_7_days_totals.clear()
    get_group_totals.clear()


def clear_membership_caches():
    """Call right after join/leave/create group so sidebar + compare list refresh."""
    get_my_memberships.clear()
    get_group_members.clear()


# ============================================================
# ACCOUNT CREATION
# ============================================================
def create_new():
    st.write("create new account")
    new_email = st.text_input("enter your email adress")
    new_pas = st.text_input("enter password: ", type="password")
    new_user = st.text_input("enter Display name: ")

    if st.button("confirm"):
        try:
            supabase.auth.sign_up({
                "email": new_email,
                "password": new_pas,
                "options": {"data": {"pending_username": new_user}}
            })

            st.session_state.show_register = False
            st.success("Account created! Check your email to confirm, then log in")
            time.sleep(0.6)
            st.rerun()
        except Exception as e:
            st.error(f"signup failed: {e}")


# ============================================================
# GRAPH + SWIM LOGGING FRAGMENT
# (only this reruns when you log/delete a swim or hit refresh —
#  the sidebar, groups list, etc. are left untouched)
# ============================================================

@st.dialog("Swimmer contribution")
def show_member_popup(member, percentage):
    pic_path = get_prof_pic(member["user_id"])
    pic = make_circular(pic_path, border_color=member["colour"] or "#000000", padding=10)
    st.image(pic, width=120)
    st.markdown(f"### {member['username']}")
    st.write(f"**Distance this week:** {sum(member['values']):,} m")
    st.write(f"**Share of group total:** {percentage:.1f}%")
    if st.button("Close"):
        st.session_state["group_pie_chart"] = {"selection": {"points": []}}
        st.rerun()

@st.fragment
def render_swim_section():
    selected_group_ids = st.session_state.compare_group_ids or []
    show_group_view = len(selected_group_ids) > 0

    # ------------------------------------------------------------
    # PLOT GRAPH
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#73E6FF")
    ax.set_facecolor("#9bedff")

    weekday_labels = [(date.today() - timedelta(days=6 - i)).strftime("%A") for i in range(7)]

    if show_group_view:
        group_data = get_group_totals(tuple(selected_group_ids))

        for member in group_data:
            ax.plot(weekday_labels, member["values"], marker="o",
                     color=member["colour"] or "#000000", label=member["username"])

            if member["user_id"] == st.session_state.current_user_id:
                ax.fill_between(weekday_labels, member["values"], color=member["colour"] or "#000000", alpha=0.15)

            last_x = weekday_labels[-1]
            last_y = member["values"][-1]
            member_icon = make_circular(get_prof_pic(member["user_id"]), border_color=member["colour"] or "#000000", padding=10)
            imagebox = OffsetImage(member_icon, zoom=0.12)
            ab = AnnotationBbox(imagebox, (last_x, last_y), frameon=False)
            ax.add_artist(ab)

        ax.legend()

        all_values = [v for member in group_data for v in member["values"]]
        ax.set_ylim(0, max(all_values) * 1.1 if all_values else 1)

    else:
        labels, values = get_last_7_days_totals(st.session_state.current_user_id)

        ax.plot(labels, values, marker="o", color=st.session_state.line_colour or "#000000")
        ax.fill_between(labels, values, color=st.session_state.line_colour or "#000000", alpha=0.15)

        img = make_circular(get_prof_pic(st.session_state.current_user_id), border_color=st.session_state.line_colour or "#000000", padding=10)
        last_x, last_y = labels[-1], values[-1]
        imagebox = OffsetImage(img, zoom=0.18)
        ab = AnnotationBbox(imagebox, (last_x, last_y), frameon=False)
        ax.add_artist(ab)

        ax.set_ylim(0, max(values) * 1.1 if values else 1)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis='both', which='major', linestyle='--', alpha=0.7)
    ax.grid(True, axis='y', which='minor', linestyle='--', alpha=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.tick_params(axis='y', which='minor', length=3)

    st.pyplot(fig)
    plt.close(fig)

    # ---------- pool / sea tabs ----------
    pool_tab, open_water_tab, stopwatch_tab = st.tabs(["Pool", "open water", "stopwatch (temp)"])

    with pool_tab:

        # ------------------------------------------------------------
        # LOG SWIM
        # ------------------------------------------------------------
        with st.popover("Log swim", width="stretch"):
            lengths = st.number_input(
                "how many lengths?",
                step=10, value=0, min_value=0, width="stretch"
            )
            pool_length = st.slider(
                "Pool length (metres)",
                step=10, value=10, min_value=0, max_value=50, width="stretch"
            )
            log_date = st.date_input(
                "for which day? (default - today)",
                value=date.today(), max_value=date.today()
            )
            st.write("Time (optional)")
            a, b, c = st.columns([1,1,1])

            with a:
                time_taken_h = st.number_input("hours", min_value=0, value=0, step=1)
            with b:
                time_taken_m = st.number_input("minutes", min_value=0, max_value=59, value=0, step=1)
            with c:
                time_taken_s = st.number_input("seconds", min_value=0, max_value=59, value=0, step=1)
            total_time_s = int((time_taken_h*3600) + (time_taken_m*60) + time_taken_s)



            a1 = st.button("log swim", width="stretch", key="log_swim_btn")
            if a1:
                try:
                    mtrs = lengths * pool_length
                    supabase.table("swims").insert({
                        "user_id": st.session_state.current_user_id,
                        "swim_date": log_date.isoformat(),
                        "distance_m": mtrs,
                        "duration_seconds": total_time_s if total_time_s else None,
                    }).execute()
                    clear_swim_caches()
                    with st.spinner("adding data..."):
                        time.sleep(1.5)
                        st.success(f"added {mtrs} metres to {log_date.strftime('%A, %B %d')}!")
                    st.rerun(scope="fragment")
                except Exception as e:
                    st.error(f"error:{e}")

    cool_dividy_things1, cool_dividy_things2 = st.columns([1, 1])
    with cool_dividy_things1:
        # ------------------------------------------------------------
        # DELETE SWIM
        # ------------------------------------------------------------
        delete_last_swim = st.button("Delete last swim", width="stretch", key="delete_swim_btn")
        if delete_last_swim:
            try:
                last_swim = supabase.table("swims") \
                    .select("id")\
                    .eq("user_id", st.session_state.current_user_id)\
                    .order("created_at", desc=True)\
                    .limit(1)\
                    .execute()

                if last_swim.data:
                    swim_id = last_swim.data[0]["id"]
                    supabase.table("swims").delete().eq("id", swim_id).execute()
                    clear_swim_caches()
                    st.success("last swim deleted")
                else:
                    st.warning("no swims to delete")
                st.rerun(scope="fragment")
            except Exception as e:
                st.error(f"error: {e}")
    with cool_dividy_things2:
        if st.button("Refresh graph", width="stretch"):
            clear_swim_caches()
            with st.spinner():
                st.rerun(scope="fragment")

    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------
    col1, col2 = st.columns([1,1])
    with col1:
        if show_group_view:
            member_totals = [sum(member["values"]) for member in group_data]
            total_group_distance = sum(member_totals)
            if total_group_distance > 0:
                percentages = [
                    (t / total_group_distance * 100) if total_group_distance else 0
                    for t in member_totals
                ]

                prior_selection = st.session_state.get("group_pie_chart")
                selected_index = None
                if prior_selection and prior_selection.get("selection", {}).get("points"):
                    selected_index = prior_selection["selection"]["points"][0]["point_index"]

                colours = [member["colour"] or "#000000" for member in group_data]
                pull = [0.12 if i == selected_index else 0 for i in range(len(group_data))]
                display_colours = [
                    colours[i] if (selected_index is None or i == selected_index) else _dim_colour(colours[i])
                    for i in range(len(group_data))
                ]

                pie_fig = go.Figure(data=[go.Pie(
                    labels=[member["username"] for member in group_data],
                    values=member_totals,
                    marker=dict(colors=display_colours, line=dict(color="#73E6FF", width=2)),
                    pull=pull,
                    textinfo="none",
                    hoverinfo="skip",
                    sort=False,
                )])
                pie_fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="#73E6FF",
                    plot_bgcolor="#73E6FF",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=350,
                )

                st.plotly_chart(
                    pie_fig,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key="group_pie_chart",
                )


                if selected_index is not None:
                    show_member_popup(group_data[selected_index], percentages[selected_index])
            else:
                st.info("No swims logged by the group in the last 7 days yet.")
        else:
            st.caption("Select a group in the sidebar to see the group's contribution breakdown.")
    with col2:

        #group total

        if show_group_view:
            member_totals = [sum(member["values"]) for member in group_data]
            total_group_distance = sum(member_totals)
            st.markdown(
                f"<div style='text-align:center; font-weight:bold; font-size:1.0rem; color:#06304a;'>"
                f"Total group distance (last 7 days): {total_group_distance:,} m</div>",
                unsafe_allow_html=True
            )
        
        #individual total

        labels, values = get_last_7_days_totals(st.session_state.current_user_id)
        #im not using labels so i hope its fine to just... leave it lol
        total_individual_distance = 0
        for day in values:
            total_individual_distance = total_individual_distance + day

        st.markdown(
            f"<div style='text-align:center; font-weight:bold; font-size:1.0rem; color:#06304a;'>"
            f"Total individual distance (last 7 days): {total_individual_distance:,} m</div>",
            unsafe_allow_html=True
        )      

        #individual average pace (7 days)

        total_week_seconds = 0

        today = date.today()
        a_week_ago = today - timedelta(days=6)
        try:
            response = supabase.table("swims")\
                .select("swim_date", "duration_seconds")\
                .eq("user_id", st.session_state.current_user_id)\
                .gte("swim_date", a_week_ago.isoformat())\
                .lte("swim_date", today.isoformat())\
                .not_.is_("duration_seconds", "null")\
                .execute()

            for row in response.data:
                total_week_seconds += row["duration_seconds"]

            total_timed_week_meters = sum(row["distance_m"] for row in response.data)

            #ok now we have total distance swam (with times added on - otherwise it doesnt count) in the week, and total time taken in the week

            if total_timed_week_meters > 0:
                raw_pace_seconds = (total_week_seconds/total_timed_week_meters)
                pace_minutes, pace_seconds = divmod(int(raw_pace_seconds), 60)

                st.markdown(
                    f"<div style='text-align:center; font-weight:bold; font-size:1.0rem; color:#06304a;'>"
                    f"Average pace (last 7 days): {pace_minutes:,},{pace_seconds:,} per 100m </div>",
                    unsafe_allow_html=True
                )   

        except Exception as e:
            st.error(f"error:{e}")



        
    with open_water_tab:
        st.write("Sea swim tracking coming soon.")

    with stopwatch_tab:
        components.html("""
            <div style="display:flex; flex-direction:column; align-items:center; gap:16px; padding:16px; font-family:sans-serif;">
                <div id="sw-display" style="font-size:3rem; font-weight:bold; color:#00008B;">00:00.000</div>
                <div style="display:flex; gap:12px;">
                    <button id="sw-toggle" style="background:#ffffff; color:#06304a; border:none; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.15); font-weight:600; padding:10px 20px; cursor:pointer;">Start</button>
                    <button id="sw-reset" style="background:#ffffff; color:#06304a; border:none; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.15); font-weight:600; padding:10px 20px; cursor:pointer; opacity:0.5;" disabled>Reset</button>
                </div>
            </div>
            <script>
            let running = false;
            let startTime = 0;
            let elapsedBeforePause = 0;
            let rafId = null;

            const display = document.getElementById('sw-display');
            const toggleBtn = document.getElementById('sw-toggle');
            const resetBtn = document.getElementById('sw-reset');

            function format(ms) {
                const totalMs = Math.floor(ms);
                const minutes = String(Math.floor(totalMs / 60000)).padStart(2, '0');
                const seconds = String(Math.floor((totalMs % 60000) / 1000)).padStart(2, '0');
                const millis = String(totalMs % 1000).padStart(3, '0');
                return minutes + ":" + seconds + "." + millis;
            }

            function tick() {
                const now = performance.now();
                const elapsed = elapsedBeforePause + (now - startTime);
                display.innerText = format(elapsed);
                rafId = requestAnimationFrame(tick);
            }

            function updateResetState() {
                if (running || elapsedBeforePause === 0) {
                    resetBtn.disabled = true;
                    resetBtn.style.opacity = "0.5";
                    resetBtn.style.cursor = "default";
                } else {
                    resetBtn.disabled = false;
                    resetBtn.style.opacity = "1";
                    resetBtn.style.cursor = "pointer";
                }
            }

            toggleBtn.addEventListener('click', function() {
                if (!running) {
                    running = true;
                    startTime = performance.now();
                    rafId = requestAnimationFrame(tick);
                    toggleBtn.innerText = "Pause";
                } else {
                    running = false;
                    elapsedBeforePause += performance.now() - startTime;
                    cancelAnimationFrame(rafId);
                    toggleBtn.innerText = "Resume";
                }
                updateResetState();
            });

            resetBtn.addEventListener('click', function() {
                if (running) return;
                elapsedBeforePause = 0;
                display.innerText = "00:00.000";
                toggleBtn.innerText = "Start";
                updateResetState();
            });

            updateResetState();
            </script>
        """, height=180)






# ============================================================
# MAIN APP
# ============================================================
def main():

    if st.session_state.show_settings:
        settings_page()
        return

    # ---------- sidebar: your groups ----------
    def groups_sidebar():
        st.sidebar.header("Your Groups:")

        try:
            my_memberships = get_my_memberships(st.session_state.current_user_id)

            if my_memberships:
                for m in my_memberships:
                    group_id = m["group_id"]
                    group_name = m["groups"]["name"]

                    st.sidebar.subheader(group_name)

                    with st.sidebar.expander("Members"):
                        try:
                            members = get_group_members(group_id)
                            for member in members:
                                st.write(f"- {member['profiles']['username']}")
                        except Exception as e:
                            st.error(f"error loading members: {e}")

                        if st.sidebar.button(f"🚪 Leave {group_name}", key=f"leave_{group_id}"):
                            try:
                                supabase.table("group_members") \
                                    .delete() \
                                    .eq("group_id", group_id) \
                                    .eq("user_id", st.session_state.current_user_id) \
                                    .execute()

                                remaining = supabase.table("group_members") \
                                    .select("user_id") \
                                    .eq("group_id", group_id) \
                                    .execute()

                                if not remaining.data:
                                    supabase.table("groups").delete().eq("id", group_id).execute()
                                    st.sidebar.success(f"Left {group_name} — group deleted (no members left)")
                                else:
                                    st.sidebar.success(f"Left {group_name}")

                                clear_membership_caches()
                                st.rerun()
                            except Exception as e:
                                st.sidebar.error(f"error: {e}")
            else:
                st.sidebar.write("You're not in any groups yet")
        except Exception as e:
            st.error(f"error loading groups: {e}")

    groups_sidebar()

    # ---------- sidebar: compare-on-graph checkboxes ----------
    st.sidebar.subheader("Compare on graph")
    selected_group_ids = []

    try:
        my_memberships_for_view = get_my_memberships(st.session_state.current_user_id)

        if my_memberships_for_view:
            for m in my_memberships_for_view:
                group_id = m["group_id"]
                group_name = m["groups"]["name"]
                is_checked_default = group_id in (st.session_state.compare_group_ids or [])
                if st.sidebar.checkbox(group_name, key=f"compare_{group_id}", value=is_checked_default):
                    selected_group_ids.append(group_id)
        else:
            st.sidebar.write("Join a group to compare data")

        # persist selection if it changed since last known state
        if set(selected_group_ids) != set(st.session_state.compare_group_ids or []):
            try:
                supabase.table("profiles").update({
                    "compare_groups": selected_group_ids
                }).eq("id", st.session_state.current_user_id).execute()
                st.session_state.compare_group_ids = selected_group_ids
            except Exception as e:
                st.sidebar.error(f"error saving comparison: {e}")

    except Exception as e:
        st.sidebar.error(f"error: {e}")

    # ---------- sidebar: create a group ----------
    with st.sidebar.expander("Create a group"):
        new_group_name = st.text_input("Group name", key="new_group_name")
        is_private = st.checkbox("Make this group private")
        group_pass = None
        if is_private:
            group_pass = st.text_input("Set a group password", type="password", key="new_group_password")

        if st.button("Create group", key="create_group_btn"):
            try:
                hashed_pass = None
                if is_private:
                    hashed_pass = bcrypt.hashpw(group_pass.encode(), bcrypt.gensalt()).decode()

                group_response = supabase.table("groups").insert({
                    "name": new_group_name,
                    "created_by": st.session_state.current_user_id,
                    "is_private": is_private,
                    "group_password": hashed_pass
                }).execute()

                new_group_id = group_response.data[0]["id"]

                supabase.table("group_members").insert({
                    "group_id": new_group_id,
                    "user_id": st.session_state.current_user_id
                }).execute()

                clear_membership_caches()
                st.sidebar.success(f"Created group: {new_group_name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"error: {e}")

    # ---------- sidebar: find a group ----------
    with st.sidebar.expander("Find a group"):
        search_term = st.text_input("search group name", key="group_search")

        try:
            if search_term:
                search_results = supabase.table("groups")\
                    .select("id, name, is_private")\
                    .ilike("name", f"%{search_term}%")\
                    .execute()
            else:
                search_results = supabase.table("groups")\
                    .select("id, name, is_private, group_members(count)")\
                    .eq("is_private", False)\
                    .execute()

                sorted_groups = sorted(
                    search_results.data,
                    key=lambda g: g["group_members"][0]["count"] if g["group_members"] else 0,
                    reverse=True
                )
                search_results.data = sorted_groups[:5]

            # groups user is already in, so they can't join again
            my_groups = supabase.table("group_members")\
                .select("group_id")\
                .eq("user_id", st.session_state.current_user_id)\
                .execute()
            my_groups_id = {m["group_id"] for m in my_groups.data}

            if search_results.data:
                for g in search_results.data:
                    lock = " 🔒" if g["is_private"] else ""
                    st.write(f"{lock} {g['name']}")

                    if g["id"] in my_groups_id:
                        st.caption("already a member")
                        continue

                    join_key = f"join_{g['id']}"

                    if g["is_private"]:
                        entered_pass = st.text_input(
                            "Enter group password", type="password", key=f"pass_{g['id']}"
                        )
                    else:
                        entered_pass = None

                    if st.button("Join", key=join_key):
                        try:
                            if g["is_private"]:
                                group_row = supabase.table("groups")\
                                    .select("group_password")\
                                    .eq("id", g["id"])\
                                    .single()\
                                    .execute()

                                stored_hash = group_row.data["group_password"]

                                if not entered_pass or not bcrypt.checkpw(entered_pass.encode(), stored_hash.encode()):
                                    st.error("Incorrect password")
                                    st.stop()

                            supabase.table("group_members").insert({
                                "group_id": g["id"],
                                "user_id": st.session_state.current_user_id
                            }).execute()
                            clear_membership_caches()
                            st.success(f"Joined {g['name']}!")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"error: {e}")
            else:
                st.write("No matching groups found")

        except Exception as e:
            st.error(f"error: {e}")

    # ---------- top bar: title + profile ----------
    with st.container(key="top_bar_container"):
        top_left, top_right = st.columns([5, 1], vertical_alignment="center")
        with top_left:
            st.markdown(
                "<h1 style='font-family: Georgia, serif; font-size: clamp(22px, 7vw, 64px); "
                "font-weight: 700; color: #00008B; letter-spacing: 1px; margin: 0; white-space: nowrap;'>Swimmer</h1>",
                unsafe_allow_html=True
            )

        with top_right:
            with st.container(horizontal_alignment="center"):
                profile_pic_path = get_prof_pic(st.session_state.current_user_id)
                appealing_prof = make_circular(profile_pic_path, border_color=st.session_state.line_colour or "#000000", padding=10)
                st.session_state.profile_pic = appealing_prof
                img_b64 = img_to_base64(appealing_prof)

                st.markdown(f"""
                    <style>
                    div[class*="st-key-pfp_btn"] button {{
                        background-image: url("data:image/png;base64,{img_b64}");
                        background-size: cover;
                        background-position: center;
                        box-sizing: border-box !important;
                        width: 60px !important;
                        height: 60px !important;
                        min-width: 60px !important;
                        max-width: 60px !important;
                        padding: 0 !important;
                        border: none;
                        border-radius: 50%;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                        margin: 0 !important;
                        display: block;
                        transition: transform 0.1s ease;
                    }}
                    div[class*="st-key-pfp_btn"] button:hover {{
                        transform: scale(1.06);
                        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
                    }}
                    div[class*="st-key-pfp_btn"] button p {{
                        display: none !important;
                    }}
                    </style>
                """, unsafe_allow_html=True)

                if st.button("pfp", key="pfp_btn"):
                    st.session_state.show_settings = True
                    st.rerun()

                st.markdown(
                    f"<div style='text-align: center; font-weight: bold; white-space: nowrap;'>{st.session_state.current_user}</div>",
                    unsafe_allow_html=True
                )

    # ---------- graph + logging (fragment: reruns on its own) ----------
    render_swim_section()


# ============================================================
# SETTINGS PAGE
# ============================================================
def settings_page():
    st.header("Settings")
    with st.container(horizontal_alignment="center"):
        current_pic_url = get_prof_pic(st.session_state.current_user_id)
        fresh_prof_pic = make_circular(current_pic_url, border_color=st.session_state.line_colour or "#000000", padding=10)
        st.image(fresh_prof_pic)
        st.markdown(
            f"<div style='text-align: center; font-weight: bold; font-size: 28px;'>{st.session_state.current_user}</div>",
            unsafe_allow_html=True
        )

    st.subheader("Change looks")
    uploaded = st.file_uploader("Upload new profile picture", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        if st.button("Upload picture"):
            try:
                file_bytes = uploaded.read()
                unique_filename = f"profile_{int(time.time())}.png"
                file_path = f"{st.session_state.current_user_id}/{unique_filename}"

                supabase.storage.from_("avatars").upload(
                    file_path, file_bytes, {"content-type": "image/png"}
                )

                supabase.table("profiles").update({
                    "avatar_path": file_path
                }).eq("id", st.session_state.current_user_id).execute()

                get_prof_pic.clear()
                make_circular.clear()
                st.success("Profile picture updated!")
                st.rerun()
            except Exception as e:
                st.error(f"upload failed: {e}")

    if st.button("Change line colour"):
        st.session_state.show_lc_change = True

    if st.session_state.show_lc_change:
        col5, col6 = st.columns([1, 1])
        with col5:
            options = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]
            current_index = options.index(st.session_state.line_colour) if st.session_state.line_colour in options else None
            selected_colour = st.selectbox(label="Select colour", options=options, index=current_index)
        with col6:
            hex_input = str(st.text_input("Or enter 6 digit hex code here"))
        if st.button("submit"):
            a4 = hex_input if hex_input else selected_colour
            try:
                supabase.table("profiles").update({
                    "line_colour": a4
                }).eq("id", st.session_state.current_user_id).execute()
                st.session_state.line_colour = a4
                st.session_state.show_lc_change = False
                st.rerun()
            except Exception as e:
                st.error(f"error: {e}")

    st.subheader("Change details")
    if st.button("Change display name"):
        st.session_state.show_change_username = True

    if st.session_state.show_change_username:
        new_user = st.text_input("Enter new display name")
        if st.button("Confirm"):
            try:
                supabase.table("profiles").update({
                    "username": new_user
                }).eq("id", st.session_state.current_user_id).execute()
                st.session_state.current_user = new_user
                st.session_state.show_change_username = False
                st.success("Display name updated!")
                st.rerun()
            except Exception as e:
                st.error(f"error: {e}")

    if st.button("Change password"):
        st.session_state.show_change_password = True

    if st.session_state.show_change_password:
        new_pass = st.text_input("New password", type="password")
        if st.button("Confirm"):
            try:
                supabase.auth.update_user({"password": new_pass})
                st.session_state.show_change_password = False
                st.success("Password updated!")
                st.rerun()
            except Exception as e:
                st.error(f"error: {e}")
    
    if st.button(":red[Delete my account]", key="delete_account_btn"):
        st.session_state.confirm_delete = True

    if st.session_state.get("confirm_delete"):
        st.warning("This is permanent and cannot be undone.")
        if st.button("Yes, permanently delete my account"):
            try:
                supabase.rpc("delete_own_account").execute()
                supabase.auth.sign_out()
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.success("Account deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"error: {e}")

    if st.sidebar.button("←back to home", width = "stretch"):
        st.session_state.show_settings = False
        st.rerun()


# ============================================================
# LOGIN
# ============================================================
def check_login(email, psk):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": psk
        })
        user_id = response.user.id

        profile = supabase.table("profiles").select("username, line_colour, compare_groups").eq("id", user_id).execute()

        if not profile.data:
            pending_username = response.user.user_metadata.get("pending_username", "New Swimmer")
            supabase.table("profiles").insert({
                "id": user_id,
                "username": pending_username
            }).execute()
            username = pending_username
            line_colour = None
            compare_groups = []
        else:
            username = profile.data[0]["username"]
            line_colour = profile.data[0]["line_colour"]
            compare_groups = profile.data[0].get("compare_groups") or []

        return "success", username, user_id, line_colour, compare_groups
    except Exception as e:
        return f"error: {e}", None, None, None, None


if not st.session_state.logged_in:
    st.header("Login")
    email = st.text_input("Email")
    psk = st.text_input("Password", type="password")

    if st.button("Log in"):
            result, username, user_id, line_colour, compare_groups = check_login(email, psk)
            if result == "success":
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.current_user_id = user_id
                st.session_state.line_colour = line_colour
                st.session_state.compare_group_ids = compare_groups
                st.session_state.login_status = None
                st.rerun()
            else:
                st.session_state.login_status = result

    if st.session_state.login_status:
        st.error(st.session_state.login_status)
        if st.button("Register new account?"):
            st.session_state.show_register = True

    if st.session_state.show_register:
        create_new()

else:
    main()

    if st.sidebar.button("Log out", width="stretch"):
        supabase.auth.sign_out()
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()