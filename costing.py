import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Medical Drone Cost Calculator – Flight Hour Model", layout="wide")

# ---------- Helpers ----------
def fmt_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return "—"

def battery_wear_per_flight(pack_cost, usable_kwh, cycle_life, kwh_per_flight):
    if usable_kwh <= 0 or cycle_life <= 0:
        return 0.0
    return (pack_cost / (usable_kwh * cycle_life)) * kwh_per_flight

def diesel_rate_per_kwh(diesel_price_per_liter, sfc_l_per_kwh):
    return diesel_price_per_liter * sfc_l_per_kwh

def lognormal_mu_for_mean(mean_minutes, sigma):
    if mean_minutes <= 0:
        mean_minutes = 1e-6
    return math.log(mean_minutes) - 0.5*(sigma**2)

def interpolate_return_mean(n_stations, mean_low=56.0, mean_high=22.0, low_threshold=5, high_threshold=25):
    if n_stations < low_threshold:
        return mean_low
    if n_stations > high_threshold:
        return mean_high
    span = high_threshold - low_threshold
    frac = (n_stations - low_threshold) / span
    return mean_low + frac * (mean_high - mean_low)

def expected_distance_km_from_time(mean_minutes, kmh):
    return kmh * (mean_minutes / 60.0)

def cap_flight_time(time_minutes, max_time=120.0):
    return min(time_minutes, max_time)

# ---------- Sidebar Inputs ----------
st.sidebar.title("Inputs")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Base Case", "Monte Carlo", "Sensitivity"], index=0)

with st.sidebar.expander("Mission & Energy Model", expanded=True):
    mode = st.radio("Energy basis", ["Distance-based", "Time-distribution-based"], index=1,
                    help="Use fixed distance OR sample flight times from a long-tailed distribution (lognormal).")
    cruise_speed_kmh = st.number_input("Cruise speed (km/h)", 20.0, 200.0, 90.0, step=5.0)
    wh_per_km = st.number_input("Cruise draw (Wh/km)", 1.0, 50.0, 5.0, step=0.5)
    vtol_overhead_wh = st.number_input("VTOL + reserves overhead (Wh/flight)", 0, 5000, 200, step=10)
    
    max_flight_time = st.number_input("Maximum flight time (minutes)", 60, 240, 120, step=5,
                                      help="Operational limit for total flight time (safety/battery constraint)")

    if mode == "Distance-based":
        distance_km_rt = st.number_input("Round-trip distance (km)", 1.0, 10000.0, 160.0, step=5.0)
        kwh_per_flight = (distance_km_rt * wh_per_km + vtol_overhead_wh) / 1000.0
        st.caption(f"Energy/flight: **{kwh_per_flight:.3f} kWh** (distance-based)")
    else:
        st.markdown("**Time distribution settings (lognormal, long-tailed)**")
        mean_out_min = st.number_input("Mean one-way time (minutes)", 1.0, 240.0, 23.0, step=1.0)
        n_stations = st.number_input("Number of base stations", 1, 100, 5, step=1)
        mean_ret_min = interpolate_return_mean(n_stations, mean_low=56.0, mean_high=22.0, low_threshold=5, high_threshold=25)
        st.caption(f"Return time mean set to **{mean_ret_min:.1f} min** based on **{n_stations}** stations")
        st.caption(f"⚠️ **Flight time capped at {max_flight_time} minutes**")
        sigma_out = st.slider("Skew (sigma) outbound", 0.2, 1.5, 0.8, step=0.05)
        sigma_ret = st.slider("Skew (sigma) return", 0.2, 1.5, 0.8, step=0.05)
        mu_out = lognormal_mu_for_mean(mean_out_min, sigma_out)
        mu_ret = lognormal_mu_for_mean(mean_ret_min, sigma_ret)

with st.sidebar.expander("Throughput & Scaling", expanded=True):
    monthly_flights_required = st.number_input("Flights required per month (total demand)", 0, 100000, 60, step=5)
    
    st.markdown("**UAS Market Share (% of blood transported by drone)**")
    use_market_share = st.checkbox("Apply UAS market share reduction", value=True,
                                   help="Not all blood is transported by UAS - ground transport handles majority")
    
    if use_market_share:
        market_share_low = st.number_input("Market share - Low (%)", 1, 100, 10, step=1) / 100
        market_share_high = st.number_input("Market share - High (%)", 1, 100, 20, step=1) / 100
        market_share_mode = st.number_input("Market share - Most likely (%)", 1, 100, 15, step=1) / 100
        
        effective_flights_low = int(monthly_flights_required * market_share_low)
        effective_flights_high = int(monthly_flights_required * market_share_high)
        effective_flights_mode = int(monthly_flights_required * market_share_mode)
        
        st.caption(f"""
        📊 **Effective monthly flights (with market share):**
        - Low (10%): {effective_flights_low} flights/month
        - Mode (15%): {effective_flights_mode} flights/month  
        - High (20%): {effective_flights_high} flights/month
        - **Base case uses mode value**
        """)
    else:
        market_share_low = 1.0
        market_share_high = 1.0
        market_share_mode = 1.0
    
    operating_days_per_month = st.number_input("Operating days per month", 1, 31, 26, step=1)
    ops_hours_per_day = st.number_input("Operational hours per day", 1, 24, 10, step=1)
    cycle_time_hours = st.number_input("Avg cycle time (hrs)", 0.25, 24.0, 2.0, step=0.25)

with st.sidebar.expander("Power Source", expanded=True):
    power_choice = st.selectbox("Select power", ["Grid", "Diesel generator"])
    base_grid_rate = st.number_input("Grid electricity price ($/kWh)", 0.00, 5.00, 0.10, step=0.01)
    diesel_price_per_liter = st.number_input("Diesel price ($/L)", 0.00, 10.00, 1.10, step=0.05)
    sfc_l_per_kwh = st.number_input("Generator SFC (L/kWh)", 0.05, 1.00, 0.30, step=0.01)
    if power_choice == "Grid":
        effective_rate = base_grid_rate
    else:
        effective_rate = diesel_rate_per_kwh(diesel_price_per_liter, sfc_l_per_kwh)

with st.sidebar.expander("Capital (USD)", expanded=False):
    st.caption("Items marked *scale per aircraft*.")
    airframe = st.number_input("Airframe (Trinity F90+)*", 0.0, 1e6, 21500.0, step=50.0)
    
    st.markdown("**Airframe Lifespan (Flight Hours)**")
    expected_airframe_hours = st.number_input("Expected airframe hours", 500, 5000, 1500, step=100,
                                              help="Total flight hours before airframe retirement (typical: 1,000-2,000)")
    use_flight_hour_amort = st.checkbox("Use flight-hour-based amortization", value=True,
                                        help="Amortize based on actual flight hours vs. calendar time")
    
    batteries = st.number_input("Batteries (2 extra packs)*", 0.0, 1e6, 2400.0, step=50.0)
    charging = st.number_input("Charging & power*", 0.0, 1e6, 900.0, step=50.0)
    comms = st.number_input("BVLOS comms*", 0.0, 1e6, 1000.0, step=50.0)
    antennas = st.number_input("Antennas/mast/RC spares*", 0.0, 1e6, 900.0, step=50.0)
    tools = st.number_input("Tools/spares*", 0.0, 1e6, 800.0, step=50.0)
    laptop = st.number_input("Rugged GCS laptop (site-level)", 0.0, 1e6, 2500.0, step=50.0)
    fridge = st.number_input("Base fridge (site-level)", 0.0, 1e6, 1000.0, step=50.0)
    cold_chain = st.number_input("Cold box + PCM*", 0.0, 1e6, 650.0, step=50.0)
    scale_laptop = st.checkbox("Scale laptop per aircraft", value=False)
    scale_fridge = st.checkbox("Scale fridge per aircraft", value=False)

with st.sidebar.expander("Per-flight & Battery", expanded=True):
    battery_pack_cost = st.number_input("Battery pack cost ($)", 0.0, 1e6, 1095.0, step=5.0)
    usable_kwh = st.number_input("Usable capacity (kWh)", 0.1, 5.0, 0.8, step=0.05)
    cycle_life = st.number_input("Cycle life (charges)", 1, 5000, 400, step=10)
    cold_chain_consumables = st.number_input("Cold chain consumables ($/flight)", 0.0, 1000.0, 1.00, step=0.10)
    wear_parts = st.number_input("Wear parts ($/flight)", 0.0, 1000.0, 0.50, step=0.10)
    data_per_flight = st.number_input("Telemetry data ($/flight)", 0.0, 1000.0, 0.10, step=0.05)

with st.sidebar.expander("Monthly Ops & Staffing", expanded=False):
    cellular_data = st.number_input("Cellular data ($/month)", 0.0, 1e6, 50.0, step=5.0)
    cellular_scales = st.checkbox("Cellular scales with aircraft", value=True)
    insurance_annual = st.number_input("Annual insurance ($/year)", 0.0, 1e6, 3000.0, step=50.0)
    software_subs = st.number_input("Software subs ($/mo)", 0.0, 1e6, 50.0, step=5.0)
    facility_rent = st.number_input("Facility rent ($/mo)", 0.0, 1e6, 100.0, step=10.0)
    pilot_salary_per_fte = st.number_input("Pilot salary per FTE ($/mo)", 0.0, 1e6, 800.0, step=10.0)
    pilot_fte_per_ac = st.number_input("Pilot FTE per aircraft", 0.0, 50.0, 4.8, step=0.1)
    tech_salary_per_fte = st.number_input("Tech salary per FTE ($/mo)", 0.0, 1e6, 400.0, step=10.0)
    tech_fte_per_ac = st.number_input("Tech FTE per aircraft", 0.0, 50.0, 4.8, step=0.1)
    cold_chain_salary_per_fte = st.number_input("CC salary per FTE ($/mo)", 0.0, 1e6, 300.0, step=10.0)
    cold_chain_fte_per_ac = st.number_input("CC FTE per aircraft", 0.0, 50.0, 4.8, step=0.1)
    site_overhead_staff_cost = st.number_input("Site overhead staff ($/mo)", 0.0, 1e6, 0.0, step=10.0)

# ---------- Scaling & Base Calculations ----------
# Calculate flight hours per mission
if mode == "Distance-based":
    flight_time_hours = (distance_km_rt / cruise_speed_kmh)
else:
    exp_out_min = np.exp(mu_out + 0.5*(sigma_out**2))
    exp_ret_min = np.exp(mu_ret + 0.5*(sigma_ret**2))
    exp_total_min = exp_out_min + exp_ret_min
    exp_total_min = cap_flight_time(exp_total_min, max_flight_time)
    flight_time_hours = exp_total_min / 60.0

# Calculate effective flights with market share
effective_monthly_flights = int(monthly_flights_required * market_share_mode)

# Calculate annual flight hours per aircraft
annual_flight_hours_per_aircraft = effective_monthly_flights * 12 * flight_time_hours

# Determine aircraft needed per station based on flight hours
if use_flight_hour_amort and annual_flight_hours_per_aircraft > expected_airframe_hours:
    aircraft_per_station_from_hours = math.ceil(annual_flight_hours_per_aircraft / expected_airframe_hours)
else:
    aircraft_per_station_from_hours = 1

# Compare with capacity-based aircraft calculation
capacity_per_aircraft_per_day = math.floor(ops_hours_per_day / cycle_time_hours) if cycle_time_hours > 0 else 0
capacity_per_aircraft_per_month = capacity_per_aircraft_per_day * operating_days_per_month
aircraft_from_capacity = math.ceil(effective_monthly_flights / max(1, capacity_per_aircraft_per_month)) if effective_monthly_flights > 0 else 0

# Use the maximum of the two
aircraft_needed = max(aircraft_from_capacity, aircraft_per_station_from_hours)

include_spare = st.sidebar.checkbox("Add +1 spare aircraft", value=True)
aircraft_total = aircraft_needed + (1 if include_spare else 0)

# Calculate airframe lifespan and replacement schedule
if annual_flight_hours_per_aircraft > 0:
    airframe_lifespan_years = expected_airframe_hours / annual_flight_hours_per_aircraft
    airframe_lifespan_months = airframe_lifespan_years * 12
else:
    airframe_lifespan_years = 10
    airframe_lifespan_months = 120

# Per-aircraft capital (excluding airframe for now)
per_aircraft_capex_no_airframe = batteries + charging + comms + antennas + tools + cold_chain
site_capex = (laptop * (aircraft_total if scale_laptop else 1)) + (fridge * (aircraft_total if scale_fridge else 1))

# Airframe amortization
if use_flight_hour_amort:
    actual_amort_months = max(6, min(60, airframe_lifespan_months))
    airframe_monthly_amort = (airframe * max(1, aircraft_total)) / actual_amort_months
else:
    actual_amort_months = 36
    airframe_monthly_amort = (airframe * max(1, aircraft_total)) / actual_amort_months

# Other equipment amortization (still use 36 months)
other_equipment_capex = per_aircraft_capex_no_airframe * max(1, aircraft_total) + site_capex
other_equipment_monthly_amort = other_equipment_capex / 36

# Total capital and amortization
total_capital = (airframe * max(1, aircraft_total)) + other_equipment_capex
monthly_capital_amort = airframe_monthly_amort + other_equipment_monthly_amort

# Energy per flight (expected) for Base Case display
if mode == "Distance-based":
    expected_kwh_per_flight = (distance_km_rt * wh_per_km + vtol_overhead_wh) / 1000.0
    expected_distance_km = distance_km_rt
else:
    expected_distance_km = cruise_speed_kmh * (exp_total_min / 60.0)
    expected_kwh_per_flight = (expected_distance_km * wh_per_km + vtol_overhead_wh) / 1000.0

battery_degradation = battery_wear_per_flight(battery_pack_cost, usable_kwh, cycle_life, expected_kwh_per_flight)
electricity_per_flight = expected_kwh_per_flight * effective_rate
per_flight_cost = battery_degradation + electricity_per_flight + cold_chain_consumables + wear_parts + data_per_flight

cellular_total = cellular_data * (aircraft_total if cellular_scales else 1)
base_station_power_cost = effective_rate * 18.25
monthly_insurance = insurance_annual / 12.0
monthly_fixed_ops = cellular_total + base_station_power_cost + facility_rent + monthly_insurance + software_subs

monthly_variable_cost = per_flight_cost * effective_monthly_flights

monthly_staff = (pilot_salary_per_fte * pilot_fte_per_ac + tech_salary_per_fte * tech_fte_per_ac + 
                cold_chain_salary_per_fte * cold_chain_fte_per_ac) * max(1, aircraft_total)
monthly_staff += site_overhead_staff_cost

total_monthly = monthly_fixed_ops + monthly_variable_cost + monthly_staff + monthly_capital_amort
annual_total = total_monthly * 12.0

ops_cost_per_flight = (monthly_fixed_ops + monthly_variable_cost + monthly_staff) / effective_monthly_flights if effective_monthly_flights > 0 else 0.0
total_cost_per_flight = total_monthly / effective_monthly_flights if effective_monthly_flights > 0 else 0.0
cost_per_km = (total_cost_per_flight / expected_distance_km) if expected_distance_km > 0 else 0.0

# ---------- Header & KPIs ----------
st.title("🚁 Medical Drone Cost Calculator – Flight Hour Model")
st.subheader("Time distributions with flight-hour-based aircraft replacement")

if use_flight_hour_amort and aircraft_per_station_from_hours > 1:
    st.warning(f"""
    ⚠️ **High Utilization Alert:** 
    - Annual flight hours per aircraft: {annual_flight_hours_per_aircraft:,.0f} hours
    - Airframe lifespan: {expected_airframe_hours:,.0f} hours
    - **{aircraft_per_station_from_hours} aircraft required per station** for fleet rotation
    - Airframe replacement every {airframe_lifespan_months:.1f} months
    """)

kpi_cols = st.columns(6)
kpi_cols[0].metric("Aircraft total", f"{aircraft_total}", 
                   delta=f"{aircraft_per_station_from_hours} per station" if aircraft_per_station_from_hours > 1 else None)
kpi_cols[1].metric("Effective flights/mo", f"{effective_monthly_flights}" + (f" ({market_share_mode*100:.0f}%)" if use_market_share else ""))
kpi_cols[2].metric("Per‑flight (ops)", fmt_currency(ops_cost_per_flight))
kpi_cols[3].metric("Per‑flight (total)", fmt_currency(total_cost_per_flight))
kpi_cols[4].metric("Per‑km (total)", fmt_currency(cost_per_km))
kpi_cols[5].metric("Monthly total", fmt_currency(total_monthly))

caption_text = f"Power rate: **${effective_rate:.2f}/kWh**  •  Model: **{mode}**  •  Cruise: **{cruise_speed_kmh:.0f} km/h**"
if use_market_share:
    caption_text += f"  •  UAS share: **{market_share_mode*100:.0f}%**"
if use_flight_hour_amort:
    caption_text += f"  •  Airframe life: **{airframe_lifespan_months:.1f} months**"
st.caption(caption_text)

st.markdown("---")

# ---------- Display based on mode ----------
if analysis_mode == "Base Case":
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("### Per‑flight breakdown (expected)")
        pf = pd.DataFrame({
            "Component": ["Battery wear", "Electricity", "Cold chain", "Wear parts", "Data"],
            "USD/flight": [battery_degradation, electricity_per_flight, cold_chain_consumables, wear_parts, data_per_flight],
        })
        st.dataframe(pf, use_container_width=True)

        st.markdown("### Monthly breakdown")
        mo = pd.DataFrame({
            "Category": ["Fixed ops", "Variable ops", "Staff", "Capital amortization"],
            "USD/month": [monthly_fixed_ops, monthly_variable_cost, monthly_staff, monthly_capital_amort],
        })
        st.dataframe(mo, use_container_width=True)

    with right:
        st.markdown("### Capital summary")
        
        if use_flight_hour_amort:
            cap = pd.DataFrame({
                "Item": ["Airframe (flight-hour based)", "Other equipment × N", "Site‑level"],
                "USD": [
                    airframe * max(1, aircraft_total),
                    per_aircraft_capex_no_airframe * max(1, aircraft_total),
                    site_capex
                ],
            })
        else:
            cap = pd.DataFrame({
                "Item": ["Per‑aircraft × N", "Site‑level"],
                "USD": [
                    (airframe + per_aircraft_capex_no_airframe) * max(1, aircraft_total), 
                    site_capex
                ],
            })
        
        st.dataframe(cap, use_container_width=True)
        st.write("**Total capital (scaled):**", fmt_currency(total_capital))
        
        if use_flight_hour_amort:
            st.markdown("**Amortization Details:**")
            st.caption(f"• Airframe: {airframe_lifespan_months:.1f} months (flight-hour based)")
            st.caption(f"• Other equipment: 36 months (calendar-based)")
            st.caption(f"• Monthly airframe amort: {fmt_currency(airframe_monthly_amort)}")
            st.caption(f"• Monthly other amort: {fmt_currency(other_equipment_monthly_amort)}")
            st.caption(f"• **Total monthly amort: {fmt_currency(monthly_capital_amort)}**")
            
            if aircraft_per_station_from_hours > 1:
                st.info(f"""
                **Fleet Rotation:**
                - {aircraft_per_station_from_hours} aircraft needed for continuous ops
                - Each aircraft flies ~{expected_airframe_hours:,.0f} hours
                - Replace {aircraft_per_station_from_hours / airframe_lifespan_years:.1f} aircraft/year
                """)
    
    if use_flight_hour_amort:
        st.markdown("---")
        st.markdown("### ✈️ Flight Hour Utilization Analysis")
        
        util_col1, util_col2, util_col3 = st.columns(3)
        
        with util_col1:
            st.metric("Flight hours/mission", f"{flight_time_hours:.2f} hours")
            st.metric("Annual hours/aircraft", f"{annual_flight_hours_per_aircraft:,.0f} hours")
        
        with util_col2:
            st.metric("Airframe lifespan", f"{airframe_lifespan_years:.2f} years")
            utilization_pct = (annual_flight_hours_per_aircraft / expected_airframe_hours) * 100
            st.metric("Annual utilization", f"{utilization_pct:.1f}%", 
                     delta="Optimal: 60-80%" if utilization_pct < 100 else "⚠️ Over-utilized")
        
        with util_col3:
            aircraft_replacements_per_year = max(1, aircraft_total) / airframe_lifespan_years
            st.metric("Aircraft replacements/year", f"{aircraft_replacements_per_year:.2f}")
            annual_airframe_cost = aircraft_replacements_per_year * airframe
            st.metric("Annual airframe cost", fmt_currency(annual_airframe_cost))
        
        st.markdown("---")
        st.markdown("### 📊 Amortization Comparison")
        
        calendar_based_amort = (airframe * max(1, aircraft_total)) / 36
        flight_hour_based_amort = airframe_monthly_amort
        
        comparison_df = pd.DataFrame({
            'Method': ['Calendar-based (36 months)', 'Flight-hour-based (actual)'],
            'Airframe Amort ($/month)': [calendar_based_amort, flight_hour_based_amort],
            'Implied Lifespan (months)': [36, airframe_lifespan_months],
            'Cost per Flight': [
                (calendar_based_amort + other_equipment_monthly_amort + monthly_fixed_ops + monthly_variable_cost + monthly_staff) / effective_monthly_flights if effective_monthly_flights > 0 else 0,
                total_cost_per_flight
            ]
        })
        
        comparison_df['Airframe Amort ($/month)'] = comparison_df['Airframe Amort ($/month)'].apply(fmt_currency)
        comparison_df['Implied Lifespan (months)'] = comparison_df['Implied Lifespan (months)'].round(1)
        comparison_df['Cost per Flight'] = comparison_df['Cost per Flight'].apply(fmt_currency)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        amort_diff_pct = ((flight_hour_based_amort - calendar_based_amort) / calendar_based_amort) * 100 if calendar_based_amort > 0 else 0
        
        if amort_diff_pct > 10:
            st.error(f"""
            🚨 **Critical Finding:** Flight-hour-based amortization is **{amort_diff_pct:+.1f}%** higher than calendar-based!
            - Airframes are wearing out faster than traditional 3-year depreciation assumes
            - High utilization requires more frequent aircraft replacement
            - Budget should reflect actual flight hour consumption, not just calendar time
            """)
        elif amort_diff_pct < -10:
            st.success(f"""
            ✅ **Good News:** Flight-hour-based amortization is **{amort_diff_pct:.1f}%** lower than calendar-based.
            - Low utilization means airframes last longer in calendar time
            - Could extend depreciation period or reduce aircraft fleet size
            """)
        else:
            st.info("ℹ️ Flight-hour and calendar-based amortization are roughly aligned.")

    if mode == "Time-distribution-based":
        st.markdown("---")
        st.markdown("### 📊 Flight Time Distribution Preview")
        
        n_samples = 5000
        rng = np.random.default_rng(seed=42)
        outbound_times = rng.lognormal(mean=mu_out, sigma=sigma_out, size=n_samples)
        return_times = rng.lognormal(mean=mu_ret, sigma=sigma_ret, size=n_samples)
        total_times = outbound_times + return_times
        
        total_times_capped = np.array([cap_flight_time(t, max_flight_time) for t in total_times])
        n_capped = np.sum(total_times > max_flight_time)
        pct_capped = (n_capped / n_samples) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time Statistics (minutes)**")
            stats_df = pd.DataFrame({
                'Leg': ['Outbound', 'Return', 'Total (uncapped)', 'Total (capped)'],
                'Mean': [np.mean(outbound_times), np.mean(return_times), np.mean(total_times), np.mean(total_times_capped)],
                'Median': [np.median(outbound_times), np.median(return_times), np.median(total_times), np.median(total_times_capped)],
                'P95': [np.percentile(outbound_times, 95), np.percentile(return_times, 95), 
                       np.percentile(total_times, 95), np.percentile(total_times_capped, 95)],
                'Max': [np.max(outbound_times), np.max(return_times), np.max(total_times), np.max(total_times_capped)]
            })
            for col in ['Mean', 'Median', 'P95', 'Max']:
                stats_df[col] = stats_df[col].round(1)
            st.dataframe(stats_df, use_container_width=True)
            
            if n_capped > 0:
                st.warning(f"⚠️ **{n_capped} flights ({pct_capped:.1f}%) exceed {max_flight_time} min cap**")
        
        with col2:
            st.markdown("**Distance Statistics (km)**")
            distances = cruise_speed_kmh * (total_times_capped / 60.0)
            dist_stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'P95', 'Max'],
                'Round-trip (km)': [
                    np.mean(distances),
                    np.median(distances),
                    np.percentile(distances, 95),
                    np.max(distances)
                ]
            })
            dist_stats_df['Round-trip (km)'] = dist_stats_df['Round-trip (km)'].round(1)
            st.dataframe(dist_stats_df, use_container_width=True)
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Histogram(x=total_times_capped, nbinsx=50, name='Total Flight Time', 
                                        marker_color='lightblue'))
        fig_time.add_vline(x=np.mean(total_times_capped), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {np.mean(total_times_capped):.1f} min")
        fig_time.add_vline(x=max_flight_time, line_dash="solid", line_color="darkred",
                          annotation_text=f"Cap: {max_flight_time} min", line_width=2)
        fig_time.add_vline(x=np.percentile(total_times_capped, 95), line_dash="dot", line_color="orange",
                          annotation_text=f"95th: {np.percentile(total_times_capped, 95):.1f} min")
        fig_time.update_layout(
            title=f"Distribution of Total Flight Times (Capped at {max_flight_time} min)",
            xaxis_title="Flight Time (minutes)",
            yaxis_title="Frequency",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.info(f"""
        **Model Settings:**
        - Outbound mean: {mean_out_min:.1f} min (σ={sigma_out})
        - Return mean: {mean_ret_min:.1f} min (σ={sigma_ret}) - adjusted for {n_stations} stations
        - Combined mean (capped): {np.mean(total_times_capped):.1f} min → {expected_distance_km:.1f} km RT
        - **Flight time cap: {max_flight_time} minutes** ({pct_capped:.1f}% of flights affected)
        - Long-tailed distribution captures variability in delivery distances and weather conditions
        """)

elif analysis_mode == "Monte Carlo":
    st.markdown("## 🎲 Monte Carlo Simulation (time‑driven)")
    st.caption("Samples from lognormal flight time distributions and stochastic UAS market share")
    
    with st.expander("⚙️ Configure Simulation", expanded=True):
        iters = st.number_input("Iterations", 1000, 100000, 20000, step=1000)
        if mode == "Time-distribution-based":
            info_text = f"""
            **Current Settings:**
            - Outbound: Lognormal(μ={mu_out:.2f}, σ={sigma_out})
            - Return: Lognormal(μ={mu_ret:.2f}, σ={sigma_ret})
            - Return time adjusts based on {n_stations} stations
            """
            if use_market_share:
                info_text += f"""
            - **UAS Market Share:** Triangular({market_share_low*100:.0f}%, {market_share_mode*100:.0f}%, {market_share_high*100:.0f}%)
            - Base demand: {monthly_flights_required} flights/month
            - Effective demand: {effective_flights_low}-{effective_flights_high} flights/month
            """
            st.info(info_text)

    if st.button("🚀 Run Monte Carlo", type="primary"):
        with st.spinner("Running simulation..."):
            rng = np.random.default_rng(seed=42)
            costs = []
            energies = []
            distances = []
            actual_flights_list = []

            if mode == "Time-distribution-based":
                mu_o, mu_r = mu_out, mu_ret
                s_o, s_r = sigma_out, sigma_ret

            for _ in range(int(iters)):
                if use_market_share:
                    sampled_share = rng.triangular(market_share_low, market_share_mode, market_share_high)
                    flights_this_iter = int(monthly_flights_required * sampled_share)
                    flights_this_iter = max(1, flights_this_iter)
                else:
                    flights_this_iter = monthly_flights_required
                
                actual_flights_list.append(flights_this_iter)
                
                if mode == "Distance-based":
                    kwh = (distance_km_rt * wh_per_km + vtol_overhead_wh) / 1000.0
                    dist_km = distance_km_rt
                else:
                    t_out = rng.lognormal(mean=mu_o, sigma=s_o)
                    t_ret = rng.lognormal(mean=mu_r, sigma=s_r)
                    t_total = cap_flight_time(t_out + t_ret, max_flight_time)
                    dist_km = cruise_speed_kmh * (t_total / 60.0)
                    kwh = (dist_km * wh_per_km + vtol_overhead_wh) / 1000.0

                c_batt = battery_wear_per_flight(battery_pack_cost, usable_kwh, cycle_life, kwh)
                c_elec = kwh * effective_rate
                c_pf = c_batt + c_elec + cold_chain_consumables + wear_parts + data_per_flight

                monthly_var_i = c_pf * flights_this_iter
                total_monthly_i = monthly_fixed_ops + monthly_var_i + monthly_staff + monthly_capital_amort
                cost_per_flight_i = total_monthly_i / flights_this_iter
                
                costs.append(cost_per_flight_i)
                energies.append(kwh)
                distances.append(dist_km)

            costs = np.array(costs)
            energies = np.array(energies)
            distances = np.array(distances)
            actual_flights_array = np.array(actual_flights_list)
            
            mean_c = np.mean(costs)
            med_c = np.median(costs)
            std_c = np.std(costs)
            ci_95 = np.percentile(costs, [2.5, 97.5])
            ci_90 = np.percentile(costs, [5, 95])

            st.markdown("### 📊 Results Summary")
            stat_cols = st.columns(5)
            stat_cols[0].metric("Mean", fmt_currency(mean_c))
            stat_cols[1].metric("Median", fmt_currency(med_c))
            stat_cols[2].metric("Std Dev", fmt_currency(std_c))
            stat_cols[3].metric("95% CI", f"{fmt_currency(ci_95[0])} - {fmt_currency(ci_95[1])}")
            stat_cols[4].metric("90% CI", f"{fmt_currency(ci_90[0])} - {fmt_currency(ci_90[1])}")

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Histogram(x=costs, nbinsx=60, marker_color='lightblue', name='Cost per Flight'))
            fig_cost.add_vline(x=mean_c, line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {fmt_currency(mean_c)}")
            fig_cost.add_vline(x=ci_95[0], line_dash="dot", line_color="orange",
                              annotation_text="95% CI Lower")
            fig_cost.add_vline(x=ci_95[1], line_dash="dot", line_color="orange",
                              annotation_text="95% CI Upper")
            fig_cost.update_layout(
                title="Distribution of Cost per Flight",
                xaxis_title="USD/flight",
                yaxis_title="Frequency",
                showlegend=False,
                height=380
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            st.markdown("### 📈 Percentile Analysis")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            perc_data = []
            for p in percentiles:
                perc_data.append({
                    'Percentile': f"{p}th",
                    'Cost/Flight': fmt_currency(np.percentile(costs, p)),
                    'Flights/Month': f"{np.percentile(actual_flights_array, p):.0f}",
                    'Energy (kWh)': f"{np.percentile(energies, p):.2f}",
                    'Distance (km)': f"{np.percentile(distances, p):.1f}"
                })
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if use_market_share:
                    fig_flights = go.Figure()
                    fig_flights.add_trace(go.Histogram(x=actual_flights_array, nbinsx=30, marker_color='lightcoral'))
                    fig_flights.update_layout(
                        title=f"Monthly Flights Distribution (UAS Share: {market_share_low*100:.0f}-{market_share_high*100:.0f}%)",
                        xaxis_title="Flights per Month",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig_flights, use_container_width=True)
                else:
                    fig_energy = go.Figure()
                    fig_energy.add_trace(go.Histogram(x=energies, nbinsx=50, marker_color='lightgreen'))
                    fig_energy.update_layout(
                        title="Energy Consumption Distribution",
                        xaxis_title="kWh per Flight",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig_energy, use_container_width=True)
            
            with col2:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=distances, nbinsx=50, marker_color='lightyellow'))
                fig_dist.update_layout(
                    title="Distance Distribution",
                    xaxis_title="Round-trip Distance (km)",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            insight_text = f"""
            **Key Insights:**
            - Mean cost per flight: {fmt_currency(mean_c)} (95% CI: {fmt_currency(ci_95[0])} - {fmt_currency(ci_95[1])})
            - Mean energy: {np.mean(energies):.2f} kWh (range: {np.min(energies):.2f} - {np.max(energies):.2f})
            - Mean distance: {np.mean(distances):.1f} km (range: {np.min(distances):.1f} - {np.max(distances):.1f})
            - Coefficient of variation: {(std_c/mean_c)*100:.1f}%
            """
            
            if use_market_share:
                insight_text += f"""
            - **UAS market share applied:** {market_share_low*100:.0f}-{market_share_high*100:.0f}% (mode: {market_share_mode*100:.0f}%)
            - Mean flights/month: {np.mean(actual_flights_array):.0f} (range: {np.min(actual_flights_array):.0f} - {np.max(actual_flights_array):.0f})
            - **Fixed costs spread over fewer flights** → higher per-flight cost
            """
            
            st.success(insight_text)

elif analysis_mode == "Sensitivity":
    st.markdown("### 🔍 Sensitivity Analysis")
    st.info("Sensitivity analysis works on expected values shown in Base Case. For stochastic effects including time variability, use Monte Carlo mode.")

st.markdown("---")
st.caption("Medical Drone Calculator v5.0 - Flight Hour Model with UAS Market Share")
